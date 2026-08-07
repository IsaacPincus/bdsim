#!/usr/bin/env bash
#
# One-command build: virtual environment, dependencies, C++ core, Python
# extension, tests.
#
#   ./setup.sh                  # everything, into ./.venv
#   ./setup.sh --native         # add -march=native (faster, not portable)
#   ./setup.sh --lapack         # use BLAS/LAPACK for the HI dense kernels
#   ./setup.sh --no-tests       # skip ctest
#   ./setup.sh --jobs 4         # limit parallel compile jobs
#
# If a virtual environment is already active, that one is used and none is
# created. Re-running is cheap: CMake rebuilds only what changed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
RUN_TESTS=1
CMAKE_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --native)   CMAKE_EXTRA+=(-DBDSIM_NATIVE=ON); shift ;;
    --lapack)   CMAKE_EXTRA+=(-DBDSIM_LAPACK=ON); shift ;;
    --no-tests) RUN_TESTS=0; shift ;;
    --jobs)     JOBS="$2"; shift 2 ;;
    # the header comment, up to the first line that is not a comment
    -h|--help)  sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# --- prerequisites ---------------------------------------------------------
say "checking prerequisites"
missing=()
command -v cmake   >/dev/null || missing+=(cmake)
command -v c++     >/dev/null || command -v g++ >/dev/null || missing+=(g++)
command -v python3 >/dev/null || missing+=(python3)
if [[ ${#missing[@]} -gt 0 ]]; then
  die "missing: ${missing[*]}
On Debian/Ubuntu/WSL:
    sudo apt update && sudo apt install -y build-essential cmake python3 python3-venv
On macOS (Homebrew):
    xcode-select --install && brew install cmake python"
fi
python3 - <<'PY' || die "Python 3.9 or newer is required"
import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
printf '  cmake   %s\n' "$(cmake --version | head -1 | awk '{print $3}')"
printf '  python  %s\n' "$(python3 --version | awk '{print $2}')"

# --- virtual environment ---------------------------------------------------
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  say "using the active virtual environment: $VIRTUAL_ENV"
  PY="$VIRTUAL_ENV/bin/python"
else
  if [[ ! -d .venv ]]; then
    say "creating .venv"
    python3 -m venv .venv || die "could not create a venv.
On Debian/Ubuntu this usually means: sudo apt install -y python3-venv"
  else
    say "reusing .venv"
  fi
  PY="$ROOT/.venv/bin/python"
fi

say "installing Python dependencies"
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install --quiet numpy h5py nanobind
# Editable install so `import bdsim` resolves from anywhere -- in scripts and in
# editors. It packages only the Python layer; the compiled part is built below.
"$PY" -m pip install --quiet -e .

# --- build -----------------------------------------------------------------
say "configuring"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBDSIM_PYTHON=ON \
  -DPython_EXECUTABLE="$PY" \
  "${CMAKE_EXTRA[@]}"

say "building (-j $JOBS)"
cmake --build build -j "$JOBS"

# nanobind_add_module writes the extension straight into python/bdsim/.
EXT="$(ls python/bdsim/_bdsim*.so python/bdsim/_bdsim*.pyd 2>/dev/null | head -1 || true)"
[[ -n "$EXT" ]] || die "the extension was not produced; check the build output above"
printf '  extension: %s\n' "$EXT"

# --- check -----------------------------------------------------------------
if [[ $RUN_TESTS -eq 1 ]]; then
  say "running the C++ tests"
  ctest --test-dir build --output-on-failure
fi

say "checking the Python package"
"$PY" - <<'PY'
import bdsim
phys = bdsim.PhysParams(); phys.number_of_beads = 10
sim = bdsim.SimParams()
sim.dt, sim.time_start, sim.time_end = 0.01, 0.0, 1.0
R = bdsim.gaussian_chain(phys.number_of_beads, seed=1)
out = bdsim.integrate(R, phys, sim, bdsim.Rng(1))
print(f"  integrated a {phys.number_of_beads}-bead chain to t=1, "
      f"Rg^2 = {bdsim.radius_of_gyration_sq(out):.4f}")
PY

say "done"
cat <<EOF

Activate the environment in new shells with:
    source ${VIRTUAL_ENV:-$ROOT/.venv}/bin/activate

Then try:
    python examples/demo.py
    python examples/all_options.py --audit
EOF

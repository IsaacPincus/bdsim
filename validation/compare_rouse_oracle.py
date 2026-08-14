"""Compare the C++ integrator with the Fortran core, at a matched tolerance.

Runs build/fortran_oracle (the vendored SingleChainBD physics, LAPACK only) and
the same case through the Python bindings, and reports the difference. Both are
run at the same implicit_loop_exit_tolerance, which is the whole point: the
recorded regression oracle was produced at 1e-6, where the corrector's
fixed-point iteration is still ~1e-8 short of the true solution, so the two
codes cannot agree better than that no matter how faithful the port is.
Tightening both sides together should drive the difference down with the
tolerance -- if it plateaus, something else really does differ.

    cmake -S . -B build -DBDSIM_FORTRAN=ON -DBDSIM_PYTHON=ON && cmake --build build
    python validation/compare_rouse_oracle.py
    python validation/compare_rouse_oracle.py --tol 1e-6 1e-8 1e-10 1e-12
    python validation/compare_rouse_oracle.py --tol 1e-12 --emit-cpp
"""
import argparse, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

N_BEADS = 10

# The initial configuration of tests.f90::test_rouse_chain_eq.
R0 = np.array([
    [0.0, 0.0, 0.0],
    [-6.586250662803650e-02, -4.911733418703079e-02, 2.791636288166050e-01],
    [-6.753685772418980e-01, 3.519757017493250e-01, -6.340299546718600e-01],
    [-2.844688922166820e00, -2.239686943590640e00, -7.809645235538480e-01],
    [-1.818384915590290e00, -2.201230965554710e00, 9.345296323299410e-01],
    [-3.210149317979810e00, -9.372534379363060e-01, -4.681020081043240e-01],
    [-1.490711003541950e00, -5.971195027232170e-01, -1.509506493806840e00],
    [-1.587433911859990e00, -1.017917685210700e00, -1.456502400338650e00],
    [-6.204553022980690e-01, -1.219155095517640e00, -7.125178799033161e-01],
    [-8.804024234414100e-01, -3.662307761609550e00, -7.348611745983360e-01]])


def run_fortran(exe, tol):
    """-> (ans1, ans2), each (N, 3). Fortran writes one bead per line."""
    # Fortran needs a d-exponent literal for its list-directed read.
    out = subprocess.run([str(exe), f"{tol:.6e}".replace("e", "d")],
                         capture_output=True, text=True, check=True).stdout
    blocks, cur = [], None
    for line in out.splitlines():
        if line.startswith("# ans"):
            cur = []
            blocks.append(cur)
        elif line.startswith("#") or not line.strip():
            continue
        elif cur is not None:
            cur.append([float(x) for x in line.split()])
    if len(blocks) != 2:
        raise RuntimeError(f"expected two configurations, parsed {len(blocks)}:\n{out}")
    return np.array(blocks[0]), np.array(blocks[1])


def run_cpp(tol):
    """The same case through the bindings: two integrations, one RNG stream."""
    import bdsim
    phys = bdsim.PhysParams()
    phys.number_of_beads = N_BEADS
    phys.spring.type = bdsim.Spring.Hook
    phys.spring.sqrtb = 1000.0
    phys.spring.natural_length = 0.0
    phys.hstar = 0.0
    phys.ev.type = bdsim.EV.None_          # "None" is a Python keyword
    phys.bend.type = bdsim.Bending.None_
    phys.flow = bdsim.Flow()                       # EQ: zero velocity gradient

    sim = bdsim.SimParams()
    sim.time_start, sim.time_end, sim.dt = 0.0, 1.0, 0.1
    sim.implicit_loop_tol = tol
    sim.update_center_of_mass = True

    rng = bdsim.Rng(5)
    rng.reset(5)
    a1 = bdsim.integrate(R0.copy(), phys, sim, rng)   # same continued stream
    a2 = bdsim.integrate(a1.copy(), phys, sim, rng)
    return a1, a2


def emit_cpp(a1, a2):
    """Print the arrays as C++ initialisers for test_integrator_rouse.cpp."""
    for name, a in (("ans1", a1), ("ans2", a2)):
        print(f"    std::vector<Vec3> {name} = {{")
        rows = [f"        {{{v[0]:.17e},{v[1]:.17e},{v[2]:.17e}}}" for v in a]
        print(",\n".join(rows) + "};")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="build/fortran_oracle")
    ap.add_argument("--tol", type=float, nargs="+",
                    default=[1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    ap.add_argument("--emit-cpp", action="store_true",
                    help="print the tightest result as C++ initialisers")
    args = ap.parse_args()

    exe = pathlib.Path(args.oracle)
    if not exe.exists():
        sys.exit(f"{exe} not found -- configure with -DBDSIM_FORTRAN=ON and build")

    print(f"  {'corrector tol':>14} {'max|C++ - Fortran|':>20} {'integration 2':>16}")
    last = None
    for tol in args.tol:
        f1, f2 = run_fortran(exe, tol)
        c1, c2 = run_cpp(tol)
        d1 = float(np.max(np.abs(c1 - f1)))
        d2 = float(np.max(np.abs(c2 - f2)))
        print(f"  {tol:14.0e} {d1:20.3e} {d2:16.3e}")
        last = (c1, c2)

    print("\n  If these fall with the tolerance, the two codes agree and the old "
          "1e-6\n  regression floor was the corrector's own truncation. If they "
          "plateau,\n  the residual is a real difference worth chasing.")
    if args.emit_cpp and last:
        print()
        emit_cpp(*last)


if __name__ == "__main__":
    main()

"""Compare the C++ integrator against the Fortran core, at a matched tolerance.

Runs build/fortran_oracle (the vendored SingleChainBD physics, LAPACK only) and
the identical case through the Python bindings, and reports the difference as the
corrector tolerance is tightened.

Three cases:

  rouse     ten Hookean beads, free draining, no flow. The setup of
            tests.f90::test_rouse_chain_eq, and the case the recorded regression
            oracle came from.

  full      everything coupled: fifteen beads, FENE-Fraenkel springs with a
            non-zero natural length (the cubic implicit solve on a shifted
            bracket), a bending potential, Lennard-Jones excluded volume,
            hydrodynamic interaction via Cholesky, and simple shear. The starting
            chain has overlapping beads, so the RPY overlapping branch is used.

  singular  not a numeric comparison but an error-handling one: a real
            configuration whose RPY tensor is not positive definite. Checks that
            both codes detect the breakdown, and at the same leading minor.

Why the tolerance sweep. The corrector is a fixed-point iteration stopped on its
increment; at the 1e-6 the recorded oracle used it is still ~3e-8 short of the
true solution, so two independent implementations cannot agree more closely than
that. Running both sides together at 1e-12 removes the floor. If the difference
falls with the tolerance the codes agree; if it plateaus, the residual is real.

    cmake -S . -B build -DBDSIM_FORTRAN=ON -DBDSIM_PYTHON=ON && cmake --build build -j
    python validation/compare_fortran_oracle.py --case full
    python validation/compare_fortran_oracle.py --case rouse --tol 1e-12 --emit-cpp
    python validation/compare_fortran_oracle.py --case singular
"""
import argparse, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

# tests.f90::test_rouse_chain_eq
R0_ROUSE = np.array([
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

# The same literals as fortran/oracle.f90::setup_full.
R0_FULL = np.array([
    [2.65804947113558121e-01, -1.51536415609088593e+00, 8.89717445791181061e-01],
    [-3.18394386622953773e-01, -2.64563101057035599e+00, 3.05587343736082095e-01],
    [-1.06546947390811186e+00, -2.81050526243381427e+00, -8.66886662385306539e-01],
    [-3.80618256309927572e-01, -3.82549992982231757e+00, -1.88090201782288413e-01],
    [-5.64124385340994294e-01, -2.59876961675185969e+00, 4.61107949479781398e-01],
    [-4.44828448905099816e-01, -1.75076197700389291e+00, 1.56865096105042268e+00],
    [1.09391750381430619e-01, -5.96707210933647536e-01, 1.00208448868080779e+00],
    [-4.45784116176970946e-01, 6.04428098566049954e-01, 5.44860555072567698e-01],
    [2.64959429939093938e-02, -1.65351468695490667e-01, -5.24901550975400810e-01],
    [-9.06054690540470387e-02, 1.14938236930606941e+00, -5.82515809656913852e-02],
    [-2.22990478928380753e-01, 2.45965545676120811e+00, 4.16783833978659535e-01],
    [1.21010683852806888e-01, 2.02117200844250133e+00, -8.67504087616699393e-01],
    [7.08211160006220863e-01, 3.22939227477250590e+00, -4.73296795910092771e-01],
    [1.69168368716962858e+00, 3.53380767141162089e+00, -1.42203967709754631e+00],
    [6.10216843728933034e-01, 2.91075275304230985e+00, -7.87822021056476252e-01]])


# The configuration of fortran/oracle.f90::setup_singular -- the real state the
# C++ integrator reached after 86860 steps of a 10 kbp DNA run once the corrector
# stopped converging. The chain has collapsed onto two points (11 beads and 10,
# 2.615 apart) with 42 pairs closer than 1e-6, so the RPY tensor has
# near-duplicate rows and is not positive definite in double precision.
R0_SINGULAR = np.array([
    [3.37650519427394511e+03, 1.24952177883608638e+04, 3.20617475621448102e+04],
    [3.37624995705613946e+03, 1.24942727222040558e+04, 3.20593226363807626e+04],
    [3.37650519236580794e+03, 1.24952177878348302e+04, 3.20617475625506631e+04],
    [3.37624995686883312e+03, 1.24942727219768185e+04, 3.20593226364890361e+04],
    [3.37650519256962343e+03, 1.24952177881101998e+04, 3.20617475624218932e+04],
    [3.37624995602158970e+03, 1.24942727220022407e+04, 3.20593226365683076e+04],
    [3.37650519275310990e+03, 1.24952177881040006e+04, 3.20617475624049948e+04],
    [3.37624995676256776e+03, 1.24942727216464009e+04, 3.20593226366290000e+04],
    [3.37650519310864456e+03, 1.24952177884205630e+04, 3.20617475622441998e+04],
    [3.37624995673909280e+03, 1.24942727225040362e+04, 3.20593226362972237e+04],
    [3.37650519343873839e+03, 1.24952177891145147e+04, 3.20617475619389988e+04],
    [3.37624995507601216e+03, 1.24942727224786813e+04, 3.20593226364821530e+04],
    [3.37650519262076159e+03, 1.24952177891598494e+04, 3.20617475620074329e+04],
    [3.37624995656296505e+03, 1.24942727235068232e+04, 3.20593226359249456e+04],
    [3.37650519312531151e+03, 1.24952177882188316e+04, 3.20617475623210667e+04],
    [3.37624995772563670e+03, 1.24942727234146205e+04, 3.20593226358385036e+04],
    [3.37650519213303596e+03, 1.24952177875682719e+04, 3.20617475626790474e+04],
    [3.37624995747488310e+03, 1.24942727227043870e+04, 3.20593226361416964e+04],
    [3.37650519216934890e+03, 1.24952177873712444e+04, 3.20617475627520143e+04],
    [3.37624995681034625e+03, 1.24942727225795206e+04, 3.20593226362603054e+04],
    [3.37650519254607570e+03, 1.24952177876328660e+04, 3.20617475626104024e+04]])

SINGULAR_HSTAR = 1.18408796570133590e-01


def run_fortran(exe, case, tol, nblocks):
    """-> (list of (N,3) configurations, steps_per_block)."""
    # A d-exponent literal, for the Fortran list-directed read.
    out = subprocess.run([str(exe), case, f"{tol:.6e}".replace("e", "d"),
                          str(nblocks)],
                         capture_output=True, text=True, check=True).stdout
    blocks, cur, per_block = [], None, None
    for line in out.splitlines():
        if line.startswith("# ans"):
            cur = []
            blocks.append(cur)
        elif line.startswith("# steps_per_block"):
            per_block = int(line.split("=")[1])
        elif line.startswith("#") or not line.strip():
            continue
        elif cur is not None:
            try:
                row = [float(x) for x in line.split()]
            except ValueError:
                continue           # e.g. the dpotrf warning, not a bead
            if len(row) == 3:
                cur.append(row)
    if len(blocks) != nblocks:
        raise RuntimeError(f"asked for {nblocks} configurations, parsed "
                           f"{len(blocks)}:\n{out}")
    return [np.array(b) for b in blocks], per_block


def build_phys(case):
    """PhysParams mirroring fortran/oracle.f90, plus (R0, seed, dt, t_end)."""
    import bdsim
    phys = bdsim.PhysParams()

    if case == "rouse":
        phys.number_of_beads = 10
        phys.spring.type = bdsim.Spring.Hook
        phys.spring.sqrtb = 1000.0
        phys.spring.natural_length = 0.0
        phys.hstar = 0.0
        phys.ev.type = bdsim.EV.None_              # "None" is a Python keyword
        phys.bend.type = bdsim.Bending.None_
        phys.flow = bdsim.Flow()                   # EQ: zero velocity gradient
        return phys, R0_ROUSE, 5, 0.1, 1.0

    if case == "full":
        phys.number_of_beads = 15
        phys.spring.type = bdsim.Spring.FENEFraenkel
        phys.spring.sqrtb = 3.0
        phys.spring.natural_length = 1.5

        phys.bend.type = bdsim.Bending.OneMinusCosTheta
        phys.bend.stiffness = 1.5

        phys.ev.type = bdsim.EV.LJ
        phys.ev.zstar = 1.0
        phys.ev.dstar = 1.0
        phys.ev.min_cutoff = 0.7
        phys.ev.max_cutoff = 1.5
        phys.ev.contour_dist_for_EV = 1
        # The Fortran derives its good/poor-solvent EV branch from the flow type
        # (EQ -> good). Under shear it takes the poor-solvent branch, which is
        # what equilibration=False selects here.
        phys.equilibration = False

        phys.hstar = 0.25
        phys.hi_method = bdsim.DelSMethod.Cholesky
        phys.ncheb_multiplier = 1.0
        phys.fd_err_max = 0.0025

        phys.flow = bdsim.flows.shear(1.0)
        return phys, R0_FULL, 11, 0.005, 0.5

    if case == "singular":
        phys.number_of_beads = 21
        phys.spring.type = bdsim.Spring.Hook
        phys.spring.sqrtb = 1000.0
        phys.spring.natural_length = 0.0
        phys.ev.type = bdsim.EV.None_
        phys.bend.type = bdsim.Bending.None_
        phys.hstar = SINGULAR_HSTAR
        phys.hi_method = bdsim.DelSMethod.Cholesky
        phys.flow = bdsim.Flow()
        return phys, R0_SINGULAR, 6, 0.01, 0.0        # a single step

    raise SystemExit(f"unknown case {case!r}")


def run_cpp(case, tol, nblocks=2):
    """nblocks consecutive integrations on one continued RNG stream."""
    import bdsim
    phys, R0, seed, dt, t_end = build_phys(case)

    sim = bdsim.SimParams()
    sim.time_start, sim.time_end, sim.dt = 0.0, t_end, dt
    sim.implicit_loop_tol = tol
    sim.update_center_of_mass = True

    rng = bdsim.Rng(seed)
    rng.reset(seed)
    out, R = [], R0.copy()
    for _ in range(nblocks):
        R = bdsim.integrate(R, phys, sim, rng)
        out.append(R.copy())
    return out


def emit_cpp(a1, a2):
    for name, a in (("ans1", a1), ("ans2", a2)):
        print(f"    std::vector<Vec3> {name} = {{")
        print(",\n".join(f"        {{{v[0]:.17e},{v[1]:.17e},{v[2]:.17e}}}" for v in a)
              + "};")
        print()


def singular_check(exe):
    """Do both codes detect the degenerate diffusion tensor, in the same place?

    The C++ raises; the Fortran, with the local modification in gsipc.f90, prints
    a dpotrf warning and carries on. Upstream it prints nothing at all and the
    run continues with a partially factored matrix -- which is the point.

    LAPACK's INFO is the 1-based order of the leading minor that failed; the C++
    reports a 0-based pivot index. Agreement means INFO == pivot + 1.
    """
    import re
    print("case: singular -- a configuration whose RPY tensor is not positive definite\n")

    # ---- Fortran ----
    proc = subprocess.run([str(exe), "singular", "1.0d-6", "1"],
                          capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    m = re.search(r"not positive.*?minor\s+(\d+)", text, re.S)
    f_info = int(m.group(1)) if m else None
    print("  Fortran:")
    if f_info is not None:
        print(f"    dpotrf reported INFO = {f_info} (leading minor, 1-based)")
        print(f"    exit status {proc.returncode} -- it continued anyway, using a "
              f"partial factorisation")
    elif "WARNING" in text:
        print("    warned, but the message could not be parsed:")
        print("   ", "\n    ".join(text.strip().splitlines()[:4]))
    else:
        print("    NO diagnostic. Either the local modification to gsipc.f90 is")
        print("    missing, or this build's LAPACK accepted the matrix.")

    # ---- C++ ----
    print("\n  C++:")
    c_pivot = None
    try:
        run_cpp("singular", 1e-6, nblocks=1)
        print("    no exception -- the factorisation succeeded")
    except RuntimeError as e:
        msg = str(e)
        m = re.search(r"pivot (\d+)", msg)
        c_pivot = int(m.group(1)) if m else None
        print(f"    raised: {msg.split('.')[0]}.")

    # ---- verdict ----
    print()
    if f_info is not None and c_pivot is not None:
        if f_info == c_pivot + 1:
            print(f"  Both codes fail at the same place: Fortran minor {f_info} "
                  f"== C++ pivot {c_pivot} + 1.")
            print("  Same matrix, same breakdown -- the difference is only that "
                  "upstream\n  says nothing and keeps going.")
            return 0
        print(f"  Both detect a failure, but at different places: Fortran minor "
              f"{f_info}, C++ pivot {c_pivot} (expected {f_info - 1}).")
        print("  The two codes are building different diffusion tensors from the "
              "same\n  configuration -- worth chasing.")
        return 1
    if f_info is None and c_pivot is not None:
        print("  The C++ detects the failure and the Fortran does not: exactly the")
        print("  silent-corruption behaviour, if gsipc.f90 is unmodified here.")
        return 1
    print("  Inconclusive -- see above.")
    return 1


def compare(exe, case, tol, nblocks):
    """-> (list of max|dR| per block, steps_per_block, list of chain scales)."""
    fort, per_block = run_fortran(exe, case, tol, nblocks)
    cpp = run_cpp(case, tol, nblocks)
    diffs, scales = [], []
    for c, f in zip(cpp, fort):
        if c.shape != f.shape:
            sys.exit(f"shape mismatch: C++ {c.shape} vs Fortran {f.shape}")
        diffs.append(float(np.max(np.abs(c - f))))
        scales.append(float(np.max(np.abs(f))))
    return diffs, per_block, scales, cpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="build/fortran_oracle")
    ap.add_argument("--case", choices=["rouse", "full", "singular"],
                    default="full")
    ap.add_argument("--tol", type=float, nargs="+",
                    default=[1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    ap.add_argument("--blocks", type=int, default=2,
                    help="consecutive integrations on one RNG stream; the "
                         "difference is reported after each, so >2 shows how "
                         "the agreement evolves over a longer trajectory")
    ap.add_argument("--emit-cpp", action="store_true",
                    help="print the tightest result as C++ initialisers")
    ap.add_argument("--check", type=float, metavar="MAXDIFF",
                    help="assert mode: run the tightest tolerance only and exit "
                         "non-zero if any block differs by more than MAXDIFF")
    args = ap.parse_args()

    exe = pathlib.Path(args.oracle)
    if not exe.exists():
        sys.exit(f"{exe} not found -- configure with -DBDSIM_FORTRAN=ON and build")

    # The singular case is a yes/no about error detection, not a numeric
    # comparison: there is no meaningful "difference" when one side refuses to
    # produce an answer at all.
    if args.case == "singular":
        return singular_check(exe)

    # ---- assert mode, for ctest ----
    if args.check is not None:
        tol = min(args.tol)
        diffs, per_block, _, _ = compare(exe, args.case, tol, args.blocks)
        worst = max(diffs)
        ok = worst <= args.check
        print(f"case {args.case}, corrector tol {tol:.0e}, {args.blocks} block(s) "
              f"of {per_block} steps")
        print(f"  worst max|dR| over blocks = {worst:.3e}  (limit {args.check:.0e})")
        print("  C++ and Fortran agree" if ok else
              "  DIFFERENCE EXCEEDS THE LIMIT -- the codes have diverged")
        return 0 if ok else 1

    print(f"  case: {args.case}, {args.blocks} block(s)\n")
    if args.blocks <= 2:
        print(f"  {'corrector tol':>14} {'max|dR| block 1':>18} "
              f"{'block 2':>14} {'rel. to chain size':>19}")
    last = None
    for tol in args.tol:
        diffs, per_block, scales, cpp = compare(exe, args.case, tol, args.blocks)
        last = cpp
        if args.blocks <= 2:
            d2 = diffs[-1]
            print(f"  {tol:14.0e} {diffs[0]:18.3e} {d2:14.3e} "
                  f"{d2/scales[-1]:19.3e}")
        else:
            print(f"  corrector tol {tol:.0e}   ({per_block} steps per block)")
            print(f"    {'block':>6} {'steps':>8} {'max|dR|':>12} "
                  f"{'growth':>9} {'rel. to chain':>14}")
            for i, d in enumerate(diffs, 1):
                growth = (f"{d/diffs[i-2]:8.2f}x" if i > 1 and diffs[i-2] > 0
                          else "       --")
                print(f"    {i:6d} {i*per_block:8d} {d:12.3e} {growth} "
                      f"{d/scales[i-1]:14.3e}")
            print()

    if args.blocks > 2:
        print("  A steady multiplicative growth is the physics, not a bug: the\n"
              "  trajectory is chaotic, so any difference -- here, the corrector's\n"
              "  truncation -- is amplified at the Lyapunov rate. What matters is\n"
              "  that it starts at round-off and grows smoothly, rather than\n"
              "  jumping, which would mean the two codes took different branches.")
    else:
        print("\n  Falling with the tolerance => the codes agree and the residual is\n"
              "  the corrector's own truncation. A plateau => a real difference.")
    if args.emit_cpp and last:
        print()
        emit_cpp(last[0], last[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())

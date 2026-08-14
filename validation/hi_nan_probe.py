"""Locate the first non-finite number in an HI run, and say what produced it.

Reproduces the failing viscosity_sweep_HI point with the same coarse-graining and
matched h*, then integrates ONE trajectory step by step, checking the state after
every step. On the first non-finite value it reports the step, the time, and the
state of the configuration just before -- closest approach between beads, bond
lengths against the FENE-Fraenkel bounds, and whether the RPY diffusion matrix at
that configuration is still positive definite.

    python validation/hi_nan_probe.py --L 10000 --Ns 20 --wi 0.3
"""
import argparse, copy, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

LP_DNA, BP_NM = 147.0, 0.34


def rpy_matrix(R, hstar):
    """The RPY tensor as the C++ builds it, in code units (3N x 3N)."""
    n = len(R)
    a = hstar * np.sqrt(np.pi)          # bead radius in l_H units
    D = np.zeros((3 * n, 3 * n))
    for i in range(n):
        D[3*i:3*i+3, 3*i:3*i+3] = np.eye(3)
        for j in range(i + 1, n):
            r = R[i] - R[j]
            d = np.linalg.norm(r)
            e = np.outer(r, r) / (d * d) if d > 0 else np.zeros((3, 3))
            if d >= 2 * a:
                c1 = 0.75 * (a / d) * (1 + 2 * a * a / (3 * d * d))
                c2 = 0.75 * (a / d) * (1 - 2 * a * a / (d * d))
            else:
                c1 = 1 - 9 * d / (32 * a)
                c2 = 3 * d / (32 * a)
            blk = c1 * np.eye(3) + c2 * e
            D[3*i:3*i+3, 3*j:3*j+3] = blk
            D[3*j:3*j+3, 3*i:3*i+3] = blk
    return D


def describe(R, hstar, p, label):
    n = len(R)
    a = hstar * np.sqrt(np.pi)
    dmin, pair = np.inf, None
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(R[i] - R[j]))
            if d < dmin:
                dmin, pair = d, (i, j)
    Q = np.linalg.norm(np.diff(R, axis=0), axis=1)
    lo, hi = p.sigma_H - p.dQ_H, p.sigma_H + p.dQ_H
    D = rpy_matrix(R, hstar)
    ev = np.linalg.eigvalsh(D)
    print(f"  {label}")
    print(f"    closest pair {pair}: r = {dmin:.4g}   (2a = {2*a:.4g}, "
          f"{'OVERLAPPING' if dmin < 2*a else 'separated'})")
    print(f"    bond lengths: min {Q.min():.4g}  max {Q.max():.4g}   "
          f"bounds ({lo:.4g}, {hi:.4g})"
          + ("   <-- OUT OF BOUNDS" if Q.min() <= lo or Q.max() >= hi else ""))
    print(f"    RPY eigenvalues: min {ev.min():.6g}  max {ev.max():.6g}  "
          f"cond {ev.max()/max(ev.min(), 1e-300):.4g}"
          + ("   <-- NOT POSITIVE DEFINITE" if ev.min() <= 0 else ""))
    return ev.min()


def main():
    import bdsim
    from bdsim import coarse_grain as cg, dynamics as dyn

    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=10000.0)
    ap.add_argument("--Ns", type=int, default=20)
    ap.add_argument("--wi", type=float, default=0.3)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--loop-tol", type=float, default=1e-4)
    ap.add_argument("--n-traj", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--viscosity", type=float, default=1.0e-3)
    ap.add_argument("--max-H-pN-per-nm", type=float, default=0.05)
    args = ap.parse_args()

    kT = dyn.KB * args.temperature
    cap = args.max_H_pN_per_nm * BP_NM ** 2 / (kT * 1e21)
    p = cg.spring_parameters(args.L, LP_DNA, args.Ns, bending="match_chain", max_H=cap)
    rh = dyn.dna_hydrodynamic_radius_nm(args.L, args.temperature, args.viscosity,
                                        length_unit_nm=BP_NM)
    hstar, info = dyn.hstar_for_target_rh(p, rh, length_unit_nm=BP_NM)
    u = dyn.physical_units(p, hstar, args.temperature, args.viscosity,
                           length_unit_nm=BP_NM, hydrodynamic_radius_nm=rh)
    tau1_s, tau1_code = dyn.free_draining_relaxation_time(p, u)

    print(f"L={args.L:g} Ns={args.Ns}: h*={hstar:.4g}  a={info['bead_radius_nm']:.3g} nm  "
          f"sigma={p.sigma_H:.4g} dQ={p.dQ_H:.4g} C={p.C:.4g}")
    print(f"  bead radius in code units a = h* sqrt(pi) = {hstar*np.sqrt(np.pi):.4g}, "
          f"spring bounds ({p.sigma_H-p.dQ_H:.4g}, {p.sigma_H+p.dQ_H:.4g})")
    print(f"  => a/sigma = {hstar*np.sqrt(np.pi)/max(p.sigma_H,1e-30):.3g}\n")

    phys = cg.to_phys_params(p, hstar=hstar, hi_method=bdsim.DelSMethod.Cholesky)
    phys.flow = bdsim.flows.shear(args.wi / tau1_code)
    sim = bdsim.SimParams()
    sim.dt = args.dt
    sim.implicit_loop_tol = args.loop_tol

    init = bdsim.Initial("fene_fraenkel_bending",
                         dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))
    from bdsim.ensemble import INITIALIZERS

    def steps(n):
        """A SimParams that advances exactly n steps."""
        s = bdsim.SimParams()
        s.dt, s.implicit_loop_tol = sim.dt, sim.implicit_loop_tol
        s.time_start = 0.0
        s.time_end = (n - 1) * sim.dt          # the loop is inclusive
        s.update_center_of_mass = True
        return s

    def from_start(seed, m):
        """State after exactly m steps, always replayed from the beginning.

        Deliberately does NOT snapshot and restore the generator: Rng::restore
        does not reinstate the lazily-computed scale factor, so a restored
        stream is not the original one. Replaying from step 0 is slower but is
        the only way to be sure the trajectory is the same one.
        """
        R = np.asarray(INITIALIZERS[init.method](phys.number_of_beads, seed,
                                                 dict(init.kwargs)),
                       dtype=np.float64)
        if m == 0:
            return R
        return bdsim.integrate(R, phys, steps(m), bdsim.Rng(seed))

    print(f"scanning {args.n_traj} trajectories x {args.max_steps} steps "
          f"(chunks of {args.chunk}; the failing step is then found by bisection,\n"
          f" replaying from step 0 each time)\n")
    for traj in range(args.n_traj):
        seed = 1 + traj
        R = np.asarray(INITIALIZERS[init.method](phys.number_of_beads, seed,
                                                 dict(init.kwargs)), dtype=np.float64)
        rng = bdsim.Rng(seed)
        done, hit = 0, None
        while done < args.max_steps:
            n = min(args.chunk, args.max_steps - done)
            R = bdsim.integrate(R, phys, steps(n), rng)
            done += n
            if not np.all(np.isfinite(R)):
                hit = done
                break
        if hit is None:
            print(f"  trajectory {traj} (seed {seed}): {args.max_steps} steps clean",
                  flush=True)
            continue

        print(f"  trajectory {traj} (seed {seed}): non-finite by step {hit}, "
              f"bisecting...", flush=True)
        lo, hi = hit - args.chunk, hit           # lo is known good, hi known bad
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if np.all(np.isfinite(from_start(seed, mid))):
                lo = mid
            else:
                hi = mid
        prev = from_start(seed, lo)
        after = from_start(seed, hi)
        print(f"\n*** first non-finite: trajectory {traj} (seed {seed}), step {hi}, "
              f"t = {hi*args.dt:.6g} ({hi*args.dt/tau1_code:.3g} tau_1)\n")
        describe(prev, hstar, p, "configuration ENTERING the failing step:")
        bad = np.where(~np.isfinite(after).all(axis=1))[0]
        print(f"\n    beads that went non-finite: {bad.tolist()}"
              f"  ({len(bad)} of {len(after)})")
        np.save("nan_config.npy", prev)
        print("    configuration saved to nan_config.npy")
        return 1
    print("\nno non-finite values seen")
    return 0


if __name__ == "__main__":
    sys.exit(main())

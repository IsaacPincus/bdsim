"""Measure the force-extension curve of a chain by actually pulling on it.

Equal and opposite forces are applied to the two end beads (zero net force, so the
chain stretches without drifting), the run is equilibrated at that force, and the
mean end-to-end extension along the pulling axis is sampled. This is the honest
version of the curve: it includes the bending potential, finite extensibility, the
discreteness of the chain and (optionally) hydrodynamic interaction -- none of
which are captured by inferring a force law from a moment fit.

Two modes:

  --check    Hookean chain, where the answer is exact. Ns springs of stiffness
             H = 1 in series each carry the full applied force, so <R_z> = Ns * f
             regardless of Ns. Any deviation is a bug or poor equilibration.

  --dna      Coarse-grained DNA (see coarse_grain.py) compared with Marko-Siggia.
             This is the measurement that answers "does the model follow MS?" --
             see the discussion of why it should not at high force.

Usage:
    python validation/force_extension.py --check
    python validation/force_extension.py --dna --L 25000 --Ns 30 --plot
"""
import argparse, math, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim
from bdsim import coarse_grain as cg
from bdsim.parallel import parallel_map

LP_DNA = 147.0


def rouse_relaxation_time(n_beads):
    """Longest Rouse relaxation time of a free-draining Hookean chain, in the
    code's time unit (lambda_H = zeta/4H = 1):  lambda_1 = 1 / sin^2(pi/2N).

    This sets the timescale a pulling experiment must respect. Equilibrating for
    less than a few times this leaves the chain still relaxing towards its
    stretched steady state, and the measured extension comes out systematically
    LOW -- which looks exactly like a model that is too stiff. Springs that are
    stiffer than Hookean relax faster, so this is a conservative upper bound.
    """
    return 1.0 / math.sin(math.pi / (2.0 * n_beads)) ** 2


def _pull_worker(job):
    """Mean extension along z of one trajectory held at constant force."""
    (phys, dt, f, t_eq, t_run, n_samples, seed, init_method, init_kwargs) = job
    phys.external.add_stretch(0, -1, (0.0, 0.0, f))
    n = phys.number_of_beads
    R = bdsim.ensemble.INITIALIZERS[init_method](n, seed, init_kwargs)
    rng = bdsim.Rng(seed)

    sim = bdsim.SimParams()
    sim.dt = dt
    sim.implicit_loop_tol = 1e-5
    sim.update_center_of_mass = True

    # equilibrate at this force, then sample
    seg = bdsim.ensemble._segment(sim, 0.0, t_eq)
    R = bdsim.integrate(R, phys, seg, rng)
    t = t_eq
    vals = []
    step = t_run / n_samples
    for _ in range(n_samples):
        R = bdsim.integrate(R, phys, bdsim.ensemble._segment(sim, t, t + step), rng)
        t += step
        vals.append(R[-1, 2] - R[0, 2])
    return float(np.mean(vals))


def measure(phys, dt, forces, *, t_eq, t_run, n_samples=40, n_traj=8, seed=0,
            initial=None, backend="processes", n_workers=None):
    """Mean extension <R_z> at each force. Returns (mean, stderr) arrays."""
    initial = initial or bdsim.Initial()
    jobs = []
    for f in forces:
        for i in range(n_traj):
            jobs.append((phys, dt, f, t_eq, t_run, n_samples, seed + i,
                         initial.method, dict(initial.kwargs)))
    out = parallel_map(_pull_worker, jobs, backend=backend, n_workers=n_workers)
    out = np.asarray(out).reshape(len(forces), n_traj)
    return out.mean(axis=1), out.std(axis=1, ddof=1) / math.sqrt(n_traj)


def run_check(args):
    """Hookean chain: <R_z> = Ns * f exactly."""
    Ns = args.Ns
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.Hook
    phys.spring.sqrtb = 1.0e6
    phys.number_of_beads = Ns + 1
    phys.flow = bdsim.flows.equilibrium()

    forces = [0.25, 0.5, 1.0, 2.0]
    m, e = measure(phys, args.dt, forces, t_eq=args.t_eq, t_run=args.t_run,
                   n_traj=args.n_traj, backend=args.backend)
    print(f"Hookean chain, Ns = {Ns}   (exact: <R_z> = Ns * f)")
    print(f"{'f':>8} {'<R_z> measured':>18} {'exact':>10} {'ratio':>8}")
    for f, mm, ee in zip(forces, m, e):
        print(f"{f:8.3f} {mm:12.4f} +/-{ee:5.4f} {Ns*f:10.4f} {mm/(Ns*f):8.4f}")


def run_dna(args):
    p = cg.spring_parameters(args.L, LP_DNA, args.Ns, bending="match_chain")
    phys = cg.to_phys_params(p, hstar=args.hstar)
    lH = p.dQ / p.dQ_H                    # Hookean length unit, in base pairs
    print(p.summary())
    print(f"\n  crossover force f* lp/kT ~ (lp/ls)^2 = {(LP_DNA/p.ls)**2:.3g}\n")

    # forces in kT per base pair -> Hookean units: f_H = f * lH
    fl = np.array(args.forces)            # f * lp / kT
    f_phys = fl / LP_DNA
    f_H = f_phys * lH

    init = bdsim.Initial("fene_fraenkel_bending",
                         dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))
    m, e = measure(phys, args.dt, f_H, t_eq=args.t_eq, t_run=args.t_run,
                   n_traj=args.n_traj, initial=init, backend=args.backend)
    x = m * lH / args.L                   # fractional extension R/L
    xe = e * lH / args.L

    def ms_x(v):
        lo, hi = 0.0, 1 - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.25 / (1 - mid) ** 2 - 0.25 + mid < v: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)

    print(f"{'f lp/kT':>10} {'x measured':>18} {'x Marko-Siggia':>15} {'ratio':>8}")
    for v, xx, ee in zip(fl, x, xe):
        xm = ms_x(v)
        print(f"{v:10.4g} {xx:12.4f} +/-{ee:5.4f} {xm:15.4f} {xx/xm:8.3f}")

    if args.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.linspace(1e-4, 0.999, 500)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot([0.25 / (1 - u) ** 2 - 0.25 + u for u in xs], xs, "k-",
                label="Marko-Siggia (WLC)")
        ax.errorbar(fl, x, xe, fmt="o", color="C1", capsize=3,
                    label=f"model, $N_s$={args.Ns} (pulled)")
        ax.axvline((LP_DNA / p.ls) ** 2, ls=":", color="0.5",
                   label=r"$f^*l_p/k_BT\sim(l_p/l_s)^2$")
        ax.set_xscale("log"); ax.set_xlabel(r"$f\,l_p/k_\mathrm{B}T$")
        ax.set_ylabel("$R/L$"); ax.set_ylim(0, 1.05)
        ax.set_title(f"L = {args.L} bp, $N_s$ = {args.Ns}"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dna", action="store_true")
    ap.add_argument("--L", type=float, default=25000)
    ap.add_argument("--Ns", type=int, default=20)
    ap.add_argument("--hstar", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--t-eq", type=float, default=None,
                    help="default: 3 x the Rouse relaxation time")
    ap.add_argument("--t-run", type=float, default=None,
                    help="default: 6 x the Rouse relaxation time")
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--backend", default="processes")
    ap.add_argument("--forces", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    lam = rouse_relaxation_time(a.Ns + 1)
    if a.t_eq is None: a.t_eq = 3.0 * lam
    if a.t_run is None: a.t_run = 6.0 * lam
    print(f"# Rouse relaxation time ~ {lam:.1f}; t_eq = {a.t_eq:.1f}, "
          f"t_run = {a.t_run:.1f}, dt = {a.dt:g}  "
          f"({int((a.t_eq + a.t_run) / a.dt)} steps/trajectory)\n")
    if a.check: run_check(a)
    if a.dna: run_dna(a)
    if not (a.check or a.dna): ap.error("choose --check and/or --dna")

"""Validate polymer shear viscosity vs a reference FENE + HI figure.

Two curves read off the reference figure (FENE, h* = 0.3, no EV, N = 20):
    dQ = 2   (sqrtb = 2):    eta_p ~ 30, 17, 1.7   at gamma_dot = 1e-2, 1, 1e2
    dQ = 100 (sqrtb = 100):  eta_p ~ 75, 110, 6     at gamma_dot = 1e-2, 1, 1e2

These are short, exploratory runs -- big error bars are expected. Timestep and
run length are chain- and shear-aware (small dt at high shear / for the stiff
dQ=2 spring; long equilibration for the near-Hookean dQ=100). Blown-up
trajectories (stiff-spring instability) are rejected.

Usage:  python viscosity_vs_shear.py [rate]      # rate in {0.01, 1, 100}; omit = all
"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import numpy as np
import bdsim
from bdsim.parallel import parallel_map

N = 20
TARGETS = {2: {0.01: 30.0, 1.0: 17.0, 100.0: 1.7},
           100: {0.01: 75.0, 1.0: 110.0, 100.0: 6.0}}
RESULTS = pathlib.Path(__file__).with_name("viscosity_results.json")

# (dt, t_eq, t_end, n_traj) per (dQ, rate). dQ=2 relaxes fast (short runs, small
# dt for stability); dQ=100 is near-Hookean (long relaxation, needs equilibration).
def settings(dQ, rate):
    if dQ <= 5:   # stiff, fast-relaxing spring
        return {0.01:  (0.005, 30.0, 90.0, 16),
                1.0:   (0.005, 20.0, 60.0, 16),
                100.0: (2e-4,  0.1,  0.5,  20)}[rate]
    else:         # near-Hookean, slow-relaxing spring (lambda_1 ~ 45)
        return {0.01:  (0.02, 180.0, 400.0, 12),
                1.0:   (0.01, 150.0, 300.0, 12),
                100.0: (2e-4, 0.3,   1.0,   16)}[rate]


def _traj(job):
    """One trajectory's viscosity (top-level so it is picklable for workers)."""
    dQ, rate, dt, samples, seed = job
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.FENE
    phys.spring.sqrtb = float(dQ); phys.spring.natural_length = 0.0
    phys.number_of_beads = N
    phys.hstar = 0.3; phys.hi_method = bdsim.DelSMethod.Cholesky
    phys.flow = bdsim.flows.shear(rate)
    sim = bdsim.SimParams(); sim.dt = dt; sim.implicit_loop_tol = 1e-4
    rng = bdsim.Rng(seed)
    R0 = bdsim.gaussian_chain(N, seed=seed)
    sxy = [bdsim.kramers_stress(R, bdsim.total_force(R, phys))[0, 1]
           for _t, R in bdsim.trajectory_samples(R0, phys, sim, rng, samples)]
    return -np.mean(sxy) / rate


def measure(dQ, rate, backend="processes"):
    dt, t_eq, t_end, ntraj = settings(dQ, rate)
    samples = list(np.linspace(t_eq, t_end, 18))
    jobs = [(dQ, rate, dt, samples, 1 + i) for i in range(ntraj)]
    raw = parallel_map(_traj, jobs, backend=backend)
    etas = np.array([e for e in raw if np.isfinite(e) and abs(e) < 1e4])  # reject blow-ups
    n_bad = ntraj - len(etas)
    mean = float(etas.mean())
    err = float(etas.std(ddof=1) / np.sqrt(len(etas))) if len(etas) > 1 else float("nan")
    return mean, err, n_bad


def main():
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    rates = [float(sys.argv[1])] if len(sys.argv) > 1 else [100.0, 1.0, 0.01]
    for rate in rates:
        for dQ in (2, 100):
            t0 = time.perf_counter()
            m, e, nbad = measure(dQ, rate)
            tgt = TARGETS[dQ][rate]
            data[f"{dQ}_{rate}"] = dict(dQ=dQ, rate=rate, eta=m, err=e, target=tgt)
            bad = f" ({nbad} rejected)" if nbad else ""
            print(f"dQ={dQ:>3} gamma_dot={rate:<6}: eta_p = {m:7.2f} +/- {e:5.2f}"
                  f"   (figure ~ {tgt}){bad}   [{time.perf_counter()-t0:.1f}s]", flush=True)
    RESULTS.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

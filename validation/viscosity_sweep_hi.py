"""Viscosity sweep WITH hydrodynamic interaction.

The companion to `viscosity_sweep.py`, which runs free-draining. The difference
is one line of physics and a large difference in what the answer means:

  free draining   a = R_H/N is chosen so the chain's diffusivity is right, but
                  the internal relaxation spectrum is Rouse. Cheap, and the
                  shapes are indicative only.

  this script     h* is solved so the chain's Kirkwood hydrodynamic radius
                  equals the measured R_H, which constrains the whole spectrum
                  rather than the centre-of-mass motion alone. 30-40x slower per
                  step, and the only version worth comparing with experiment.

Shear rates are still placed using the FREE-DRAINING longest relaxation time,
because that one is known in closed form. Under HI it is an estimate rather than
the chain's actual tau_1, so the Wi labels are nominal. The physically meaningful
abscissa is `wi_eta = gammadot * lambda_eta,0`, built on the zero-shear
viscometric time that `annotate()` extrapolates from the measured curve --
plot with `xkey="wi_eta"`.

The post-processing (derive, annotate, the Carreau zero-shear fit, summarise,
make_plot) is imported from viscosity_sweep rather than duplicated: the analysis
of a viscosity curve does not care how the mobility was computed.

All the parameters live in `CONFIG` below. Edit it and run the file:

    python validation/viscosity_sweep_hi.py

or drive it from your own script:

    from viscosity_sweep_hi import Config, run, plot
    run(Config(L=(48502.0,), Ns=(30,), wi=(1.0, 3.0, 10.0), n_traj=200,
               out="lambda_hi.json"))
    plot("lambda_hi.json", xkey="wi_eta")

Results accumulate in the JSON file and each (L, Ns) pair is replaced when
re-run, so a long sweep can be done in pieces.
"""
import json, os, sys, pathlib
from dataclasses import dataclass

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "python"))
sys.path.insert(0, str(_HERE))

# The analysis is shared with the free-draining sweep; only the physics differs.
from viscosity_sweep import (derive, annotate, zero_shear_viscosity,   # noqa: F401
                             summarise, make_plot, _err_of)           # noqa: F401

LP_DNA, BP_NM, G_PER_MOL_PER_BP, N_A = 147.0, 0.34, 650.0, 6.02214076e23


@dataclass
class Config:
    """Everything the sweep needs. Edit CONFIG below, or build your own."""

    # What to sweep.
    L: tuple = (48502.0,)                # DNA contour lengths, base pairs
    Ns: tuple = (30,)                    # springs per chain
    wi: tuple = (1.0, 3.0, 10.0)         # nominal Weissenberg numbers

    out: str = "validation/viscosity_sweep_hi.json"

    # Physical conditions.
    temperature: float = 298.15          # K
    viscosity: float = 1.0e-3            # solvent, Pa s
    max_H_pN_per_nm: float = 0.05        # cap on the spring constant; None to disable

    # Hydrodynamic interaction.
    hi_method: str = "cholesky"          # "cholesky" (exact) or "chebyshev" (cheaper, large N)
    hstar_n_configs: int = 2000          # equilibrium configurations used to solve for h*

    # Time stepping and sampling. See the note on stability below: these are
    # deliberately more conservative than the free-draining defaults.
    dt: float = 0.05
    implicit_loop_tol: float = 1e-4
    eq_taus: float = 3.0                 # equilibration, in units of tau_1
    run_taus: float = 8.0                # production, in units of tau_1
    n_samples: int = 80
    n_traj: int = 4

    variance_reduction: bool = False     # pays below Wi ~ 0.1, hurts above ~0.3

    # Keep the configurations as well. Costs no extra simulation, only disk and
    # memory: one bdsim run directory per (L, Ns, Wi), readable with
    # bdsim.read_run(). None to disable.
    save_traj: str = None
    save_forces: bool = True

    x: str = "wi"                        # plotting abscissa: "wi" or "wi_eta"


CONFIG = Config()
"""The parameters this script runs with. Edit here, then run the file."""


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------
# The defaults above are dt = 0.005 and implicit_loop_tol = 1e-6, against
# dt = 0.01 and 1e-4 for the free-draining sweep. That is not caution for its own
# sake. At dt = 0.01 / 1e-4 a 10 kbp chain with matched h* loses about a quarter
# of its trajectories: the implicit corrector stops converging, the bonds are
# clamped onto the FENE bound where the force is ~1e6 H, the chain is thrown a
# long way in one step, and the RPY tensor built from the wreck is no longer
# positive definite. Both the corrector warning and the Cholesky failure are
# reported now, and run_ensemble refuses to average a badly-thinned ensemble, so
# this shows up as an error rather than a quietly biased number -- but it is
# still a wasted run.
#
# If you see "implicit corrector did not converge" during a sweep, reduce dt
# before trusting anything, and treat the failed-trajectory count as the
# diagnostic. HI makes this worse than free draining because the mobility couples
# every bead, so the corrector's fixed-point iteration has a larger spectral
# radius at the same dt.


def _hi_method(name):
    import bdsim
    return {"cholesky": bdsim.DelSMethod.Cholesky,
            "chebyshev": bdsim.DelSMethod.Chebyshev}[name.lower()]


def one_case(L_bp, Ns, wi_list, cfg):
    """One (L, Ns): coarse-grain, match h*, then sweep the shear rates."""
    import bdsim
    from bdsim import coarse_grain as cg
    from bdsim import dynamics as dyn

    kT = dyn.KB * cfg.temperature
    cap = (cfg.max_H_pN_per_nm * BP_NM ** 2 / (kT * 1e21)
           if cfg.max_H_pN_per_nm else None)
    p = cg.spring_parameters(L_bp, LP_DNA, Ns, bending="match_chain", max_H=cap)

    rh_nm = dyn.dna_hydrodynamic_radius_nm(L_bp, cfg.temperature, cfg.viscosity,
                                           length_unit_nm=BP_NM)
    # THE difference from the free-draining sweep: solve for the h* whose
    # Kirkwood radius matches the measurement, instead of setting a = R_H/N.
    hstar, info = dyn.hstar_for_target_rh(p, rh_nm, length_unit_nm=BP_NM,
                                          n_configs=cfg.hstar_n_configs)
    u = dyn.physical_units(p, hstar, cfg.temperature, cfg.viscosity,
                           length_unit_nm=BP_NM, hydrodynamic_radius_nm=rh_nm)
    # Free-draining closed form, used only to place the shear rates. Under HI it
    # estimates tau_1 rather than giving it; lambda_eta,0 from the fitted curve is
    # the timescale to quote.
    tau1_s, tau1_code = dyn.free_draining_relaxation_time(p, u)

    phys = cg.to_phys_params(p, hstar=hstar, hi_method=_hi_method(cfg.hi_method))
    init = bdsim.Initial("fene_fraenkel_bending",
                         dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))
    sim = bdsim.SimParams()
    sim.dt = cfg.dt
    sim.implicit_loop_tol = cfg.implicit_loop_tol

    M_kg = L_bp * G_PER_MOL_PER_BP * 1e-3
    # a/<Q>: bead radius against bond length, both in Hookean units. Past ~0.5 the
    # beads overlap and the RPY tensor is outside the regime it is meant for.
    lH_bp = p.dQ / p.dQ_H
    a_over_Q = np.sqrt(np.pi) * hstar / (np.sqrt(p.Q2_segment) / lH_bp)
    if cfg.variance_reduction and max(wi_list) > 0.3:
        print("  note: variance reduction is being used above Wi ~ 0.3, where it "
              "costs more than it saves", flush=True)
    print(f"  L={L_bp:g} Ns={Ns}: ls/lp={p.ls/LP_DNA:.2f} sqrtb={p.dQ_H:.3g} "
          f"C={p.C:.3g} | h*={hstar:.4g} a={info['bead_radius_nm']:.3g}nm "
          f"lam_H={u['lambda_H']:.3g}s tau1={tau1_s:.4g}s "
          f"D={u['D']*1e12:.4g}um2/s", flush=True)
    if np.isfinite(a_over_Q) and a_over_Q > 0.5:
        print(f"    warning: a/<Q> = {a_over_Q:.2f} -- the beads overlap, and the "
              f"RPY tensor is being used outside its intended regime", flush=True)

    out = []
    for wi in wi_list:
        gd_code = wi / tau1_code
        phys.flow = bdsim.flows.shear(gd_code)
        t_eq, t_run = cfg.eq_taus * tau1_code, cfg.run_taus * tau1_code
        samples = np.linspace(t_eq, t_eq + t_run, cfg.n_samples)

        traj_dir = None
        if cfg.save_traj:
            traj_dir = os.path.join(cfg.save_traj, f"L{L_bp:.0f}_Ns{Ns}_wi{wi:g}")

        series = _measure(phys, sim, gd_code, samples, init, cfg, traj_dir,
                          dict(L=L_bp, Ns=Ns, wi=wi, hstar=hstar,
                               lambda_H=u["lambda_H"], tau1_s=tau1_s,
                               gammadot_per_s=gd_code / u["lambda_H"],
                               RH_nm=rh_nm,
                               bead_radius_nm=info["bead_radius_nm"],
                               temperature=cfg.temperature,
                               solvent_viscosity=cfg.viscosity))

        interval = samples[1] - samples[0]
        stats, se_between = bdsim.trajectory_ensemble_stats(series, interval)
        # Convert stress -> intrinsic viscosity. The factor is positive and
        # independent of the measurement, so the error transforms with it; do NOT
        # reconstruct the error bar as (value x relative error), because the value
        # itself can come out negative when the run is noise-dominated.
        to_intrinsic = kT * u["lambda_H"] * N_A / (M_kg * cfg.viscosity) * 1000.0
        intr = stats.mean * to_intrinsic
        intr_err = abs(stats.stderr * to_intrinsic)
        rel = stats.stderr / abs(stats.mean) if stats.mean else float("nan")
        rec = dict(L=L_bp, Ns=Ns, wi=wi, gammadot_per_s=gd_code / u["lambda_H"],
                   lambda_eta_s=dyn.viscometric_relaxation_time(stats.mean, u),
                   eta_code=stats.mean, eta_err_code=stats.stderr,
                   eta_over_n=stats.mean * kT * u["lambda_H"],
                   intrinsic_mL_g=intr, intrinsic_err_mL_g=intr_err,
                   rel_err=rel, g=stats.g, n_eff=stats.n_effective,
                   tau1_s=tau1_s, D_um2_s=u["D"] * 1e12, lambda_H=u["lambda_H"],
                   hstar=hstar, bead_radius_nm=info["bead_radius_nm"],
                   RH_nm=rh_nm, n_traj_used=int(np.shape(series)[0]),
                   traj_dir=traj_dir, warning=stats.warning)
        out.append(rec)
        flag = "  <-- non-positive: noise-dominated, not a measurement" if intr <= 0 else ""
        lost = cfg.n_traj - rec["n_traj_used"]
        flag += f"  [{lost} trajectories lost]" if lost else ""
        print(f"    Wi={wi:5.3g}  gd={rec['gammadot_per_s']:9.3g}/s  "
              f"[eta]={intr:9.4g} +/- {intr_err:8.3g} mL/g   "
              f"(g={stats.g:.1f}, N_eff={stats.n_effective:.0f}){flag}", flush=True)
    return out


def _measure(phys, sim, rate, samples, init, cfg, traj_dir, manifest_extra):
    """The viscosity series, optionally keeping the configurations too.

    Without `save_traj` this is one call into bdsim.rheology. With it, the
    per-trajectory function returns the snapshots alongside the series and the
    files are written here, in the parent: that keeps the worker function free of
    any notion of which trajectory it is, at the cost of holding the snapshots in
    memory (n_traj x n_samples x N x 3 x 8 bytes -- 8 MB for 50 x 300 x 21).
    """
    import bdsim
    if not traj_dir:
        return bdsim.rheology.shear_viscosity_series(
            phys, sim, rate, cfg.n_traj, samples, seed=1, initial=init,
            variance_reduction=cfg.variance_reduction, backend="processes")

    from bdsim import storage
    results = bdsim.run_ensemble(
        phys, sim, cfg.n_traj, viscosity_series_with_states,
        args=(list(samples), rate, cfg.save_forces), seed=1, initial=init,
        backend="processes")

    os.makedirs(traj_dir, exist_ok=True)
    steps = [int(round(t / sim.dt)) for t in samples]
    files, series = [], []
    for index, (vals, positions, forces) in enumerate(results):
        series.append(vals)
        path = storage.trajectory_path(traj_dir, index)
        storage.write_trajectory(
            path, steps, list(samples), positions, forces,
            attrs={"index": index, "dt": float(sim.dt),
                   "n_beads": int(phys.number_of_beads),
                   "shear_rate_code": float(rate),
                   **{k: v for k, v in manifest_extra.items()
                      if isinstance(v, (int, float, bool, str))}},
            compression="gzip")
        files.append(os.path.basename(path))
    storage.write_manifest(traj_dir, {
        "n_trajectories": len(files), "seed": 1, "dt": float(sim.dt),
        "sample_times": list(map(float, samples)),
        "write_forces": bool(cfg.save_forces),
        "shear_rate_code": float(rate),
        "initial": {"method": init.method, "kwargs": init.kwargs},
        "files": files, **manifest_extra})
    np.save(os.path.join(traj_dir, "viscosity_series.npy"), np.asarray(series))
    return np.asarray(series)


def viscosity_series_with_states(R0, phys, sim, rng, sample_times, rate,
                                 write_forces):
    """Per-trajectory: the viscosity series AND the configurations behind it.

    A `run_ensemble` function, so it must live at module level to be picklable.
    Returns (series, positions, forces_or_None). The configurations come from the
    same integration that produces the viscosity, at no extra simulation cost, and
    the random stream is consumed exactly as `bdsim.rheology.viscosity_series`
    consumes it -- so for a given seed the two agree.
    """
    from bdsim import properties as props
    from bdsim._bdsim import total_force
    from bdsim.ensemble import trajectory_samples

    vals, positions, forces = [], [], ([] if write_forces else None)
    for _t, R in trajectory_samples(R0, phys, sim, rng, sample_times):
        F = total_force(R, phys)
        vals.append(-float(props.kramers_stress(R, F)[0, 1]) / rate)
        positions.append(np.array(R, dtype=np.float64, copy=True))
        if write_forces:
            forces.append(np.asarray(F, dtype=np.float64))
    return vals, positions, forces


def run(cfg: Config = None):
    """Run the sweep described by `cfg`, appending to its JSON file.

    Returns the full list of records. Safe to interrupt: the JSON is rewritten
    after each (L, Ns).
    """
    import time
    cfg = cfg or CONFIG

    recs = json.load(open(cfg.out)) if os.path.exists(cfg.out) else []
    n_points = len(cfg.L) * len(cfg.Ns) * len(cfg.wi)
    print(f"plan: {len(cfg.L)} length(s) x {len(cfg.Ns)} discretisation(s) x "
          f"{len(cfg.wi)} shear rate(s) = {n_points} points, "
          f"{cfg.n_traj} trajectories each, HI on ({cfg.hi_method})")
    print(f"      dt = {cfg.dt:g}, implicit_loop_tol = {cfg.implicit_loop_tol:g}")
    if min(cfg.wi) < 0.3 and not cfg.variance_reduction:
        print("      note: Wi < 0.3 is noise-dominated without variance_reduction")
    if cfg.save_traj:
        print(f"      trajectories -> {cfg.save_traj}/")
    print()

    done, t_start = 0, time.time()
    for L in cfg.L:
        for Ns in cfg.Ns:
            recs = [r for r in recs if not (r["L"] == L and r["Ns"] == Ns)]
            recs += one_case(L, Ns, list(cfg.wi), cfg)
            json.dump(recs, open(cfg.out, "w"), indent=1)
            done += len(cfg.wi)
            elapsed = time.time() - t_start
            if done < n_points:
                eta = elapsed / done * (n_points - done)
                print(f"    [{done}/{n_points} points, {elapsed/60:.1f} min elapsed, "
                      f"~{eta/60:.0f} min remaining]\n", flush=True)
    print(f"\n{len(recs)} records -> {cfg.out}  "
          f"({(time.time()-t_start)/60:.1f} min)")
    return recs


def plot(path: str = None, xkey: str = None):
    """Summarise and plot an existing sweep. No simulation, no compiled extension."""
    path = path or CONFIG.out
    xkey = xkey or CONFIG.x
    recs = json.load(open(path))
    summarise(recs)
    print()
    make_plot(recs, os.path.splitext(path)[0] + ".png", xkey=xkey,
              title="DNA with HI, $h^*$ matched to $R_H$; Wi placed on the "
                    "free-draining $\\tau_1$")
    return recs


if __name__ == "__main__":
    run()
    plot()

"""Free-draining viscosity sweep: several DNA lengths, several discretisations.

Runs the full pipeline -- static coarse-graining, dynamic matching, real units --
with hydrodynamic interaction switched off, which is 30-40x cheaper per step and so
lets a whole sweep be done quickly. The diffusivity is matched exactly (a = R_H/N,
Section 11.5), so the timescale is meaningful even though the internal relaxation
spectrum is Rouse rather than Zimm; treat the curves as indicative shapes, not as
quantitative rheology.

Shear rates are set by Weissenberg number using the free-draining longest
relaxation time, which is known in closed form, so the same Wi means the same
degree of stretching at every L and Ns and the curves are comparable.

All the parameters live in `CONFIG` below. Edit it and run the file:

    python validation/viscosity_sweep.py

or drive it from your own script, which is the point of the Config object:

    from viscosity_sweep import Config, run, plot
    run(Config(L=(2000.0, 10000.0), Ns=(20, 30), n_traj=200,
               out="my_sweep.json"))
    plot("my_sweep.json", xkey="wi_eta")

Results accumulate in the JSON file and each (L, Ns) pair is replaced when re-run,
so a long sweep can be done in pieces.
"""
import json, math, os, sys, pathlib
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
# bdsim is imported lazily inside one_case(): plotting an existing sweep should not
# require the compiled extension to be built.

LP_DNA, BP_NM, G_PER_MOL_PER_BP, N_A = 147.0, 0.34, 650.0, 6.02214076e23


def one_case(L_bp, Ns, wi_list, cfg):
    import bdsim
    from bdsim import coarse_grain as cg
    from bdsim import dynamics as dyn

    kT = dyn.KB * cfg.temperature
    cap = (cfg.max_H_pN_per_nm * BP_NM ** 2 / (kT * 1e21)
           if cfg.max_H_pN_per_nm else None)
    p = cg.spring_parameters(L_bp, LP_DNA, Ns, bending="match_chain", max_H=cap)

    rh_nm = dyn.dna_hydrodynamic_radius_nm(L_bp, cfg.temperature, cfg.viscosity,
                                           length_unit_nm=BP_NM)
    hs_nom, info = dyn.free_draining_units(p, rh_nm, cfg.temperature,
                                           cfg.viscosity, BP_NM)
    u = dyn.physical_units(p, hs_nom, cfg.temperature, cfg.viscosity,
                           length_unit_nm=BP_NM, hydrodynamic_radius_nm=rh_nm)
    tau1_s, tau1_code = dyn.free_draining_relaxation_time(p, u)

    # HI OFF for the run; hs_nom was only for the unit conversion.
    phys = cg.to_phys_params(p, hstar=0.0)
    init = bdsim.Initial("fene_fraenkel_bending",
                         dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))
    sim = bdsim.SimParams(); sim.dt = cfg.dt; sim.implicit_loop_tol = 1e-4

    M_kg = L_bp * G_PER_MOL_PER_BP * 1e-3
    if cfg.variance_reduction and max(wi_list) > 0.3:
        print("  note: variance reduction is being used above Wi ~ 0.3, where it "
              "costs more than it saves", flush=True)
    print(f"  L={L_bp:g} Ns={Ns}: ls/lp={p.ls/LP_DNA:.2f} sqrtb={p.dQ_H:.3g} "
          f"C={p.C:.3g} | a={info['bead_radius_nm']:.3g}nm lam_H={u['lambda_H']:.3g}s "
          f"tau1={tau1_s:.4g}s D={u['D']*1e12:.4g}um2/s", flush=True)

    out = []
    for wi in wi_list:
        gd_code = wi / tau1_code
        phys.flow = bdsim.flows.shear(gd_code)
        t_eq, t_run = cfg.eq_taus * tau1_code, cfg.run_taus * tau1_code
        samples = np.linspace(t_eq, t_eq + t_run, cfg.n_samples)
        series = bdsim.rheology.shear_viscosity_series(
            phys, sim, gd_code, cfg.n_traj, samples, seed=1, initial=init,
            variance_reduction=cfg.variance_reduction, backend="processes")
        interval = samples[1] - samples[0]
        stats, se_between = bdsim.trajectory_ensemble_stats(series, interval)
        # Convert stress -> intrinsic viscosity. The factor is positive and
        # independent of the measurement, so the error transforms with it; do NOT
        # reconstruct the error bar as (value x relative error), because the value
        # itself can come out negative when the run is noise-dominated.
        to_intrinsic = kT * u["lambda_H"] * N_A / (M_kg * cfg.viscosity) * 1000.0
        eta_over_n = stats.mean * kT * u["lambda_H"]
        intr = stats.mean * to_intrinsic
        intr_err = abs(stats.stderr * to_intrinsic)
        rel = stats.stderr / abs(stats.mean) if stats.mean else float("nan")
        # lambda_eta here is eta_p(gammadot)/(n k_B T), i.e. AT THIS SHEAR RATE --
        # it thins with eta_p. The zero-shear lambda_eta,0 needs the whole curve
        # extrapolated to gammadot -> 0; annotate() does that per (L, Ns).
        rec = dict(L=L_bp, Ns=Ns, wi=wi, gammadot_per_s=gd_code / u["lambda_H"],
                   lambda_eta_s=dyn.viscometric_relaxation_time(stats.mean, u),
                   eta_code=stats.mean, eta_err_code=stats.stderr,
                   eta_over_n=eta_over_n, intrinsic_mL_g=intr,
                   intrinsic_err_mL_g=intr_err,
                   rel_err=rel, g=stats.g, n_eff=stats.n_effective,
                   tau1_s=tau1_s, D_um2_s=u["D"] * 1e12, lambda_H=u["lambda_H"],
                   warning=stats.warning)
        out.append(rec)
        flag = "  <-- non-positive: noise-dominated, not a measurement" if intr <= 0 else ""
        print(f"    Wi={wi:5.3g}  gd={rec['gammadot_per_s']:9.3g}/s  "
              f"[eta]={intr:9.4g} +/- {intr_err:8.3g} mL/g   "
              f"(g={stats.g:.1f}, N_eff={stats.n_effective:.0f}){flag}", flush=True)
    return out


def derive(r):
    """Fill in every quantity derivable from what is stored.

    The expensive part of a sweep is the simulation, and each record already
    carries enough to reconstruct the derived quantities: eta in code units, the
    time unit lambda_H, and the shear rate. So nothing here needs a re-run --
    including for records written before a given field existed.

    Adds, where they are missing:
      eta_err_code       absolute error on eta in code units
      lambda_eta_s       eta_p(gammadot)/(n kT) in seconds -- the SHEAR-RATE
                         DEPENDENT viscometric time, which shear-thins with the
                         viscosity it is built from. The zero-shear lambda_eta,0
                         is a property of the group, not of one record: see
                         zero_shear_viscosity() and annotate().
      lambda_eta_err_s   its error
      intrinsic_err_mL_g absolute error on [eta]
    """
    d = dict(r)
    nan = float("nan")
    lam_H = d.get("lambda_H")

    if "eta_err_code" not in d and "eta_code" in d and "rel_err" in d:
        d["eta_err_code"] = abs(d["eta_code"] * d["rel_err"])

    # eta_p/(n kT) is exactly eta in code units; lambda_H converts it to seconds.
    if lam_H is not None and "eta_code" in d:
        d["lambda_eta_s"] = d["eta_code"] * lam_H
    if lam_H is not None and "eta_err_code" in d:
        d["lambda_eta_err_s"] = abs(d["eta_err_code"]) * lam_H

    if "intrinsic_err_mL_g" not in d:
        d["intrinsic_err_mL_g"] = abs(d.get("intrinsic_mL_g", nan)
                                      * d.get("rel_err", nan))
    d.pop("wi_eta", None)   # group-level; annotate() puts back a correct one
    return d


def _carreau(wi, eta0, lam, n):
    return eta0 * (1.0 + (lam * wi) ** 2) ** ((n - 1.0) / 2.0)


def zero_shear_viscosity(rs):
    """Estimate eta_p0 in code units for one (L, Ns) group.

    eta_p is measured at a finite shear rate, so eta_p/(n kT) thins along with the
    viscosity it is built from -- it is not lambda_eta,0. Getting the zero-shear
    limit means extrapolating gammadot -> 0, which is a property of the whole
    curve rather than of any single run.

    A Carreau form eta = eta0 [1 + (lam Wi)^2]^{(n-1)/2} is fitted in log space,
    weighted by the error bars. With fewer than three usable points, or if the fit
    fails or runs away, the lowest-Wi point is used instead -- which is biased low
    by however much that point has already thinned.

    Returns (eta0_code, method, note) with method in {"carreau", "lowest-Wi",
    "none"} and note a human-readable caveat (empty if none applies).
    """
    good = [r for r in rs if r.get("eta_code", 0) > 0 and r.get("wi", 0) > 0]
    good.sort(key=lambda z: z["wi"])
    if not good:
        return None, "none", "no positive eta"
    lo = good[0]
    fallback = (lo["eta_code"], "lowest-Wi",
                f"single-point stand-in at Wi={lo['wi']:g}; biased low by however "
                f"much that point has thinned")
    if len(good) < 3:
        return fallback

    wi = np.array([r["wi"] for r in good])
    eta = np.array([r["eta_code"] for r in good])
    err = np.array([abs(r.get("eta_err_code", np.nan)) for r in good])
    sig = np.where(np.isfinite(err) & (err > 0), err / eta, 0.1)  # log-space sigma

    try:
        from scipy.optimize import least_squares
        resid = lambda p: (np.log(_carreau(wi, *p)) - np.log(eta)) / sig
        p0 = [eta[0] * 1.5, 1.0 / wi[0], 0.5]
        fit = least_squares(resid, p0, bounds=([1e-12, 1e-6, 0.0],
                                               [np.inf, 1e6, 1.0]))
        eta0 = float(fit.x[0])
    except Exception:
        return fallback

    # Sanity checks. With no data in the Newtonian plateau the Carreau fit is an
    # extrapolation, and a three-point fit through pure thinning can send lam and
    # eta0 off together without the residual noticing. Two guards catch that:
    #   - eta0 far above the lowest measured point is not a measurement;
    #   - lambda_eta,0 cannot much exceed tau_1. For a Rouse chain
    #     lambda_eta,0 = (pi^2/12) tau_1 ~ 0.82 tau_1 (Section 11.7), so anything
    #     past ~2 tau_1 is the fit running away, not physics.
    if not np.isfinite(eta0) or eta0 <= 0:
        return fallback
    if eta0 > 10.0 * eta[0]:
        return (fallback[0], fallback[1],
                f"Carreau fit rejected: it lifted eta0 {eta0/eta[0]:.1f}x above "
                f"the lowest-Wi point, with no plateau data to pin it down. "
                f"{fallback[2]}")
    lam_H, tau1 = lo.get("lambda_H"), lo.get("tau1_s")
    if lam_H and tau1 and eta0 * lam_H > 2.0 * tau1:
        return (fallback[0], fallback[1],
                f"Carreau fit rejected: gave lambda_eta,0 = "
                f"{eta0 * lam_H / tau1:.1f} tau_1, but it cannot much exceed "
                f"tau_1 (Rouse: 0.82 tau_1). {fallback[2]}")

    note = ""
    if wi[0] > 0.5:
        note = (f"extrapolated from Wi >= {wi[0]:g}, with no data in the "
                f"Newtonian plateau -- indicative only")
    if lam_H and tau1:
        note = (note + "; " if note else "") + \
            f"lambda_eta,0 = {eta0 * lam_H / tau1:.2f} tau_1 (Rouse: 0.82)"
    return eta0, "carreau", note


def annotate(records):
    """derive() each record, then add the group-level zero-shear quantities.

    Adds per (L, Ns) group: eta0_code, lambda_eta0_s = eta_p0/(n kT), the method
    used to get it, and wi_eta = gammadot * lambda_eta,0 -- a Weissenberg number
    built on a constant per curve, so switching the x axis to it rescales each
    curve rigidly rather than distorting it.
    """
    rs = [derive(r) for r in records]
    groups = {}
    for r in rs:
        groups.setdefault((r["L"], r["Ns"]), []).append(r)
    for g in groups.values():
        g.sort(key=lambda z: z["wi"])
        eta0, method, note = zero_shear_viscosity(g)
        # The same tau_1 check applied to whatever survived, fit or fallback: a
        # lambda_eta,0 well past tau_1 means the measurement and the free-draining
        # tau_1 disagree, which is worth seeing rather than hiding.
        lam_H0, tau1_0 = g[0].get("lambda_H"), g[0].get("tau1_s")
        if eta0 and lam_H0 and tau1_0 and eta0 * lam_H0 > 2.0 * tau1_0:
            note = (note + "; " if note else "") + \
                (f"lambda_eta,0 = {eta0 * lam_H0 / tau1_0:.1f} tau_1, far above "
                 f"the Rouse 0.82 -- eta_p and the free-draining tau_1 disagree here")
        for r in g:
            lam_H = r.get("lambda_H")
            if eta0 is None or not lam_H:
                continue
            r["eta0_code"] = eta0
            r["lambda_eta0_s"] = eta0 * lam_H
            r["zero_shear_method"] = method
            r["zero_shear_note"] = note
            if "gammadot_per_s" in r:
                r["wi_eta"] = r["gammadot_per_s"] * r["lambda_eta0_s"]
    return rs


def _err_of(r):
    """Absolute error on [eta], tolerating records written before it was stored."""
    return abs(derive(r).get("intrinsic_err_mL_g", float("nan")))


def summarise(records):
    """Print the derived quantities for a sweep."""
    rs = annotate(records)
    nan = float("nan")
    print(f"{'L(kbp)':>8} {'Ns':>4} {'Wi':>6} {'Wi_eta':>7} {'gd (1/s)':>10} "
          f"{'[eta] mL/g':>12} {'+/-':>9} {'lam_eta (s)':>12} "
          f"{'lam_eta,0':>11} {'tau1 (s)':>10}")
    for r in sorted(rs, key=lambda z: (z["L"], z["Ns"], z["wi"])):
        print(f"{r['L']/1000:8.3g} {r['Ns']:4d} {r['wi']:6.3g} "
              f"{r.get('wi_eta', nan):7.3g} {r['gammadot_per_s']:10.3g} "
              f"{r['intrinsic_mL_g']:12.4g} {r.get('intrinsic_err_mL_g', nan):9.3g} "
              f"{r.get('lambda_eta_s', nan):12.4g} "
              f"{r.get('lambda_eta0_s', nan):11.4g} "
              f"{r.get('tau1_s', nan):10.3g}")
    print("\nlam_eta is eta_p(gammadot)/(n k_B T) at that shear rate, so it thins "
          "with the viscosity.\nlam_eta,0 is the gammadot -> 0 limit, one value "
          "per (L, Ns):")
    seen = set()
    for r in sorted(rs, key=lambda z: (z["L"], z["Ns"])):
        key = (r["L"], r["Ns"])
        if key in seen or "lambda_eta0_s" not in r:
            continue
        seen.add(key)
        print(f"  L={r['L']/1000:g} kbp, Ns={r['Ns']}: lam_eta,0 = "
              f"{r['lambda_eta0_s']:.4g} s  [{r['zero_shear_method']}]"
              + (f"\n      {r['zero_shear_note']}" if r.get("zero_shear_note") else ""))


def make_plot(records, path, xkey="wi", title=None):
    """Plot a sweep. xkey selects the abscissa:

      "wi"      Wi = gammadot * tau_1, the longest relaxation time (default)
      "wi_eta"  Wi = gammadot * lambda_eta,0, the ZERO-SHEAR viscometric time

    `title` overrides the figure heading (the HI sweep passes its own).

    Both are a constant per curve, so the choice rescales each curve rigidly.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = sorted({r["L"] for r in records})
    Nss = sorted({r["Ns"] for r in records})
    bad = [r for r in records if not (r["intrinsic_mL_g"] > 0)]
    if bad:
        print(f"note: {len(bad)} of {len(records)} points have [eta] <= 0 "
              f"(noise-dominated, typically the lowest Wi). They are dropped from "
              f"the log axes; the values remain in the JSON.")
        for r in bad:
            print(f"      L={r['L']:g} Ns={r['Ns']} Wi={r['wi']:g}: "
                  f"[eta] = {r['intrinsic_mL_g']:.4g} +/- {_err_of(r):.4g}")
    records = annotate(records)
    xlab = {"wi": r"$\mathrm{Wi} = \dot\gamma\,\tau_1$",
            "wi_eta": r"$\mathrm{Wi}_\eta = \dot\gamma\,\lambda_{\eta,0}$"}[xkey]
    missing = [r for r in records if not np.isfinite(r.get(xkey, np.nan))]
    if missing:
        print(f"note: {len(missing)} points have no {xkey}; falling back to Wi for them")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    marks = {20: "o", 30: "s", 40: "^", 10: "v"}
    for i, L in enumerate(Ls):
        col = f"C{i}"
        for Ns in Nss:
            rs = sorted([r for r in records if r["L"] == L and r["Ns"] == Ns
                         and r["intrinsic_mL_g"] > 0], key=lambda r: r["wi"])
            if not rs:
                continue
            x = np.array([r.get(xkey, r["wi"]) for r in rs])
            it = np.array([r["intrinsic_mL_g"] for r in rs])
            er = np.array([_err_of(r) for r in rs])
            le = np.array([r.get("lambda_eta_s", np.nan) for r in rs])
            lee = np.array([r.get("lambda_eta_err_s", np.nan) for r in rs])
            ax[0].errorbar(x, it, er, fmt=marks.get(Ns, "o") + "-", color=col,
                           capsize=3, ms=5,
                           label=f"{L/1000:g} kbp, $N_s$={Ns}")
            ax[1].errorbar(x, it / it[0], er / it[0], fmt=marks.get(Ns, "o") + "-",
                           color=col, capsize=3, ms=5)
            if np.any(np.isfinite(le)):
                ax[2].errorbar(x, le, lee, fmt=marks.get(Ns, "o") + "-",
                               color=col, capsize=3, ms=5)
            # the zero-shear limit the finite-shear points are thinning away from
            l0 = rs[0].get("lambda_eta0_s")
            if l0:
                ax[2].axhline(l0, color=col, ls=":", lw=1, alpha=0.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(xlab); ax[0].set_ylabel(r"$[\eta]$ (mL/g)")
    ax[0].set_title("intrinsic viscosity"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(xlab); ax[1].set_ylabel(r"$[\eta]/[\eta]_{\mathrm{first}}$")
    ax[1].set_title("shear thinning, normalised"); ax[1].grid(alpha=0.3)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel(xlab); ax[2].set_ylabel(r"$\lambda_\eta$ (s)")
    ax[2].set_title(r"$\eta_p(\dot\gamma)/(nk_BT)$; dotted: $\lambda_{\eta,0}$")
    ax[2].grid(alpha=0.3)
    fig.suptitle(title or
                 "Free-draining DNA, dynamics matched (indicative, not quantitative)")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"saved {path}")


@dataclass
class Config:
    """Everything the sweep needs. Edit CONFIG below, or build your own."""

    # What to sweep.
    L: tuple = (48502.0,)          # DNA contour lengths, base pairs
    Ns: tuple = (20, 30)           # springs per chain
    wi: tuple = (0.3, 1.0, 3.0, 10.0)   # Weissenberg numbers

    # Where results go. Records accumulate, and each (L, Ns) pair is replaced
    # when re-run, so a long sweep can be done in pieces.
    out: str = "validation/viscosity_sweep.json"

    # Physical conditions.
    temperature: float = 298.15    # K
    viscosity: float = 1.0e-3      # solvent, Pa s
    max_H_pN_per_nm: float = 0.05  # cap on the spring constant; None for no cap

    # Time stepping and sampling.
    dt: float = 0.01
    eq_taus: float = 3.0           # equilibration, in units of tau_1
    run_taus: float = 8.0          # production, in units of tau_1
    n_samples: int = 80
    n_traj: int = 8

    # Pair each run with an equilibrium one on the same random stream and
    # subtract. Worth it below Wi ~ 0.1, harmful above ~0.3.
    variance_reduction: bool = False

    # Plotting: "wi" = gammadot*tau_1, "wi_eta" = gammadot*lambda_eta,0.
    x: str = "wi"


CONFIG = Config()
"""The parameters this script runs with. Edit here, then `python viscosity_sweep.py`,
or import and call `run(Config(...))` from your own script."""


def run(cfg: Config = None):
    """Run the sweep described by `cfg`, appending to its JSON file.

    Returns the full list of records (the ones just computed plus any already in
    the file). Safe to interrupt: the JSON is rewritten after each (L, Ns).
    """
    import time
    cfg = cfg or CONFIG

    recs = json.load(open(cfg.out)) if os.path.exists(cfg.out) else []
    n_points = len(cfg.L) * len(cfg.Ns) * len(cfg.wi)
    print(f"plan: {len(cfg.L)} length(s) x {len(cfg.Ns)} discretisation(s) x "
          f"{len(cfg.wi)} shear rate(s) = {n_points} points, "
          f"{cfg.n_traj} trajectories each")
    if min(cfg.wi) < 0.3 and not cfg.variance_reduction:
        print("      note: Wi < 0.3 is noise-dominated without variance_reduction")
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
    """Summarise and plot an existing sweep. No simulation, no compiled extension.

    Everything derivable is reconstructed from what is stored, so this works on
    records written before a given field existed.
    """
    path = path or CONFIG.out
    xkey = xkey or CONFIG.x
    recs = json.load(open(path))
    summarise(recs)
    print()
    make_plot(recs, os.path.splitext(path)[0] + ".png", xkey=xkey)
    return recs


if __name__ == "__main__":
    run()
    plot()

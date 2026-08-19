"""Polymer viscosity of lambda-DNA in real units, at several shear rates.

Puts the whole pipeline together: static coarse-graining (spring + bending), the
dynamic match (h* from the measured hydrodynamic radius), and the unit conversions
(l_H, F_H, lambda_H) needed to quote a viscosity in Pa.s rather than code units.

What is computed
----------------
The simulation gives the Kramers-Kirkwood stress per chain, in units of kT, so

    eta_p,code = -<tau_xy> / gammadot_code                     (a time, in lambda_H)
    eta_p / n  = eta_p,code * kT * lambda_H                    (Pa.s.m^3)
    [eta]      = (eta_p/n) * N_A / (M * eta_s)                 (m^3/kg -> mL/g)

with n the chain number density and M the molar mass. eta_p itself needs a
concentration; the concentration-independent quantities are eta_p/n and the
intrinsic viscosity [eta], which are what this reports.

Shear rates are specified in the lab frame (s^-1) and converted with
gammadot_code = gammadot_real * lambda_H. The Weissenberg number is reported using
the viscometric relaxation time lambda_eta = eta_p/(n kT) from the lowest rate,
which is self-consistent and needs no separate relaxation measurement.

Usage:
    python validation/lambda_dna_viscosity.py --quick          # pipeline smoke test
    python validation/lambda_dna_viscosity.py --n-traj 64      # production
"""
import math, sys, pathlib
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim
from bdsim import coarse_grain as cg
from bdsim import dynamics as dyn

LAMBDA_DNA_BP = 48502
LP_DNA = 147.0
BP_NM = 0.34
G_PER_MOL_PER_BP = 650.0
N_A = 6.02214076e23


def run_case(L_bp, Ns, rates_per_s, cfg):
    kT = dyn.KB * cfg.temperature

    # --- static, with the H cap to keep lambda_H (and so the timestep) usable ---
    cap = None
    if cfg.max_H_pN_per_nm:
        cap = cfg.max_H_pN_per_nm * BP_NM ** 2 / (kT * 1e21)   # pN/nm -> kT/bp^2
    p = cg.spring_parameters(L_bp, LP_DNA, Ns, bending="match_chain", max_H=cap)

    # --- dynamic: h* from the measured hydrodynamic radius ---
    rh_nm = dyn.dna_hydrodynamic_radius_nm(L_bp, cfg.temperature, cfg.viscosity,
                                           length_unit_nm=BP_NM)
    hstar, info = dyn.hstar_for_target_rh(p, rh_nm, length_unit_nm=BP_NM,
                                          n_configs=cfg.n_configs)
    u = dyn.physical_units(p, hstar, cfg.temperature, cfg.viscosity,
                           length_unit_nm=BP_NM, hydrodynamic_radius_nm=rh_nm)

    print(f"\n=== Ns = {Ns} ===")
    print(f"  {p.branch};  ls/lp = {p.ls/LP_DNA:.3f}  sqrtb = {p.dQ_H:.4g}  "
          f"sigma_H = {p.sigma_H:.4g}  C = {p.C:.4g}")
    print(f"  H = {p.H*kT*1e21/BP_NM**2:.4g} pN/nm   h* = {hstar:.4g}   "
          f"a = {info['bead_radius_nm']:.4g} nm   R_H = {rh_nm:.4g} nm")
    print(f"  l_H = {u['l_H']*1e9:.4g} nm   F_H = {u['force_H']*1e12:.4g} pN   "
          f"lambda_H = {u['lambda_H']:.4g} s")

    M_kg_per_mol = L_bp * G_PER_MOL_PER_BP * 1e-3
    phys = cg.to_phys_params(p, hstar=hstar, hi_method=bdsim.DelSMethod.Cholesky)
    init = bdsim.Initial("fene_fraenkel_bending",
                         dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))

    sim = bdsim.SimParams()
    sim.dt = cfg.dt
    sim.implicit_loop_tol = 1e-4

    print(f"\n  {'gd (1/s)':>10} {'gd (code)':>11} {'[eta] (mL/g)':>20} {'Wi':>7} "
          f"{'g':>6} {'N_eff':>7} {'SE(traj)':>9}")
    results = []
    for gd_real in rates_per_s:
        gd_code = gd_real * u["lambda_H"]
        phys.flow = bdsim.flows.shear(gd_code)
        samples = np.linspace(cfg.t_eq, cfg.t_eq + cfg.t_run, cfg.n_samples)
        series = bdsim.rheology.shear_viscosity_series(
            phys, sim, gd_code, cfg.n_traj, samples, seed=1,
            initial=init, backend="processes")
        interval = (samples[1] - samples[0]) if len(samples) > 1 else 1.0
        stats, se_between = bdsim.trajectory_ensemble_stats(series, interval)
        m, e = stats.mean, stats.stderr
        eta_over_n = m * kT * u["lambda_H"]          # Pa.s.m^3
        intrinsic = eta_over_n * N_A / (M_kg_per_mol * cfg.viscosity) * 1000.0
        results.append((gd_real, gd_code, m, eta_over_n, intrinsic, stats, se_between))
    lam_eta = results[0][2]                          # eta_p,code at lowest rate = a time
    for gd_real, gd_code, m, eon, intr, stt, seb in results:
        rel = stt.stderr / abs(stt.mean) if stt.mean else float("nan")
        print(f"  {gd_real:10.4g} {gd_code:11.4g} {intr:12.4g} +/-{intr*rel:6.3g} "
              f"{gd_code*lam_eta:7.3g} {stt.g:6.2f} {stt.n_effective:7.1f} "
              f"{seb/abs(stt.mean)*intr if stt.mean else float('nan'):9.3g}")
    print("  (error bars are autocorrelation-corrected; SE(traj) is the independent")
    print("   between-trajectory estimate -- they should agree)")
    for _g, _c, _m, _e, _i, stt, _s in results:
        if stt.warning:
            print(f"  NOTE: {stt.warning}")
            break
    # lambda_eta = M eta_p0 / (c N_A kT) = eta_p0/(n kT): the viscometric
    # relaxation time. It is a weighted sum over the whole spectrum, and for a
    # free-draining Rouse chain equals (N^2-1)/3 in code units -- about 0.82 tau_1,
    # i.e. SHORTER than the slowest mode.
    lam_eta_s = dyn.viscometric_relaxation_time(lam_eta, u)
    rouse = dyn.rouse_viscometric_time(p.n_springs + 1)
    print(f"  lambda_eta = eta_p0/(n kT) at the lowest rate")
    print(f"    = {lam_eta:.4g} code units = {lam_eta_s:.4g} s")
    print(f"    (free-draining Rouse value for N={p.n_springs+1} would be "
          f"{rouse:.4g} code units; ratio {lam_eta/rouse:.3g})")

    # Guard: the chain must be equilibrated for several relaxation times at each
    # rate, or the low-rate points are still relaxing out of their coiled initial
    # state while the high-rate ones have already stretched -- which shows up as
    # spurious shear THICKENING.
    if cfg.t_eq < 3.0 * lam_eta:
        print(f"\n  *** WARNING: t_eq = {cfg.t_eq:g} is only "
              f"{cfg.t_eq/lam_eta:.2f} relaxation times. Use t_eq >~ "
              f"{3*lam_eta:.0f} and t_run >~ {6*lam_eta:.0f}. Until then the")
        print("      apparent shear-rate dependence is an equilibration artefact,")
        print("      not rheology. ***")
    thickening = all(results[i][3] < results[i+1][3] for i in range(len(results)-1))
    if thickening and len(results) > 1:
        print("  *** and indeed eta_p rises with shear rate here, which a polymer")
        print("      solution does not do: treat these numbers as a smoke test. ***")


@dataclass
class Config:
    """Everything this script needs. Edit CONFIG below, or build your own."""

    L: float = LAMBDA_DNA_BP       # contour length, base pairs
    Ns: tuple = (30, 40)           # springs per chain
    rates: tuple = None            # shear rates in 1/s; None => a decade around 1/lambda

    temperature: float = 298.15    # K
    viscosity: float = 1.0e-3      # solvent, Pa s
    max_H_pN_per_nm: float = 0.05  # cap on the spring constant; 0 or None to disable

    n_configs: int = 1500          # Monte-Carlo configurations for the static check
    dt: float = 1e-3
    t_eq: float = 800.0
    t_run: float = 1600.0
    n_samples: int = 40
    n_traj: int = 32


CONFIG = Config()
"""The parameters this script runs with. Edit here, or call run(Config(...))."""

QUICK = Config(n_configs=300, dt=5e-3, t_eq=40.0, t_run=80.0,
               n_samples=12, n_traj=8)
"""A tiny configuration that checks the pipeline end to end. NOT converged."""


def run(cfg: Config = None):
    """Measure the viscosity of lambda-DNA at each Ns in `cfg`."""
    cfg = cfg or CONFIG
    print(f"lambda-DNA: L = {cfg.L:g} bp = {cfg.L*BP_NM*1e-3:.3g} um, "
          f"M = {cfg.L*G_PER_MOL_PER_BP*1e-3:.4g} kg/mol")

    for Ns in cfg.Ns:
        if cfg.rates is None:
            # a decade around the estimated relaxation rate
            rh = dyn.dna_hydrodynamic_radius_nm(cfg.L, cfg.temperature,
                                                cfg.viscosity, length_unit_nm=BP_NM)
            lam_est = 6 * math.pi * cfg.viscosity * (rh * 1e-9) ** 3 / (
                dyn.KB * cfg.temperature)
            use = [0.1 / lam_est, 1.0 / lam_est, 10.0 / lam_est]
        else:
            use = list(cfg.rates)
        run_case(cfg.L, Ns, use, cfg)


if __name__ == "__main__":
    run()

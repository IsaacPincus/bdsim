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
import argparse, math, sys, pathlib
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


def run_case(L_bp, Ns, rates_per_s, args):
    kT = dyn.KB * args.temperature

    # --- static, with the H cap to keep lambda_H (and so the timestep) usable ---
    cap = None
    if args.max_H_pN_per_nm:
        cap = args.max_H_pN_per_nm * BP_NM ** 2 / (kT * 1e21)   # pN/nm -> kT/bp^2
    p = cg.spring_parameters(L_bp, LP_DNA, Ns, bending="match_chain", max_H=cap)

    # --- dynamic: h* from the measured hydrodynamic radius ---
    rh_nm = dyn.dna_hydrodynamic_radius_nm(L_bp, args.temperature, args.viscosity,
                                           length_unit_nm=BP_NM)
    hstar, info = dyn.hstar_for_target_rh(p, rh_nm, length_unit_nm=BP_NM,
                                          n_configs=args.n_configs)
    u = dyn.physical_units(p, hstar, args.temperature, args.viscosity,
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
    sim.dt = args.dt
    sim.implicit_loop_tol = 1e-4

    print(f"\n  {'gd (1/s)':>10} {'gd (code)':>11} {'[eta] (mL/g)':>20} {'Wi':>7} "
          f"{'g':>6} {'N_eff':>7} {'SE(traj)':>9}")
    results = []
    for gd_real in rates_per_s:
        gd_code = gd_real * u["lambda_H"]
        phys.flow = bdsim.flows.shear(gd_code)
        samples = np.linspace(args.t_eq, args.t_eq + args.t_run, args.n_samples)
        series = bdsim.shear_viscosity_series(
            phys, sim, gd_code, args.n_traj, samples, seed=1,
            initial=init, backend="processes")
        interval = (samples[1] - samples[0]) if len(samples) > 1 else 1.0
        stats, se_between = bdsim.trajectory_ensemble_stats(series, interval)
        m, e = stats.mean, stats.stderr
        eta_over_n = m * kT * u["lambda_H"]          # Pa.s.m^3
        intrinsic = eta_over_n * N_A / (M_kg_per_mol * args.viscosity) * 1000.0
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
    if args.t_eq < 3.0 * lam_eta:
        print(f"\n  *** WARNING: t_eq = {args.t_eq:g} is only "
              f"{args.t_eq/lam_eta:.2f} relaxation times. Use t_eq >~ "
              f"{3*lam_eta:.0f} and t_run >~ {6*lam_eta:.0f}. Until then the")
        print("      apparent shear-rate dependence is an equilibration artefact,")
        print("      not rheology. ***")
    thickening = all(results[i][3] < results[i+1][3] for i in range(len(results)-1))
    if thickening and len(results) > 1:
        print("  *** and indeed eta_p rises with shear rate here, which a polymer")
        print("      solution does not do: treat these numbers as a smoke test. ***")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=LAMBDA_DNA_BP)
    ap.add_argument("--Ns", type=int, nargs="+", default=[30, 40])
    ap.add_argument("--rates", type=float, nargs="+", default=None,
                    help="shear rates in s^-1")
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--viscosity", type=float, default=1.0e-3)
    ap.add_argument("--max-H-pN-per-nm", type=float, default=0.05,
                    help="cap on the spring constant (0 to disable)")
    ap.add_argument("--n-configs", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--t-eq", type=float, default=800.0)
    ap.add_argument("--t-run", type=float, default=1600.0)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run: checks the pipeline, NOT converged")
    args = ap.parse_args()

    if args.quick:
        args.n_configs, args.dt = 300, 5e-3
        args.t_eq, args.t_run, args.n_samples, args.n_traj = 40.0, 80.0, 12, 8

    print(f"lambda-DNA: L = {args.L:g} bp = {args.L*BP_NM*1e-3:.3g} um, "
          f"M = {args.L*G_PER_MOL_PER_BP*1e-3:.4g} kg/mol")
    if args.quick:
        print("*** --quick: short run, results are NOT converged ***")

    rates = args.rates
    for Ns in args.Ns:
        # default rates: pick a decade around the estimated relaxation rate
        if rates is None:
            p0 = cg.spring_parameters(args.L, LP_DNA, Ns, bending="match_chain")
            rh = dyn.dna_hydrodynamic_radius_nm(args.L, args.temperature, args.viscosity,
                                                length_unit_nm=BP_NM)
            lam_est = 6 * math.pi * args.viscosity * (rh * 1e-9) ** 3 / (
                dyn.KB * args.temperature)
            use = [0.1 / lam_est, 1.0 / lam_est, 10.0 / lam_est]
        else:
            use = rates
        run_case(args.L, Ns, use, args)


if __name__ == "__main__":
    main()

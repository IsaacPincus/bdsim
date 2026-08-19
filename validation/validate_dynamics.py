"""Matching chain DYNAMICS across levels of coarse-graining.

The static scheme (validate_coarse_graining.py) fixes the chain's size. It leaves
the friction free, so a model can have the right structure and completely the
wrong timescale. This script closes that gap and answers the question that
motivates it:

    given a DNA fragment of length L, produce a bead-spring model that reproduces
    both its structure and its dynamics, with an answer that does not depend on
    how many springs I chose to use.

The procedure:

  1. Static coarse-graining fixes the springs and the bending constant (given Ns).
  2. The chain's hydrodynamic radius is computed from its own equilibrium
     configurations via the Kirkwood formula. It depends on h* and on the static
     structure only -- no dynamics need be run.
  3. h* is solved so that R_H matches the experimental value for that fragment.

Because step 2 depends on Ns (a chain of 10 big beads and one of 80 small beads
shield each other differently), h* comes out Ns-dependent -- that is expected and
is the point. What must NOT depend on Ns is the physical prediction: the
diffusivity (matched by construction) and, the real test, the longest relaxation
time (not matched by construction, so it is a genuine check).

EXPERIMENTAL INPUT. The target R_H must come from measurement. Supply it either
directly (--rh-nm) or as a power law R_H = R0 (L/L0)^nu with your own calibration;
the defaults below are placeholders and should be replaced with values from the
source you are modelling.

Usage:
    python validation/validate_dynamics.py --L 2000 --Ns 5 10 20 40
    python validation/validate_dynamics.py --L 2000 --Ns 5 10 20 --relax
"""
import math, sys, pathlib
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim
from bdsim import coarse_grain as cg
from bdsim import dynamics as dyn

LP_DNA = 147.0          # base pairs
BP_NM = 0.34            # nm per base pair


def target_rh_nm(L_bp, R0_nm, L0_bp, nu):
    """Experimental hydrodynamic radius, as a power law in contour length.

    R_H = R0 (L/L0)^nu. For DNA in a good solvent nu is around 0.6; R0 must come
    from a measured diffusivity via Stokes-Einstein.
    """
    return R0_nm * (L_bp / L0_bp) ** nu


def end_to_end_autocorrelation(phys, dt, t_total, n_samples, n_traj, initial,
                               seed=0, backend="processes"):
    """<R(0).R(t)>/<R^2>, averaged over trajectories and time origins."""
    from bdsim.parallel import parallel_map

    def job(i):
        return (phys, dt, t_total, n_samples, seed + i,
                initial.method, dict(initial.kwargs))

    def worker(j):
        phys_, dt_, T, ns, sd, method, kw = j
        R = bdsim.ensemble.INITIALIZERS[method](phys_.number_of_beads, sd, kw)
        rng = bdsim.Rng(sd)
        sim = bdsim.SimParams(); sim.dt = dt_; sim.implicit_loop_tol = 1e-5
        step = T / ns
        t = 0.0
        vecs = [np.asarray(R)[-1] - np.asarray(R)[0]]
        for _ in range(ns):
            R = bdsim.integrate(R, phys_, bdsim.ensemble._segment(sim, t, t + step), rng)
            t += step
            vecs.append(np.asarray(R)[-1] - np.asarray(R)[0])
        return np.asarray(vecs)

    globals()["_acf_worker"] = worker
    out = parallel_map(_acf_worker, [job(i) for i in range(n_traj)], backend=backend)
    n = len(out[0])
    acf = np.zeros(n)
    cnt = np.zeros(n)
    for v in out:
        for lag in range(n):
            d = (v[lag:] * v[:n - lag]).sum(axis=1)
            acf[lag] += d.sum(); cnt[lag] += len(d)
    acf /= np.maximum(cnt, 1)
    return acf / acf[0], np.arange(n) * (t_total / n_samples)


@dataclass
class Config:
    """Edit CONFIG below, or call run(Config(...)) from your own script."""

    L: float = 2000.0              # contour length, base pairs
    Ns: tuple = (5, 10, 20, 40)    # discretisations to compare
    rh_nm: float = None            # measured hydrodynamic radius; None => DNA calibration
    kuhn_nm: float = dyn.DSDNA_KUHN_NM
    temperature: float = 298.15    # K
    viscosity: float = 1.0e-3      # solvent, Pa s
    n_configs: int = 3000          # Monte-Carlo configurations for the Kirkwood average
    relax: bool = False            # also measure the relaxation time (slow)
    dt: float = None               # None => chosen from the chain
    n_traj: int = 16


CONFIG = Config()


def run(cfg: Config = None):
    """Check that the dynamic matching reproduces the target diffusivity."""
    cfg = cfg or CONFIG

    if cfg.rh_nm:
        rh_nm, src = cfg.rh_nm, "supplied"
        D_exp = dyn.experimental_hydrodynamic_radius  # unused; kept for clarity
        D_exp_val = dyn.KB * cfg.temperature / (6 * math.pi * cfg.viscosity * rh_nm * 1e-9)
    else:
        D_exp_val = dyn.dna_diffusivity(cfg.L, cfg.temperature, cfg.viscosity,
                                        cfg.kuhn_nm, BP_NM)
        rh_nm = dyn.dna_hydrodynamic_radius_nm(cfg.L, cfg.temperature,
                                               cfg.viscosity, cfg.kuhn_nm, BP_NM)
        src = "dsDNA calibration (Zimm scale x measured D/D_Zimm)"
    L_um = cfg.L * BP_NM * 1e-3
    dz = dyn.zimm_diffusivity(cfg.L * BP_NM * 1e-9, cfg.kuhn_nm * 1e-9,
                              cfg.temperature, cfg.viscosity)
    print(f"L = {cfg.L:g} bp = {L_um:.4g} um,  lp = {LP_DNA:g} bp,  b = {cfg.kuhn_nm:g} nm")
    print(f"  D_Zimm       = {dz*1e12:.4g} um^2/s")
    print(f"  D/D_Zimm     = {D_exp_val/dz:.4g}   [{src}]")
    print(f"  D experiment = {D_exp_val*1e12:.4g} um^2/s")
    print(f"  target R_H   = {rh_nm:.4g} nm\n")

    print("STEP 1-2: static fit fixes l_H (nm); STEP 3: h* is solved so the model's")
    print("R_H matches in NANOMETRES -- R_H is a length, so no time unit is involved.\n")
    print(f"{'Ns':>4} {'ls/lp':>7} {'l_H (nm)':>9} {'sqrtb':>7} {'C':>7} | {'h*':>8} "
          f"{'a (nm)':>8} {'R_H (nm)':>9}")
    rows = []
    for Ns in cfg.Ns:
        p = cg.spring_parameters(cfg.L, LP_DNA, Ns, bending="match_chain")
        try:
            hstar, info = dyn.hstar_for_target_rh(p, rh_nm, length_unit_nm=BP_NM,
                                                  n_configs=cfg.n_configs)
        except ValueError as e:
            print(f"{Ns:4d} {p.ls/LP_DNA:7.3f} {p.dQ_H:8.3f} {p.C:8.3f} |  {e}")
            continue
        rows.append((Ns, p, hstar, info))
        print(f"{Ns:4d} {p.ls/LP_DNA:7.3f} {info['hookean_unit_nm']:9.3f} {p.dQ_H:7.3f} "
              f"{p.C:7.3f} | {hstar:8.4f} {info['bead_radius_nm']:8.3f} "
              f"{info['achieved_RH_hookean']*info['hookean_unit_nm']:9.3f}")

    print("\n  h* varies strongly with Ns -- as it must: fewer, larger beads shield")
    print("  each other differently. The physical prediction is what should not vary.")

    # STEP 4: real units. Both Hookean units move with the coarse-graining -- l_H
    # with the static fit and lambda_H with h* -- so a diffusivity quoted in code
    # units is NOT comparable between rows: each is in a different second.
    print("\nSTEP 4: real units. lambda_H is the simulation's second; it varies by")
    print("orders of magnitude, and cancels against the code-unit diffusivity.\n")
    print(f"{'Ns':>4} | {'l_H (nm)':>9} {'F_H (pN)':>9} {'lambda_H (s)':>13} | "
          f"{'D_K (code)':>10} {'D (um^2/s)':>11} {'H (pN/nm)':>10}")
    for Ns, p, hstar, info in rows:
        u = dyn.physical_units(p, hstar, cfg.temperature, cfg.viscosity,
                               length_unit_nm=BP_NM, hydrodynamic_radius_nm=rh_nm)
        print(f"{Ns:4d} | {u['l_H']*1e9:9.4g} {u['force_H']*1e12:9.4g} "
              f"{u['lambda_H']:13.3e} | {info['D_kirkwood_hookean']:10.5f} "
              f"{u['D']*1e12:11.4f} {u['spring_H']*1e3:10.4g}")
    print("\n  The three unit conversions are l_H (length), F_H (force) and lambda_H")
    print("  (time); energies are in kT. Multiply any simulation output by these.")
    print("\n  D is identical for every Ns: it is fixed by R_H through")
    print("  Stokes-Einstein, D = kT/(6 pi eta_s R_H), and R_H was matched in nm.")
    print("  Use lambda_H to convert any simulated time into seconds.")

    # physical diffusivity: D_phys = D_hookean * lH^2 / tau_H, and the Hookean time
    # unit itself depends on the friction, so compare the dimensionless combination
    # that is actually fixed: R_H (matched) and R_g / R_H (a pure shape ratio).
    print(f"\n{'Ns':>4} {'R_H (nm)':>10} {'Rg (nm)':>10} {'Rg/R_H':>9} {'a/<Q>':>8}  overlap?")
    for Ns, p, hstar, info in rows:
        N = Ns + 1
        cfgs = [bdsim.fene_fraenkel_bending_chain(N, p.sigma_H, p.dQ_H, p.C, seed=i)
                for i in range(1000)]
        rg = np.mean([bdsim.radius_of_gyration_sq(c) for c in cfgs]) ** 0.5
        Qbar = np.mean([np.linalg.norm(np.diff(c, axis=0), axis=1).mean() for c in cfgs])
        lH = info["hookean_unit_nm"]
        a_over_Q = math.sqrt(math.pi) * hstar / Qbar
        flag = "  BEADS OVERLAP" if a_over_Q > 0.5 else ""
        print(f"{Ns:4d} {info['achieved_RH_hookean']*lH:10.3f} {rg*lH:10.3f} "
              f"{rg/info['achieved_RH_hookean']:9.3f} {a_over_Q:8.3f}{flag}")

    print("\n  Note on h*: the familiar 'h* <~ 0.3' rule is for flexible springs, where")
    print("  the Hookean unit is comparable to the bond length. Stiff FENE-Fraenkel")
    print("  springs have l_H much smaller than <Q>, so h* > 1 can be perfectly")
    print("  physical. The meaningful check is a/<Q>, the bead radius relative to the")
    print("  bond length: beyond ~0.5 the beads overlap and the RPY tensor is being")
    print("  used outside the regime it is meant for.")

    if cfg.relax and rows:
        print(f"\n{'Ns':>4} {'tau_1 (Hookean)':>16} {'tau_1 * D_K / R_H^2':>21}")
        print("  (the dimensionless combination should be Ns-independent if the")
        print("   dynamics are properly matched)")
        for Ns, p, hstar, info in rows:
            N = Ns + 1
            phys = cg.to_phys_params(p, hstar=hstar,
                                     hi_method=bdsim.DelSMethod.Cholesky)
            init = bdsim.Initial("fene_fraenkel_bending",
                                 dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))
            dt = cfg.dt or min(1e-3, 0.02 / max(1.0, p.dQ_H ** 2))
            lam_guess = dyn_relax_guess(N)
            acf, tt = end_to_end_autocorrelation(phys, dt, 6 * lam_guess, 60,
                                                 cfg.n_traj, init)
            # integral of the normalised ACF up to its first crossing of 1/e^2
            keep = acf > 0.05
            tau = float(np.trapezoid(acf[keep], tt[keep])) if hasattr(np, "trapezoid") \
                else float(np.trapz(acf[keep], tt[keep]))
            rh = info["achieved_RH_hookean"]
            print(f"{Ns:4d} {tau:16.3f} {tau*info['D_kirkwood_hookean']/rh**2:21.4f}")


def dyn_relax_guess(n_beads):
    """Crude Rouse estimate of the longest relaxation time, for sizing runs."""
    return 1.0 / math.sin(math.pi / (2.0 * n_beads)) ** 2


if __name__ == "__main__":
    run()

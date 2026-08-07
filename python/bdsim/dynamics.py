"""Matching chain dynamics: hydrodynamic radius, diffusivity, and h*.

The static coarse-graining (`coarse_grain`) fixes the chain's *size*. It says
nothing about how fast it moves: that is set by the bead hydrodynamic radius,
through h*. This module closes the loop by choosing h* so the model reproduces a
target diffusivity -- and, crucially, does so in a way that is consistent across
different levels of discretisation.

Units and conventions
---------------------
In the code's Hookean units a free bead has diffusivity D = 1/4 (the SDE carries
1/sqrt(2) on the noise, so <dR^2> = 3 dt/2 per bead), and the bead hydrodynamic
radius is a = sqrt(pi) h* (Section 1.2 of docs/theory.tex). The Kirkwood
approximation to the chain's centre-of-mass diffusivity is

    D_K = (1/4N^2) [ N + sum_{mu != nu} <tr(D_mu,nu)/3> ]                     (1)

and the corresponding hydrodynamic radius, defined by D = a/(4 R_H) so that a
single bead gives R_H = a, is

    R_H = a N^2 / ( N + sum_{mu != nu} <tr(D_mu,nu)/3> ).                     (2)

The trace of the RPY tensor collapses to the Oseen form -- the dyadic corrections
cancel exactly -- leaving a function of the separation alone:

    tr(D_mu,nu)/3 = a/r          for r >= 2a  (well separated)
                  = 1 - r/(4a)   for r <  2a  (overlapping),                  (3)

which is continuous at r = 2a (both give 1/2). In the well-separated regime (2)
reduces to the familiar Kirkwood expression

    1/R_H = 1/(N a) + (1/N^2) sum_{mu != nu} <1/r_mu,nu>,                     (4)

which is linear in 1/a and can be inverted for a in closed form. The general case
is solved by bisection, since overlapping beads break that linearity.

What this buys, and what it does not
------------------------------------
Equation (2) depends on the configuration only through <tr/3>, i.e. only on the
*static* structure -- which the coarse-graining has already fixed. So for a given
target R_H there is a unique h* per discretisation, obtained without running any
dynamics. Matching R_H matches the diffusivity by construction; whether it also
matches the longest relaxation time is a separate question, and one worth checking
(see `validation/validate_dynamics.py`).
"""
import math
from dataclasses import dataclass

import numpy as np

__all__ = ["rpy_trace", "kirkwood_sum", "hydrodynamic_radius", "kirkwood_diffusivity",
           "solve_hstar", "hstar_for_target_rh", "com_diffusivity",
           "experimental_hydrodynamic_radius",
           "RodlikeUnits", "to_rodlike_units",
           "free_draining_bead_radius", "free_draining_units",
           "free_draining_relaxation_time",
           "viscometric_relaxation_time", "viscometric_time_from_viscosity",
           "viscometric_time_from_intrinsic", "rouse_viscometric_time"]


def rpy_trace(r, a):
    """(1/3) tr of the RPY off-diagonal block at separation r, bead radius a.

    Equation (3): the Oseen form far away, regularised for overlapping beads.
    Accepts array input.
    """
    r = np.asarray(r, dtype=float)
    if a <= 0.0:
        return np.zeros_like(r)
    return np.where(r >= 2.0 * a, a / np.maximum(r, 1e-300), 1.0 - r / (4.0 * a))


def kirkwood_sum(configs, a):
    """<sum_{mu != nu} tr(D_mu,nu)/3> averaged over a set of configurations.

    `configs` is an iterable of (N, 3) arrays. This is the only place the chain's
    structure enters the diffusivity.
    """
    total, n = 0.0, 0
    for R in configs:
        R = np.asarray(R, dtype=float)
        d = R[:, None, :] - R[None, :, :]
        r = np.sqrt((d * d).sum(-1))
        iu = np.triu_indices(len(R), k=1)
        total += 2.0 * rpy_trace(r[iu], a).sum()      # both orderings
        n += 1
    return total / max(n, 1)


def hydrodynamic_radius(configs, hstar, n_beads=None):
    """Kirkwood hydrodynamic radius of the chain, in Hookean units (Eq. 2)."""
    configs = list(configs)
    N = n_beads or len(np.asarray(configs[0]))
    a = math.sqrt(math.pi) * hstar
    S = kirkwood_sum(configs, a)
    return a * N * N / (N + S)


def kirkwood_diffusivity(configs, hstar, n_beads=None):
    """Kirkwood centre-of-mass diffusivity in Hookean units (Eq. 1).

    Free draining (h* = 0) gives 1/(4N) exactly.
    """
    configs = list(configs)
    N = n_beads or len(np.asarray(configs[0]))
    a = math.sqrt(math.pi) * hstar
    S = kirkwood_sum(configs, a)
    return (N + S) / (4.0 * N * N)


def solve_hstar(configs, target_rh, n_beads=None, lo=1e-6, hi=2.0):
    """h* such that the chain's Kirkwood R_H equals `target_rh` (Hookean units).

    R_H increases monotonically with h*, from 0 towards the non-draining ceiling
    R_H(h* -> inf) = N^2 / sum<1/r>, so a target above that ceiling is
    unreachable: the chain cannot be made to diffuse more slowly than its own
    fully-shielded limit. Raises ValueError in that case, which is a real physical
    statement -- it means this discretisation is too coarse to represent the
    molecule's friction.
    """
    configs = list(configs)
    N = n_beads or len(np.asarray(configs[0]))

    def rh(h):
        return hydrodynamic_radius(configs, h, N)

    while rh(hi) < target_rh and hi < 1e4:
        hi *= 2.0
    if rh(hi) < target_rh:
        raise ValueError(
            f"target R_H = {target_rh:.4g} exceeds the non-draining ceiling for "
            f"N = {N} beads; use more beads or check the target")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rh(mid) < target_rh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def experimental_hydrodynamic_radius(D, temperature=298.15, viscosity=1.0e-3):
    """Stokes-Einstein R_H (metres) from a measured diffusivity D (m^2/s)."""
    kB = 1.380649e-23
    return kB * temperature / (6.0 * math.pi * viscosity * D)


def hstar_for_target_rh(params, target_rh_physical, length_unit_nm=0.34,
                        n_configs=2000, seed=0):
    """h* for a coarse-grained chain whose real hydrodynamic radius is known.

    `params` is a coarse_grain.SpringParameters. `target_rh_physical` is in
    nanometres; `length_unit_nm` converts the units of L and lp (base pairs by
    default) to nm. Samples the model's own equilibrium configurations, converts
    the target into Hookean units, and solves.

    Returns (hstar, info) where info carries the intermediate quantities.
    """
    from . import initial

    N = params.n_springs + 1
    lH_bp = params.dQ / params.dQ_H                  # Hookean unit, in units of L
    lH_nm = lH_bp * length_unit_nm
    target_rh_H = target_rh_physical / lH_nm

    configs = [initial.fene_fraenkel_bending_chain(N, params.sigma_H, params.dQ_H,
                                                   params.C, seed=seed + i)
               for i in range(n_configs)]
    hstar = solve_hstar(configs, target_rh_H, N)
    info = {
        "n_beads": N,
        "hookean_unit_nm": lH_nm,
        "target_RH_hookean": target_rh_H,
        "achieved_RH_hookean": hydrodynamic_radius(configs, hstar, N),
        "D_kirkwood_hookean": kirkwood_diffusivity(configs, hstar, N),
        "bead_radius_nm": math.sqrt(math.pi) * hstar * lH_nm,
    }
    return hstar, info


# --------------------------------------------------------------------------
# Direct measurement, as a check on the Kirkwood approximation
# --------------------------------------------------------------------------

def _diffusion_worker(job):
    phys, sim_dt, t_total, n_samples, seed, method, kwargs = job
    import numpy as _np
    from . import ensemble as _ens
    from ._bdsim import SimParams, Rng, integrate

    N = phys.number_of_beads
    R = _ens.INITIALIZERS[method](N, seed, kwargs)
    rng = Rng(seed)
    sim = SimParams()
    sim.dt = sim_dt
    sim.implicit_loop_tol = 1e-5
    sim.update_center_of_mass = False        # the COM motion IS the measurement

    step = t_total / n_samples
    t = 0.0
    coms, times = [_np.asarray(R).mean(axis=0)], [0.0]
    for _ in range(n_samples):
        R = integrate(R, phys, _ens._segment(sim, t, t + step), rng)
        t += step
        coms.append(_np.asarray(R).mean(axis=0))
        times.append(t)
    return _np.asarray(times), _np.asarray(coms)


def com_diffusivity(phys, dt, t_total, n_samples=60, n_traj=32, seed=0,
                    initial=None, backend="processes", n_workers=None,
                    lag_range=(0.02, 0.12)):
    """Centre-of-mass diffusivity measured directly from BD trajectories.

    The mean-square displacement is accumulated over *all* time origins, not just
    the start of each run: with `n_samples` recorded positions that is O(n^2)
    displacement pairs instead of n, which is what makes the measurement precise
    enough to test the Kirkwood estimate at all. A single-origin estimator on the
    same data has a relative error of order 20%.

    `lag_range` selects the fraction of the window used for the straight-line fit
    ``<|dr|^2> = 6 D t``. The default is deliberately SHORT, for two reasons that
    happen to coincide:

      * Statistics. Displacements at lag tau taken from a run of length T give
        only ~T/tau independent samples however many origins are used, so long
        lags are expensive. Short lags give far tighter error bars.
      * Definition. The Kirkwood formula is the instantaneous chain mobility
        averaged over the equilibrium ensemble, i.e. the SHORT-time diffusivity
        (lag << the chain relaxation time). That is the quantity to compare it
        with. The long-time diffusivity is slightly lower -- the difference is the
        well-known (small, ~1-2% for flexible chains) correction by which Kirkwood
        overestimates -- and is obtained by passing a lag_range beyond the
        relaxation time.

    Returns (D, stderr).
    """
    from . import ensemble as _ens
    from .parallel import parallel_map

    initial = initial or _ens.Initial()
    jobs = [(phys, dt, t_total, n_samples, seed + i, initial.method,
             dict(initial.kwargs)) for i in range(n_traj)]
    out = parallel_map(_diffusion_worker, jobs, backend=backend, n_workers=n_workers)

    times = out[0][0]
    step = times[1] - times[0]
    n = len(times)
    lo = max(1, int(lag_range[0] * n))
    hi = max(lo + 2, int(lag_range[1] * n))

    lags = np.arange(lo, hi)
    per_traj = np.zeros((len(out), len(lags)))
    for k, (_t, coms) in enumerate(out):
        for j, L in enumerate(lags):
            d = coms[L:] - coms[:-L]                 # every origin at this lag
            per_traj[k, j] = (d * d).sum(axis=1).mean()

    mean = per_traj.mean(axis=0)
    tlag = lags * step
    slope = float(np.dot(tlag, mean) / np.dot(tlag, tlag))   # fit through origin
    # trajectory-to-trajectory scatter of the same slope estimate
    slopes = np.array([np.dot(tlag, per_traj[k]) / np.dot(tlag, tlag)
                       for k in range(len(out))])
    return slope / 6.0, float(slopes.std(ddof=1) / math.sqrt(len(out))) / 6.0


# --------------------------------------------------------------------------
# Real units
# --------------------------------------------------------------------------
# The Hookean units are built from the spring itself, so BOTH of them move when
# the coarse-graining changes:
#
#     l_H = sqrt(kT/H)          length   -- fixed by the STATIC coarse-graining
#     lambda_H = zeta/(4H)      time     -- depends on h*, through zeta = 6 pi eta a
#
# That is why the diffusivity quoted in code units is not comparable between
# discretisations: each row of such a table is in a different second. The chain of
# reasoning that makes the match well posed is:
#
#   1. the static fit fixes  l_H  in nanometres (l_H = dQ/sqrt(2c), in units of L);
#   2. R_H is a LENGTH, so the model's R_H (in l_H) converts to nanometres with no
#      reference to time at all -- this is what `solve_hstar` matches;
#   3. Stokes-Einstein then gives the real diffusivity,
#
#          D = kT / (6 pi eta_s R_H),
#
#      which is therefore matched in m^2/s the moment R_H is matched in metres,
#      independently of Ns. Substituting a = sqrt(pi) h* l_H and R_H = a/(4 D_H)
#      into D = D_H l_H^2 / lambda_H reproduces exactly this, which is the
#      consistency check that the unit bookkeeping is right;
#   4. lambda_H in seconds then follows, and it is what converts a relaxation time
#      measured in the simulation into a relaxation time in seconds.

KB = 1.380649e-23           # J/K
N_AVOGADRO = 6.02214076e23  # 1/mol


def physical_units(params, hstar, temperature=298.15, viscosity=1.0e-3,
                   length_unit_nm=0.34, hydrodynamic_radius_nm=None):
    """Convert a coarse-grained model into SI units.

    `params` is a coarse_grain.SpringParameters, `hstar` the value that matched the
    target hydrodynamic radius. Returns a dict of real-world quantities:

      l_H          Hookean length unit (m)      -- from the static fit alone
      force_H      F_H = sqrt(H kT) = kT/l_H (N) -- the force unit
      lambda_H     zeta/(4H) (s)                -- the time unit
      bead_radius  a = sqrt(pi) h* l_H (m)
      spring_H     H = kT / l_H^2 (N/m)
      zeta         6 pi eta_s a (kg/s)
      D            kT/(6 pi eta_s R_H) (m^2/s)  -- Stokes-Einstein
      R_H          hydrodynamic radius (m)

    With l_H, F_H and lambda_H every simulation quantity converts to SI: a length
    times l_H, a force times F_H, a time times lambda_H, an energy times kT.

    D is Ns-independent whenever R_H is, which is the whole point: lambda_H and the
    code-unit diffusivity both vary with the discretisation, and cancel.
    """
    kT = KB * temperature
    lH = (params.dQ / params.dQ_H) * length_unit_nm * 1e-9        # m
    a = math.sqrt(math.pi) * hstar * lH                           # m
    H = kT / lH ** 2                                              # N/m
    zeta = 6.0 * math.pi * viscosity * a                          # kg/s
    lambda_H = zeta / (4.0 * H)                                   # s
    out = {"l_H": lH, "force_H": kT / lH, "lambda_H": lambda_H,
           "bead_radius": a, "spring_H": H, "zeta": zeta, "kT": kT}
    if hydrodynamic_radius_nm is not None:
        RH = hydrodynamic_radius_nm * 1e-9
        out["R_H"] = RH
        out["D"] = kT / (6.0 * math.pi * viscosity * RH)
    return out


def seconds(t_hookean, units):
    """Convert a time from simulation units to seconds."""
    return t_hookean * units["lambda_H"]


# --------------------------------------------------------------------------
# Experimental calibration for DNA
# --------------------------------------------------------------------------
# The reference scale is Zimm's result for a flexible, non-draining, ideal coil,
#
#     D_Zimm = 8 / (3 sqrt(6 pi^3)) * kT / (eta sqrt(L b)),                   (Z)
#
# with L the contour length and b the Kuhn length (b = 2 l_p; b ~ 100 nm for
# dsDNA). Real DNA departs from (Z) in both directions: short fragments are
# rod-like and diffuse FASTER than the coil formula predicts, while long ones are
# swollen by excluded volume and diffuse SLOWER, so D/D_Zimm falls below 1 and
# keeps drifting as L^(1/2 - nu) with nu ~ 0.588.
#
# DSDNA_D_OVER_DZIMM tabulates that ratio.
#
#     WARNING -- these numbers were read off a published figure (dsDNA with
#     excluded volume, fitted by a discrete wormlike chain with b/d = 36,
#     d = 2.9 nm, against dynamic light scattering, sedimentation and
#     single-molecule data). They are good to perhaps a few percent and are
#     provided so the pipeline runs end to end. For quantitative work substitute
#     your own digitisation, or a direct fit to the source data, by passing
#     `ratio_table=` to the functions below.

DSDNA_KUHN_NM = 100.0        # b = 2 l_p for dsDNA in excess salt
BP_PER_NM = 1.0 / 0.34

# (contour length in micrometres, D / D_Zimm)
DSDNA_D_OVER_DZIMM = (
    (0.010, 3.00), (0.020, 2.80), (0.050, 2.55), (0.100, 2.35),
    (0.200, 2.10), (0.500, 1.75), (1.000, 1.55), (2.000, 1.40),
    (5.000, 1.22), (10.00, 1.12), (20.00, 1.04), (50.00, 0.96),
    (100.0, 0.92), (200.0, 0.88), (500.0, 0.84), (1000., 0.80),
)


def zimm_diffusivity(L_m, kuhn_m, temperature=298.15, viscosity=1.0e-3):
    """Zimm diffusivity of an ideal non-draining coil, Eq. (Z), in m^2/s."""
    kT = KB * temperature
    pref = 8.0 / (3.0 * math.sqrt(6.0 * math.pi ** 3))
    return pref * kT / (viscosity * math.sqrt(L_m * kuhn_m))


def d_over_dzimm(L_um, ratio_table=None):
    """Interpolate D/D_Zimm at contour length L_um (micrometres).

    Log-log interpolation, clamped at the ends of the table.
    """
    tab = ratio_table or DSDNA_D_OVER_DZIMM
    xs = np.log([t[0] for t in tab])
    ys = np.log([t[1] for t in tab])
    return float(np.exp(np.interp(math.log(L_um), xs, ys)))


def dna_diffusivity(L_bp, temperature=298.15, viscosity=1.0e-3,
                    kuhn_nm=DSDNA_KUHN_NM, length_unit_nm=0.34, ratio_table=None):
    """Experimental dsDNA centre-of-mass diffusivity (m^2/s) for a fragment of
    L_bp base pairs: the Zimm scale times the measured ratio."""
    L_m = L_bp * length_unit_nm * 1e-9
    dz = zimm_diffusivity(L_m, kuhn_nm * 1e-9, temperature, viscosity)
    return dz * d_over_dzimm(L_m * 1e6, ratio_table)


def dna_hydrodynamic_radius_nm(L_bp, temperature=298.15, viscosity=1.0e-3,
                               kuhn_nm=DSDNA_KUHN_NM, length_unit_nm=0.34,
                               ratio_table=None):
    """Experimental dsDNA hydrodynamic radius (nm), via Stokes-Einstein."""
    D = dna_diffusivity(L_bp, temperature, viscosity, kuhn_nm, length_unit_nm,
                        ratio_table)
    return experimental_hydrodynamic_radius(D, temperature, viscosity) * 1e9


# --------------------------------------------------------------------------
# Rodlike units
# --------------------------------------------------------------------------
# Hookean units are built from the spring (l_H = sqrt(kT/H)), which is awkward for
# a stiff, nearly-rod-like spring: l_H is then far smaller than the bond itself, so
# every length comes out as a large number and h* loses its usual interpretation.
# The rodlike scaling instead uses a length taken from the chain,
#
#     l_R = sigma            (the natural length), or
#     l_R = sqrt(<Q^2>)      (the r.m.s. bond length),
#
# with the time and force units built from it in the usual diffusive way,
#
#     lambda_R = zeta l_R^2 / kT,        F_R = kT / l_R .
#
# Conversions from Hookean to rodlike then follow from the ratios of the units.
# Writing sigma_H and dQ_H for the spring parameters as the code stores them, and
# noting that H = 1 and kT = 1 in Hookean units:
#
#     length:   X_R = X_H * (l_H / l_R)          l_H/l_R = 1/s,  s = l_R in units of l_H
#     force:    F_R = F_H * (l_R / l_H)
#     time:     t_R = t_H * (lambda_H / lambda_R) = t_H * l_H^2 / (4 l_R^2) = t_H/(4 s^2)
#               (the factor 4 is the one in lambda_H = zeta/4H)
#     shear:    gammadot_R = gammadot_H * 4 s^2
#
# and the spring parameters become
#
#     H*_R    = H l_R^2 / kT = s^2,        dQ*_R = dQ_H / s,      sigma*_R = sigma_H / s,
#     h*_R    = 3a / (4 l_R) = 3 sqrt(pi) h* / (4 s),
#
# using a = sqrt(pi) h* l_H. With l_R = sigma this gives sigma*_R = 1 by
# construction and H*_R = sigma_H^2, and dQ*_R = dQ_H/sigma_H = 1/q0 is the inverse
# of the reduced natural length the solver works with. The bending constant C is
# already dimensionless and is unchanged.


@dataclass
class RodlikeUnits:
    scale: str                 # "sigma" or "rms"
    s: float                   # l_R expressed in Hookean units
    length_factor: float       # multiply a Hookean length by this
    force_factor: float        # multiply a Hookean force by this
    time_factor: float         # multiply a Hookean time by this
    shear_factor: float        # multiply a Hookean shear rate by this
    H_R: float                 # dimensionless spring constant H l_R^2 / kT
    sigma_R: float             # natural length in rodlike units
    dQ_R: float                # extensibility in rodlike units
    hstar_R: float = float("nan")   # 3a/(4 l_R), if h* was supplied
    C: float = float("nan")         # bending constant (dimensionless, unchanged)

    def __str__(self):
        return (f"rodlike units (scaled by {self.scale}): l_R = {self.s:.6g} l_H\n"
                f"  H*_R = {self.H_R:.6g}   sigma*_R = {self.sigma_R:.6g}   "
                f"dQ*_R = {self.dQ_R:.6g}   h*_R = {self.hstar_R:.6g}   C = {self.C:.6g}\n"
                f"  length x {self.length_factor:.6g}   force x {self.force_factor:.6g}   "
                f"time x {self.time_factor:.6g}   shear rate x {self.shear_factor:.6g}")


def to_rodlike_units(params, hstar=None, scale="sigma", mean_square_Q=None):
    """Convert a coarse-grained model from Hookean to rodlike units.

    `params` is a coarse_grain.SpringParameters. `scale` selects the length:

      "sigma" -- the spring's natural length (sigma*_R = 1 by construction).
                 Undefined for a FENE spring, where sigma = 0.
      "rms"   -- sqrt(<Q^2>) of the equilibrium bond-length distribution, which is
                 always available and is the sensible choice when sigma is small.

    Time is scaled by lambda_R = zeta l_R^2 / kT, following the thesis convention
    lambda*_R = L^2 zeta / kT with L the rod length.
    """
    from .coarse_grain import ff_moment

    if scale == "sigma":
        s = params.sigma_H
        if not (s > 0.0):
            raise ValueError("sigma = 0 (FENE limit): rodlike scaling by sigma is "
                             "undefined; use scale='rms'")
    elif scale == "rms":
        if mean_square_Q is None:
            # <Q^2> of the equilibrium distribution, in Hookean units
            q0 = params.sigma_H
            mean_square_Q = ff_moment(q0, params.dQ_H, params.c, 2)
        s = math.sqrt(mean_square_Q)
    else:
        raise ValueError("scale must be 'sigma' or 'rms'")

    hs_R = (3.0 * math.sqrt(math.pi) * hstar / (4.0 * s)
            if hstar is not None else float("nan"))
    return RodlikeUnits(
        scale=scale, s=s,
        length_factor=1.0 / s,
        force_factor=s,
        time_factor=1.0 / (4.0 * s * s),
        shear_factor=4.0 * s * s,
        H_R=s * s,
        sigma_R=params.sigma_H / s,
        dQ_R=params.dQ_H / s,
        hstar_R=hs_R,
        C=params.C,
    )


# --------------------------------------------------------------------------
# Free-draining matching (HI off)
# --------------------------------------------------------------------------
# With hydrodynamic interaction switched off the diffusion tensor is the identity,
# so the centre-of-mass diffusivity is exactly
#
#     D = kT / (N zeta) = kT / (6 pi eta_s N a),
#
# with no dependence on configuration at all. Matching a measured D therefore needs
# no Monte-Carlo sampling and no root find -- the bead radius follows in closed
# form,
#
#     a = R_H / N ,                                                          (FD)
#
# since R_H = kT/(6 pi eta_s D). Equivalently R_H = N a, which is the h* -> 0 limit
# of the Kirkwood expression (11.2): the beads no longer shield one another, so the
# chain's friction is just N times a bead's.
#
# A point worth being clear about. The *simulation* is run with hstar = 0, so the
# RPY tensor is never built. But the bead radius still has to be declared, because
# it sets the friction and hence the time unit lambda_H = zeta/4H. So (FD) returns
# a NOMINAL h*, used only for the unit conversions; it is not passed to the
# integrator. Setting hstar = 0 in the simulation and h*_nominal in the bookkeeping
# is the whole trick.
#
# What this buys and costs:
#
#   * the diffusivity is matched exactly, and the run is much faster (no tensor to
#     build, no square root to take -- Table 5.1 puts HI at 40-60x the cost of the
#     rest of a step);
#   * the RELAXATION spectrum is not matched. Only the centre-of-mass motion is
#     constrained by (FD); the internal modes of a free-draining chain follow Rouse
#     rather than Zimm dynamics and couple to flow differently. The longest time
#     happens to land in the right range (see free_draining_relaxation_time), but
#     that is one number out of a spectrum. Use free draining to shake down a
#     workflow and to place runs at a sensible Weissenberg number; measure rheology
#     with HI.


def free_draining_bead_radius(n_beads, target_rh_nm):
    """Bead radius (nm) giving the target hydrodynamic radius without HI: a = R_H/N."""
    return target_rh_nm / float(n_beads)


def free_draining_units(params, target_rh_physical, temperature=298.15,
                        viscosity=1.0e-3, length_unit_nm=0.34):
    """Unit conversions for a free-draining chain matched to a measured diffusivity.

    Returns (hstar_nominal, info). `hstar_nominal` is NOT to be given to the
    integrator -- run with hstar = 0 -- it exists so that `physical_units` can
    compute the friction and hence lambda_H. `info` mirrors `hstar_for_target_rh`.
    """
    N = params.n_springs + 1
    lH_nm = (params.dQ / params.dQ_H) * length_unit_nm
    a_nm = free_draining_bead_radius(N, target_rh_physical)
    hstar_nominal = a_nm / (math.sqrt(math.pi) * lH_nm)
    info = {
        "n_beads": N,
        "hookean_unit_nm": lH_nm,
        "target_RH_hookean": target_rh_physical / lH_nm,
        "achieved_RH_hookean": target_rh_physical / lH_nm,   # exact by construction
        "D_kirkwood_hookean": 1.0 / (4.0 * N),               # exact, free draining
        "bead_radius_nm": a_nm,
        "free_draining": True,
    }
    return hstar_nominal, info


def free_draining_relaxation_time(params, units, n_beads=None):
    """Longest relaxation time of the free-draining chain, in seconds.

    For a free-draining Hookean chain the slowest Rouse mode is
    lambda_1 = 1/sin^2(pi/2N) in code units, needing no fitted prefactor; this
    converts it with lambda_H. Quote it so that a free-draining run can be set up
    at a chosen Weissenberg number, and so the timescale can be compared against
    an HI run later.

    Empirically this comes out close to Ns-independent, and in the right range:
    for lambda-DNA it gives ~0.10 s across Ns = 10 to 40. The naive worry -- that
    Rouse N^2 scaling would inflate it -- does not materialise, because lambda_H
    shrinks in step (a = R_H/N, and l_H falls with the discretisation), and the two
    nearly cancel.

    That is a happy accident about tau_1 alone, and is NOT a licence to trust
    free-draining rheology. Zimm and Rouse chains differ in the whole spectrum of
    relaxation modes and in how those modes couple to flow, not just in the slowest
    one. Use this to place a run at a sensible Weissenberg number; measure the
    rheology with HI.
    """
    N = n_beads or (params.n_springs + 1)
    tau_code = 1.0 / math.sin(math.pi / (2.0 * N)) ** 2
    return tau_code * units["lambda_H"], tau_code


# --------------------------------------------------------------------------
# Viscometric relaxation time
# --------------------------------------------------------------------------
# The zero-shear polymer viscosity defines a relaxation time
#
#     lambda_eta = M eta_p0 / (c N_A kT) ,                                  (V)
#
# with M the molar mass, c the mass concentration, and eta_p0 the polymer
# contribution to the zero-shear viscosity. Since the chain number density is
# n = c N_A / M, this is simply
#
#     lambda_eta = eta_p0 / (n kT) ,
#
# which is exactly what the simulation reports: `shear_viscosity` returns
# -<tau_xy>/gammadot with tau_xy in units of kT, i.e. eta_p/(n kT), already a time
# in code units. Multiplying by lambda_H converts it to seconds. The concentration
# cancels, so this is a single-chain property despite being defined through a
# solution viscosity.
#
# In terms of the intrinsic viscosity [eta] = eta_p0/(c eta_s), which is what
# experiments usually quote, (V) becomes
#
#     lambda_eta = M eta_s [eta] / (N_A kT) ,
#
# again independent of concentration.
#
# lambda_eta is a weighted sum over the whole relaxation spectrum, not the slowest
# mode alone, and there is a factor of two in it that is easy to lose.
#
# For a free-draining Hookean chain the Rouse mode amplitudes decay with
# tau_p = 1/sin^2(p pi / 2N) in code units -- that is what the end-to-end vector
# correlation measures, and what `free_draining_relaxation_time` returns. But the
# STRESS is quadratic in the mode amplitudes, so a stress correlation decays at
# TWICE the rate, with time constant tau_p/2. Hence
#
#     lambda_eta = sum_{p=1}^{N-1} tau_p/2
#                = (1/2) sum_p 1/sin^2(p pi / 2N)
#                = (N^2 - 1)/3        (exactly),
#
# which tends to (pi^2/12) tau_1 ~ 0.82 tau_1 for a long chain -- shorter than the
# slowest mode, not longer. Using sum_p tau_p without the factor of 1/2 gives
# 2(N^2-1)/3 and is wrong by exactly two; a direct measurement of a Hookean chain
# (N = 11) gives 42.5 +/- 2.6 against 40.0 for (N^2-1)/3 and 80.0 without the
# factor. `rouse_viscometric_time` provides the correct value as a check.


def viscometric_relaxation_time(eta_p0_code, units):
    """lambda_eta in seconds, from the simulation's zero-shear viscosity.

    `eta_p0_code` is eta_p/(n kT) as returned by `shear_viscosity` (already a time
    in code units, so this is just a unit conversion); `units` comes from
    `physical_units`.
    """
    return eta_p0_code * units["lambda_H"]


def viscometric_time_from_viscosity(eta_p0, concentration, molar_mass,
                                    temperature=298.15):
    """lambda_eta (s) from Eq. (V), in SI.

    eta_p0        polymer contribution to the zero-shear viscosity (Pa s)
    concentration mass concentration c (kg/m^3)
    molar_mass    M (kg/mol)
    """
    return molar_mass * eta_p0 / (concentration * N_AVOGADRO * KB * temperature)


def viscometric_time_from_intrinsic(intrinsic_mL_g, molar_mass_kg_per_mol,
                                    temperature=298.15, viscosity=1.0e-3):
    """lambda_eta (s) from an intrinsic viscosity in mL/g.

    Uses lambda_eta = M eta_s [eta] / (N_A kT); the concentration cancels, so this
    is the form to use with tabulated [eta] values.
    """
    intrinsic_m3_per_kg = intrinsic_mL_g * 1e-3        # mL/g -> m^3/kg
    return (molar_mass_kg_per_mol * viscosity * intrinsic_m3_per_kg
            / (N_AVOGADRO * KB * temperature))


def rouse_viscometric_time(n_beads):
    """lambda_eta of a free-draining Hookean chain, in code units (exact).

    (1/2) sum_{p=1}^{N-1} 1/sin^2(p pi / 2N) = (N^2 - 1)/3, the factor of one half
    coming from the stress being quadratic in the mode amplitudes (see above).
    Tends to (pi^2/12) tau_1 for a long chain. Useful as a check on a measured
    value, and as the zero-shear viscosity of a Rouse chain in these units.
    """
    N = int(n_beads)
    return 0.5 * sum(1.0 / math.sin(p * math.pi / (2.0 * N)) ** 2 for p in range(1, N))


"""Coarse-graining a wormlike chain (e.g. DNA) onto a FENE-Fraenkel bead-spring chain.

Given a DNA fragment of contour length ``L`` and persistence length ``lp``, and a
choice of ``n_springs``, this produces the spring and bending parameters for a
bdsim run: the FENE-Fraenkel natural length and extensibility, and the bending
stiffness.

The scheme (after Sunthar & Prakash, Saadat & Khomami, and the author's thesis):

  1. Each of the ``Ns`` segments carries contour length ``ls = L/Ns``. The spring
     is made exactly as extensible as the contour it represents,

         sigma + dQ = ls,                                                    (1)

     which leaves one free shape parameter.
  2. That parameter is fixed by matching the segment's mean-square extension to
     the wormlike-chain result over the same contour length,

         <Q^2>_FF = <R^2>_WLC(ls, lp).                                       (2)

  3. The bending stiffness C is chosen so that the angle between successive
     springs reproduces the wormlike-chain correlation (see `bending_constant`).

Units. ``L`` and ``lp`` may be in any consistent unit (base pairs, nm, m) --
everything that reaches the simulation is dimensionless. Quote DNA as
``L = 25000`` bp with ``lp = 147`` bp, or in nm; only the ratio matters.

The two numbers the simulation needs are in *Hookean* units (see docs/theory.tex
Section 1.2): ``sigma_H`` -> ``SpringParams.natural_length`` and ``dQ_H`` ->
``SpringParams.sqrtb``.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

__all__ = ["wlc_mean_square", "wlc_fourth_moment", "wlc_sixth_moment",
           "wlc_reduced_moment", "moment_importance", "important_moment_orders",
           "fit_spring_to_moments", "MAX_MOMENT_ORDER", "HKFit", "hk_fit", "hk_pdf", "hk_force",
           "marko_siggia_force", "ff_moment",
           "bending_constant", "bending_for_chain", "mean_cos_theta",
           "discrete_chain_mean_square",
           "SpringParameters", "spring_parameters", "to_phys_params"]


# --------------------------------------------------------------------------
# Wormlike-chain reference moments
# --------------------------------------------------------------------------

def wlc_mean_square(L, lp):
    """<R^2> of a Kratky-Porod wormlike chain of contour length L, persistence lp."""
    x = L / lp
    return 2.0 * lp * lp * (x - 1.0 + math.exp(-x))


def wlc_fourth_moment(L, lp):
    """<R^4> of a Kratky-Porod wormlike chain (Hamprecht & Kleinert, D=3).

    Verified against both limits: -> L^4 as L/lp -> 0 (rigid rod) and
    -> (5/3)<R^2>^2 = 20 (L lp)^2 / 3 as L/lp -> inf (Gaussian coil).
    """
    t = L / lp
    val = (4.0 / 27.0) * (-54.0 * (t + 4.0) * math.exp(-t)
                          + 45.0 * t * t - 156.0 * t + 214.0
                          + 2.0 * math.exp(-3.0 * t))
    return val * lp ** 4


# --------------------------------------------------------------------------
# Analytic end-to-end distribution from the moments (Hamprecht & Kleinert)
# --------------------------------------------------------------------------
# Hamprecht & Kleinert parameterise the wormlike-chain end-to-end distribution as
#
#     P(r) = [beta / B((3+k)/beta, m+1)] r^(k+2) (1 - r^beta)^m,   r = R/L,      (HK 3.1)
#
# (a radial distribution on 0 <= r <= 1, so it already carries the r^2 Jacobian),
# whose moments are known in closed form,
#
#     <r^2n> = Gamma((3+k+2n)/beta) Gamma((3+k)/beta + m + 1)
#            / [Gamma((3+k)/beta) Gamma((3+k+2n)/beta + m + 1)].                 (HK 3.2)
#
# Fitting the three parameters (k, beta, m) to three exact moments therefore turns
# a moment list into a closed-form distribution -- and, via the free energy, into a
# closed-form force-extension curve. The same machinery applies to the coarse-
# grained chain by feeding it that chain's (Monte-Carlo) moments, so model and
# target can be compared as smooth analytic curves rather than histograms.

from ._wlc_moment_data import _SERIES, _TERMS

# The closed forms cancel catastrophically for stiff chains (O(1) terms cancelling
# to O(t^2n)) while the Taylor series truncates for flexible ones; they agree best
# around t = 0.8, which is where we switch. Worst-case accuracy there is ~1e-13 for
# n <= 3, ~1e-11 (n=4), ~1e-7 (n=5), ~1e-5 (n=6).
_T_SWITCH = 0.8
MAX_MOMENT_ORDER = 6


def wlc_reduced_moment(L, lp, n):
    """<r^{2n}> = <R^{2n}>/L^{2n} for a wormlike chain, n = 1 .. MAX_MOMENT_ORDER."""
    if n not in _SERIES:
        raise ValueError(f"n must be 1..{MAX_MOMENT_ORDER}; regenerate the data "
                         "table from wlc_moments.py for higher orders")
    t = L / lp
    if t < _T_SWITCH:
        return sum(c * t ** i for i, c in enumerate(_SERIES[n]))
    tot = 0.0
    for rate, poly in _TERMS[n]:
        tot += sum(c * t ** i for i, c in enumerate(poly)) * math.exp(-rate * t)
    return tot / t ** (2 * n)


def moment_importance(L, lp, n_max_order=MAX_MOMENT_ORDER):
    """Hamprecht-Kleinert's measure of how informative each moment is.

    A moment is informative to the extent that it differs from what a *flat*
    distribution on r in [0,1] would give, <r^2n>_flat = 1/(2n+2). The ratio
    <r^2n> / <r^2n>_flat peaks at some order n_max (roughly 4 lp/L), and those are
    the moments worth fitting: low moments of a stiff chain all sit near 1 and
    carry almost no information about the shape.

    Returns a list of (n, ratio).
    """
    return [(n, wlc_reduced_moment(L, lp, n) * (2 * n + 2))
            for n in range(1, n_max_order + 1)]


def important_moment_orders(L, lp, count=3, n_max_order=MAX_MOMENT_ORDER):
    """The `count` most informative moment orders for this stiffness.

    Follows Hamprecht & Kleinert: take n_max = argmax of the importance ratio; if
    n_max <= 1 use the lowest orders (1, 2, 3, ...), otherwise take the window
    centred on n_max. Orders are clipped to what the moment table supports.
    """
    imp = moment_importance(L, lp, n_max_order)
    n_peak = max(imp, key=lambda p: p[1])[0]
    if n_peak <= 1:
        orders = list(range(1, count + 1))
    else:
        half = (count - 1) // 2
        lo = max(1, n_peak - half)
        orders = list(range(lo, lo + count))
    orders = [n for n in orders if n <= n_max_order]
    while len(orders) < count and orders and orders[0] > 1:
        orders.insert(0, orders[0] - 1)
    return tuple(orders)


def fit_spring_to_moments(ls, lp, orders=(1, 2), sigma0=None, c0=None):
    """Solve for the FENE-Fraenkel (sigma, c) that reproduce the given WLC moment
    orders of a segment of contour length `ls`, subject to sigma + dQ = ls.

    Two free parameters, so exactly two moments can be matched. Which two should
    come from `important_moment_orders`. Returns (sigma, c, residual) or None.
    """
    if len(orders) != 2:
        raise ValueError("exactly two moments can be matched by (sigma, c)")
    targets = [wlc_reduced_moment(ls, lp, n) * ls ** (2 * n) for n in orders]
    tlog = [math.log(t) for t in targets]

    def res(x):
        sig = ls / (1.0 + math.exp(-x[0]))          # sigma in (0, ls)
        cc = math.exp(x[1])
        return np.array([math.log(ff_moment(sig, ls - sig, cc, 2 * n)) - t
                         for n, t in zip(orders, tlog)]), sig, cc

    R2n = wlc_reduced_moment(ls, lp, 1)
    s_guess = sigma0 if sigma0 else 0.5 * ls
    c_guess = c0 if c0 else max(1e-3, _shape_exponent(ls, lp, R2n))
    x = np.array([math.log(s_guess / (ls - s_guess)), math.log(c_guess)])
    try:
        f, sig, cc = res(x)
    except (ValueError, OverflowError):
        return None
    for _ in range(200):
        J = np.empty((2, 2))
        try:
            for j in range(2):
                h = 1e-6 * max(1.0, abs(x[j]))
                xp = x.copy(); xp[j] += h
                J[:, j] = (res(xp)[0] - f) / h
            dx = np.linalg.solve(J, -f)
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return None
        step, moved = 1.0, False
        for _ in range(60):
            xn = x + step * dx
            if abs(xn[0]) > 40.0 or abs(xn[1]) > 40.0:
                step *= 0.5; continue
            try:
                fn, sign, ccn = res(xn)
            except (ValueError, OverflowError):
                step *= 0.5; continue
            if np.all(np.isfinite(fn)) and np.linalg.norm(fn) < np.linalg.norm(f):
                moved = True; break
            step *= 0.5
        if not moved:
            break
        x, f, sig, cc = xn, fn, sign, ccn
        if np.linalg.norm(f, np.inf) < 1e-12:
            break
    r = float(np.linalg.norm(f, np.inf))
    return (sig, cc, r) if r < 1e-6 else None


def wlc_sixth_moment(L, lp):
    """<R^6> of a wormlike chain."""
    return wlc_reduced_moment(L, lp, 3) * L ** 6


@dataclass
class HKFit:
    """Parameters of the Hamprecht-Kleinert distribution P(r) ~ r^(k+2)(1-r^b)^m."""
    k: float
    beta: float
    m: float
    residual: float           # max |log <r^2n>_fit - log <r^2n>_target|
    orders: tuple = (1, 2, 3)


def _hk_log_moment(n, k, beta, m):
    a = (3.0 + k) / beta
    b = (3.0 + k + 2.0 * n) / beta
    return (math.lgamma(b) + math.lgamma(a + m + 1.0)
            - math.lgamma(a) - math.lgamma(b + m + 1.0))


def hk_fit(targets, orders=(1, 2, 3)):
    """Fit (k, beta, m) so the HK distribution reproduces the given reduced moments.

    `targets[i]` is <r^{2 orders[i]}> with r = R/L, so all values are in (0, 1).
    Solved by damped Newton in (k, ln beta, ln(m+1)), which keeps beta > 0 and
    m > -1 automatically. Returns an HKFit, or None if it fails to converge.
    """
    tlog = [math.log(v) for v in targets]

    def residual(x):
        k, beta, m = x[0], math.exp(x[1]), math.exp(x[2]) - 1.0
        return np.array([_hk_log_moment(n, k, beta, m) - t
                         for n, t in zip(orders, tlog)])

    # Gaussian-limit guess: P ~ r^2 exp(-m r^2) has <r^2> = 3/(2m).
    m0 = max(1.0, 1.5 / max(targets[0], 1e-6))
    best = None
    for start in ([0.0, math.log(2.0), math.log(1.0 + m0)],
                  [0.0, math.log(4.0), math.log(2.0)],
                  [1.0, math.log(8.0), math.log(1.5)]):
        x = np.array(start, dtype=float)
        try:
            f = residual(x)
        except (ValueError, OverflowError):
            continue
        for _ in range(300):
            J = np.empty((3, 3))
            ok = True
            for j in range(3):
                h = 1e-6 * max(1.0, abs(x[j]))
                xp = x.copy(); xp[j] += h
                try:
                    J[:, j] = (residual(xp) - f) / h
                except (ValueError, OverflowError):
                    ok = False; break
            if not ok:
                break
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break
            step, moved = 1.0, False
            for _ in range(60):
                xn = x + step * dx
                # keep the parameters in a sane box
                if xn[0] <= -2.99 or abs(xn[1]) > 20.0 or xn[2] > 40.0 or xn[2] < -20.0:
                    step *= 0.5; continue
                try:
                    fn = residual(xn)
                except (ValueError, OverflowError):
                    step *= 0.5; continue
                if np.all(np.isfinite(fn)) and np.linalg.norm(fn) < np.linalg.norm(f):
                    moved = True; break
                step *= 0.5
            if not moved:
                break
            x, f = xn, fn
            if np.linalg.norm(f, np.inf) < 1e-12:
                break
        r = float(np.linalg.norm(f, np.inf))
        if best is None or r < best[1]:
            best = (x.copy(), r)
    if best is None or best[1] > 1e-4:
        return None
    x, r = best
    return HKFit(k=x[0], beta=math.exp(x[1]), m=math.exp(x[2]) - 1.0,
                 residual=r, orders=tuple(orders))


def hk_pdf(r, fit):
    """Normalised P(r) on 0 <= r <= 1 for an HKFit (r = R/L)."""
    r = np.asarray(r, dtype=float)
    a = (3.0 + fit.k) / fit.beta
    lognorm = math.log(fit.beta) - (math.lgamma(a) + math.lgamma(fit.m + 1.0)
                                    - math.lgamma(a + fit.m + 1.0))
    out = np.zeros_like(r)
    ok = (r > 0.0) & (r < 1.0)
    rr = r[ok]
    out[ok] = np.exp(lognorm + (fit.k + 2.0) * np.log(rr)
                     + fit.m * np.log1p(-rr ** fit.beta))
    return out


def hk_force(r, fit, L):
    """Force-extension f(R) implied by an HKFit, in kT per unit length.

    The free energy of the chain with its ends held at separation R is
    A(R) = -kT ln[P(R)/R^2] (dividing out the angular Jacobian), so the force
    needed to hold that extension is f = dA/dR, giving

        f(r) L / kT = m beta r^(beta-1) / (1 - r^beta)  -  k / r .

    In the Gaussian limit (k -> 0, beta -> 2) this reduces to f = 3 kT R / <R^2>.

    CAVEAT -- do not use this for quantitative force-extension work. It is a
    *consequence* of a three-moment fit, and low moments constrain the bulk of the
    distribution, not the r -> 1 tail that sets the high-force response. The
    divergence as r -> 1 has the right form only because it was built into the
    ansatz (1-r^beta)^m, with m fitted to moments that barely see that region.
    The k/r term likewise makes it meaningless as r -> 0. Use it to *compare* two
    distributions on equal footing, not to predict a stretching curve; for that,
    compute the force-extension directly in the fixed-force ensemble.
    """
    r = np.asarray(r, dtype=float)
    out = np.full_like(r, np.nan)
    ok = (r > 0.0) & (r < 1.0)
    rr = r[ok]
    out[ok] = (fit.m * fit.beta * rr ** (fit.beta - 1.0) / (1.0 - rr ** fit.beta)
               - fit.k / rr) / L
    return out


def marko_siggia_force(x, lp):
    """Marko-Siggia interpolation for the WLC, f lp/kT = 1/(4(1-x)^2) - 1/4 + x.

    Returns f in kT per unit length. Valid for a flexible chain (L >> lp); shown
    only as a familiar reference curve.
    """
    x = np.asarray(x, dtype=float)
    return (0.25 / (1.0 - x) ** 2 - 0.25 + x) / lp


# --------------------------------------------------------------------------
# FENE-Fraenkel equilibrium moments
# --------------------------------------------------------------------------
# psi(Q) ~ Q^2 [1 - (Q-sigma)^2/dQ^2]^c   on   max(0, sigma-dQ) <= Q <= sigma+dQ
#
# with c = H dQ^2 / (2 kT) the shape exponent. Substituting u = (Q-sigma)/dQ and
# then u = sin(t) removes the endpoint singularity of (1-u^2)^c, so a plain
# uniform trapezoid rule in t is accurate for any c > 0.

def _ff_grid(sigma, dQ, c, n=2001):
    u_lo = max(-1.0, -sigma / dQ) if dQ > 0 else -1.0
    t = np.linspace(math.asin(min(1.0, max(-1.0, u_lo))), 0.5 * math.pi, n)
    u = np.sin(t)
    Q = sigma + dQ * u
    # weight = Q^2 (1-u^2)^c du/dt = Q^2 cos(t)^(2c+1)
    w = Q ** 2 * np.cos(t) ** (2.0 * c + 1.0)
    return Q, t, w


def ff_moment(sigma, dQ, c, order=2):
    """<Q^order> for the FENE-Fraenkel equilibrium distribution."""
    Q, t, w = _ff_grid(sigma, dQ, c)
    Z = np.trapezoid(w, t) if hasattr(np, "trapezoid") else np.trapz(w, t)
    num = np.trapezoid(w * Q ** order, t) if hasattr(np, "trapezoid") else np.trapz(w * Q ** order, t)
    return float(num / Z)


# --------------------------------------------------------------------------
# Bending stiffness
# --------------------------------------------------------------------------

def mean_cos_theta(C):
    """<cos(theta)> for the potential phi_b/kT = C(1-cos theta): the Langevin function."""
    if C < 1e-8:
        return C / 3.0
    return 1.0 / math.tanh(C) - 1.0 / C


def _invert_mean_cos(target):
    """Solve <cos theta>(C) = target by bisection. target in (0, 1)."""
    if target <= 1e-12:
        return 0.0
    if target >= 1.0 - 1e-12:
        return 1e6
    lo, hi = 1e-8, 1.0
    while mean_cos_theta(hi) < target and hi < 1e7:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_cos_theta(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Saadat & Khomami's fit coefficients (their Eq. for the bending parameter).
_SK = (-1.237, 0.8105, -1.0243, 0.4595)


def bending_constant(ls, lp, method="saadat"):
    """Bending stiffness C for the potential phi_b/kT = C(1 - cos theta).

    method:
      "saadat"   -- Saadat & Khomami's fit in nu = ls/lp = 2 N_K,s (the default,
                    and what the original scheme uses).
      "langevin" -- match the *continuous* WLC tangent correlation exactly,
                    <cos theta> = exp(-ls/lp).
      "yamakawa" -- match the *discrete*-chain persistence length,
                    lp/ls = (1 + <cos theta>) / (2 (1 - <cos theta>)).
      "match_chain" -- solve for C so the *whole discretised chain* reproduces
                    <R^2>_WLC(L, lp). Handled in `spring_parameters` (it needs
                    Ns and <Q^2>), not here.

    The three agree when ls << lp and diverge as the segment becomes flexible,
    because they match different things. "saadat" is a fit to the discretised
    chain's statistics, not to the local correlation, so do not expect
    <cos theta> to equal exp(-ls/lp).
    """
    nu = ls / lp
    if method == "saadat":
        p1, p2, p3, p4 = _SK
        num = 1.0 + p1 * nu + p2 * nu * nu
        den = nu + p3 * nu * nu + p4 * nu ** 3
        return num / den
    if method == "langevin":
        return _invert_mean_cos(math.exp(-nu))
    if method == "yamakawa":
        r = 2.0 / nu                      # 2 lp / ls
        return _invert_mean_cos(max(0.0, (r - 1.0) / (r + 1.0)))
    raise ValueError(f"unknown bending method {method!r}")


def bending_for_chain(n_springs, Q1, Q2, target_R2):
    """Solve for C so the discretised chain's <R^2> equals `target_R2`.

    Matching each segment's <Q^2> to its own piece of WLC contour does NOT make
    the assembled chain reproduce <R^2>_WLC(L, lp): the chain value also depends
    on the angular correlation, and a bending constant fitted to local statistics
    gets that wrong by up to tens of percent. This inverts the chain relation
    instead, trading exact local correlation for the correct global size.

    Returns (C, achieved_R2, ok). `ok` is False when the target is outside the
    range reachable with C >= 0, in which case C is clamped to the nearest end.
    """
    lo_cs, hi_cs = 0.0, 1.0 - 1e-12
    r_lo = discrete_chain_mean_square(n_springs, Q1, Q2, lo_cs)
    r_hi = discrete_chain_mean_square(n_springs, Q1, Q2, hi_cs)
    if target_R2 <= r_lo:
        return 0.0, r_lo, False
    if target_R2 >= r_hi:
        return 1e6, r_hi, False
    for _ in range(200):                    # <R^2> increases monotonically with cs
        mid = 0.5 * (lo_cs + hi_cs)
        if discrete_chain_mean_square(n_springs, Q1, Q2, mid) < target_R2:
            lo_cs = mid
        else:
            hi_cs = mid
    cs = 0.5 * (lo_cs + hi_cs)
    return _invert_mean_cos(cs), discrete_chain_mean_square(n_springs, Q1, Q2, cs), True


def discrete_chain_mean_square(n_springs, Q1, Q2, cs):
    """<R^2> of a discrete chain of `n_springs` bonds with independent lengths
    (mean Q1 = <Q>, mean square Q2 = <Q^2>) and Markovian angles with
    <cos theta> = cs.

    Note the two different length moments: the diagonal terms carry <Q^2> but the
    cross terms carry <Q>^2, because bond length and direction are independent.
    Using <Q^2> in both (as a fixed-bond-length freely-rotating-chain formula
    does) overestimates <R^2>.
    """
    Ns = int(n_springs)
    if abs(1.0 - cs) < 1e-12:
        return Ns * Q2 + Q1 * Q1 * Ns * (Ns - 1)
    geom = Ns * cs / (1.0 - cs) - cs * (1.0 - cs ** Ns) / (1.0 - cs) ** 2
    return Ns * Q2 + 2.0 * Q1 * Q1 * geom


# --------------------------------------------------------------------------
# The parameter set
# --------------------------------------------------------------------------

@dataclass
class SpringParameters:
    # inputs
    L: float
    lp: float
    n_springs: int
    # segment
    ls: float
    N_ks: float                  # ls / (2 lp), Kuhn steps per spring
    # spring, in the same units as L and lp
    sigma: float
    dQ: float
    c: float                     # shape exponent  H dQ^2 / 2kT
    H: float                     # spring constant, in kT / [length]^2
    # spring, in Hookean units -> feed these straight to bdsim
    sigma_H: float               # -> SpringParams.natural_length
    dQ_H: float                  # -> SpringParams.sqrtb
    # bending
    C: float                     # -> BendingParams.stiffness
    cos_theta: float             # <cos theta> implied by C
    # diagnostics
    branch: str                  # "fene-fraenkel" or "fene" (sigma driven to 0)
    R2_segment_target: float     # <R^2>_WLC(ls)
    Q2_segment: float            # <Q^2> achieved
    R4_segment_target: float
    Q4_segment: float
    R2_chain_target: float       # <R^2>_WLC(L)
    R2_chain_predicted: float    # from the discretised chain
    q0_reduced: float            # sigma_H / dQ_H, the code's reduced natural length

    def summary(self):
        m2 = self.Q2_segment / self.R2_segment_target - 1.0
        m4 = self.Q4_segment / self.R4_segment_target - 1.0
        mc = self.R2_chain_predicted / self.R2_chain_target - 1.0
        return (
            f"L = {self.L:g}, lp = {self.lp:g}, Ns = {self.n_springs}\n"
            f"  segment      ls = {self.ls:.4g}   N_K,s = {self.N_ks:.4g}   ({self.branch})\n"
            f"  spring       sigma = {self.sigma:.4g}   dQ = {self.dQ:.4g}   "
            f"c = {self.c:.4g}   H = {self.H:.4g} kT/len^2\n"
            f"  -> bdsim     natural_length = {self.sigma_H:.6g}   sqrtb = {self.dQ_H:.6g}   "
            f"(q0 = {self.q0_reduced:.4g})\n"
            f"  bending      C = {self.C:.4g}   <cos theta> = {self.cos_theta:.4f}\n"
            f"  fit quality  <Q^2>/<R^2>_seg - 1 = {m2:+.2e}   "
            f"<Q^4>/<R^4>_seg - 1 = {m4:+.2e}\n"
            f"               chain <R^2> vs WLC   = {mc:+.2%}"
        )

    def as_dict(self):
        return asdict(self)


def _shape_exponent(ls, lp, R2n):
    """The shape exponent c = H dQ^2 / 2kT for a segment.

    Sunthar & Prakash give, in the flexible (FENE) limit,
        H = (kT/dQ^2) (3 ls^2/<R^2> - 5)     i.e.   c = (3/R2n - 5)/2
    with R2n = <R^2>/ls^2. This goes NEGATIVE once R2n > 3/5 (equivalently
    ls^2/<R^2> < 5/3), which happens for stiff segments, ls/lp < 1.772.

    A stiff segment should become *harder*, not softer, so the correction must
    grow as ls/lp -> 0. Adding lp/ls does that and vanishes in the flexible
    limit, leaving the Sunthar-Prakash result intact:
        c = (3/R2n - 5)/2 + lp/ls
    """
    return (3.0 / R2n - 5.0) / 2.0 + lp / ls


def spring_parameters(L, lp, n_springs, bending="saadat", match_fourth_moment=False,
                      max_H=None):
    """Spring + bending parameters for one DNA fragment.

    Parameters
    ----------
    L, lp
        contour and persistence length, in any consistent unit.
    n_springs
        number of springs Ns (the chain has Ns + 1 beads).
    bending
        "saadat" (default), "langevin", "yamakawa", or "match_chain"
        -- see `bending_constant` and `bending_for_chain`.
    max_H
        optional ceiling on the spring constant, in kT per (unit of L)^2.
        Stiff segments drive H up without bound, and H sets the time unit
        through lambda_H = zeta/4H: doubling H halves the number of real
        seconds a timestep covers, so an uncapped H makes reaching the
        chain's relaxation time arbitrarily expensive. Capping it trades
        an artificially soft spring for a usable timestep.

        The trade is cheap in exactly the regime where it bites. Once a
        segment is much shorter than the persistence length its length
        barely fluctuates, and the chain's conformation -- and hence its
        dynamics -- is governed by the bending potential between segments,
        not by the spring. The second-moment condition is still enforced:
        c is pinned to the cap and sigma re-solved, so <Q^2> still matches
        the wormlike chain, with a broader spring at a shorter sigma.
    match_fourth_moment
        if True, additionally solve for the shape exponent c so that
        <Q^4> matches the WLC fourth moment, instead of taking c from
        the closed-form expression. This is better conditioned only in
        the crossover region; it degenerates for stiff segments (where
        both moments collapse onto the rigid-rod values and the two
        conditions stop being independent), so it falls back
        automatically.
    """
    Ns = int(n_springs)
    if Ns < 1:
        raise ValueError("n_springs must be >= 1")
    ls = L / Ns
    R2 = wlc_mean_square(ls, lp)
    R4 = wlc_fourth_moment(ls, lp)
    R2n = R2 / ls ** 2                       # <R^2>/ls^2, in (0, 1)

    c = _shape_exponent(ls, lp, R2n)
    branch = "fene-fraenkel"

    def q2_residual(sig, cc):
        return ff_moment(sig, ls - sig, cc, 2) - R2

    # Solve <Q^2> = <R^2> for sigma, with sigma + dQ = ls.
    lo, hi = 1e-12 * ls, ls * (1.0 - 1e-12)
    f_lo, f_hi = q2_residual(lo, c), q2_residual(hi, c)
    if f_lo * f_hi > 0.0:
        # No root with sigma > 0: the segment is flexible enough that the FENE
        # limit (sigma = 0) applies. Drop the stiffness correction, which is
        # only there to rescue the stiff branch.
        sigma = 0.0
        c = max((3.0 / R2n - 5.0) / 2.0, 1e-6)
        branch = "fene"
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if q2_residual(lo, c) * q2_residual(mid, c) <= 0.0:
                hi = mid
            else:
                lo = mid
        sigma = 0.5 * (lo + hi)

    if match_fourth_moment and branch == "fene-fraenkel":
        # Two-parameter solve: (sigma, c) matching <Q^2> and <Q^4>.
        ok = True
        cc = c
        for _ in range(60):
            a, b = 1e-12 * ls, ls * (1.0 - 1e-12)
            if q2_residual(a, cc) * q2_residual(b, cc) > 0.0:
                ok = False
                break
            for _ in range(120):
                mid = 0.5 * (a + b)
                if q2_residual(a, cc) * q2_residual(mid, cc) <= 0.0:
                    b = mid
                else:
                    a = mid
            s_try = 0.5 * (a + b)
            r4 = ff_moment(s_try, ls - s_try, cc, 4) - R4
            if abs(r4 / R4) < 1e-10:
                sigma, c = s_try, cc
                break
            # secant-free damped update on c (r4 decreases with c)
            eps = max(1e-6, 1e-4 * cc)
            r4b = ff_moment(s_try, ls - s_try, cc + eps, 4) - R4
            deriv = (r4b - r4) / eps
            if abs(deriv) < 1e-30:
                ok = False
                break
            cc_new = cc - 0.5 * r4 / deriv
            if not (cc_new > 1e-8) or not math.isfinite(cc_new):
                ok = False
                break
            sigma, c, cc = s_try, cc_new, cc_new
        if not ok:
            c = _shape_exponent(ls, lp, R2n)   # fall back

    dQ = ls - sigma
    H = 2.0 * c / dQ ** 2                    # in kT / [length]^2

    if max_H is not None and H > max_H:
        # Pin H to the cap. With sigma + dQ = ls this ties c to sigma through
        # c = max_H (ls - sigma)^2 / 2, leaving one unknown for the one remaining
        # condition <Q^2> = <R^2>, which is re-solved here.
        def resid(sig):
            cc = 0.5 * max_H * (ls - sig) ** 2
            if cc <= 0.0:
                return float("inf")
            return ff_moment(sig, ls - sig, cc, 2) - R2
        a, b = 1e-12 * ls, ls * (1.0 - 1e-9)
        fa, fb = resid(a), resid(b)
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
            for _ in range(200):
                mid = 0.5 * (a + b)
                if resid(a) * resid(mid) <= 0.0:
                    b = mid
                else:
                    a = mid
            sigma = 0.5 * (a + b)
            dQ = ls - sigma
            c = 0.5 * max_H * dQ ** 2
            H = max_H
            branch += " (H capped)"
        else:
            branch += " (H cap not attainable)"
    # Hookean units: l_H = sqrt(kT/H) = dQ / sqrt(2c)
    dQ_H = math.sqrt(2.0 * c)
    sigma_H = sigma * math.sqrt(2.0 * c) / dQ

    Q2 = ff_moment(sigma, dQ, c, 2)
    Q4 = ff_moment(sigma, dQ, c, 4)
    Q1 = ff_moment(sigma, dQ, c, 1)

    if bending == "match_chain":
        C, _achieved, ok = bending_for_chain(Ns, Q1, Q2, wlc_mean_square(L, lp))
        if not ok:
            branch += " (chain <R^2> unreachable with C>=0; C clamped)"
    else:
        C = bending_constant(ls, lp, bending)
    cs = mean_cos_theta(C)

    return SpringParameters(
        L=L, lp=lp, n_springs=Ns, ls=ls, N_ks=ls / (2.0 * lp),
        sigma=sigma, dQ=dQ, c=c, H=H,
        sigma_H=sigma_H, dQ_H=dQ_H,
        C=C, cos_theta=cs, branch=branch,
        R2_segment_target=R2, Q2_segment=Q2,
        R4_segment_target=R4, Q4_segment=Q4,
        R2_chain_target=wlc_mean_square(L, lp),
        R2_chain_predicted=discrete_chain_mean_square(Ns, Q1, Q2, cs),
        q0_reduced=(sigma_H / dQ_H if dQ_H > 0 else float("inf")),
    )


def to_phys_params(p, hstar=0.0, hi_method=None):
    """Build a bdsim.PhysParams from a SpringParameters (needs the compiled core)."""
    from . import _bdsim as core
    from . import flows
    phys = core.PhysParams()
    phys.spring.type = core.Spring.FENEFraenkel
    phys.spring.sqrtb = p.dQ_H
    phys.spring.natural_length = p.sigma_H
    phys.number_of_beads = p.n_springs + 1
    phys.bend.type = core.Bending.OneMinusCosTheta
    phys.bend.stiffness = p.C
    phys.hstar = hstar
    if hi_method is not None:
        phys.hi_method = hi_method
    phys.flow = flows.equilibrium()
    return phys


# --------------------------------------------------------------------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="FENE-Fraenkel + bending parameters for a wormlike chain (DNA).")
    ap.add_argument("--L", type=float, required=True, help="contour length")
    ap.add_argument("--lp", type=float, default=147.0,
                    help="persistence length (default 147, i.e. bp for DNA)")
    ap.add_argument("--Ns", type=int, nargs="+", required=True, help="number of springs")
    ap.add_argument("--bending", default="saadat",
                    choices=["saadat", "langevin", "yamakawa", "match_chain"])
    ap.add_argument("--match-fourth-moment", action="store_true")
    args = ap.parse_args()
    for ns in args.Ns:
        p = spring_parameters(args.L, args.lp, ns, bending=args.bending,
                              match_fourth_moment=args.match_fourth_moment)
        print(p.summary())
        print()


if __name__ == "__main__":
    _main()

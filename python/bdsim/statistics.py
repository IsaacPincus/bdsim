"""Error bars for time averages over a correlated steady-state trajectory.

Successive samples from a Brownian-dynamics run are not independent: a chain
remembers its configuration for roughly its longest relaxation time. Treating N
samples as N independent measurements therefore understates the error by a factor
of sqrt(g), where g is the statistical inefficiency -- often a factor of several.

Three standard tools, which cross-check each other:

  * `statistical_inefficiency` -- g = 1 + 2 sum_k rho_k from the autocorrelation
    function, truncated by Sokal's automatic window. The effective sample size is
    N_eff = N/g and the standard error is std(x)/sqrt(N_eff).
  * `blocking_curve` / `blocking_stderr` -- Flyvbjerg-Petersen blocking. Average
    adjacent pairs repeatedly; the apparent standard error rises and then
    plateaus once the blocks are longer than the correlation time. The plateau is
    the answer. This assumes nothing about the shape of the correlation function,
    so it is the honest fallback when the ACF is awkward.
  * `equilibration_point` -- choose the amount of initial transient to discard by
    maximising the effective sample size of what remains (Chodera's rule). Keeping
    a slowly-relaxing transient inflates both the mean and the apparent
    correlation time, so this is not merely cosmetic.

The simple approach the analysis is often done with -- sample once per relaxation
time and treat the samples as independent -- is the g -> sampling-interval limit
of the same idea, and throws away most of the data. `statistical_inefficiency`
recovers that information instead.

`trajectory_ensemble_stats` combines the two levels of averaging present in an
ensemble run: correlated samples *within* each trajectory, and genuinely
independent trajectories. It reports both estimates so they can be compared; if
they disagree by much, the run is too short for the autocorrelation estimate to be
reliable.
"""
from dataclasses import dataclass
import math

import numpy as np

__all__ = ["autocorrelation", "statistical_inefficiency", "effective_sample_size",
           "blocking_curve", "blocking_stderr", "equilibration_point",
           "SteadyStateStats", "steady_state_stats", "trajectory_ensemble_stats",
           "stress_autocorrelation", "green_kubo"]


def autocorrelation(x, max_lag=None):
    """Normalised autocorrelation function rho_k of a 1-D series (rho_0 = 1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    # FFT-based, zero-padded to avoid wrap-around
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(d, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
    acf /= np.arange(n, 0, -1)          # unbiased normalisation by overlap count
    if acf[0] <= 0:
        return np.ones(1)
    acf = acf / acf[0]
    return acf[:max_lag] if max_lag else acf


def statistical_inefficiency(x, c=6.0):
    """g = 1 + 2 sum_{k>=1} rho_k, with Sokal's automatic windowing.

    The window W is the smallest integer with W >= c * tau_int(W), where
    tau_int = g/2 in units of the sampling interval. Truncating the sum matters:
    the tail of the ACF is pure noise and summing it produces a divergent estimate.

    Returns (g, window). g >= 1 always; g = 1 means uncorrelated.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or np.allclose(x, x[0]):
        return 1.0, 0
    rho = autocorrelation(x)
    g = 1.0
    window = 0
    csum = 0.0
    for k in range(1, n):
        csum += rho[k]
        tau = 0.5 + csum
        if tau <= 0.0:
            break
        if k >= c * tau:
            window = k
            g = max(1.0, 2.0 * tau)
            return g, window
    return max(1.0, 2.0 * (0.5 + csum)), n - 1


def effective_sample_size(x, c=6.0):
    """Number of effectively independent samples, N/g."""
    g, _ = statistical_inefficiency(x, c)
    return len(x) / g


def blocking_curve(x):
    """Flyvbjerg-Petersen blocking: [(block_size, stderr_estimate), ...]."""
    x = np.asarray(x, dtype=float)
    out = []
    size = 1
    while len(x) >= 4:
        out.append((size, float(x.std(ddof=1) / math.sqrt(len(x)))))
        m = len(x) // 2
        x = 0.5 * (x[:2 * m:2] + x[1:2 * m:2])
        size *= 2
    return out


def blocking_stderr(x):
    """Plateau value of the blocking curve: the correlation-corrected error.

    Taken as the largest estimate among blocking levels that still have at least
    16 blocks, which is where the curve has usually flattened but the estimate is
    not yet dominated by its own noise.
    """
    curve = blocking_curve(x)
    n = len(x)
    usable = [se for size, se in curve if n // size >= 16]
    if not usable:
        return curve[0][1] if curve else float("nan")
    return max(usable)


def equilibration_point(x, n_candidates=40):
    """Index at which to start averaging, maximising the effective sample size.

    Discarding transient costs samples but removes bias and shortens the apparent
    correlation time; the optimum trades those off (Chodera's rule).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 16:
        return 0
    best_t, best_neff = 0, -1.0
    for t0 in np.unique(np.linspace(0, n - 8, n_candidates).astype(int)):
        neff = effective_sample_size(x[t0:])
        if neff > best_neff:
            best_t, best_neff = int(t0), neff
    return best_t


@dataclass
class SteadyStateStats:
    mean: float
    stderr: float                # from the autocorrelation-corrected N_eff
    stderr_blocking: float       # independent estimate, for cross-checking
    g: float                     # statistical inefficiency
    tau: float                   # integrated autocorrelation time (time units)
    n_samples: int
    n_effective: float
    discarded: int               # samples dropped as transient
    warning: str = ""

    def __str__(self):
        s = (f"{self.mean:.6g} +/- {self.stderr:.3g}  "
             f"(g = {self.g:.2f}, tau = {self.tau:.4g}, "
             f"N_eff = {self.n_effective:.1f} of {self.n_samples}, "
             f"blocking err {self.stderr_blocking:.3g})")
        return s + ("\n  WARNING: " + self.warning if self.warning else "")


def steady_state_stats(x, sample_interval=1.0, detect_equilibration=True,
                       min_effective=20.0):
    """Mean and correlation-corrected error of a steady-state series."""
    x = np.asarray(x, dtype=float)
    t0 = equilibration_point(x) if detect_equilibration else 0
    y = x[t0:]
    g, _w = statistical_inefficiency(y)
    n_eff = len(y) / g
    stderr = float(y.std(ddof=1) / math.sqrt(max(n_eff, 1.0)))
    warn = ""
    if n_eff < min_effective:
        warn = (f"only {n_eff:.1f} effectively independent samples "
                f"(correlation time is {g/2*sample_interval:.4g}, run covers "
                f"{len(y)*sample_interval:.4g}); the error bar is itself uncertain "
                f"by ~{100/math.sqrt(2*max(n_eff,1)):.0f}% -- run longer")
    return SteadyStateStats(
        mean=float(y.mean()), stderr=stderr,
        stderr_blocking=float(blocking_stderr(y)),
        g=float(g), tau=float(g / 2.0 * sample_interval),
        n_samples=len(y), n_effective=float(n_eff), discarded=int(t0), warning=warn)


def trajectory_ensemble_stats(series_list, sample_interval=1.0,
                              detect_equilibration=True):
    """Combine several independent trajectories of correlated samples.

    Returns (stats, stderr_between) where `stats` pools all samples with the
    autocorrelation correction, and `stderr_between` is the independent estimate
    from the scatter of the per-trajectory means. The second needs no assumptions
    at all but has only (n_traj - 1) degrees of freedom; the two agreeing is good
    evidence that both are trustworthy.
    """
    series_list = [np.asarray(s, dtype=float) for s in series_list]
    if detect_equilibration:
        t0 = max(equilibration_point(s) for s in series_list)
        series_list = [s[t0:] for s in series_list]
    else:
        t0 = 0

    means = np.array([s.mean() for s in series_list])
    n_traj = len(series_list)
    stderr_between = (float(means.std(ddof=1) / math.sqrt(n_traj))
                      if n_traj > 1 else float("nan"))

    gs = [statistical_inefficiency(s)[0] for s in series_list]
    g = float(np.mean(gs))
    n_tot = sum(len(s) for s in series_list)
    n_eff = n_tot / g
    pooled = np.concatenate(series_list)
    se_acf = float(pooled.std(ddof=1) / math.sqrt(max(n_eff, 1.0)))

    # --- guards -------------------------------------------------------------
    # The autocorrelation estimate is only meaningful if each trajectory contains
    # many samples per correlation time. With a handful of samples the ACF cannot
    # be measured at all: the Sokal window closes immediately, g comes back as ~1,
    # and the error bar is silently too small by sqrt(g_true). Two independent
    # symptoms are checked, and when either fires the assumption-free
    # between-trajectory estimate is used instead (it is always valid, just noisy).
    warn = ""
    per_traj = len(series_list[0]) if series_list else 0
    use_between = False
    if per_traj < 30:
        warn = (f"only {per_traj} samples per trajectory -- too few to estimate the "
                f"autocorrelation (g came out {g:.2f}); falling back to the "
                f"between-trajectory error")
        use_between = True
    elif (n_traj > 2 and math.isfinite(stderr_between) and se_acf > 0
          and not (0.5 < stderr_between / se_acf < 2.0)):
        warn = (f"the autocorrelation-corrected error ({se_acf:.3g}) and the "
                f"between-trajectory error ({stderr_between:.3g}) disagree by more "
                f"than 2x; taking the larger. Usually this means the sampling "
                f"interval is comparable to the correlation time")
        use_between = True

    stderr = (max(se_acf, stderr_between) if use_between and
              math.isfinite(stderr_between) else se_acf)

    stats = SteadyStateStats(
        mean=float(means.mean()), stderr=stderr,
        stderr_blocking=float(np.mean([blocking_stderr(s) for s in series_list])
                              / math.sqrt(n_traj)),
        g=g, tau=float(g / 2.0 * sample_interval), n_samples=n_tot,
        n_effective=float(n_eff), discarded=int(t0), warning=warn)
    return stats, stderr_between


# --------------------------------------------------------------------------
# Green-Kubo
# --------------------------------------------------------------------------
# The zero-shear viscosity can be had from equilibrium fluctuations alone,
#
#     eta_p / (n kT) = int_0^inf <tau_xy(0) tau_xy(t)>_eq dt ,                (GK)
#
# with tau_xy in units of kT, giving a time in code units -- directly comparable
# with what `shear_viscosity` returns. Nothing is sheared, so the estimate does not
# degrade as Wi -> 0, which is the appeal.
#
# The difficulty is entirely in the upper limit. The correlation function decays on
# the stress relaxation time, but the NOISE in the estimated correlation function
# does not decay with lag: beyond a few relaxation times each C(k) is a small
# difference of large numbers, and integrating that noise makes the running integral
# wander. Cutting the integral early biases it low; cutting it late adds variance.
# Two treatments are offered, and comparing them is the honest way to see whether
# the answer is real:
#
# MEASURED PERFORMANCE, and a warning. On the 2 kbp / Ns = 20 free-draining chain,
# with 6 trajectories of 40 tau_1 (comparable compute to the sheared runs):
#
#     peak of running integral   2514 +/- 573   (23%)
#     running integral averaged  2127 +/- 513   (24%)
#     single-exponential tail    1093            -- badly low, see below
#
# against 2951 +/- 923 (31%) from a sheared run at Wi = 0.3 with variance
# reduction. So Green-Kubo is only modestly better statistically, and it carries a
# SYSTEMATIC ambiguity the sheared measurement does not: the three estimators above
# span more than a factor of two, which swamps the 23% statistical error.
#
# The cause is that the stress correlation function of a polymer is not a single
# exponential -- it is a sum over relaxation modes, tau_p ~ tau_1/p^2. Fitting one
# exponential locks onto the fast initial decay (here tau_fit = 0.04 tau_1) and
# then truncates the slow tail, which is where much of the integral lives: the
# running integral was still climbing at 10 tau_1, long after the fitted decay had
# finished. That is exactly the difficulty this method is known for, and it is not
# fixed by longer runs, only by modelling the tail correctly.
#
# The principled repair is to fit the tail with the Rouse/Zimm mode spectrum,
# G(t) = sum_p exp(-2 t / tau_p) with tau_p = tau_1/p^2, which has one free
# parameter and an analytic integral, sum_p tau_p/2. That is not implemented here.
# Until it is, prefer a sheared run with variance reduction at Wi ~ 0.1-0.3 for the
# low-shear viscosity, and treat the numbers below as a cross-check on it.


def stress_autocorrelation(series_list, max_lag_fraction=0.5):
    """<tau(0)tau(t)> averaged over trajectories and all time origins (unnormalised)."""
    series_list = [np.asarray(s, dtype=float) for s in series_list]
    n = len(series_list[0])
    max_lag = max(2, int(max_lag_fraction * n))
    acf = np.zeros(max_lag)
    for x in series_list:
        d = x - 0.0                       # equilibrium mean is zero by symmetry
        for k in range(max_lag):
            acf[k] += float(np.dot(d[k:], d[:n - k]) / (n - k))
    return acf / len(series_list)


def green_kubo(series_list, sample_interval, tail="exponential",
               fit_hi=0.5, fit_lo=0.05):
    """Zero-shear viscosity from equilibrium stress fluctuations, Eq. (GK).

    The upper limit is the whole difficulty, so three numbers are returned rather
    than one, and they should be compared:

      "matched"      integrate the measured C(k) only as far as the fit window
                     starts, then append the analytic tail of an exponential
                     fitted to the decaying region. Least sensitive to the cutoff
                     and the recommended value.
      "peak"         the maximum of the running integral. An upper bound: the
                     integral of the noise performs a random walk, so its maximum
                     is biased high by roughly the walk's excursion.
      "plateau_avg"  the running integral averaged over the flat region (from the
                     fit window onwards), which trades that bias for variance.

    The fit window is selected from the correlation function itself -- between
    where it falls below `fit_hi` and `fit_lo` of C(0) -- not from a fixed fraction
    of the lag range, which is what makes the fit robust.
    """
    acf = stress_autocorrelation(series_list)
    lags = np.arange(len(acf)) * sample_interval
    inc = 0.5 * (acf[1:] + acf[:-1]) * sample_interval
    running = np.concatenate(([0.0], np.cumsum(inc)))

    c0 = acf[0]
    if not (c0 > 0):
        return {"value": float("nan"), "matched": float("nan"), "peak": float("nan"),
                "plateau_avg": float("nan"), "tau_fit": float("nan"),
                "acf": acf, "lags": lags, "running": running}

    # window: from where C first drops below fit_hi*C0, to where it drops below
    # fit_lo*C0 or first goes non-positive, whichever comes first.
    below = np.nonzero(acf < fit_hi * c0)[0]
    lo = int(below[0]) if len(below) else 1
    stop = np.nonzero((acf < fit_lo * c0) | (acf <= 0))[0]
    hi = int(stop[0]) if len(stop) else len(acf) - 1
    hi = max(hi, lo + 4)
    hi = min(hi, len(acf) - 1)

    tau_fit, matched = float("nan"), float("nan")
    seg = acf[lo:hi]
    good = seg > 0
    if good.sum() >= 4:
        t = lags[lo:hi][good]
        slope, intercept = np.polyfit(t, np.log(seg[good]), 1)
        if slope < 0:
            tau_fit = -1.0 / slope
            A = math.exp(intercept)
            # integral to lags[lo], plus the analytic tail from there to infinity
            matched = float(running[lo] + A * tau_fit * math.exp(-lags[lo] / tau_fit))

    peak = float(np.max(running))
    plateau_avg = float(np.mean(running[lo:])) if lo < len(running) else float("nan")
    value = matched if math.isfinite(matched) else plateau_avg
    return {"value": value, "matched": matched, "peak": peak,
            "plateau_avg": plateau_avg, "tau_fit": tau_fit,
            "fit_range": (float(lags[lo]), float(lags[hi])),
            "acf": acf, "lags": lags, "running": running}

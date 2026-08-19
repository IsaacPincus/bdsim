"""Rheological measurements: shear stress, viscosity, and variance reduction.

Deliberately separate from `bdsim.ensemble`. The core run layer knows how to
propagate chains and reduce them with a function; it has no opinion about what
that function computes. This module supplies the rheological ones.

Two levels, and you can use either:

  * per-trajectory functions -- `viscosity_series`, `stress_series`,
    `viscosity_series_vr` -- which have the signature `run_ensemble` expects and
    can be passed to it directly, or called on a single chain.
  * thin wrappers -- `shear_viscosity_series`, `equilibrium_stress_series` --
    which just call `run_ensemble` with the matching function and stack the
    result. Four lines each; they exist for convenience, not to hide anything.

All of these are module-level functions so that `backend="processes"` can pickle
them.

The per-sample series is returned rather than averaged, because successive
samples along a trajectory are correlated and their raw scatter understates the
uncertainty. Feed the array to `bdsim.statistics.trajectory_ensemble_stats`,
which corrects for that and cross-checks against the between-trajectory scatter.
"""
import copy

import numpy as np

from ._bdsim import Flow, integrate, total_force
from . import properties as props
from .ensemble import _segment, run_ensemble, trajectory_samples


# --------------------------------------------------------------------------
# Per-trajectory functions: pass these to run_ensemble
# --------------------------------------------------------------------------

def stress_series(R0, phys, sim, rng, sample_times):
    """tau_xy at each sample time, for one chain.

    With `phys.flow` at zero this is the equilibrium stress fluctuation that
    Green-Kubo integrates; under flow it is the raw stress behind the viscosity.
    """
    return [float(props.kramers_stress(R, total_force(R, phys))[0, 1])
            for _t, R in trajectory_samples(R0, phys, sim, rng, sample_times)]


def viscosity_series(R0, phys, sim, rng, sample_times, rate):
    """Per-sample polymer viscosity contribution -tau_xy/rate, for one chain."""
    return [-float(props.kramers_stress(R, total_force(R, phys))[0, 1]) / rate
            for _t, R in trajectory_samples(R0, phys, sim, rng, sample_times)]


def viscosity_series_vr(R0, phys, sim, rng, sample_times, rate):
    """`viscosity_series` with an equilibrium control chain subtracted.

    A second chain is propagated at zero flow from the same initial
    configuration and driven by the SAME random stream, and its shear stress is
    subtracted sample by sample:

        eta_p = -<tau_xy - tau_xy^eq> / gammadot .

    Unbiased, because <tau_xy^eq> = 0 identically by symmetry, but the two chains
    see the same Brownian kicks and so fluctuate together, and most of the noise
    cancels in the difference.

    The pair stays in lockstep because the integrator draws exactly 3N deviates
    per step regardless of configuration, and nothing else consumes the stream
    (the implicit solve and the Chebyshev series are both deterministic).

    WHEN TO USE IT. The benefit depends entirely on how correlated the pair
    stays, and shear destroys that correlation: the flow rotates and eventually
    tumbles the chain, and the sheared and unsheared copies fall out of phase.
    Measured variance ratios (2 kbp, Ns = 20, free draining), where >2 is needed
    just to pay for the doubled cost:

        Wi = 0.01   146x        Wi = 0.3    1.2x
        Wi = 0.03    31x        Wi = 3      0.6x  (worse than not using it)

    So this is a low-shear tool. Below Wi ~ 0.1 it makes the near-equilibrium
    region accessible at all -- a direct measurement there returns noise, because
    the signal falls off as Wi while the stress fluctuations do not. Above
    Wi ~ 0.3 it adds independent noise and should be left off.

    Being unbiased, it agrees with the direct estimate where both work: at Wi = 3,
    224 +/- 147 against 332 +/- 159 (0.50 sigma); at Wi = 0.3, 3883 +/- 1259
    against 2951 +/- 923.

    Even 146x is not a licence to push arbitrarily low. The signal itself scales
    as Wi, so the sampling needed still grows as Wi -> 0; at Wi = 0.01 the
    residual noise remains larger than the answer for the run lengths used here.
    """
    phys_eq = copy.deepcopy(phys)
    phys_eq.flow = Flow()                       # zero velocity gradient

    R_f = np.asarray(R0, dtype=np.float64)
    R_e = R_f.copy()
    # An independent generator on the identical stream. deepcopy goes through the
    # pickle protocol and builds a new object; calling __setstate__ on a live one
    # is rejected by nanobind. (This relies on Rng state round-tripping, which it
    # did not before the am_ fix -- a restored generator used to emit only zeros.)
    rng_e = copy.deepcopy(rng)

    out, t = [], sim.time_start
    for ts in sample_times:
        seg = _segment(sim, t, ts)
        R_f = integrate(R_f, phys, seg, rng)
        R_e = integrate(R_e, phys_eq, seg, rng_e)
        tau_f = props.kramers_stress(R_f, total_force(R_f, phys))[0, 1]
        tau_e = props.kramers_stress(R_e, total_force(R_e, phys))[0, 1]
        out.append(-float(tau_f - tau_e) / rate)
        t = ts
    return out


# --------------------------------------------------------------------------
# Convenience wrappers over run_ensemble
# --------------------------------------------------------------------------

def shear_viscosity_series(phys, sim, rate, n_traj, sample_times, *,
                           variance_reduction=False, **kwargs):
    """(n_traj, n_samples) array of per-sample viscosity contributions.

    `phys.flow` must already be a shear flow at `rate`. Remaining keyword
    arguments (seed, initial, backend, n_workers, on_error, ...) go to
    `run_ensemble`.

    Equivalent to calling `run_ensemble` yourself:

        run_ensemble(phys, sim, n_traj, viscosity_series,
                     args=(list(sample_times), rate))
    """
    fn = viscosity_series_vr if variance_reduction else viscosity_series
    out = run_ensemble(phys, sim, n_traj, fn,
                       args=(list(sample_times), rate), **kwargs)
    return np.asarray(out)


def equilibrium_stress_series(phys, sim, n_traj, sample_times, **kwargs):
    """(n_traj, n_samples) array of tau_xy(t) with no flow, for Green-Kubo.

    `phys.flow` should be the zero tensor: Green-Kubo extracts the zero-shear
    viscosity from equilibrium fluctuations. Feed the result to
    `bdsim.statistics.green_kubo`.
    """
    out = run_ensemble(phys, sim, n_traj, stress_series,
                       args=(list(sample_times),), **kwargs)
    return np.asarray(out)


def shear_viscosity(phys, sim, rate, n_traj, sample_times, **kwargs):
    """Steady-state polymer shear viscosity as (mean, stderr).

    The quick answer. It averages each trajectory's samples first and then treats
    the trajectories as independent, which is only honest if the sample window is
    long compared with the relaxation time. For a defensible error bar use
    `shear_viscosity_series` and `bdsim.statistics.trajectory_ensemble_stats`.
    """
    from .ensemble import mean_stderr
    series = shear_viscosity_series(phys, sim, rate, n_traj, sample_times, **kwargs)
    return mean_stderr([float(np.mean(s)) for s in series])

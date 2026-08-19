"""Running trajectories and ensembles -- the single place that decides *what runs*.

There are exactly two entry points, because an ensemble is only ever one of two
things: something you want on disk, or something you want reduced to numbers.

  * `simulate`      -- run independent chains and write each one's snapshots
                       (positions, optionally forces) to HDF5.
  * `run_ensemble`  -- run independent chains and reduce each with a function
                       you supply. Everything else -- equilibrium averages,
                       stress, viscosity, tumbling statistics -- is that
                       function. The rheological ones live in `bdsim.rheology`.

A run is specified by four things, matching the natural questions a user asks:

  1. how to start        -> `Initial` (initial-config method + its kwargs, seed)
  2. the physics         -> `PhysParams`
  3. the time stepping   -> `SimParams` (dt, ...) and `n_steps`
  4. what to record      -> `Output` (directory, write_every, write_forces)

Trajectories are independent and seeded by index (seed + i), so results do not
depend on the number of workers. Parallelism is `parallel` (serial / processes).
A trajectory that fails is dropped and reported rather than taking the run down
with it; see `MAX_FAILED_FRACTION`.
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from ._bdsim import PhysParams, SimParams, Rng, integrate, total_force
from .initial import (gaussian_chain, fene_fraenkel_chain,
                      fene_fraenkel_chain_aligned_x, fene_fraenkel_bending_chain)
from . import storage
from .parallel import parallel_map


# --------------------------------------------------------------------------
# Run specification (inputs 1 and 4 above; phys/sim are the C++ param objects)
# --------------------------------------------------------------------------

# Initial-configuration methods, by name (picklable: workers look up the string).
def _init_gaussian(n, seed, kw):              return gaussian_chain(n, seed=seed, **kw)
def _init_fene_fraenkel(n, seed, kw):         return fene_fraenkel_chain(n, seed=seed, **kw)
def _init_fene_fraenkel_x(n, seed, kw):       return fene_fraenkel_chain_aligned_x(n, seed=seed, **kw)
def _init_fene_fraenkel_bending(n, seed, kw): return fene_fraenkel_bending_chain(n, seed=seed, **kw)

INITIALIZERS = {
    "gaussian": _init_gaussian,
    "fene_fraenkel": _init_fene_fraenkel,
    "fene_fraenkel_x": _init_fene_fraenkel_x,
    "fene_fraenkel_bending": _init_fene_fraenkel_bending,
}


@dataclass
class Initial:
    """How to build each trajectory's starting chain.

    `method` is a key of INITIALIZERS (e.g. "gaussian", "fene_fraenkel"); `kwargs`
    are passed through to that constructor (e.g. sigma=..., dQ=...). The per-chain
    seed is provided by the runner (seed + trajectory index).
    """
    method: str = "gaussian"
    kwargs: dict = field(default_factory=dict)


@dataclass
class Output:
    """What to record. If `directory` is set, `simulate` writes one HDF5 file per
    trajectory there, taking a snapshot every `write_every` steps (plus the
    initial and final states); `write_forces` adds the total bead forces.
    """
    directory: str = None
    write_every: int = 0          # steps between snapshots (0 => only start + end)
    write_forces: bool = False
    compression: str = "gzip"


# --------------------------------------------------------------------------
# Time stepping
# --------------------------------------------------------------------------

def _segment(sim: SimParams, t0: float, t1: float) -> SimParams:
    """Copy `sim` but run only over [t0, t1] (the flow clock advances with it)."""
    s = SimParams()
    s.dt = sim.dt
    s.implicit_loop_tol = sim.implicit_loop_tol
    s.update_center_of_mass = sim.update_center_of_mass
    s.time_start = t0
    s.time_end = t1
    return s


def n_steps_of(sim: SimParams) -> int:
    """Number of integrator steps implied by a SimParams' time span and dt."""
    return int(round((sim.time_end - sim.time_start) / sim.dt))


def _snapshot(step, R, phys, sim, write_forces):
    t = sim.time_start + step * sim.dt
    F = total_force(R, phys) if write_forces else None
    return step, t, np.array(R, dtype=np.float64, copy=True), F


def _iter_snapshots(R0, phys, sim, rng, n_steps, write_every, write_forces):
    """Integrate one chain `n_steps` steps, yielding (step, time, R, F_or_None)
    at step 0, then every `write_every` steps, and at the final step.

    A chunk of k steps is run as a segment of length (k-1)*dt: the integrator does
    exactly k steps for that span, and successive chunks share the flow clock, so
    chunked stepping equals one continuous run (same RNG stream, same steps).
    """
    R = np.asarray(R0, dtype=np.float64)
    w = write_every if (write_every and write_every > 0) else n_steps
    step = 0
    yield _snapshot(step, R, phys, sim, write_forces)
    while step < n_steps:
        chunk = min(w, n_steps - step)
        t0 = sim.time_start + step * sim.dt
        R = integrate(R, phys, _segment(sim, t0, t0 + (chunk - 1) * sim.dt), rng)
        step += chunk
        yield _snapshot(step, R, phys, sim, write_forces)


def trajectory_samples(R0, phys: PhysParams, sim: SimParams, rng: Rng, sample_times):
    """Integrate one chain, recording its configuration at each of `sample_times`.

    Integrates in segments between successive sample times; the RNG stream carries
    across segments. Returns a list of (time, (N,3) array) pairs.
    """
    R = np.asarray(R0, dtype=np.float64)
    out, t = [], sim.time_start
    for ts in sample_times:
        R = integrate(R, phys, _segment(sim, t, ts), rng)
        out.append((ts, R.copy()))
        t = ts
    return out


# --------------------------------------------------------------------------
# simulate: run an ensemble and write trajectories to HDF5
# --------------------------------------------------------------------------

def _phys_summary(phys: PhysParams) -> dict:
    """Small JSON/attr-friendly description of the physical parameters."""
    return {
        "n_beads": int(phys.number_of_beads),
        "spring_type": str(phys.spring.type),
        "sqrtb": float(phys.spring.sqrtb),
        "natural_length": float(phys.spring.natural_length),
        "hstar": float(phys.hstar),
        "hi_method": str(phys.hi_method),
    }


def _simulate_worker(job):
    (index, phys, sim, n_steps, directory, write_every, write_forces,
     compression, seed, method, kwargs) = job
    R0 = INITIALIZERS[method](phys.number_of_beads, seed, kwargs)
    rng = Rng(seed)
    steps, times, positions = [], [], []
    forces = [] if write_forces else None
    for st, t, R, F in _iter_snapshots(R0, phys, sim, rng, n_steps, write_every, write_forces):
        steps.append(st)
        times.append(t)
        positions.append(R)
        if write_forces:
            forces.append(F)
    attrs = {"index": index, "seed": seed, "n_steps": n_steps,
             "write_every": write_every or n_steps, "dt": float(sim.dt),
             **{f"phys_{k}": v for k, v in _phys_summary(phys).items()}}
    path = storage.trajectory_path(directory, index)
    storage.write_trajectory(path, steps, times, positions, forces,
                             attrs=attrs, compression=compression)
    return os.path.basename(path)


def simulate(phys: PhysParams, sim: SimParams, *, n_traj: int = 1, seed: int = 0,
             n_steps: int = None, initial: Initial = None, output: Output = None,
             backend: str = "serial", n_workers: int = None,
             on_error: str = "skip", max_failed_fraction: float = None):
    """Run `n_traj` independent chains and write their snapshots to HDF5.

    Steps are `n_steps` (default: implied by sim.time_end/dt). Each trajectory is
    seeded by `seed + i` and started with `initial`; snapshots are written to
    `output.directory` every `output.write_every` steps. Returns a `storage.Run`
    reader over the resulting directory.
    """
    initial = initial or Initial()
    output = output or Output()
    if output.directory is None:
        raise ValueError("output.directory must be set (the run folder to write into)")
    n_steps = int(n_steps) if n_steps is not None else n_steps_of(sim)
    os.makedirs(output.directory, exist_ok=True)

    jobs = [(i, phys, sim, n_steps, output.directory, output.write_every,
             output.write_forces, output.compression, seed + i,
             initial.method, dict(initial.kwargs)) for i in range(n_traj)]
    files = parallel_map(_simulate_worker, jobs, backend=backend,
                         n_workers=n_workers, on_error=on_error)
    files, failed = _drop_failures(files, "trajectories", max_failed_fraction)

    manifest = {
        "n_trajectories": len(files),
        "n_trajectories_requested": n_traj,
        "n_trajectories_failed": len(failed),
        "seed": seed,
        "n_steps": n_steps,
        "dt": float(sim.dt),
        "write_every": output.write_every or n_steps,
        "write_forces": bool(output.write_forces),
        "initial": {"method": initial.method, "kwargs": initial.kwargs},
        "phys": _phys_summary(phys),
        "files": list(files),
    }
    storage.write_manifest(output.directory, manifest)
    return storage.read_run(output.directory)


# --------------------------------------------------------------------------
# Reducing an ensemble to numbers (no per-step output)
#
# One primitive, run_ensemble, which maps a function of your choosing over the
# trajectories. There is deliberately no registry of named properties and no
# built-in "shear viscosity" entry point: a property is just a function you
# write, and the rheological ones live in bdsim.rheology.
# --------------------------------------------------------------------------

MAX_FAILED_FRACTION = 0.1
"""Above this share of failed trajectories an ensemble refuses to return a value.

A trajectory fails when the integrator gives up on it -- in practice the implicit
corrector stops converging, the chain is thrown apart, and the diffusion tensor
that follows is no longer positive definite. That trajectory's data is worthless,
so dropping it is right. But the failures are not random: they happen to the
most-stretched chains, so discarding many of them biases the ensemble towards the
well-behaved ones, and the answer would look plausible while being wrong. A high
failure rate is a statement about `dt` and `implicit_loop_tol`, not about the
polymer, so past this fraction the run stops instead of reporting a number.
"""


def _drop_failures(results, what, max_fraction=None):
    """Split `parallel_map` output into successes and failures.

    Reports what failed on stderr, and raises if too many did. Returns the
    successful results and the list of Failures.
    """
    from .parallel import Failure
    ok = [r for r in results if not isinstance(r, Failure)]
    bad = [r for r in results if isinstance(r, Failure)]
    if not bad:
        return ok, bad

    n = len(results)
    frac = len(bad) / n
    seen, lines = set(), []
    for f in bad:
        key = f.message.split("(")[0]
        if key not in seen:
            seen.add(key)
            lines.append(f"    trajectory {f.index}: {f.message.splitlines()[0]}")

    print(f"bdsim: {len(bad)} of {n} {what} failed and were dropped "
          f"({100 * frac:.0f}%). Distinct causes:", file=sys.stderr)
    for line in lines[:3]:
        print(line, file=sys.stderr)
    print("  Look for an earlier corrector non-convergence warning: that is where\n"
          "  a failing trajectory actually goes wrong. Reducing dt or tightening\n"
          "  implicit_loop_tol is the usual fix.", file=sys.stderr)

    limit = MAX_FAILED_FRACTION if max_fraction is None else max_fraction
    if frac > limit:
        raise RuntimeError(
            f"{len(bad)} of {n} {what} failed ({100 * frac:.0f}%), above the "
            f"{100 * limit:.0f}% limit. The survivors are the trajectories that "
            f"happened not to blow up, so averaging them would bias the result "
            f"towards well-behaved chains. Reduce dt or tighten "
            f"implicit_loop_tol; raise bdsim.ensemble.MAX_FAILED_FRACTION (or "
            f"pass max_failed_fraction) if you really want the partial ensemble.")
    if not ok:
        raise RuntimeError(f"every one of the {n} {what} failed")
    return ok, bad


def mean_stderr(values):
    """(mean, standard error of the mean) of a 1-D sequence.

    The plain estimate, valid when the values are independent -- one number per
    trajectory, say. It is NOT valid for successive samples along one trajectory,
    which are correlated; use `bdsim.statistics.trajectory_ensemble_stats` for
    those.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, stderr


def _run_worker(job):
    """Build this trajectory's starting chain and hand it to the user function."""
    fn, phys, sim, seed, method, kwargs, args = job
    R0 = INITIALIZERS[method](phys.number_of_beads, seed, kwargs)
    return fn(R0, phys, sim, Rng(seed), *args)


def final_state(R0, phys, sim, rng):
    """A `per_trajectory` function: integrate the chain and return its final (N,3).

    The simplest useful one, and the building block for equilibrium averages:

        Rs = run_ensemble(phys, sim, 200, final_state)
        rg2 = [radius_of_gyration_sq(R) for R in Rs]
        mean, err = mean_stderr(rg2)
    """
    return np.asarray(integrate(R0, phys, sim, rng), dtype=np.float64)


def sampled_states(R0, phys, sim, rng, sample_times):
    """A `per_trajectory` function: the configuration at each of `sample_times`.

    Returns a list of (N,3) arrays. `run_ensemble(..., args=(times,))` passes the
    times through.
    """
    return [R for _t, R in trajectory_samples(R0, phys, sim, rng, sample_times)]


def run_ensemble(phys: PhysParams, sim: SimParams, n_traj: int,
                 per_trajectory, *, args=(), seed: int = 0,
                 initial: Initial = None, backend="serial", n_workers=None,
                 on_error="skip", max_failed_fraction=None):
    """Run `n_traj` independent chains, reducing each one with your own function.

    This is the whole ensemble layer. `per_trajectory` is called once per chain as

        per_trajectory(R0, phys, sim, rng, *args)

    with `R0` the starting configuration built from `initial` and `rng` already
    seeded with `seed + i`, and may return anything: a number, an array, a tuple.
    Results come back as a list in trajectory order, with failed trajectories
    dropped and reported (see MAX_FAILED_FRACTION).

    Because trajectories are seeded by index, the result does not depend on how
    many workers ran it.

    One constraint with `backend="processes"`: `per_trajectory` is sent to the
    worker processes by pickle, so it must be a module-level function --- a
    lambda, a closure, or a function defined inside another function will not
    pickle. Anything that varies between runs goes in `args`.

    `initial` is an `Initial` (default: gaussian). Starting from a configuration
    already drawn from the model's equilibrium matters whenever the springs are
    stiff or have a large natural length: a gaussian start is then so far from
    equilibrium that a short run measures the relaxation, not the equilibrium.

    Examples
    --------
    Equilibrium averages::

        Rs = run_ensemble(phys, sim, 200, final_state, backend="processes")
        mean, err = mean_stderr([radius_of_gyration_sq(R) for R in Rs])

    Anything else you want per trajectory is a function you write::

        def max_extension(R0, phys, sim, rng):
            R = integrate(R0, phys, sim, rng)
            return float(np.ptp(R[:, 0]))

        vals = run_ensemble(phys, sim, 200, max_extension)
    """
    initial = initial or Initial()
    jobs = [(per_trajectory, phys, sim, seed + i, initial.method,
             dict(initial.kwargs), tuple(args)) for i in range(n_traj)]
    out = parallel_map(_run_worker, jobs, backend=backend, n_workers=n_workers,
                       on_error=on_error)
    out, _ = _drop_failures(out, "trajectories", max_failed_fraction)
    return out

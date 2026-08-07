"""Parallel execution for embarrassingly-parallel trajectory ensembles.

Trajectories are independent, so an ensemble is a map over per-trajectory work
followed by an optional reduce. `parallel_map` runs that map with one of two
backends:

  * "serial"    - a plain loop (no dependencies; easiest to debug).
  * "processes" - multiprocessing across the cores of one machine (stdlib); the
                  simplest way to use every core of a workstation or one node.

Reproducibility: callers partition work by trajectory index (seed = base + i), so
the combined result is identical no matter how many workers are used.

`fn` must be a top-level (importable) function and `items` must be picklable --
the bdsim parameter objects (PhysParams, SimParams, Rng, ...) are picklable.
"""


def parallel_map(fn, items, *, backend="serial", n_workers=None):
    """Apply `fn` to each of `items`, returning results in order.

    backend="serial" runs a loop; backend="processes" fans out over `n_workers`
    processes (None => all cores). Results are returned in input order either way.
    """
    items = list(items)

    if backend == "serial":
        return [fn(x) for x in items]

    if backend == "processes":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(fn, items))

    raise ValueError(f"unknown backend {backend!r} (use 'serial' or 'processes')")

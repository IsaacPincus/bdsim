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


class Failure:
    """Placeholder for a work item whose function raised.

    Returned in place of a result when `parallel_map` is called with
    `on_error="skip"`. One trajectory can fail for reasons that say nothing about
    the others -- a degenerate configuration after the implicit corrector stops
    converging, say -- and on a long ensemble it is usually better to lose that
    trajectory than the whole run. The exception is kept so the caller can report
    what happened instead of guessing.
    """

    __slots__ = ("index", "error")

    def __init__(self, index, error):
        self.index = index
        self.error = error

    def __repr__(self):
        return f"Failure(index={self.index}, error={self.error!r})"

    @property
    def message(self):
        return f"{type(self.error).__name__}: {self.error}"


class _Guarded:
    """fn wrapped so exceptions come back as Failure.

    A class rather than a closure because the processes backend has to pickle it.
    """

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, item):
        try:
            return self.fn(item)
        except Exception as exc:            # deliberately broad: see Failure
            return Failure(None, exc)


def parallel_map(fn, items, *, backend="serial", n_workers=None,
                 on_error="raise"):
    """Apply `fn` to each of `items`, returning results in order.

    backend="serial" runs a loop; backend="processes" fans out over `n_workers`
    processes (None => all cores). Results are returned in input order either way.

    on_error="raise" (the default) lets an exception propagate, aborting the map.
    on_error="skip" catches it and puts a `Failure` in that slot instead, so one
    bad item does not destroy the rest of the work; the caller is then
    responsible for noticing them. `Failure.index` gives the position in `items`.
    """
    items = list(items)
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    call = fn if on_error == "raise" else _Guarded(fn)

    if backend == "serial":
        out = [call(x) for x in items]

    elif backend == "processes":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            out = list(ex.map(call, items))

    else:
        raise ValueError(f"unknown backend {backend!r} (use 'serial' or 'processes')")

    for i, r in enumerate(out):
        if isinstance(r, Failure):
            r.index = i
    return out

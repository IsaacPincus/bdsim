# Running in parallel

Trajectories are independent, so an ensemble is a map (per-trajectory work) then
a reduce (average) or a write (to HDF5). `bdsim.parallel.parallel_map` runs the
map with one of two backends, and the driver functions (`simulate`,
`run_ensemble`, `ensemble.shear_viscosity`) take a `backend=` argument.

**Trajectories are seeded by index** (`seed = base + i`), so the combined result
is *identical regardless of the number of workers* -- parallelism changes the
speed, never the science.

## Backends

- `backend="serial"` -- a plain loop. Easiest to debug; the default.
- `backend="processes"` -- Python's `multiprocessing` (stdlib, no extra deps),
  saturating every core of the machine:

```python
res = bdsim.run_ensemble(phys, sim, n_traj=200, backend="processes")     # n_workers=... optional
run = bdsim.simulate(phys, sim, n_traj=200, output=out, backend="processes")
```

Measured ~1.8x on 2 cores (near-ideal); scales with core count. This covers a
workstation or a single cluster node (tens of cores).

> Earlier revisions had an `mpi` backend for multi-node clusters. It was removed
> to keep the driver simple. For a cluster the dependency-free option is a SLURM
> **job array**: launch P single-core jobs, each running a slice of the seeds
> (`simulate(..., seed=base + shard, n_traj=per_shard)` or filter `i % nshards`),
> and combine the HDF5 run directories afterwards. Because trajectory *i* is fully
> determined by `seed = base + i`, any sharding gives the same set of trajectories.

## Notes

- Functions/objects sent to workers must be picklable. The bdsim parameter
  objects (PhysParams, SimParams, Rng, Flow, ...) are picklable; custom worker
  functions must be top-level (importable), not lambdas/closures.
- Keep threaded BLAS from oversubscribing when you fork many processes: set
  `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS=1`) so each trajectory process
  stays single-threaded and the cores go to trajectories, not nested BLAS.

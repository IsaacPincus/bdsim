# Running simulations

:::{admonition} Complete option reference
:class: tip
`examples/all_options.py` sets **every** parameter the code accepts, with its
default, meaning and valid values, and runs a check that the combination works.
It also self-audits against the compiled module (`--audit`), so it cannot fall out
of date when the bindings change. Use it as the lookup table; use this page for the
workflow.
:::

## The four questions

A run is specified by four things, and the API is shaped around them:

| Question | Object |
|---|---|
| How do the chains start? | `bdsim.Initial` |
| What is the physics? | `bdsim.PhysParams` |
| How is time stepped? | `bdsim.SimParams` (+ `n_steps`) |
| What is recorded? | `bdsim.Output` |

## Physics

```python
import bdsim

phys = bdsim.PhysParams()
phys.number_of_beads = 20

phys.spring.type = bdsim.Spring.FENEFraenkel   # Hook, FENE, ILC, WLC,
phys.spring.sqrtb = 5.0                        # Fraenkel, FENEFraenkel, WLCbounded
phys.spring.natural_length = 1.0               # sigma; 0 makes FENE-Fraenkel = FENE

phys.bend.type = bdsim.Bending.OneMinusCosTheta
phys.bend.stiffness = 1.5                      # C in phi_b/kT = C(1 - cos theta)

phys.hstar = 0.2                               # 0 = free draining
phys.hi_method = bdsim.DelSMethod.Cholesky     # or Chebyshev

phys.flow = bdsim.flows.shear(1.0)             # equilibrium(), shear(),
                                               # uniaxial_extension(), planar_extension()
```

`sqrtb` and `natural_length` are in Hookean units. For a real molecule you do not
choose them by hand --- see [coarse-graining](coarse_graining.md).

:::{admonition} Which HI method?
:class: tip
`Cholesky` is exact and unconditionally stable, and is **faster** below about
$N=100$. `Chebyshev` wins above that. Both are validated against each other.
:::

## Time stepping

```python
sim = bdsim.SimParams()
sim.dt = 0.01
sim.time_end = 100.0
sim.implicit_loop_tol = 1e-4     # convergence of the implicit bond solve
sim.update_center_of_mass = True # subtract the centre of mass each step
```

:::{admonition} The timestep and the relaxation time
:class: warning
The single most common mistake is running for less than a relaxation time and
believing the answer. The longest Rouse time is
$\lambda_1 = 1/\sin^2(\pi/2N)$ in simulation units --- for $N=21$ that is
**179**, so `time_end = 10` is a twentieth of one relaxation time. Equilibrate for
$\gtrsim 3\lambda_1$ and sample over $\gtrsim 6\lambda_1$.

```python
from bdsim.dynamics import free_draining_relaxation_time
```

Under-equilibration does not look like noise: it produces smooth, plausible,
*wrong* curves --- for example an apparent shear-thickening viscosity.
:::

## Initial configuration

```python
initial = bdsim.Initial("gaussian")
initial = bdsim.Initial("fene_fraenkel", dict(sigma=1.0, dQ=4.0))
initial = bdsim.Initial("fene_fraenkel_bending",
                        dict(sigma=1.0, dQ=4.0, stiffness=1.5))
```

Starting from the model's own equilibrium matters whenever the springs are stiff
or have a large natural length: a Gaussian start is then so far from equilibrium
that a short run measures the relaxation, not the equilibrium.

## Running an ensemble

For averaged scalars, with no per-step output:

```python
res = bdsim.run_ensemble(phys, sim, n_traj=200, seed=0,
                         properties=("Rsq", "Rg_sq"),
                         initial=initial, backend="processes")
mean, stderr = res["Rsq"]
```

For trajectories written to disk, see [Storage](storage.md).

## Parallelism

`backend="serial"` or `"processes"`; `n_workers=None` uses all cores. Trajectories
are seeded by index, so **results are identical whatever the worker count**.

```bash
export OMP_NUM_THREADS=1     # stop a threaded BLAS oversubscribing the cores
```

For a cluster, use a SLURM job array over seed ranges rather than MPI: because
trajectory *i* is fully determined by `seed + i`, any sharding gives the same set.

## Applying an external force

```python
phys.external.add_stretch(0, -1, (0.0, 0.0, 5.0))   # +/-5 z on the two ends
phys.external.add_constant(3, (1.0, 0.0, 0.0))      # constant force, bead 3
phys.external.add_time_varying(0, times, forces)    # interpolated protocol
```

Negative indices count from the end. `add_stretch` applies equal and opposite
forces, so the chain stretches without drifting. External forces are excluded from
`total_force`, which continues to mean the chain's own force.

## A complete script

```python
import bdsim, numpy as np

phys = bdsim.PhysParams()
phys.number_of_beads = 21
phys.spring.type = bdsim.Spring.FENE
phys.spring.sqrtb = 7.0
phys.hstar = 0.15
phys.hi_method = bdsim.DelSMethod.Cholesky
phys.flow = bdsim.flows.shear(0.5)

tau = 1.0 / np.sin(np.pi / (2 * phys.number_of_beads)) ** 2   # ~179
sim = bdsim.SimParams()
sim.dt = 0.01
sim.implicit_loop_tol = 1e-4

samples = np.linspace(3 * tau, 9 * tau, 100)          # after equilibration
series = bdsim.shear_viscosity_series(
    phys, sim, 0.5, n_traj=32, sample_times=samples,
    initial=bdsim.Initial("gaussian"), backend="processes")

stats, se_between = bdsim.trajectory_ensemble_stats(series, samples[1] - samples[0])
print(stats)
```

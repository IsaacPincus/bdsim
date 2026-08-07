# Overview

## What the code is

`bdsim` integrates the Itô stochastic differential equation for a bead--spring
chain,

$$\mathrm{d}\bm{R} = \left[\bm{K}\cdot\bm{R} + \tfrac14 \bm{D}\cdot\bm{F}^\phi\right]\mathrm{d}t^* + \tfrac{1}{\sqrt2}\bm{B}\cdot\mathrm{d}\bm{W},$$

with a semi-implicit predictor--corrector step: the spring force is solved
implicitly so that finitely extensible springs stay stable at usable timesteps,
while excluded volume, bending, external forces and the flow are explicit.

It is a port of an older Fortran code, and is cross-checked against it: the force
kernel agrees to machine precision, and ensemble rheology agrees within statistics
(see `validation/compare_fortran.py`).

## The two layers, and why the split is where it is

```
     ┌─────────────────────────── Python ───────────────────────────┐
     │  what to run · how to start · what to record · how to analyse │
     │                                                               │
     │  ensemble   coarse_grain   dynamics   storage   statistics    │
     └───────────────────────────┬───────────────────────────────────┘
                                 │  nanobind  (bindings/module.cpp)
     ┌───────────────────────────┴───────────────────────────────────┐
     │                          C++ core                             │
     │  one job: advance a chain through time, correctly and fast     │
     │                                                               │
     │  integrator · spring · hydrodynamics · mobility · bending ·   │
     │  excluded_volume · flow · external_force · rng · numerics      │
     └───────────────────────────────────────────────────────────────┘
```

**The C++ core does one thing: advance a configuration through time.** It holds no
opinion about sampling, output, ensembles or units. `time_integrate_chain(R, phys,
sim, rng)` takes a configuration and returns it advanced, with the random stream
carried in the `Rng` object so consecutive calls continue seamlessly.

**Everything else is Python**, because everything else changes often: which
observables to sample, how to store them, how many trajectories, how to average,
how to convert to real units. Putting that in C++ would fossilise decisions that
are still moving.

The seam is deliberately narrow, and it has one important property: **integrating
in segments is identical to integrating in one go**. The RNG stream persists across
calls and the flow clock advances with the segment, so sampling every $N$ steps
does not perturb the trajectory. That is what lets the Python layer take control of
sampling without paying for it.

## Reproducibility

Trajectories in an ensemble are seeded by index (`seed + i`), so **the result does
not depend on how the work is distributed**. Serial and multiprocessing runs give
bit-identical output; the number of workers changes the speed, never the science.

## Units

The core works in Hookean units, built from the spring: length
$l_H = \sqrt{k_BT/H}$, time $\lambda_H = \zeta/4H$, force $F_H = k_BT/l_H$,
energy $k_BT$. Nothing dimensional ever enters the C++.

Converting to laboratory units is a Python-side concern and is handled by
{mod}`bdsim.dynamics`, which returns all three scales once the chain's friction has
been fixed. See the [coarse-graining tutorial](tutorials/coarse_graining.md) --- it
is the part most often got wrong, because *both* Hookean units move when the
discretisation changes.

## Where things live

| Directory | Contents |
|---|---|
| `src/` | the C++ core |
| `bindings/` | `module.cpp`, the nanobind interface |
| `python/bdsim/` | the Python package |
| `tests/` | one self-contained C++ executable per unit, run by `ctest` |
| `validation/` | physics validation and comparison scripts (not unit tests) |
| `examples/` | short runnable demonstrations |
| `docs/` | this documentation and `theory.tex` |
| `bench/` | timing harnesses |
| `fortran_ref/` | the legacy executable and its input deck, for cross-checking |

## What to read next

- Running your first simulation: [Running simulations](tutorials/running.md)
- Saving and reading trajectories: [Storage](tutorials/storage.md)
- Getting numbers with defensible error bars: [Post-processing](tutorials/postprocessing.md)
- Modelling a real molecule: [Coarse-graining DNA](tutorials/coarse_graining.md)
- The physics: `docs/theory.tex`

# The C++ layer

This describes what each source file is responsible for and how they fit together.
For the mathematics, see `docs/theory.tex`; the chapter references below point into
it.

## Why this is hand-written rather than generated

The headers are documented in prose at the top of each file, which is where a
developer reading the source will look. Extracting that with Doxygen would mean
converting every comment to `///` markup across a dozen headers, for an API that
**users never call** --- the user-facing surface is the Python package. So this page
is a guide to the architecture, and the headers themselves are the reference.

If you do want generated C++ API pages, the route is Doxygen + Breathe:
`doxygen` produces XML, `breathe` pulls it into Sphinx, and you add a
`breathe_projects` entry to `conf.py`. Nothing here prevents that; it simply is not
set up, because the cost/benefit did not favour it.

## Dependency order

```
types.hpp ──> vec.hpp ──> config.hpp ──┬──> spring.hpp ──────┐
                                       ├──> bending.hpp      │
                                       └──> excluded_volume  ├──> model.hpp ──> integrator
              linalg ──> hydrodynamics ───> mobility ────────┤
              flow.hpp ─────────────────────────────────────┤
              external_force.hpp ───────────────────────────┘
              rng.hpp, numerics.hpp  (used throughout)
```

## Files

### Foundations

`types.hpp`
: `dp` (double), `i32`, physical constants, and the enums (`Spring`, `EV`,
  `Bending`, `DelSMethod`). Ported from the Fortran `modules.f90`.

`vec.hpp`
: `Vec3`, `Mat3`, and `Vec3Field` --- a per-bead field with arithmetic, so the
  integrator can be written as `R + dt*(...)` and read like the SDE.

`config.hpp` / `.cpp`
: `ChainGeometry` (adjacent bond vectors and their lengths) and `ChainState`
  (positions plus geometry). Computing the geometry once per step and passing it
  around is what removed the old $N\times N$ scratch arrays.

`numerics.hpp` / `.cpp`
: Cubic root finding with a bracketed bisection fallback, Newton polishing, and
  `rtsafe`. The fallback is not decoration: the closed-form cubic returns a
  spurious out-of-range root for extreme coefficients, which used to let a bond
  escape its finite extensibility (theory §4.4).

`rng.hpp` / `.cpp`
: A bit-exact port of the Numerical Recipes `ran_1`, kept so the Fortran
  regression tests compare at machine precision. The single-precision scale factor
  `am` is deliberate --- do not "fix" it (theory ch. 8).

### Physics

`spring.hpp` / `.cpp`
: The seven force laws, and `solve_implicit_r`, which solves
  $r(1 + \tfrac{\Delta t}{4}f(r)) = \gamma$ by one of three techniques depending on
  the law: closed form, cubic, or `rtsafe`. `solve_connector` wraps the reduced-unit
  bookkeeping so the integrator never unpacks spring internals (theory ch. 4).

`bending.hpp` / `.cpp`
: Bending force for $\phi_b/k_BT = C(1-\cos\theta)$. The $\sin\theta$ factors
  cancel analytically for this potential, which is why no small-angle guard is
  needed (theory ch. 6).

`excluded_volume.hpp` / `.cpp`
: Pairwise Gaussian, Lennard-Jones and SDK potentials. The equilibration flag is
  fixed for a run and baked into the object at construction (theory ch. 7).

`hydrodynamics.hpp` / `.cpp`
: The RPY diffusion tensor, and the Chebyshev square root of it --- spectral bounds
  from a warm-started Lanczos iteration with Sturm bisection, then the three-term
  recurrence. The bounds **must** contain the spectrum or the series amplifies
  instead of square-rooting; this is the subtlest code in the project (theory §5.4).

`mobility.hpp` / `.cpp`
: `Mobility` (the configuration-independent model: $h^*$ and method) produces a
  `Diffusion` (the operator at one configuration). Owns the fluctuation--dissipation
  check and the Cholesky fallback, and the warm-start cache that persists between
  steps.

`linalg.hpp` / `.cpp`
: Dense matrix, mat-vec, Cholesky. Hand-rolled, with an optional LAPACK backend
  behind the same signatures (`-DBDSIM_LAPACK=ON`).

`flow.hpp` / `.cpp`
: The velocity-gradient tensor $\bm{\kappa}(t)$: constant, or interpolated from a
  table of samples. All the old named flow types are just particular tensors.

`external_force.hpp` / `.cpp`
: User-applied per-bead forces, constant or time-interpolated. Deliberately kept
  out of `non_spring_force`, so `total_force` still means "the chain's own force"
  --- the Kramers stress and the Fortran cross-check both depend on that
  (theory §2.3).

### Assembly

`model.hpp`
: `PhysParams` is the user's description; `PhysicalModel` is the prepared model
  built from it, owning the force objects, flow, mobility and chain size. New
  explicit force types are added in exactly one place:
  `PhysicalModel::non_spring_force`.

`integrator.hpp` / `.cpp`
: The predictor--corrector, written stage by stage (`recenter`, `predictor`,
  `upsilon`, `solve_connectors`) so it reads like the SDE it solves. Holds only a
  `PhysicalModel` and a `SimParams` (theory ch. 3).

`bindings/module.cpp`
: The nanobind interface. Parameter objects are picklable so they can be shipped
  to worker processes; bead positions and forces cross as `(N,3)` float64 arrays.

## Design rules worth knowing

- **Stateful things are classes, pure transforms are free functions.**
  `ExcludedVolume` and `ChainIntegrator` are objects; the spring laws, config
  helpers and flow evaluation are functions.
- **Functions return their results** rather than writing into out-parameters,
  including the solvers. Data flow is visible at the call site.
- **All physics lives in `PhysicalModel`.** The integrator asks it for the spring
  force, the non-spring force, $\bm{D}\cdot\bm{F}$, $\bm{B}\cdot\mathrm{d}\bm{W}$
  and $\bm{\kappa}$, and knows nothing else.
- **Correctness is anchored on physics, not on the Fortran.** The legacy regression
  is kept as a close cross-check and currently passes at $10^{-6}$, but it does not
  constrain the design.

## Tests

`ctest` runs 14 self-contained executables. Three are worth knowing about:

- `test_integrator_rouse` reproduces the Fortran regression exactly, including two
  chained integrations on one continued RNG stream.
- `test_hi_stability` guards the blow-up mode: bonds bounded for $\gamma$ up to
  $10^{20}$, the Chebyshev square root finite and satisfying
  fluctuation--dissipation on stretched configurations, and warm-started spectral
  bounds still bracketing the spectrum against exact eigenvalues.
- `test_external_force` verifies applied forces through the integrator by
  differencing two runs on the same random stream, which cancels the noise and
  recovers the exact linear-response result to six digits.

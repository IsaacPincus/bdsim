# bdsim

Brownian dynamics of bead-spring polymer chains: a C++ core with a Python
driver. Semi-implicit predictor-corrector integration of the Itô SDE

    dR = [K·R + ¼ D·F] dt + (1/√2) B·dW

with hydrodynamic interaction (Rotne-Prager-Yamakawa, Cholesky or Chebyshev
square roots), seven spring force laws, bending and excluded-volume potentials,
HDF5 trajectory output, and a DNA coarse-graining layer that maps a real
contour length and persistence length onto matched static *and* dynamic
model parameters.

It is a port of [SingleChainBD](https://github.com/IsaacPincus/SingleChainBD)
and is validated against it — the forces agree bit-for-bit to 1e-6 and the
dynamics agree statistically.

## Quick start

```bash
git clone https://github.com/IsaacPincus/bdsim.git
cd bdsim
./setup.sh
```

That creates a virtual environment in `.venv`, installs the Python
dependencies, builds the C++ core and the Python extension, and runs the test
suite. It takes a couple of minutes. You need a C++17 compiler, CMake ≥ 3.18
and Python ≥ 3.9:

```bash
sudo apt update && sudo apt install -y build-essential cmake python3 python3-venv   # Debian/Ubuntu/WSL
xcode-select --install && brew install cmake python                                 # macOS
```

Then, in that environment:

```bash
source .venv/bin/activate
python examples/demo.py                    # equilibrium ensemble + shear with HI
python examples/all_options.py --audit     # every input option, annotated
```

`./setup.sh --help` lists the options: `--native` (build with `-march=native`),
`--lapack` (BLAS/LAPACK for the dense HI kernels), `--fortran` (also build the
Fortran reference, see below), `--no-tests`, `--jobs N`. If
a virtual environment is already active it is used as-is rather than creating
`.venv`. Re-running is cheap — CMake rebuilds only what changed.

<details>
<summary>Building by hand instead</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy h5py nanobind
pip install -e .                     # so `import bdsim` resolves anywhere

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBDSIM_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake locates nanobind by asking the active interpreter, so no `-Dnanobind_DIR`
is needed as long as the environment with `nanobind` installed is the one you
build against. The extension is written straight into `python/bdsim/`, so there
is no copy step. Omit `-DBDSIM_PYTHON=ON` to build only the C++ core and tests.

</details>

## A first simulation

```python
import bdsim

phys = bdsim.PhysParams()
phys.number_of_beads = 10
phys.flow = bdsim.flows.shear(1.0)
phys.hstar = 0.3                        # 0 for free draining

sim = bdsim.SimParams()
sim.dt = 0.01

run = bdsim.simulate(
    phys, sim,
    n_traj=16, seed=0, n_steps=1000,
    initial=bdsim.Initial("fene_fraenkel", dict(sigma=1.0, dQ=2.0)),
    output=bdsim.Output(directory="myrun", write_every=100, write_forces=True),
    backend="processes",
)

run = bdsim.read_run("myrun")           # read back from disk
times, Rg2, err = bdsim.ensemble_average(run, bdsim.radius_of_gyration_sq)
```

A run is four things: how to start (`Initial`), the physics (`PhysParams`), the
time stepping (`SimParams` + `n_steps`), and what to record (`Output`).
`simulate` writes one HDF5 file per trajectory, snapshotting every
`write_every` steps. Stepping in chunks is exact: a chunked run reproduces one
continuous run step-for-step on the same RNG stream, so `write_every` never
perturbs the trajectory. `run_ensemble` is the other entry point: it reduces each
trajectory with a function you supply, so an ensemble average is

```python
Rs = bdsim.run_ensemble(phys, sim, 200, bdsim.final_state, backend="processes")
mean, err = bdsim.mean_stderr([bdsim.radius_of_gyration_sq(R) for R in Rs])
```

and anything else you want per trajectory is a function you write. Rheological
measurements (stress, viscosity, variance reduction) are in `bdsim.rheology`,
deliberately not in the core run layer. A trajectory the integrator gives up on
is dropped and reported rather than taking the run down with it.

## Documentation

```bash
pip install sphinx myst-parser sphinx-rtd-theme
make -C docs html            # -> docs/_build/html/index.html
make -C docs theory          # -> docs/theory.pdf, the mathematics
```

The HTML docs cover an overview, the C++ architecture, the generated Python
API, and tutorials on running simulations, HDF5 storage, post-processing with
correlation-corrected error bars, and coarse-graining a DNA fragment. They
build without compiling the extension (the core is mocked), so they work in a
plain checkout.

[`docs/theory.tex`](docs/theory.tex) is the mathematical reference: the SDE and
the semi-implicit scheme, every spring law and its implicit solve, the HI
machinery, potentials, samplers, observables, the unit systems, and the DNA
coarse-graining and dynamic-matching procedure.

## Layout

```
src/            C++ core
  rng            bit-exact Numerical Recipes RNG
  numerics       cubic root, rtsafe, Newton polish
  config         chain geometry: bonds, lengths, unit vectors, cosines
  spring         7 force laws + the semi-implicit connector solve
  bending        bending force object
  excluded_volume  Gauss / LJ / SDK
  external_force user-applied per-bead forces, constant or time-varying
  flow           kappa tensor, constant or time-interpolated
  linalg         dense kernels; optional BLAS/LAPACK backend
  hydrodynamics  RPY tensor, Chebyshev square root, Lanczos spectral bounds
  mobility       free-draining / Cholesky / Chebyshev dispatch
  model          PhysParams + PhysicalModel (owns all the physics)
  integrator     ChainIntegrator: the SDE step, stage by stage
bindings/       nanobind module
python/bdsim/   Python driver
  flows, initial, properties     kappa tensors, start configurations, observables
  ensemble, parallel, storage    simulate + run_ensemble, HDF5 I/O, post-processing
  rheology                       stress, viscosity, variance reduction
  coarse_grain                   WLC -> FENE-Fraenkel parameters for DNA
  dynamics                       hydrodynamic radius, unit systems, relaxation times
  statistics                     autocorrelation, blocking, steady-state error bars
tests/          one self-contained executable per unit, run by ctest
examples/       demo.py, all_options.py
validation/     Fortran comparison, physics checks, coarse-graining and
                dynamics validation, viscosity sweeps (free-draining and HI)
docs/           Sphinx documentation + theory.tex
fortran/        the original SingleChainBD physics (LAPACK only) + oracle driver
bench/          integrator timings
```

## Validation

```bash
ctest --test-dir build --output-on-failure       # 14 unit tests
python validation/validate_physics.py            # equilibrium moments vs Rouse theory
python validation/relaxation_scaling.py --cap 0.05   # relaxation time scaling (no simulation)
./build/bench                                    # timings
```

### The Fortran reference

The original SingleChainBD physics is vendored in `fortran/` and builds from
this repository with **LAPACK as its only dependency** -- no MPI, no NetCDF,
which were needed by the drivers rather than by the physics:

```bash
sudo apt install gfortran liblapack-dev
cmake -S . -B build -DBDSIM_FORTRAN=ON -DBDSIM_PYTHON=ON && cmake --build build -j
python validation/compare_fortran_oracle.py --case full
```

That runs the same case through both codes at a matched corrector tolerance and
reports the difference as the tolerance is tightened. See
[fortran/README.md](fortran/README.md) for what is vendored and why, and for the
reason the recorded regression tolerance is 1e-6.

`validation/compare_fortran.py` is the older, statistical cross-check against a
full `sens` ensemble. It needs a built `sens` from
[SingleChainBD](https://github.com/IsaacPincus/SingleChainBD); the binary and its
NetCDF output are not tracked here (see `fortran_ref/README.md`). Correctness, physics validation and timings are
described in [BENCHMARKS.md](BENCHMARKS.md); profiling guidance is in
[PROFILING.md](PROFILING.md) and parallelism in [PARALLEL.md](PARALLEL.md).

## Design notes

- Everything is double precision, matching the Fortran `DBprec == DOBL`. The
  one deliberate single-precision detail is the RNG scale factor `am` (see
  `src/rng.hpp`), kept single on purpose for bit-exactness.
- Stateful things are classes (`ExcludedVolume`, `ChainIntegrator`); pure
  transforms are free functions (config, spring laws, bending, flow eval).
- Functions return their results rather than writing into out-parameters,
  including the solvers, so data flow is visible.
- All physics lives in `PhysicalModel`, built once from `PhysParams`. The
  integrator holds only a model and a `SimParams` and asks the model for
  everything: spring force, non-spring force, `D·F`, `B·dW`, kappa. New force
  types are added in one place, `PhysicalModel::non_spring_force`.
- Sampling, property calculation and trajectory output are deliberately outside
  the integrator. The Python driver owns them and integrates in segments; the
  RNG stream persists across calls, so segmented runs are seamless.
- The implicit spring solve lives in the spring module, organised by its three
  solution families — closed form (Hookean, Fraenkel), cubic (FENE, ILC, WLC,
  FENE-Fraenkel) and transcendental root-find (modified WLC) — so it is clear
  which technique each law uses and why.
- Hydrodynamic interaction sits behind the `Mobility` seam. Cholesky matches
  the Fortran oracle exactly; Chebyshev is validated as a genuine symmetric
  square root (fluctuation-dissipation, and `B·(B·X) == D·X`).

## Licence

MIT — see [LICENSE](LICENSE).

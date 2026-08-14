# The Fortran reference core

The original [SingleChainBD](https://github.com/IsaacPincus/SingleChainBD)
physics, vendored so the two codes can be compared without leaving this
repository. **The only dependency is LAPACK** — no MPI, no NetCDF.

That is possible because the dependencies were confined to the drivers, not the
physics:

| file | MPI | NetCDF | vendored |
|---|---|---|---|
| `modules.f90` | – | – | yes |
| `utils.f90` | – | – | yes |
| `properties.f90` | – | – | yes |
| `InitialPos.f90` | – | – | yes |
| `gsipc.f90` | – | – | yes |
| `DataOutput.f90` | – | 45 refs | no |
| `sensemble.f90` | 7 calls | yes | no |
| `tests.f90` | – | – | no (needs FRUIT) |

`sensemble.f90` is the ensemble driver: it needs MPI and writes NetCDF, and none
of that is required to run the physics. `tests.f90` is not vendored either, for a
different reason — it depends on the FRUIT unit-testing framework. `oracle.f90`
here replaces it for the one case that matters, with no external dependencies.

The five vendored files are unmodified **except for one clearly-marked block** in
`gsipc.f90::get_delS_cholesky`, which prints a warning when `dpotrf` returns a
non-zero `INFO`. Upstream that value is declared and never read. Search for
`LOCAL MODIFICATION` to find it; it is worth upstreaming. To refresh the other
files, copy them across from SingleChainBD again.

## Build

```bash
cmake -S . -B build -DBDSIM_FORTRAN=ON
cmake --build build -j
```

You need `gfortran` (or any Fortran compiler CMake can find) and LAPACK:

```bash
sudo apt install gfortran liblapack-dev        # Debian/Ubuntu/WSL
brew install gcc openblas                      # macOS
```

## `oracle.f90`

Runs a case and prints both final configurations at full precision — two
integrations on one continued random stream, as the regression test does.

```bash
./build/fortran_oracle [case] [tolerance] [blocks]   # default: rouse 1.0d-6 2
```

`blocks` is how many consecutive integrations to run on one continued random
stream, printing the configuration after each. Block *k* is the state after
*k* x steps_per_block steps of a single trajectory, so raising it shows how the
agreement between the two codes evolves over a longer run from one invocation.

| case | what it exercises |
|---|---|
| `rouse` | ten Hookean beads, free draining, no flow, seed 5, `dt = 0.1`, `t = 0 → 1`. The setup of `tests.f90::test_rouse_chain_eq`, reproducing the recorded oracle. |
| `singular` | not a numeric comparison but an error-handling one: a real configuration whose RPY tensor is not positive definite. See below. |
| `full` | fifteen beads, FENE-Fraenkel springs with a non-zero natural length (the cubic implicit solve on a shifted bracket), a bending potential, Lennard-Jones excluded volume, hydrodynamic interaction via Cholesky, and simple shear at `dt = 0.005`. The starting chain has overlapping beads (closest pair 0.29 against `2a = 0.89`), so the RPY overlapping branch runs too. |

`full` uses Cholesky rather than Chebyshev deliberately: Chebyshev carries an
adaptive fluctuation-dissipation loop whose iteration count can differ between
implementations, which would confound an exact comparison.

## Two traps found while writing this

- **`phys_params%hstar` is not the one the integrator reads.** `Time_Integrate_Chain`
  takes `phys_params%HI_params%hstar` (`gsipc.f90:1747`); the outer `hstar` field
  exists but is never read. `test_WLC_PU_flow_LJ_EV_bending_potential` sets only
  the outer one, so despite its name that test runs **free draining**. It still
  passes, because its recorded oracle was generated the same way.
- **The good/poor-solvent EV branch is derived from the flow type.** The Fortran
  sets `inteq = 1` for `EQ` and `0` otherwise, selecting `Rcutg` or `Rcutp`. In
  the C++ that is an independent `PhysParams::equilibration` flag. They agree
  under shear with `equilibration = False`, but an `EQ`-flow run with LJ or SDK
  excluded volume needs `equilibration = True` to match.

## The `singular` case

`dpotrf` returns `INFO > 0` when the matrix is not positive definite, and stops
there: the factor is only partially computed and the rest of the array still
holds the input. The following `dgemv` then produces a finite but **wrong**
Brownian displacement. Upstream, nothing is printed and the run continues.

The case feeds both codes a configuration that actually triggers this. It is not
synthetic: it is the state the C++ integrator reached after 86860 steps of a
coarse-grained 10 kbp DNA run, once the implicit corrector stopped converging and
threw the chain apart. The chain has collapsed onto two points -- 11 beads at one,
10 at the other, 2.615 apart -- with 42 pairs closer than 1e-6, so the RPY tensor
has near-duplicate rows.

```bash
python validation/compare_fortran_oracle.py --case singular
```

The C++ raises; the Fortran (with the local modification) warns and carries on.
LAPACK's `INFO` is the 1-based order of the failing leading minor and the C++
reports a 0-based pivot index, so agreement means `INFO == pivot + 1`. The C++
side reports **pivot 55**, so the expected Fortran value is **INFO = 56**.

This is deliberately not registered with ctest: whether a given LAPACK build
declares this particular matrix indefinite is a rounding question, and a stricter
or looser BLAS could legitimately go either way. It is a diagnostic, not a
regression.

## Why the tolerance is the point

The regression oracle recorded in `tests.f90` and in
`tests/test_integrator_rouse.cpp` was generated with
`implicit_loop_exit_tolerance = 1.d-6`. The corrector is a fixed-point iteration
stopped on its *increment*, and at that tolerance it is still ~3e-8 away from the
true solution — measurably, in both codes independently. So the two can never
agree better than ~1e-8 at that setting, and the C++ test's `1e-6` comparison
tolerance is not slack, it is the accuracy the algorithm delivers.

`validation/compare_fortran_oracle.py` runs both sides at a matched tolerance and
prints the difference as the tolerance is tightened:

```bash
python validation/compare_fortran_oracle.py --case full
python validation/compare_fortran_oracle.py --case full --tol 1e-12 --blocks 20
python validation/compare_fortran_oracle.py --case rouse --tol 1e-12 --emit-cpp
```

With `--blocks > 2` it prints a growth table instead: the difference per block,
and the factor it grew by. Expect steady multiplicative growth -- the trajectory
is chaotic, so the corrector's truncation is amplified at the Lyapunov rate.
What matters is that it starts at round-off and grows *smoothly*; a jump would
mean the two codes took different branches somewhere.

## As a test

`-DBDSIM_FORTRAN=ON -DBDSIM_PYTHON=ON` registers `fortran_cross_check` with
ctest, so `./setup.sh --fortran` runs it. It is the `full` case at a corrector
tolerance of 1e-12, two blocks, failing if any block differs by more than 1e-11
-- three orders of headroom over the 4.7e-14 measured, so a different compiler or
BLAS will not trip it but a genuine divergence will.

If the difference falls with the tolerance, the codes agree and the old 1e-6
floor was the corrector's own truncation. If it plateaus, the residual is a real
difference worth chasing. `--emit-cpp` prints the tight result as C++
initialisers, ready to paste into `test_integrator_rouse.cpp` so the regression
can be tightened from 1e-6 to ~1e-11.

## Two things worth knowing about the original

Both were found while chasing NaNs in the C++ HI runs, and both are in the
Fortran too:

- **The corrector's non-convergence is reported but not acted on.** `gsipc.f90`
  exits the loop on `lt_count > 100*Nbeads` and prints `Loop exceeded ...`. If
  that message appears in old output, the trajectory had already failed. The C++
  now prints an equivalent warning.
- **`dpotrf`'s `INFO` is never checked** (`gsipc.f90:1197`). On a degenerate
  configuration LAPACK returns early with the matrix only partially factored, and
  the following `dgemv` produces a finite but wrong Brownian displacement — a
  silent corruption rather than a NaN. The C++ throws instead.

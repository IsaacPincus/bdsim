# Validation & benchmarking

## Correctness

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure       # 12/12 pass
```

The suite covers, against the Fortran/MATLAB oracles: the RNG (`ran_1`), the
seven spring laws + semi-implicit solver, geometry helpers, spring/bending/EV
forces, the RPY tensor and both square-root methods, the connector-direction
identity, and the end-to-end `test_rouse_chain_eq` regression (bit-exact to 1e-6).

## Physics validation

```bash
PYTHONPATH=python python validation/validate_physics.py 400      # or 4000 for tight
```

Equilibrium chain statistics vs Rouse theory (`<R^2> = 3(N-1)`,
`<Rg^2> = (N^2-1)/(2N)` in the code's reduced units), plus a check that HI does
not change the equilibrium distribution (only the dynamics). Example (400 traj):

| N  | <R^2> (sim)     | theory | <Rg^2> (sim)    | theory |
|----|-----------------|--------|-----------------|--------|
| 5  | 11.58 +/- 0.49  | 12.00  | 2.32 +/- 0.07   | 2.40   |
| 10 | 24.91 +/- 0.95  | 27.00  | 4.68 +/- 0.12   | 4.95   |
| 20 | 55.60 +/- 2.27  | 57.00  | 9.70 +/- 0.25   | 9.97   |

Free draining / HI-Cholesky / HI-Chebyshev agree on `<R^2>` to within statistics.
(The ~5% undershoot is the expected finite-`dt` bias of the integrator; it
shrinks with smaller `dt`.)

## Timing

```bash
./build/bench            # quick, 0.5 s/case
./build/bench 20         # a few minutes total, better statistics
```

Single thread, `-O3 -march=native`, on this machine
(Intel(R) Core(TM) Ultra 7 165H). Reported as microseconds per timestep and nanoseconds per bead-step:

| case          | N   | us/step | ns/bead-step |
|---------------|-----|---------|--------------|
| free-draining | 10  | 1.8     | 177          |
| free-draining | 50  | 5.2     | 104          |
| free-draining | 100 | 8.9     | 89           |
| free-draining | 200 | 16.5    | 83           |
| HI Cholesky   | 10  | 7.1     | 712          |
| HI Cholesky   | 50  | 271     | 5427         |
| HI Cholesky   | 100 | 2307    | 23070        |
| HI Chebyshev  | 10  | 8.0     | 804          |
| HI Chebyshev  | 50  | 163     | 3259         |
| HI Chebyshev  | 100 | 952     | 9519         |

### Reading the numbers

- **Free draining is cheap and scales ~linearly** in N (ns/bead-step is roughly
  flat): ~0.06-0.6 million steps/s for N = 200-10.
- **HI is the cost centre**, dominated by the hand-rolled dense linear algebra:
  Cholesky is O((3N)^3) per step, Chebyshev O((3N)^2 * n_terms).
- **Chebyshev scales better than Cholesky** for larger chains (~2.4x faster at
  N = 100, gap widening with N), as expected.

### Where the speed is

1. **HI linear algebra** — the biggest lever. `linalg.*` is deliberately behind
   plain functions (matvec, Cholesky) so it can be dispatched to BLAS/LAPACK
   (OpenBLAS or MKL) or Eigen with no change to callers. Expect a large HI
   speed-up (multithreaded BLAS + tuned kernels).
2. **Per-step allocations** — the `Vec3Field`-by-value returns allocate; a small
   reusable workspace would help the free-draining hot path.
3. **Ensemble parallelism** — trajectories are independent, so wall-clock
   throughput scales with cores via Python multiprocessing (no MPI needed).

## BLAS/LAPACK backend for HI

The HI dense kernels (matvec, Cholesky) can be routed through BLAS/LAPACK instead
of the hand-rolled loops:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBDSIM_LAPACK=ON
```

The two backends give **identical results** (the lower Cholesky factor is unique,
and `cholesky_lower`/`lower_matvec` are a matched pair). Whether it is *faster*
depends entirely on the BLAS:

- With **reference (Netlib) BLAS** — single-threaded, untuned — there is no
  speedup; the `-O3 -march=native` hand-rolled loops are already competitive, and
  BLAS call overhead makes it marginally slower at small N. (This is what the CI
  sandbox has, so the numbers above use the hand-rolled backend.)
- With a **tuned BLAS (OpenBLAS or MKL)** — multithreaded, vectorised kernels —
  expect a large HI speedup at moderate-to-large N, since Cholesky is O((3N)^3).
  Link it via the usual `libblas`/`liblapack` alternatives, or point CMake at MKL.

So the backend is plumbed and correct; the performance lever is which BLAS you
link, not the code path.

### Stiff FENE-Fraenkel spring (sqrtb = 3, natural length = 10)

A stiff, non-zero-rest-length spring costs more per step than FENE, because the
FENE-Fraenkel implicit solve (analytic cubic root + polish) is heavier and a
stiff spring needs more corrector iterations. Free-draining, us/step:

| case         | N=10 | N=50 | N=100 |
|--------------|------|------|-------|
| FENE free    | 1.5  | 4.3  | 7.5   |
| stiff-FF free| 2.1  | 7.2  | 12.9  |

(~1.4x FENE.) Under HI the spring cost is dwarfed by the linear algebra.

See [PROFILING.md](PROFILING.md) for how to attribute time to source lines.

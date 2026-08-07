# Profiling

Build with symbols so profilers can attribute time to source lines:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBDSIM_PYTHON=ON \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
cmake --build build -j
```

`RelWithDebInfo` keeps `-O2`/`-O3` optimisation but adds `-g` line tables.

## C++ core (the hot loops)

**perf** (Linux, low overhead) on the benchmark binary:

```bash
perf record -g ./build/bench 10
perf report                      # interactive; 'a' to annotate a function to source
```

Look for time in `matvec` / `cholesky_lower` (HI), `solve_connectors`, and the
`Vec3Field` allocations. For a specific case, edit `bench/bench.cpp` to run just
that one.

**callgrind** (instruction-level call graph, ~50x slower but exact):

```bash
valgrind --tool=callgrind ./build/bench 0.2
kcachegrind callgrind.out.*      # visual call graph + per-line costs
```

**allocations** (the by-value `Vec3Field` returns):

```bash
valgrind --tool=massif ./build/bench 0.2      # heap over time
# or: heaptrack ./build/bench 0.2
```

## Python + C++ together (end-to-end)

**py-spy** is ideal for this mixed stack — sampling, no code changes, and with
`--native` it shows the C++ frames inside `_bdsim` too:

```bash
pip install py-spy
py-spy record --native -o profile.svg -- python examples/demo.py    # flame graph
py-spy top   --native --            python examples/demo.py          # live top
```

**cProfile** for the pure-Python driver overhead (ensemble loops, numpy):

```bash
python -m cProfile -s cumtime examples/demo.py
```

## What to expect

- Free-draining runs: time is in the force kernels, the connector solve, and the
  per-step `Vec3Field` allocations.
- HI runs: dominated by the dense linear algebra (`cholesky_lower`, `matvec`).
  If that is the bottleneck, the fix is the BLAS/LAPACK backend
  (`-DBDSIM_LAPACK=ON` with OpenBLAS/MKL) rather than micro-optimisation -- see
  BENCHMARKS.md.
- Ensembles: mostly independent trajectories, so wall-clock is best improved with
  process-level parallelism, not per-step tuning.

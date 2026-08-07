# bdsim Python package

Driver layer over the compiled C++ core (`_bdsim`, built via nanobind).

## Build the extension

From the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBDSIM_PYTHON=ON \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
cmake --build build -j
cp build/_bdsim*.so python/bdsim/        # place the extension in the package
```

(`pip install nanobind` first if needed.)

## Use

```bash
PYTHONPATH=python python examples/demo.py
```

or in code:

```python
import bdsim
phys = bdsim.PhysParams(); phys.spring.type = bdsim.Spring.FENE
phys.number_of_beads = 10; phys.hstar = 0.1
phys.flow = bdsim.flows.shear(2.0)
sim = bdsim.SimParams(); sim.time_end, sim.dt = 1.0, 0.01
rng = bdsim.Rng(1)
R = bdsim.integrate(bdsim.gaussian_chain(10, seed=1), phys, sim, rng)
```

## Modules

- `flows` — kappa tensors: `equilibrium`, `shear`, `uniaxial_extension`, `planar_extension`
- `initial` — `gaussian_chain(n, ...)` initial configurations
- `properties` — `end_to_end_sq`, `radius_of_gyration_sq`, `gyration_tensor`, `stretch`
- `ensemble` — `trajectory_samples(...)` (time series) and `run_ensemble(...)` (averaged)

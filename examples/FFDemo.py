"""Minimal bdsim demo: an equilibrium ensemble and a shear trajectory.

Build the extension first (from the project root):

    cmake -S . -B build -DBDSIM_PYTHON=ON \
          -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
    cmake --build build -j
    cp build/_bdsim*.so python/bdsim/

then run:  PYTHONPATH=python python examples/demo.py
"""
import numpy as np
from ..python import bdsim
# import ..python.bdsim

def equilibrium_ensemble():
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.FENEFraenkel
    phys.spring.sqrtb = 1000.0
    phys.spring.natural_length = 0.0
    phys.number_of_beads = 10
    phys.flow = bdsim.flows.equilibrium()

    sim = bdsim.SimParams()
    sim.time_end, sim.dt = 5.0, 0.05
    sim.implicit_loop_tol = 1e-4

    res = bdsim.run_ensemble(phys, sim, n_traj=200, seed=1)
    print("Equilibrium Hookean chain, N=10:")
    for name, (mean, err) in res.items():
        print(f"  <{name}> = {mean:8.4f} +/- {err:.4f}")


def shear_with_hi():
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.FENE
    phys.spring.sqrtb = 50.0
    phys.number_of_beads = 10
    phys.hstar = 0.15                        # hydrodynamic interaction on
    phys.hi_method = bdsim.DelSMethod.Chebyshev
    phys.flow = bdsim.flows.shear(2.0)

    sim = bdsim.SimParams()
    sim.time_end, sim.dt = 2.0, 0.01

    rng = bdsim.Rng(42)
    R0 = bdsim.gaussian_chain(10, seed=42)
    print("\nFENE chain under shear (rate 2) with HI:")
    for t, R in bdsim.trajectory_samples(R0, phys, sim, rng, np.linspace(0.5, 2.0, 4)):
        print(f"  t={t:.2f}  R^2={bdsim.end_to_end_sq(R):7.3f}  "
              f"Rg^2={bdsim.radius_of_gyration_sq(R):7.3f}")


if __name__ == "__main__":
    equilibrium_ensemble()
    shear_with_hi()

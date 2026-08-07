"""This runs a simulation of filament thinning. 
We solve an ODE where the stress difference comes from BD simulations.
We need sufficient trajectories to get a good ensemble for the stress.
We will start with setting up a Newtonian thinning case, then add in polymer stress.
"""
import numpy as np
import scipy.integrate
import sys, pathlib
# Run standalone from anywhere: put the bdsim package (and its compiled
# _bdsim extension) on the path without needing PYTHONPATH set.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim

# the equation we want to solve is:
# 3 eta_s da/dt * 1/a = chi/a - polymer_stress_difference

# we will need to be very careful about units here. I will need a nice scheme
# to non-dimensionalise the equation above.
# I think I probably do this in terms of the relaxation times of the polymer chains
# that I will use. 


def equilibrium_ensemble():
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.Hook
    phys.spring.sqrtb = 1000.0
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

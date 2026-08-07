"""bdsim — Brownian dynamics bead-spring simulations.

The compiled core (``_bdsim``, built from the C++ via nanobind) provides the
parameter types and the ``integrate`` function. This package adds the driver
layer: initial configurations, flow helpers, per-configuration properties,
ensemble running, and HDF5 trajectory storage.

The compiled extension ``_bdsim`` must sit next to this file (CMake with
``-DBDSIM_PYTHON=ON`` builds it; copy the resulting ``_bdsim*.so`` here, or add
its build directory to ``sys.path``).
"""
from . import _bdsim as core
from ._bdsim import (
    Spring, EV, Bending, DelSMethod,
    SpringParams, EVParams, BendingParams, PhysParams, SimParams, Flow, Rng,
    ExternalForce,
    integrate, spring_force, bending_force, ev_force, total_force, external_force,
)

from . import (flows, initial, properties, ensemble, parallel, storage,
               coarse_grain, dynamics, statistics)
from .initial import (gaussian_chain, fene_fraenkel_chain, fene_fraenkel_chain_aligned_x,
                      equilibrium_chain, fene_fraenkel_bending_chain,
                      one_minus_cos_theta_angles, isotropic_directions)
from .properties import (end_to_end_sq, radius_of_gyration_sq, gyration_tensor, stretch,
                         kramers_stress, shear_viscosity as shear_viscosity_config,
                         normal_stress_differences)

# Running: config, the HDF5-writing driver, and the reducing drivers.
from .statistics import steady_state_stats, trajectory_ensemble_stats, green_kubo
from .ensemble import (Initial, Output, INITIALIZERS, n_steps_of,
                       shear_viscosity_series, stress_series,
                       simulate, trajectory_samples,
                       run_ensemble, shear_viscosity as shear_viscosity_ensemble)
# Storage: read runs/trajectories back and post-process.
from .storage import (read_run, read_trajectory, Run, Trajectory,
                      map_property, ensemble_average)
from .parallel import parallel_map
from .coarse_grain import spring_parameters

__all__ = [
    "core", "Spring", "EV", "Bending", "DelSMethod",
    "SpringParams", "EVParams", "BendingParams", "PhysParams", "SimParams",
    "Flow", "Rng", "ExternalForce", "integrate", "external_force",
    "spring_force", "bending_force", "ev_force", "total_force",
    "flows", "initial", "properties", "ensemble", "parallel", "storage",
    "coarse_grain", "dynamics", "statistics", "spring_parameters",
    "steady_state_stats", "trajectory_ensemble_stats", "shear_viscosity_series",
    "stress_series", "green_kubo",
    "gaussian_chain", "fene_fraenkel_chain", "fene_fraenkel_chain_aligned_x",
    "equilibrium_chain", "fene_fraenkel_bending_chain",
    "end_to_end_sq", "radius_of_gyration_sq", "gyration_tensor", "stretch",
    "kramers_stress", "normal_stress_differences",
    "Initial", "Output", "INITIALIZERS", "n_steps_of",
    "simulate", "trajectory_samples", "run_ensemble",
    "read_run", "read_trajectory", "Run", "Trajectory",
    "map_property", "ensemble_average", "parallel_map",
]

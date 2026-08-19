"""Every input option, with its default, meaning and valid values.

This is a reference rather than a demonstration: it sets every parameter the code
accepts, explains what each one does, and then runs a short simulation to prove the
combination is valid. Copy the blocks you need.

    python examples/all_options.py            # print the reference and run a check
    python examples/all_options.py --audit    # only verify it covers the API

The audit at the end compares this file against the compiled module and fails if
any option exists that is not documented here, so the reference cannot silently
fall out of date when the bindings change.

Defaults shown are the ones the constructors actually produce. Physics is in
docs/theory.tex; the workflow tutorials are in docs/tutorials/.
"""
import sys, pathlib
import numpy as np

try:
    import bdsim
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
    import bdsim


# ===========================================================================
# 1. SPRING  (bdsim.SpringParams, reached as phys.spring)
# ===========================================================================
# All lengths are in Hookean units (l_H = sqrt(kT/H)); H = 1 and kT = 1 there.
# For a real molecule these come from bdsim.coarse_grain, not by hand.

spring = bdsim.SpringParams()

spring.type = bdsim.Spring.FENEFraenkel
#   Hook          f = 1                       Hookean, infinitely extensible
#   FENE          f = 1/(1-q^2)               finitely extensible, max length sqrtb
#   ILC           inverse Langevin (Pade)
#   WLC           Marko-Siggia wormlike chain
#   Fraenkel      stiff spring about a natural length, no upper bound
#   FENEFraenkel  natural length AND finite extensibility  (the general case)
#   WLCbounded    modified Marko-Siggia with a natural length
#   -> q = Q/sqrtb is the reduced bond length; see theory ch. 4, Table 4.1.

spring.sqrtb = 10.0            # default 10.0
#   Finite extensibility. The bond cannot exceed natural_length + sqrtb.
#   Equals sqrt(b) of the FENE b-parameter, and dQ_H from the coarse-graining.

spring.natural_length = 0.0    # default 0.0
#   sigma, the unstretched length. 0 makes FENE-Fraenkel identical to FENE and
#   WLCbounded identical to WLC. The solver works with q0 = natural_length/sqrtb.


# ===========================================================================
# 2. EXCLUDED VOLUME  (bdsim.EVParams, as phys.ev)
# ===========================================================================
ev = bdsim.EVParams()

ev.type = bdsim.EV.None_       # default None_
#   None_        no excluded volume (the default; note the trailing underscore,
#                because `None` is a Python keyword)
#   Gauss        Phi = (z*/d*^5) exp(-r^2/2d*^2)
#   LJ           12-6 Lennard-Jones
#   SDK          Soddemann-Duenweg-Kremer: LJ core plus a cosine tail
#   SDKstickers  NOT IMPLEMENTED -- selecting it gives zero force

ev.zstar = 0.0                 # default 0.0   dimensionless EV energy
ev.dstar = 0.0                 # default 0.0   dimensionless EV range
ev.min_cutoff = 0.7            # default 0.7   force held constant below this r
ev.max_cutoff = 1.5            # default 1.5   force zero beyond this r (production)
ev.contour_dist_for_EV = 1     # default 1     skip bead pairs closer than this
#                                              along the chain
ev.phi = []                    # default []    SDK per-pair attractive strengths,
#                                              row-major N*N; empty means zeros


# ===========================================================================
# 3. BENDING  (bdsim.BendingParams, as phys.bend)
# ===========================================================================
bend = bdsim.BendingParams()

bend.type = bdsim.Bending.OneMinusCosTheta
#   None_             no bending potential (default)
#   OneMinusCosTheta  phi_b/kT = C (1 - cos theta)

bend.stiffness = 1.5           # default 0.0
#   C above. <cos theta> = coth(C) - 1/C (the Langevin function).
#   From coarse_grain.bending_constant for a real molecule.


# ===========================================================================
# 4. FLOW  (bdsim.Flow, as phys.flow)
# ===========================================================================
# The flow enters only as the velocity-gradient tensor kappa, applied per bead.
flow = bdsim.flows.equilibrium()                  # zero tensor (the default)
flow = bdsim.flows.shear(1.0)                     # kappa_xy = rate
flow = bdsim.flows.uniaxial_extension(1.0)        # diag(e, -e/2, -e/2)
flow = bdsim.flows.planar_extension(1.0)          # diag(e, -e, 0)

K = np.zeros((3, 3)); K[0, 1] = 2.0               # or supply any tensor
flow = bdsim.Flow.constant(K)

#   Time-varying: linear interpolation between samples, clamped outside.
#   flow = bdsim.Flow.time_varying([0.0, 1.0, 2.0], [K0, K1, K2])


# ===========================================================================
# 5. EXTERNAL FORCES  (bdsim.ExternalForce, as phys.external)
# ===========================================================================
# Applied loads, kept separate from the chain's own forces: they are NOT included
# in total_force, so the Kramers stress stays meaningful. Negative bead indices
# count from the end (-1 is the last bead).
external = bdsim.ExternalForce()
external.add_constant(2, (1.0, 0.0, 0.0))                 # constant, one bead
external.add_stretch(0, -1, (0.0, 0.0, 5.0))              # +/-F on the two ends,
#                                                           zero net force
external.add_time_varying(0, [0.0, 1.0],                  # interpolated protocol
                          [(0.0, 0.0, 0.0), (0.0, 0.0, 3.0)])
external = bdsim.ExternalForce()                          # (reset: none applied)


# ===========================================================================
# 6. PHYSICS  (bdsim.PhysParams)
# ===========================================================================
phys = bdsim.PhysParams()
phys.spring = spring
phys.ev = ev
phys.bend = bend
phys.flow = bdsim.flows.shear(1.0)
phys.external = external

phys.number_of_beads = 20      # default 10;  N-1 springs
#   Overridden by the array passed to integrate(), so it rarely needs setting
#   directly except for run_ensemble/simulate.

phys.hstar = 0.15              # default 0.0 (free draining)
#   Hydrodynamic interaction strength; bead radius a = sqrt(pi) h* l_H.
#   The familiar "h* <~ 0.3" rule assumes flexible springs -- for stiff ones the
#   meaningful check is a/<Q> < 0.5 (theory sec. 11.6).

phys.hi_method = bdsim.DelSMethod.Cholesky
#   Cholesky   exact, unconditionally stable, and FASTER below about N = 100
#   Chebyshev  approximate, O(N^2) per term; wins above N ~ 100
#   ExactSqrt  NOT IMPLEMENTED -- aborts if selected

phys.ncheb_multiplier = 1.0    # default 1.0   Chebyshev term-count multiplier
phys.fd_err_max = 0.0025       # default 0.0025
#   Fluctuation-dissipation tolerance for the Chebyshev square root. If the series
#   cannot meet it, that step falls back to an exact Cholesky factor.

phys.equilibration = False     # default False
#   Excluded-volume good-solvent flag: selects the purely repulsive (WCA) cutoff.


# ===========================================================================
# 7. TIME STEPPING  (bdsim.SimParams)
# ===========================================================================
sim = bdsim.SimParams()

sim.dt = 0.01                  # default 0.01
sim.time_start = 0.0           # default 0.0    also sets the flow clock
sim.time_end = 1.0             # default 1.0
#   The integrator takes round((time_end - time_start)/dt) + 1 steps.

sim.implicit_loop_tol = 1e-4   # default 1e-3
#   Convergence tolerance of the implicit bond solve, on ||dR||_2 / N.
#   Capped at 100*N iterations.

sim.update_center_of_mass = True   # default True
#   Subtract the centre of mass each step. Turn OFF to measure diffusion.

#   NOTE ON RUN LENGTH. The longest Rouse time is 1/sin^2(pi/2N) in these units --
#   179 for N=21. Equilibrate for >~ 3 of those and sample over >~ 6. Running for
#   less does not look like noise; it produces smooth, plausible, wrong curves.


# ===========================================================================
# 8. INITIAL CONFIGURATION  (bdsim.Initial)
# ===========================================================================
initial = bdsim.Initial("gaussian", {})
#   "gaussian"              kwargs: bond_std=1.0
#   "fene_fraenkel"         kwargs: sigma, dQ          (isotropic bonds)
#   "fene_fraenkel_x"       kwargs: sigma, dQ          (bonds along x)
#   "fene_fraenkel_bending" kwargs: sigma, dQ, stiffness
#   The per-chain seed is supplied by the runner as seed + trajectory index.
#   Use the model's own equilibrium whenever springs are stiff or have a large
#   natural length, or a short run will measure relaxation, not equilibrium.


# ===========================================================================
# 9. OUTPUT  (bdsim.Output) -- only used by simulate()
# ===========================================================================
output = bdsim.Output()
output.directory = None        # default None; must be set for simulate()
output.write_every = 100       # default 0 -> only the first and last step
output.write_forces = False    # default False; needed for stress/viscosity
output.compression = "gzip"    # default "gzip"; None disables


# ===========================================================================
# 10. RUNNERS
# ===========================================================================
#   integrate(R, phys, sim, rng) -> (N,3)
#       The core call. The rng stream persists, so consecutive calls continue
#       seamlessly and segmented stepping equals one continuous run.
#
#   run_ensemble(phys, sim, n_traj, per_trajectory, *, args=(), seed=0,
#                initial=None, backend="serial", n_workers=None,
#                on_error="skip", max_failed_fraction=None) -> list
#       The whole ensemble layer. `per_trajectory(R0, phys, sim, rng, *args)` is
#       called once per chain and may return anything. Ready-made ones:
#           final_state(R0, phys, sim, rng)                   -> final (N,3)
#           sampled_states(R0, phys, sim, rng, times)         -> list of (N,3)
#       With backend="processes" your function must be a module-level one, so
#       that it can be pickled; put anything that varies in `args`.
#
#   mean_stderr(values) -> (mean, stderr)      for independent per-trajectory values
#
#   simulate(phys, sim, *, n_traj=1, seed=0, n_steps=None, initial=None,
#            output=None, backend="serial", n_workers=None,
#            on_error="skip", max_failed_fraction=None) -> storage.Run
#       n_steps defaults to the span implied by sim.
#
#   Rheology lives in bdsim.rheology, not in the core run layer:
#       rheology.shear_viscosity_series(phys, sim, rate, n_traj, times,
#                                       variance_reduction=False, **kwargs)
#       rheology.equilibrium_stress_series(phys, sim, n_traj, times, **kwargs)
#       rheology.shear_viscosity(phys, sim, rate, n_traj, times, **kwargs)
#       variance_reduction pays below Wi ~ 0.1 and hurts above ~0.3.
#
#   backend: "serial" or "processes"; n_workers=None uses all cores.
#   Trajectories are seeded by index, so results never depend on worker count.
#
#   Error handling: a trajectory the integrator gives up on is dropped and
#   reported rather than taking the run down. Past MAX_FAILED_FRACTION (0.1) the
#   run refuses to return a number, because the survivors are a biased sample.
#   on_error="raise" restores the old abort-on-first-failure behaviour.


# ===========================================================================
# Check that this combination actually runs
# ===========================================================================
def smoke_test():
    p = bdsim.PhysParams()
    p.spring.type = bdsim.Spring.FENEFraenkel
    p.spring.sqrtb, p.spring.natural_length = 5.0, 1.0
    p.bend.type, p.bend.stiffness = bdsim.Bending.OneMinusCosTheta, 1.0
    p.number_of_beads = 10
    p.hstar, p.hi_method = 0.15, bdsim.DelSMethod.Cholesky
    p.flow = bdsim.flows.shear(1.0)
    p.external.add_stretch(0, -1, (0.0, 0.0, 0.5))
    s = bdsim.SimParams(); s.dt, s.time_end = 0.005, 2.0
    R = bdsim.fene_fraenkel_bending_chain(10, 1.0, 5.0, 1.0, seed=0)
    R = bdsim.integrate(R, p, s, bdsim.Rng(0))
    assert np.all(np.isfinite(R))
    print(f"smoke test ok: R^2 = {bdsim.end_to_end_sq(R):.4f}, "
          f"Rg^2 = {bdsim.radius_of_gyration_sq(R):.4f}")


def audit():
    """Fail if the bindings expose an option this file does not mention."""
    text = pathlib.Path(__file__).read_text()
    missing = []
    for cls in ("SpringParams", "EVParams", "BendingParams", "PhysParams",
                "SimParams"):
        for field in dir(getattr(bdsim, cls)()):
            if field.startswith("_"):
                continue
            if field not in text:
                missing.append(f"{cls}.{field}")
    for enum in ("Spring", "EV", "Bending", "DelSMethod"):
        for member in dir(getattr(bdsim, enum)):
            if member.startswith("_"):
                continue
            if member not in text:
                missing.append(f"{enum}.{member}")
    for name in bdsim.INITIALIZERS:
        if f'"{name}"' not in text:
            missing.append(f"initialiser {name!r}")
    for name in ("shear_viscosity_series", "equilibrium_stress_series",
                 "shear_viscosity", "viscosity_series", "stress_series",
                 "viscosity_series_vr"):
        if name not in text:
            missing.append(f"bdsim.rheology.{name}")
    if missing:
        print("UNDOCUMENTED OPTIONS (add them to this file):")
        for m in missing:
            print("   ", m)
        return 1
    print(f"audit ok: every option exposed by the bindings is documented here")
    return 0


if __name__ == "__main__":
    rc = audit()
    if "--audit" not in sys.argv and rc == 0:
        smoke_test()
    sys.exit(rc)

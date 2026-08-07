"""bdsim demo: run an ensemble, write trajectories to HDF5, read them back.

Build the extension first (from the project root):

    cmake -S . -B build -DBDSIM_PYTHON=ON \
          -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
    cmake --build build -j
    cp build/_bdsim*.so python/bdsim/

Preferred: install the package once (from the project root, venv active) so
`import bdsim` works everywhere -- running and in the editor:

    pip install -e .

Then run it:

    python examples/demo.py

Storage needs h5py:  pip install h5py   (also pulled in by `pip install -e .`)
"""
import numpy as np
import sys, pathlib, tempfile
try:
    import bdsim                      # installed (pip install -e .) -- preferred
except ModuleNotFoundError:           # fallback: run from the source tree uninstalled
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
    import bdsim

# Use every core for the ensemble; "serial" is the simple single-core fallback.
BACKEND = "processes"


def write_and_read_run():
    """Run an ensemble under shear and save every N steps to HDF5, then post-process."""
    # 1. how to start: FENE-Fraenkel equilibrium chains, seeded per trajectory
    initial = bdsim.Initial(method="fene_fraenkel", kwargs=dict(sigma=1.0, dQ=2.0))

    # 2. the physics
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.FENEFraenkel
    phys.spring.sqrtb = 2.0
    phys.spring.natural_length = 1.0
    phys.number_of_beads = 30
    phys.hstar = 0.15
    phys.hi_method = bdsim.DelSMethod.Chebyshev
    phys.flow = bdsim.flows.shear(10.0)

    # 3. the time stepping
    sim = bdsim.SimParams()
    sim.dt = 0.01

    # 4. what to record: positions + forces every 100 steps, into a run folder
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="bdsim_run_"))
    output = bdsim.Output(directory=str(out_dir), write_every=500, write_forces=True)

    run = bdsim.simulate(phys, sim, n_traj=80, seed=0, n_steps=3000,
                         initial=initial, output=output,
                         backend=BACKEND)

    print(f"Wrote {len(run)} trajectories to {out_dir}")
    print(f"  each file: steps {list(run[0].step)}, positions {run[0].positions.shape}, "
          f"forces {'yes' if run[0].forces is not None else 'no'}")

    # Post-process straight from disk: ensemble-mean stretch and viscosity vs time.
    reopened = bdsim.read_run(str(out_dir))
    times, Rg2, err = bdsim.ensemble_average(reopened, bdsim.radius_of_gyration_sq)
    print("\n  t        <Rg^2>     (stderr)")
    for t, m, e in zip(times, Rg2, err):
        print(f"  {t:5.1f}   {m:8.3f}   +/- {e:.3f}")

    # Forces are stored too, so stress/viscosity is a post-processing step:
    eta = lambda R, F: -bdsim.kramers_stress(R, F)[0, 1] / 1.0
    t2, eta_mean, eta_err = bdsim.ensemble_average(reopened, eta, use_forces=True)
    print(f"\n  polymer shear viscosity at final time: {eta_mean[-1]:.3f} +/- {eta_err[-1]:.3f}")


def equilibrium_moments():
    """Reduced-property ensemble (no per-step output): equilibrium chain moments."""
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.Hook
    phys.spring.sqrtb = 1000.0
    phys.number_of_beads = 50
    phys.flow = bdsim.flows.equilibrium()

    sim = bdsim.SimParams()
    sim.time_end, sim.dt = 5.0, 0.05
    sim.implicit_loop_tol = 1e-4

    res = bdsim.run_ensemble(phys, sim, n_traj=2000, seed=1, backend=BACKEND)
    print(f"\nEquilibrium Hookean chain, N={phys.number_of_beads}:")
    for name, (mean, err) in res.items():
        print(f"  <{name}> = {mean:8.4f} +/- {err:.4f}")


if __name__ == "__main__":
    write_and_read_run()
    equilibrium_moments()

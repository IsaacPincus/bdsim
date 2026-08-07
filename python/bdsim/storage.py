"""HDF5 trajectory storage: writing snapshots during a run and reading them back.

Layout of a *run* -- a directory holding one HDF5 file per trajectory plus a
JSON manifest describing the whole ensemble:

    run_dir/
        manifest.json          # ensemble-level parameters + file list
        traj_00000.h5
        traj_00001.h5
        ...

Each traj_NNNNN.h5 stores the snapshots taken during that trajectory:

    attrs:  index, seed, n_beads, dt, n_steps, write_every, has_forces, ...
    step        (n_snap,)          integrator step index of each snapshot
    time        (n_snap,)          simulation time of each snapshot
    positions   (n_snap, N, 3)     bead positions
    forces      (n_snap, N, 3)     total bead forces   (only if requested)

The writer is used per-trajectory inside a worker (so parallel workers never
touch the same file); the reader (`read_run` / `read_trajectory`) is lazy and
provides small helpers for post-processing across the ensemble.
"""
import json
import os
import numpy as np


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_trajectory(path, steps, times, positions, forces=None, attrs=None,
                     compression="gzip"):
    """Write one trajectory's snapshots to `path` (an .h5 file).

    steps/times are length-n_snap; positions is (n_snap, N, 3); forces is the
    same shape or None. `attrs` is an optional dict of scalar metadata.
    """
    import h5py
    steps = np.asarray(steps)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        f.attrs["has_forces"] = forces is not None
        f.create_dataset("step", data=steps)
        f.create_dataset("time", data=times)
        f.create_dataset("positions", data=positions, compression=compression)
        if forces is not None:
            f.create_dataset("forces", data=np.asarray(forces, dtype=np.float64),
                             compression=compression)


def write_manifest(directory, manifest):
    """Write the ensemble-level manifest.json into `directory`."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)


def trajectory_path(directory, index):
    """Conventional per-trajectory filename inside a run directory."""
    return os.path.join(directory, f"traj_{index:05d}.h5")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

class Trajectory:
    """Lazy reader for one traj_NNNNN.h5 file.

    Access `.step`, `.time`, `.positions` (n_snap, N, 3) and `.forces` (or None);
    `.attrs` is the metadata dict. Iterating yields (step, time, R, F_or_None).
    """
    def __init__(self, path):
        self.path = path

    def _open(self):
        import h5py
        return h5py.File(self.path, "r")

    @property
    def attrs(self):
        with self._open() as f:
            return dict(f.attrs)

    @property
    def step(self):
        with self._open() as f:
            return f["step"][()]

    @property
    def time(self):
        with self._open() as f:
            return f["time"][()]

    @property
    def positions(self):
        with self._open() as f:
            return f["positions"][()]

    @property
    def forces(self):
        with self._open() as f:
            return f["forces"][()] if "forces" in f else None

    def __len__(self):
        with self._open() as f:
            return len(f["step"])

    def __iter__(self):
        with self._open() as f:
            has_f = "forces" in f
            step, time = f["step"][()], f["time"][()]
            pos = f["positions"][()]
            frc = f["forces"][()] if has_f else None
        for i in range(len(step)):
            yield int(step[i]), float(time[i]), pos[i], (frc[i] if frc is not None else None)


class Run:
    """A whole run: the manifest plus its trajectories.

    Iterate to get `Trajectory` objects; `len(run)` is the trajectory count.
    `run.manifest` is the parsed manifest.json.
    """
    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "manifest.json")) as fh:
            self.manifest = json.load(fh)
        self.files = [os.path.join(directory, f) for f in self.manifest["files"]]

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for p in self.files:
            yield Trajectory(p)

    def __getitem__(self, i):
        return Trajectory(self.files[i])


def read_trajectory(path):
    """Open a single trajectory file."""
    return Trajectory(path)


def read_run(directory):
    """Open a run directory (reads its manifest)."""
    return Run(directory)


# --------------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------------

def map_property(run, fn, *, use_forces=False):
    """Apply `fn` to every snapshot of every trajectory in `run`.

    `fn` is fn(R) if use_forces is False, else fn(R, F). Returns (times, values)
    where `times` is the common snapshot-time array (from the first trajectory)
    and `values` is (n_traj, n_snap). Assumes all trajectories share snapshot
    times (true for a single `simulate` run).
    """
    times = None
    rows = []
    for traj in run:
        vals, ts = [], []
        for _step, t, R, F in traj:
            vals.append(fn(R, F) if use_forces else fn(R))
            ts.append(t)
        rows.append(vals)
        if times is None:
            times = np.asarray(ts)
    return times, np.asarray(rows, dtype=float)


def ensemble_average(run, fn, *, use_forces=False):
    """Ensemble mean and standard error of `fn` at each snapshot time.

    Returns (times, mean, stderr), each length n_snap.
    """
    times, values = map_property(run, fn, use_forces=use_forces)
    n = values.shape[0]
    mean = values.mean(axis=0)
    stderr = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(mean.shape, np.nan)
    return times, mean, stderr

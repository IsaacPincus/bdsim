# Storing and reading trajectories

## Writing

`simulate` runs an ensemble and writes one HDF5 file per trajectory:

```python
import bdsim

run = bdsim.simulate(
    phys, sim,
    n_traj=16, seed=0, n_steps=10000,
    initial=bdsim.Initial("fene_fraenkel", dict(sigma=1.0, dQ=4.0)),
    output=bdsim.Output(directory="myrun", write_every=100, write_forces=True),
    backend="processes",
)
```

`simulate` returns a reader for what it just wrote.

### Layout

```
myrun/
    manifest.json        ensemble parameters and the file list
    traj_00000.h5
    traj_00001.h5
    ...
```

Each file holds the snapshots of one trajectory:

| Dataset | Shape | Contents |
|---|---|---|
| `step` | `(n_snap,)` | integrator step index |
| `time` | `(n_snap,)` | simulation time |
| `positions` | `(n_snap, N, 3)` | bead positions |
| `forces` | `(n_snap, N, 3)` | total bead forces (only if requested) |

with the run parameters in the file attributes. One file per trajectory means
parallel workers never contend for the same file.

### Snapshot cadence

Snapshots are taken at steps $0$, `write_every`, $2\times$`write_every`, ...,
`n_steps`. **Chunked stepping is exact**: the chunked run is bit-identical to one
continuous run --- same RNG stream, same step count --- so `write_every` never
perturbs the trajectory. `write_every=0` records only the start and the end.

:::{admonition} Storage adds up
:class: warning
Positions are $8 \times N \times 3$ bytes per snapshot per trajectory. A
$N=100$ chain, 1000 snapshots, 64 trajectories is about 150 MB before compression
(gzip is on by default). Record forces only if you need stress.
:::

## Reading

```python
run = bdsim.read_run("myrun")
print(len(run), run.manifest["n_steps"])

traj = run[0]
traj.step         # (n_snap,)
traj.time         # (n_snap,)
traj.positions    # (n_snap, N, 3)
traj.forces       # (n_snap, N, 3) or None
traj.attrs        # dict of metadata

for step, t, R, F in traj:      # iterate snapshots
    ...
```

Reading is lazy --- the file is opened per access --- so a `Run` can be held over a
directory larger than memory, as long as you do not ask for every trajectory at
once.

## Post-processing from disk

```python
times, mean, stderr = bdsim.ensemble_average(run, bdsim.radius_of_gyration_sq)

eta = lambda R, F: -bdsim.kramers_stress(R, F)[0, 1] / 1.0
times, mean, stderr = bdsim.ensemble_average(run, eta, use_forces=True)
```

`map_property` returns the full `(n_traj, n_snap)` array if you want to do the
statistics yourself --- which you generally should, because
`ensemble_average` gives the plain scatter across trajectories and does **not**
correct for correlation in time. See [Post-processing](postprocessing.md).

## Working with the files directly

Nothing is proprietary; the files are ordinary HDF5.

```python
import h5py
with h5py.File("myrun/traj_00000.h5") as f:
    R = f["positions"][:]
    print(dict(f.attrs))
```

```bash
h5ls -r myrun/traj_00000.h5
```

# Fortran reference (`./sens`) and cross-validation

The original Fortran Brownian-dynamics code, used as a reference oracle for this
C++/Python port. Its source lives in a separate repository:

**https://github.com/IsaacPincus/SingleChainBD**

- `inputc.dat` — the input deck, tracked here because it defines the comparison
  case.
- `sens` — the compiled Fortran executable (dynamically linked; MPI + NetCDF).
  **Not tracked**: build it from SingleChainBD and either drop it in this
  directory or point `SENS_EXE` in `validation/compare_fortran.py` at it.
- `net_dt01_proc*.nc` — its NetCDF output. **Not tracked** (34 MB); regenerated
  by running `sens`, which `compare_fortran.py` does for you.

`inputc.dat` here is: N=51 FENE-Fraenkel with `sigma = 0` (so it reduces to FENE),
`sqrtb = 14.16`, simple shear at rate 1, `h* = 0.2`, **Chebyshev** HI, `dt = 0.001`,
integrated `t = 0 -> 1`, sampled at `t = 0, 0.25, 0.5, 0.75, 1.0`. Initial
configuration is the FENE-Fraenkel equilibrium (random spherical bonds).

## Running `./sens`

On WSL/Ubuntu the runtime deps are `libnetcdff` and OpenMPI:

```bash
sudo apt install libnetcdff7 openmpi-bin        # once
mpirun -n 4 ./sens                              # writes net_dt01_procNNN.nc, one per rank
```

Trajectory count is set in `inputc.dat` by `ntot` and `nblock` (trajectories per
rank = `min(ntot, nblock)`); total = ranks x that. The RNG seed is taken from the
wall clock, so each run is an independent ensemble.

The NetCDF output holds, at each sample, the bead `configuration`, the total bead
force (`Gradient`), and the cumulative `cofm`.

## Cross-validation against the new code

One script runs **both** codes and prints the comparison:

```bash
python validation/compare_fortran.py
```

All settings are a CONFIG block at the top of that script (paths, ensemble size,
physics, time stepping) -- edit it rather than passing arguments. Point `SENS_EXE`
at your own compiled `sens`. For a longer test, raise `T_END` (and `N_SAMPLES`),
and/or `N_RANKS` x `TRAJ_PER_RANK`. Set `RUN_FORTRAN`/`RUN_CPP` to `False` to reuse
an existing run and just re-analyse.

It performs three checks:

1. **Force kernel — exact.** Every Fortran configuration is fed into the C++
   `total_force` and compared with the Fortran's own `Gradient`. RNG-independent,
   so this is a true bit-level check. Result: **bit-exact**, worst relative
   difference ~2e-16 (machine epsilon).

2. **Initial configuration — distributional.** The t=0 bond lengths from both
   codes are compared with a two-sample KS test. This isolates the initial-config
   generator from the dynamics. Result: **same distribution** (e.g. KS p = 0.19
   over 20100 bonds); `<Q^2>` agrees to <1%.

3. **Dynamics — statistical.** The Fortran seeds its RNG from the wall clock
   (`sensemble.f90` reads the system time), so trajectories cannot be matched
   one-to-one; ensemble means with standard errors are the meaningful comparison.
   Reported as `|difference|/sigma` per observable per sample time.

Observables (computed identically from each code's raw positions + total force):
end-to-end `R^2`, `Rg^2`, x-stretch, polymer viscosity `eta_p = -tau_xy/gamma_dot`,
and first normal-stress difference `N1 = tau_xx - tau_yy`.

### A note on sample times

The Fortran time loop is inclusive and overshoots by one step, so its final sample
is at `T_END + dt` (e.g. 1.001 rather than 1.000) while the C++ stops exactly at
`T_END`. That is ~0.1% of the integrated time, far below the statistical error; the
script prints a NOTE when it detects this so it is not mistaken for a physics
difference.

### Result (402 Fortran vs 512 C++ trajectories)

The transient build-up under shear agrees within statistics (see
`../validation/fortran_comparison.png`):

| observable | agreement (|Δ|/σ across the 5 sample times) |
|---|---|
| eta_p (viscosity)      | < 1.0 sigma at every time (0.03 sigma at t=0.5) |
| N1 (normal stress)     | < 1.0 sigma at every time |
| R^2, Rg^2              | < 0.8 sigma at every time |
| x-stretch             | ~1.3-2.1 sigma (extreme-value statistic, noisiest of the five) |

The rheological observables (viscosity, normal stress) — the physically important
ones — are statistically indistinguishable between the two codes, and the force
kernel is bit-identical.

The x-stretch was the largest apparent deviation, including at t=0 where no
dynamics has happened yet. That was checked directly and is **noise, not a real
difference**: comparing the underlying t=0 bond-length distributions (20100 bonds
per code, far more statistical power than a per-chain extremal scalar) gives
KS p = 0.19, and with matched sample counts the t=0 x-stretch itself gives
z = 0.71, KS p = 0.98. x-stretch is a max-minus-min over 51 beads, so it is
tail-dominated and converges much more slowly than the mean-based observables --
expect it to stay the noisiest entry in the table. Check `compare_fortran.py`'s
step 2 (which runs automatically) for the definitive initial-distribution test.

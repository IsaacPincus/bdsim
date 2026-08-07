# Post-processing and error bars

Getting a number is easy. Getting a number with a defensible error bar is most of
the work, because successive samples of a trajectory are **not independent**.

## Observables

From positions alone:

```python
bdsim.end_to_end_sq(R)          # |R_N - R_1|^2
bdsim.radius_of_gyration_sq(R)  # trace of the gyration tensor
bdsim.gyration_tensor(R)        # 3x3
bdsim.stretch(R)                # extent along each axis
```

From positions and forces --- the Kramers--Kirkwood stress
$\tau_{ij} = \sum_\nu (R-R_c)_{\nu i} F_{\nu j}$:

```python
tau = bdsim.kramers_stress(R, bdsim.total_force(R, phys))
eta_p = -tau[0, 1] / gammadot
N1, N2 = bdsim.normal_stress_differences(R, F)
```

Pass whichever force you want the stress attributed to. `total_force` is the
chain's own force and **excludes** any applied external force, which is what makes
it the right thing here.

## Why the naive error bar is wrong

A chain remembers its configuration for roughly its longest relaxation time.
Treating $N$ samples as $N$ independent measurements understates the error by
$\sqrt{g}$, where $g$ is the statistical inefficiency. For a strongly correlated
series that is an order of magnitude.

```python
from bdsim import statistics as st

g, window = st.statistical_inefficiency(series)   # g = 1 + 2 sum_k rho_k
n_eff = len(series) / g
```

Validated against an AR(1) process, where $g = (1+\phi)/(1-\phi)$ exactly: the
estimator recovers $g$ to a few percent from $g=1$ to $g=99$, and the naive error
is wrong by $\sqrt{g}$ throughout.

## The recommended path

```python
series = bdsim.shear_viscosity_series(phys, sim, rate, n_traj, sample_times,
                                      initial=init, backend="processes")
interval = sample_times[1] - sample_times[0]
stats, se_between = bdsim.trajectory_ensemble_stats(series, interval)
print(stats)
```

`stats` carries the mean, the autocorrelation-corrected error, an independent
blocking estimate, $g$, $\tau$, and $N_\mathrm{eff}$. `se_between` is the scatter
of the per-trajectory means --- assumption-free, but with only
$n_\mathrm{traj}-1$ degrees of freedom.

**Agreement between the two is the check.** If they disagree, the run is too short.

:::{admonition} You need enough samples, not just enough time
:class: warning
The autocorrelation estimate needs many samples *per correlation time*. With too
few, the Sokal window closes on the first step, $g$ comes back as $\approx 1$ ---
"uncorrelated" --- and the error bar is silently too small.

This is guarded: with fewer than ~30 samples per trajectory, or if the two error
estimates disagree by more than $2\times$, the correction is rejected and the
between-trajectory error is used instead. Aim for **~20 samples per correlation
time**: sampling a window of $1600$ time units at $\tau \approx 78$ with only 40
samples is not enough.
:::

## Discarding the transient

```python
t0 = st.equilibration_point(series)     # maximises N_eff of what remains
stats = st.steady_state_stats(series, sample_interval)   # does this for you
```

Keeping a slowly-relaxing transient biases the mean *and* inflates the apparent
correlation time.

## Low shear rates

The viscosity signal scales as $\dot\gamma$ while the stress fluctuations do not,
so a direct measurement fails as $\mathrm{Wi}\to0$ --- it will happily return a
**negative** viscosity.

### Variance reduction

Pair each trajectory with an equilibrium one on the *same random stream* and
subtract:

```python
series = bdsim.shear_viscosity_series(..., variance_reduction=True)
```

Unbiased, because $\langle\tau_{xy}\rangle_\mathrm{eq}=0$ identically. Measured
variance ratios (2 kbp, $N_s=20$; $>2$ is needed just to pay for the doubled cost):

| Wi | 0.01 | 0.03 | 0.3 | 3 |
|---|---|---|---|---|
| variance ratio | 146× | 31× | 1.2× | 0.6× |

**Use it below Wi ≈ 0.1; leave it off above ≈ 0.3**, where shear has decorrelated
the pair and it merely adds noise.

### Green--Kubo

```python
ser = bdsim.stress_series(phys_eq, sim, n_traj, sample_times, initial=init)
res = bdsim.green_kubo(ser, sample_interval)
```

Honest assessment: on the test case this gave 23% statistical error against 31%
for a sheared run with variance reduction --- but its three tail treatments span
more than a factor of two, a systematic ambiguity that swamps the statistical gain.
The stress correlation function is a *sum* over modes ($\tau_p \sim \tau_1/p^2$), so
a single-exponential tail fit locks onto the fast decay and truncates the slow part
where much of the integral lives. **Prefer variance reduction; use Green--Kubo as a
cross-check.**

## Checklist before believing a number

1. Equilibrated for at least a few relaxation times?
2. Sampling window several relaxation times, with ~20 samples per correlation time?
3. Do the autocorrelation and between-trajectory errors agree?
4. Any warnings in `stats.warning`?
5. Does the answer move when you halve the timestep?
6. Does it move when you change $N_s$?

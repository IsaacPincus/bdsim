# Coarse-graining a DNA fragment

Turning a real molecule --- contour length $L$, persistence length $l_p$ --- into a
bead--spring model. This is the inverse problem: everything else in the
documentation assumes the parameters are already known.

The physics is in `docs/theory.tex` chapters 10 and 11; this is the workflow.

## Static: size

```python
from bdsim import coarse_grain as cg

p = cg.spring_parameters(L=25000, lp=147.0, n_springs=30,
                         bending="match_chain")
print(p.summary())
```

`L` and `lp` may be in any consistent unit --- base pairs, nm, m --- since only their
ratio enters. DNA is quoted here in base pairs with $l_p = 147$ bp.

Two conditions fix the spring: it is exactly as extensible as the contour it
replaces ($\sigma + \delta Q = l_s$), and it has the right mean-square extension
($\langle Q^2\rangle = \langle R^2\rangle_\mathrm{WLC}$).

The two numbers the simulation wants are `p.sigma_H` → `natural_length` and
`p.dQ_H` → `sqrtb`.

:::{admonition} Choose the bending constant with care
:class: important
`bending="saadat"` (the literature fit) matches *local* correlation but leaves the
assembled chain's $\langle R^2\rangle$ wrong by $-28\%$ to $+15\%$.
`bending="match_chain"` inverts the chain relation instead and lands within ~1%.

You cannot have both with this two-parameter family: correct local correlation or
correct global size. For anything where chain size matters, use `match_chain`.
:::

### Capping the spring constant

Stiff segments drive $H$ up without bound --- 736 pN/nm at 2 kbp with $N_s=80$ ---
and $H$ sets the time unit through $\lambda_H = \zeta/4H$, so an uncapped $H$
makes reaching the relaxation time arbitrarily expensive.

```python
p = cg.spring_parameters(L, lp, Ns, bending="match_chain", max_H=cap)
```

When the cap binds, $\sigma$ is **re-solved** so $\langle Q^2\rangle$ still matches.
The cost is negligible in the regime where it bites: capping $H$ by four orders of
magnitude moves the mean segment length in the fourth decimal place.

## Dynamic: friction

The static fit says nothing about how fast the chain moves. Fix that by matching a
measured diffusivity.

```python
from bdsim import dynamics as dyn

rh_nm = dyn.dna_hydrodynamic_radius_nm(25000)     # from the DNA calibration
hstar, info = dyn.hstar_for_target_rh(p, rh_nm)
```

The Kirkwood hydrodynamic radius depends on the configurations only through the
*static* structure, which is already fixed --- so $h^*$ follows from a Monte-Carlo
average with **no dynamics run at all**.

The experimental input is the Zimm scale times a measured ratio,
$D = D_\mathrm{Zimm}(L,b)\times[D/D_\mathrm{Zimm}](L)$.

:::{admonition} Replace the calibration table
:class: warning
The tabulated $D/D_\mathrm{Zimm}$ was read off a published figure and is good to
perhaps a few percent. It follows the light-scattering/sedimentation curve;
single-molecule values sit systematically below it. Pass `ratio_table=` with your
own digitisation before doing anything quantitative.
:::

### Free draining

With HI off the matching becomes exact and needs no sampling at all, because
$D = k_BT/(6\pi\eta_s N a)$ has no configuration dependence:

$$a = R_H / N$$

```python
hstar_nominal, info = dyn.free_draining_units(p, rh_nm)
phys = cg.to_phys_params(p, hstar=0.0)        # run with HI OFF
```

**The simulation runs with `hstar=0`**; the nominal $h^*$ exists only so the unit
conversions can compute the friction. This is 30--40× cheaper per step, matches
$D$ exactly, and gives a longest relaxation time in the right range --- but the
internal modes are Rouse rather than Zimm, so use it to place runs and shake down
a workflow, not for quantitative rheology.

## Units

```python
u = dyn.physical_units(p, hstar, hydrodynamic_radius_nm=rh_nm)
u["l_H"], u["force_H"], u["lambda_H"], u["D"]
```

Multiply a length by `l_H`, a force by `force_H`, a time by `lambda_H`; energies
are in $k_BT$.

:::{admonition} Why matching a *length* is what makes this work
:class: note
Both Hookean units move when the discretisation changes: $l_H$ with the static fit,
$\lambda_H$ with $h^*$. A diffusivity quoted in simulation units is therefore not
comparable between discretisations --- each is in a different second.

$R_H$ is a length, so it converts with $l_H$ alone, no time involved. Stokes--Einstein
then gives $D = k_BT/6\pi\eta_s R_H$ matched in m²/s automatically. Across
$N_s = 5,10,20$ the code-unit diffusivity falls 2.6×, $l_H^2$ falls 268× and
$\lambda_H$ falls 689× --- and the *combination* $D_K l_H^2/\lambda_H$ is invariant
to four figures.
:::

## Rodlike units

For a stiff spring, $l_H$ is far smaller than the bond and every length becomes a
large number. Scale by the chain instead:

```python
r = dyn.to_rodlike_units(p, hstar=hstar, scale="sigma")   # or "rms"
```

Three identities serve as checks: $\sigma^*_R = 1$, $H^*_R = \sigma_H^2$, and
$\delta Q^*_R = 1/q_0$.

## End to end

```python
import bdsim
from bdsim import coarse_grain as cg, dynamics as dyn

L, lp, Ns = 48502, 147.0, 30                      # lambda-DNA

p = cg.spring_parameters(L, lp, Ns, bending="match_chain")
rh = dyn.dna_hydrodynamic_radius_nm(L)
hstar, info = dyn.free_draining_units(p, rh)
u = dyn.physical_units(p, hstar, hydrodynamic_radius_nm=rh)
tau1_s, tau1_code = dyn.free_draining_relaxation_time(p, u)

phys = cg.to_phys_params(p, hstar=0.0)
phys.flow = bdsim.flows.shear(1.0 / tau1_code)     # Wi = 1
init = bdsim.Initial("fene_fraenkel_bending",
                     dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))

print(f"tau1 = {tau1_s:.4g} s, D = {u['D']*1e12:.4g} um^2/s, "
      f"l_H = {u['l_H']*1e9:.4g} nm")
```

## Validating

```bash
python validation/validate_coarse_graining.py --plots   # structure
python validation/validate_dynamics.py --L 2000         # h*, units, Ns-independence
python validation/viscosity_sweep.py --L 2000 10000 --Ns 20 30
```

## Known limitations

- $\langle R^2\rangle$ and $R_H$ are matched; $\langle R_g^2\rangle$ is **not**, and
  drifts ~9% over $N_s = 5$–40. Shape-sensitive observables keep a discretisation
  dependence at the ten-percent level.
- The high-force response is wrong by construction: FENE-type springs give
  $1-x \propto f^{-1}$ where the wormlike chain gives $f^{-1/2}$.
- The Hamprecht--Kleinert moment fit fails for $L/l_p \lesssim 1$, where the
  informative moments are not the lowest three.

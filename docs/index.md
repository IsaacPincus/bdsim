# bdsim

Brownian dynamics of a single bead--spring chain, with hydrodynamic interaction,
excluded volume, a bending potential and arbitrary flow. A C++ core does the
integration; a Python package owns configuration, sampling, storage and analysis.

The documentation is in four parts:

- **[Overview](overview.md)** --- what the code is, how the two layers fit together,
  and where to find things.
- **Tutorials** --- task-oriented walkthroughs, in the order you are likely to need
  them.
- **[C++ layer](cpp_layer.md)** --- what each source file is responsible for, and
  the seams between them.
- **[Python API](api/python.rst)** --- generated from the docstrings.

The mathematics is documented separately and in full in `docs/theory.tex`
(compile with `pdflatex theory.tex`): the stochastic differential equation, the
semi-implicit scheme, every spring force law, the hydrodynamic machinery, the
coarse-graining of a wormlike chain, force--extension, and the estimation of error
bars from correlated data. **Read that for the physics; read this for the code.**

```{toctree}
:maxdepth: 2
:caption: Contents

overview
installation
tutorials/running
tutorials/storage
tutorials/postprocessing
tutorials/coarse_graining
cpp_layer
api/python
```

For a complete list of every input option with its default, see
`examples/all_options.py`.

## The shortest possible example

```python
import bdsim

phys = bdsim.PhysParams()
phys.spring.type = bdsim.Spring.FENE
phys.spring.sqrtb = 50.0
phys.number_of_beads = 20
phys.flow = bdsim.flows.shear(1.0)

sim = bdsim.SimParams()
sim.time_end, sim.dt = 10.0, 0.01

Rs = bdsim.run_ensemble(phys, sim, 200, bdsim.final_state, backend="processes")
print(bdsim.mean_stderr([bdsim.end_to_end_sq(R) for R in Rs]))   # (mean, stderr)
```

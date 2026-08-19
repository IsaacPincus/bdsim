Python API
==========

Generated from the docstrings. The modules are listed in the order you are likely
to need them: configure and run, then store, then analyse, then parameterise.

Running
-------

.. automodule:: bdsim.ensemble
   :members:
   :member-order: bysource

.. automodule:: bdsim.flows
   :members:

.. automodule:: bdsim.initial
   :members:

.. automodule:: bdsim.parallel
   :members:

Storage
-------

.. automodule:: bdsim.storage
   :members:

Analysis
--------

.. automodule:: bdsim.properties
   :members:

.. automodule:: bdsim.rheology
   :members:
   :undoc-members:

.. automodule:: bdsim.statistics
   :members:
   :member-order: bysource

Parameterisation
----------------

.. automodule:: bdsim.coarse_grain
   :members:
   :member-order: bysource

.. automodule:: bdsim.dynamics
   :members:
   :member-order: bysource

Compiled core
-------------

The types below come from the nanobind extension ``bdsim._bdsim`` and are
re-exported at package level. They are documented in
:doc:`../cpp_layer` and, for the mathematics, in ``docs/theory.tex``.

``PhysParams``
    Spring, excluded volume, bending, flow, external forces, bead count,
    ``hstar``, ``hi_method``.

``SimParams``
    ``dt``, ``time_start``, ``time_end``, ``implicit_loop_tol``,
    ``update_center_of_mass``.

``ExternalForce``
    ``add_constant``, ``add_time_varying``, ``add_stretch``. Negative bead
    indices count from the end.

``Rng``
    The bit-exact ``ran_1`` stream; persists across ``integrate`` calls.

``integrate(R, phys, sim, rng)``
    Advance a configuration. Returns the new ``(N, 3)`` array.

``spring_force``, ``bending_force``, ``ev_force``, ``total_force``, ``external_force``
    Individual force contributions at a configuration. ``total_force`` is the
    chain's own force and excludes applied external forces.

External Optimizer Adapters API
===============================

Thin adapter functions that drive a mature external optimizer (SciPy, optax)
from a SpaceCore :class:`~spacecore.Functional`. They evaluate the objective
through ``F.value`` and convert the metric (Riesz) gradient ``F.grad`` to the
*coordinate* gradient an external optimizer expects with ``X.riesz`` -- the
identity on a Euclidean space and mandatory on a weighted one. The external
optimizer owns the loop, line search, and convergence; the adapter only
translates the objective and its geometry. See :doc:`/dev/adr/018_external_optimizer_adapters`.

SciPy
-----

.. autosummary::
   :nosignatures:

   spacecore.minimize_scipy
   spacecore.line_search_scipy

* ``minimize_scipy`` drives :func:`scipy.optimize.minimize`, flattening elements
  to and from SciPy's coordinate vector and supplying ``X.riesz(F.grad(x))`` as
  the Jacobian. Returns the SciPy ``OptimizeResult`` with an added ``x_element``
  field (the minimizer unflattened into ``F.domain``).
* ``line_search_scipy`` drives :func:`scipy.optimize.line_search` along a
  search direction with the same gradient handoff.

optax
-----

.. autosummary::
   :nosignatures:

   spacecore.minimize_optax

* ``minimize_optax`` runs the canonical optax update loop with pytree
  pass-through (no flatten/unflatten); the coordinate gradient is the same
  ``X.riesz(F.grad(x))`` handoff. Requires a JAX-backed domain and the optional
  ``optax`` dependency (``pip install spacecore[optax]``).

Information lost at the external boundary
-----------------------------------------

* **Structure.** SciPy sees a flat coordinate vector; ``bounds`` and
  ``constraints`` are expressed in flattened coordinates.
* **Geometry.** The external optimizer works in the flat Euclidean coordinate
  metric; the domain geometry survives only through the ``X.riesz`` gradient
  conversion.
* **Field.** The SciPy adapters require a real domain and reject complex spaces.
* **Tracing.** Jitted external solves may require a context built with
  ``check_level="none"``.

Autodoc
-------

.. autofunction:: spacecore.minimize_scipy
.. autofunction:: spacecore.line_search_scipy
.. autofunction:: spacecore.minimize_optax

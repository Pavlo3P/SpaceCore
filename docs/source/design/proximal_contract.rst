Metric proximal operations
==========================

``f.prox(x, step)`` returns the closed-form prox in the domain's own metric, or
raises ``CapabilityError``; SpaceCore never densifies or iterates to get one.

.. math::

   \operatorname{prox}^{X}_{t f}(x)
   = \arg\min_z f(z) + \frac{\|z-x\|_X^2}{2t}, \qquad t>0.

Built-in rules
--------------

=======================================  ============================================================
Functional                               Prox
=======================================  ============================================================
``SquaredL2NormFunctional(X)``           ``X.scale(1 / (1 + step), x)``; any geometry, any element
``L1NormFunctional(X)``                  soft threshold at ``step / weights``; Euclidean or
(= ``LpNormFunctional(X, 1)``)           ``WeightedInnerProduct`` only; complex entries shrink radially
``c * f`` with real ``c > 0``            ``f.prox(x, c * step)``
``f + const`` (finite real constant)     ``f.prox(x, step)``
``IndicatorFunctional(C)``               ``C.project(x)``
=======================================  ============================================================

Any other sum, product or composition ``f @ A`` has no prox.

Steps must be finite real scalars ``> 0``. Python scalars are always checked; at
``check_level="none"`` a backend scalar is accepted unchecked and its value is the
caller's responsibility. Scaling coefficients follow the same rule.

A registered prox asserts a proper lower-semicontinuous convex real objective; on a
complex space the pairing is the real inner product.

Conjugate prox
--------------

``f.conjugate_prox(p, step)`` is the Moreau decomposition in the same space; ``p`` is
a Riesz representer, not a coordinate covector.

.. math::

   f_X^*(p) &= \sup_z \operatorname{Re}\langle p,z\rangle_X - f(z),\\
   \operatorname{prox}^{X}_{t f_X^*}(p)
     &= p-t\operatorname{prox}^{X}_{f/t}(p/t).

This matches ``LinOp.rapply`` in a PDHG dual update. Example: with diagonal metric
``W`` and ``f(x) = lam * ||x||_1``, the conjugate is the indicator of
``|W p| <= lam`` and its prox clips to those bounds.

Sets and indicators
-------------------

A ``Set`` has an ``ambient`` space, ``contains(x)`` and ``project(x)``.

* ``sc.Set`` has **no cores**: a custom set overrides ``contains`` and ``project``
  directly. ``contains`` may return ``None`` (membership unavailable); ``project``
  defaults to raising ``CapabilityError``.
* Set ``is_convex = True`` to allow ``IndicatorFunctional(C)``; otherwise its
  constructor raises ``CapabilityError``.
* ``IndicatorFunctional(C)`` is ``0`` on ``C`` and ``inf`` outside. Its value raises
  ``CapabilityError`` when ``contains`` returns ``None``; its prox (any step) is
  ``C.project(x)``.
* Declare ``_fields``: a set without it cannot be constructed; it also gives
  conversion and pytree support (:doc:`extending`).
* ``NonnegativeOrthant(X)``: real ``DenseCoordinateSpace`` with Euclidean or
  diagonal geometry only; projection is ``max(x, 0)``.

There is no intersection-projection solver and no manifold structure.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx,
       geometry=sc.WeightedInnerProduct(ctx.asarray([2., 5.])))
   x = ctx.asarray([-1., 2.])

   f = 0.3 * sc.L1NormFunctional(X)
   f.prox(x, 0.5)                                                  # [-0.925, 1.97]
   sc.IndicatorFunctional(sc.NonnegativeOrthant(X)).prox(x, 0.5)   # [0., 2.]


   class UnitBall(sc.Set):
       """Closed unit ball of the ambient norm; radial projection is metric in any geometry."""

       _fields = ("ambient", "check_level")
       is_convex = True

       def contains(self, x):
           return self.ambient.norm(x) <= 1

       def project(self, x):
           return self.ambient.scale(1 / max(float(self.ambient.norm(x)), 1.0), x)


   sc.IndicatorFunctional(UnitBall(X)).prox(x, 0.5)

Adding a prox rule
------------------

Register ``rule(functional, geometry, x, step)`` with ``spacecore.mapping.register``;
the rule returns a native domain element.

.. code-block:: python

   import numpy as np
   import spacecore as sc
   from spacecore.mapping import register

   class HalfSquaredNorm(sc.Functional):
       _fields = ("dom", "ctx")

       def __init__(self, dom, ctx=None, check_level=None):
           super().__init__(dom, ctx, check_level=check_level)

       def _value_core(self, x):
           return 0.5 * self.domain.inner(x, x)

       def _grad_core(self, x):
           return x

   # The prox of ||x||_X^2 / 2 is x / (1 + step) in every geometry, so key on object.
   register("prox", HalfSquaredNorm, object,
            lambda f, geometry, x, step: f.domain.scale(1 / (1 + step), x))

   ctx = sc.Context(sc.NumpyOps(), np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   HalfSquaredNorm(X).prox(ctx.asarray([2., 4.]), 1.0)     # [1., 2.]

* Lookup walks the functional MRO, then the geometry MRO; the first match wins.
* Key on ``object`` only for a rule valid in every geometry the functional supports,
  including native spaces with no geometry object.
* A subclass that changes the objective inherits its parent's rule: re-register it.
* Register at setup, before tracing; registering later does not change already
  compiled functions.

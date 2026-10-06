Metric proximal operations
==========================

``Functional.prox(x, step)`` is an optional closed-form operation:

.. math::

   \operatorname{prox}^{X}_{t f}(x)
   = \arg\min_z f(z) + \frac{\|z-x\|_X^2}{2t}, \qquad t>0.

The domain owns the metric and the returned element's representation. A missing
rule raises ``CapabilityError``. SpaceCore does not construct a dense problem or
run an optimization algorithm to obtain a prox.

Built-in rules
--------------

* ``SquaredL2NormFunctional(X)``: ``X.scale(1 / (1 + step), x)`` for the
  squared **space** norm, including custom elements and full metrics.
* ``L1NormFunctional(X)`` (equivalently ``LpNormFunctional(X, 1)``): coordinate
  threshold ``step / weights`` for Euclidean or diagonal weighted geometry.
  Complex coordinates use radial shrinkage with real positive metric weights.
* Positive real scaling multiplies the proximal step. Finite real constant
  shifts preserve the prox. General sums, products, and linear pullbacks do
  not inherit a proximal rule.
* ``IndicatorFunctional(C)``: projection onto a declared closed convex set.

Steps must be finite real scalars strictly greater than zero. Eager checks
validate their values. At ``check_level="none"``, dynamic backend scalars are
accepted without host conversion; the caller must guarantee their values are
valid. Concrete invalid Python scalars still raise. This also applies to
positive scaling coefficients and finite real constant shifts in compiled
functional algebra. The older ``prox_l1`` / ``prox_l2sq`` helpers retain their
nonnegative-step convention, including the documented zero-step behavior.

Conjugates use Hilbert dual variables
-------------------------------------

``f.conjugate_prox(p, step)`` uses Moreau decomposition in the same space:

.. math::

   f_X^*(p) &= \sup_z \operatorname{Re}\langle p,z\rangle_X - f(z),\\
   \operatorname{prox}^{X}_{t f_X^*}(p)
     &= p-t\operatorname{prox}^{X}_{f/t}(p/t).

Here ``p`` is a Riesz representer, not a coordinate covector. For diagonal
metric ``W`` and coordinate ``f(x)=lambda*||x||_1``, the conjugate is the
indicator of ``|W p| <= lambda``. Its prox clips to those weighted bounds.
This convention matches ``LinOp.rapply`` in a PDHG dual update. A registered
prox asserts a proper lower-semicontinuous convex real objective; a complex
space uses its real inner-product pairing. Arbitrary complex-valued functions
and nonconvex extensions do not acquire this guarantee.

Sets and indicators
-------------------

``Set`` owns ``ambient`` and offers ``project(x)`` and ``contains(x)``.
Membership may return ``None`` when unavailable; the indicator's value then
raises ``CapabilityError``, while projection may still work.
``IndicatorFunctional`` requires ``is_convex = True`` and returns zero on the
set and positive infinity outside it. For every positive step its prox is the
set's metric projection.

``NonnegativeOrthant(X)`` supports real dense coordinates with Euclidean or
diagonal geometry. Its projection is coordinatewise maximum with zero. Other
geometries and representations need their own mathematically valid set
implementation. A custom ``Set`` also implements the standard ``_convert``
and pytree hooks; ambient geometry is static metadata. Sets do not imply
manifold operations or an intersection-projection solver.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx,
       geometry=sc.WeightedInnerProduct(ctx.asarray([2., 5.])))
   f = 0.3 * sc.L1NormFunctional(X)
   x = ctx.asarray([-1., 2.])
   y = f.prox(x, 0.5)  # [-0.925, 1.97]
   constraint = sc.NonnegativeOrthant(X)
   feasible = sc.IndicatorFunctional(constraint).prox(x, 0.5)  # [0., 2.]

Extending the rule table
------------------------

Use the existing ``spacecore.mapping.register`` table:

.. code-block:: python

   from spacecore.mapping import register

   def my_prox(functional, geometry, x, step):
       # Compute the exact metric prox using the representation's native operations.
       return functional.domain.native_prox(x, step)

   register("prox", MyFunctional, MyGeometry, my_prox)

Lookup follows the functional MRO, then the geometry MRO. ``object`` as the
geometry key is appropriate only for rules valid across the geometries that
the functional supports, including a native space with no separate geometry
object. Changing a subclass's objective also requires reviewing its inherited
prox rule. Register extensions during setup, before tracing; changing the table
does not invalidate an already compiled function.

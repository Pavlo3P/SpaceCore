Functionals API
===============

Functionals are scalar-valued maps on a domain space :math:`f : X \to R`.
Gradients are represented as elements of ``X`` using the domain geometry: for a
differentiable functional, ``grad(x)`` satisfies
:math:`\operatorname{Re}\langle \nabla f(x), v \rangle_X = D f(x)[v]` when implemented.
Beck, *First-Order Methods in Optimization*, Remark 3.32; on complex domains
use the real inner product as in :doc:`../design/geometry`.

Differentials
-------------

``f.derivative_at(x)`` returns a ``LinearFunctional`` acting on directions:
``df(h)`` evaluates :math:`Df(x)[h]`. ``f.value_and_derivative(x)`` returns the
value and differential together, reusing an existing fused
``value_and_grad`` implementation when adapting a gradient-based functional.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx,
       geometry=sc.WeightedInnerProduct(ctx.asarray([2., 5.])))
   f = sc.SquaredL2NormFunctional(X)
   value, df = f.value_and_derivative(ctx.asarray([1., 2.]))
   slope = df(ctx.asarray([3., 4.]))  # 46.0
   gradient = df.representer         # [1., 2.], in X's metric

Existing ``grad`` implementations remain valid. New functionals can implement
``derivative_at`` directly. The default ``grad`` then requests the derivative's
``representer``; a callable-only differential raises ``NotImplementedError``
instead of reconstructing a dense gradient. Implement both fused methods when
a native evaluator needs to share work in both interfaces.

Sums, scalar multiples, and pullbacks of linear functionals remain
``LinearFunctional`` objects. Products of objectives use the product rule.
A callable differential can be pulled back through ``A`` using only forward
application: :math:`(Df(x)\circ A)[h]=Df(x)[Ah]`. Retrieving its gradient may
additionally require a representer and the metric adjoint.

``df.scalar_field`` records whether the differential is real- or complex-linear.
An ``InnerProductFunctional`` with a real codomain, or with
``real_linear=True``, evaluates :math:`\operatorname{Re}\langle c,h\rangle`.
For ``MatrixFreeLinearFunctional``, ``real_linear=True`` declares the callable's
linearity; its output is not projected to its real part. This also permits
complex-valued, real-linear derivatives such as complex conjugation.

The default gradient adapter assumes complex linearity for a complex codomain.
Real-valued built-in coordinate objectives explicitly select the real pairing,
even when their legacy scalar codomain is complex. General complex-valued,
real-linear derivatives may have no single Riesz representer. Complex-valued
spectral objectives require an explicit differential implementation.
At a nonsmooth point, a chosen subgradient does not become a Frechet derivative;
use these interfaces where the objective is differentiable.

This API composes with existing linear operators. General nonlinear-map
linearization and manifold tangent spaces are separate extensions.

Base and composition
--------------------

Optional proximal methods
~~~~~~~~~~~~~~~~~~~~~~~~~

``Functional.prox(x, step)`` computes a registered closed-form prox in the
domain metric. ``conjugate_prox`` uses Hilbert-space Moreau decomposition.
``IndicatorFunctional`` connects a closed convex set's projection to this API.
See :doc:`../design/proximal_contract` for conventions, supported rules,
compiled execution, and custom representations.

.. autosummary::
   :nosignatures:

   spacecore.IndicatorFunctional
   spacecore.Set
   spacecore.NonnegativeOrthant

Scalar maps
~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   spacecore.functional.Functional
   spacecore.functional.ComposedFunctional
   spacecore.functional.make_functional_composed

* ``Functional`` is the base scalar-valued map contract.
* ``ComposedFunctional`` represents pullback ``f o A`` for a linear operator ``A``.
* ``make_functional_composed`` constructs the same pullback with simplifications.

Algebra
-------

Lazy nodes for combining functionals. The operator overloads on ``Functional``
(``a * F``, ``F + G``, ``F - G``, ``-F``, ``F * G``) delegate to the ``make_*``
factories, which apply local, *structural* canonicalization — they read node
types, never values.

.. autosummary::
   :nosignatures:

   spacecore.functional.ScaledFunctional
   spacecore.functional.SumFunctional
   spacecore.functional.ZeroFunctional
   spacecore.functional.ConstantFunctional
   spacecore.functional.ProductFunctional
   spacecore.functional.make_scaled_functional
   spacecore.functional.make_functional_sum
   spacecore.functional.make_shifted_functional
   spacecore.functional.make_constant_functional
   spacecore.functional.make_functional_product

* ``ScaledFunctional`` is ``a * F``; the Riesz gradient scales by ``conj(a)``.
* ``SumFunctional`` is ``F_1 + ... + F_n`` on a shared domain.
* ``ZeroFunctional`` is the additive identity, recognized by the canonicalizers.
* ``ConstantFunctional`` is ``x -> c``, the embedding of a scalar into the
  algebra; ``make_constant_functional`` collapses ``c = 0`` to ``ZeroFunctional``.
* ``ProductFunctional`` is the pointwise product ``F(x) * G(x)``, with the
  product-rule Riesz gradient
  :math:`\overline{G(x)}\, \nabla F(x) + \overline{F(x)}\, \nabla G(x)`. Because
  functionals are scalar-valued, scaling is the constant-factor case of a
  product: ``make_functional_product`` folds a ``ConstantFunctional`` factor back
  into a ``ScaledFunctional``, and a ``ZeroFunctional`` factor to zero.

For complex-linear differentials, expand ``D(FG)[h] = G DF[h] + F DG[h]``
and use conjugate-linearity in the first inner-product argument to obtain the
displayed rule. It also holds for real-valued factors using the real pairing.
Arbitrary complex-valued, real-differentiable maps need more than one gradient.

Mapping
-------

.. autosummary::
   :nosignatures:

   spacecore.Mapping

Linear functionals
------------------

.. autosummary::
   :nosignatures:

   spacecore.functional.LinearFunctional
   spacecore.functional.InnerProductFunctional
   spacecore.functional.MatrixFreeLinearFunctional

* ``LinearFunctional`` is the base class for linear maps to scalars.
* ``InnerProductFunctional`` stores a Riesz representer ``c`` and evaluates ``<c, x>``.
* ``MatrixFreeLinearFunctional`` wraps a callable evaluator; a stored gradient is optional
  and is not inferred from the callable.

Quadratic functionals
---------------------

.. autosummary::
   :nosignatures:

   spacecore.functional.QuadraticForm
   spacecore.functional.LinOpQuadraticForm

* ``QuadraticForm`` models objectives with value, gradient, and Hessian-vector action.
* ``LinOpQuadraticForm`` represents ``0.5 <x, Qx> + ell(x) + a``.

Battery functionals
-------------------

Named constructors over the existing machinery (ADR-019). Their gradients are
metric (Riesz) gradients under the domain geometry.

.. autosummary::
   :nosignatures:

   spacecore.functional.least_squares
   spacecore.functional.SquaredL2NormFunctional
   spacecore.functional.LpNormFunctional
   spacecore.functional.L1NormFunctional
   spacecore.functional.SpectralFunctional
   spacecore.functional.spectralize
   spacecore.functional.eigenvalue_space
   spacecore.functional.NuclearNormFunctional
   spacecore.functional.RealifiedFunctional
   spacecore.functional.realify
   spacecore.functional.NegativeEntropyFunctional
   spacecore.functional.KLDivergenceFunctional
   spacecore.functional.HuberFunctional

* ``least_squares`` builds the ``scale ||A x - b||^2`` objective as a ``LinOpQuadraticForm``.
* ``SquaredL2NormFunctional`` is ``1/2 ||x||_X^2`` (gradient ``x``, clean shrinkage prox).
* ``LpNormFunctional`` / ``L1NormFunctional`` are coordinate ``p``-norms.
* ``SpectralFunctional`` / ``spectralize`` lift **any** symmetric coordinate
  functional onto a Jordan spectrum: on Hermitian matrices the differentiable
  case is Lewis, "Derivatives of Spectral Functions", Thm 1.1, and convex
  subgradients are Thm 1.3. The Schatten ``p``-norm is
  ``spectralize(X, lambda s: LpNormFunctional(s, p))``, the negative von Neumann entropy
  is ``spectralize(X, NegativeEntropyFunctional)``. ``eigenvalue_space`` builds
  the real space the spectrum lives in.
* ``NuclearNormFunctional`` is the named Schatten-1 case.
* ``RealifiedFunctional`` / ``realify`` present a complex-domain functional over
  stacked real coordinates, for real-only optimizers.
* ``NegativeEntropyFunctional`` and ``KLDivergenceFunctional`` are the entropy objectives.
* ``HuberFunctional`` is the separable Huber loss.

These constructors live in the :mod:`spacecore.functional.tools` subpackage and are
re-exported from :mod:`spacecore.functional` and the top-level ``spacecore`` namespace.

Proximal and projection
-----------------------

A closed-form, metric-aware proximal primitive and its named wrappers (ADR-019).
Valid on Euclidean and diagonal metrics; a non-diagonal metric raises.

.. autosummary::
   :nosignatures:

   spacecore.functional.generalized_shrinkage
   spacecore.functional.prox_l1
   spacecore.functional.prox_l2sq
   spacecore.functional.project_nonneg

* ``generalized_shrinkage`` solves ``<c, x>_X + eps ||x - x0||^2_X + lam ||x||_1`` (optionally ``x >= 0``).
* ``prox_l1`` is the metric soft-threshold; ``prox_l2sq`` is the shrinkage ``v / (1 + t)``.
* ``project_nonneg`` is the metric projection onto the nonnegative orthant.

Autodoc
-------

.. autoclass:: spacecore.functional.Functional
   :members:
   :inherited-members:

.. autoclass:: spacecore.functional.LinearFunctional
   :members:
   :inherited-members:

.. autoclass:: spacecore.functional.InnerProductFunctional
   :members:
   :inherited-members:

.. autoclass:: spacecore.functional.MatrixFreeLinearFunctional
   :members:
   :inherited-members:

.. autoclass:: spacecore.functional.QuadraticForm
   :members:
   :inherited-members:

.. autoclass:: spacecore.functional.LinOpQuadraticForm
   :members:
   :inherited-members:

.. autoclass:: spacecore.functional.ComposedFunctional
   :members:
   :inherited-members:

.. autofunction:: spacecore.functional.make_functional_composed

.. autofunction:: spacecore.functional.least_squares

.. autoclass:: spacecore.functional.SquaredL2NormFunctional
   :members:

.. autoclass:: spacecore.functional.LpNormFunctional
   :members:

.. autofunction:: spacecore.functional.L1NormFunctional

.. autoclass:: spacecore.functional.SpectralFunctional
   :members:

.. autofunction:: spacecore.functional.spectralize

.. autofunction:: spacecore.functional.eigenvalue_space

.. autofunction:: spacecore.functional.NuclearNormFunctional

.. autoclass:: spacecore.functional.RealifiedFunctional
   :members:

.. autofunction:: spacecore.functional.realify

.. autoclass:: spacecore.functional.NegativeEntropyFunctional
   :members:

.. autoclass:: spacecore.functional.KLDivergenceFunctional
   :members:

.. autoclass:: spacecore.functional.HuberFunctional
   :members:

.. autofunction:: spacecore.functional.generalized_shrinkage

.. autofunction:: spacecore.functional.prox_l1

.. autofunction:: spacecore.functional.prox_l2sq

.. autofunction:: spacecore.functional.project_nonneg

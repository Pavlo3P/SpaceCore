Functionals API
===============

Functionals are real scalar-valued maps on a domain space :math:`f : X \to R`.
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

Functionals are real-valued. The codomain is always ``RealField``; complex
domains remain supported. With checks on (``cheap`` and above), a complex output
dtype, coefficient or offset raises ``TypeError``, even with a zero imaginary
part; this covers batched and fused evaluations and subclass overrides.
Extended-real indicator values remain valid.

Functionals pair through the domain's real inner product
``real_inner(x, y)`` :math:`=\operatorname{Re}\langle x,y\rangle_X`: ``InnerProductFunctional``
evaluates ``real_inner(c, h)`` and ``LinOpQuadraticForm`` evaluates
``real_inner(x, Qx) / 2``.
Matrix-free callables return real values themselves.

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

* ``ScaledFunctional`` is ``a * F``; the Riesz gradient scales by the real scalar ``a``.
* ``SumFunctional`` is ``F_1 + ... + F_n`` on a shared domain.
* ``ZeroFunctional`` is the additive identity, recognized by the canonicalizers.
* ``ConstantFunctional`` is ``x -> c``, the embedding of a scalar into the
  algebra; ``make_constant_functional`` collapses ``c = 0`` to ``ZeroFunctional``.
* ``ProductFunctional`` is the pointwise product ``F(x) * G(x)``, with the
  product-rule Riesz gradient
  :math:`G(x)\, \nabla F(x) + F(x)\, \nabla G(x)`. Because
  functionals are scalar-valued, scaling is the constant-factor case of a
  product: ``make_functional_product`` folds a ``ConstantFunctional`` factor back
  into a ``ScaledFunctional``, and a ``ZeroFunctional`` factor to zero.

The product rule follows from ``D(FG)[h] = G DF[h] + F DG[h]`` using the
real pairing. All values and scaling coefficients are real.

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

Norm functionals
----------------

Coordinate norms (the ones the closed-form prox rules cover). Their gradients are
metric (Riesz) gradients under the domain geometry.

.. autosummary::
   :nosignatures:

   spacecore.functional.SquaredL2NormFunctional
   spacecore.functional.LpNormFunctional
   spacecore.functional.L1NormFunctional

* ``SquaredL2NormFunctional`` is ``1/2 ||x||_X^2`` (gradient ``x``, clean shrinkage prox).
* ``LpNormFunctional`` / ``L1NormFunctional`` are coordinate ``p``-norms.

The norms live in the :mod:`spacecore.functional.tools` subpackage and are
re-exported from :mod:`spacecore.functional` and the top-level ``spacecore`` namespace.
Least squares, entropy, Huber, spectral lifts, the realified view
(``RealifiedFunctional``, ``realify``) and the shrinkage helpers are in
`spacecore-optimize <https://github.com/Pavlo3P/spacecore-optimize>`_.

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

.. autoclass:: spacecore.functional.SquaredL2NormFunctional
   :members:

.. autoclass:: spacecore.functional.LpNormFunctional
   :members:

.. autofunction:: spacecore.functional.L1NormFunctional


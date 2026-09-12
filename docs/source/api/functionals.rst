Functionals API
===============

Functionals are scalar-valued maps on a domain space :math:`f : X \to R`.
Gradients are represented as elements of ``X`` using the domain geometry: for a
differentiable functional, ``grad(x)`` satisfies
:math:`\langle \nabla f(x), v \rangle_X = D f(x)[v]` when implemented.

Base and composition
--------------------

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
   spacecore.functional.ShiftedFunctional
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
* ``ShiftedFunctional`` is the affine shift ``F + c`` (gradient unchanged).
* ``ZeroFunctional`` is the additive identity, recognized by the canonicalizers.
* ``ConstantFunctional`` is ``x -> c``, the embedding of a scalar into the
  algebra; ``make_constant_functional`` collapses ``c = 0`` to ``ZeroFunctional``.
* ``ProductFunctional`` is the pointwise product ``F(x) * G(x)``, with the
  product-rule Riesz gradient
  :math:`\overline{G(x)}\, \nabla F(x) + \overline{F(x)}\, \nabla G(x)`. Because
  functionals are scalar-valued, scaling is the constant-factor case of a
  product: ``make_functional_product`` folds a ``ConstantFunctional`` factor back
  into a ``ScaledFunctional``, and a ``ZeroFunctional`` factor to zero.

Operator families (``F · A``)
-----------------------------

.. autosummary::
   :nosignatures:

   spacecore.OperatorFamily
   spacecore.FunctionalScaledOperator
   spacecore.make_functional_scaled_operator

``F * A`` for a ``Functional`` and a ``LinOp`` is the functional-weighted map
:math:`m(x) = F(x)\,Ax`. This is **not** linear — both the scale and the
direction move with ``x`` — so it is *not* a ``LinOp``; it is an
``OperatorFamily``, a point-indexed family :math:`x \mapsto A_x`.

Each point carries two different linear operators, and they are not the same:

* ``m.at(x)`` — the **frozen member** ``F(x) · A``, an ordinary ``LinOp`` that
  composes, sums, and has an adjoint. Freezing the point is what recovers
  linearity.
* ``m.linearize_at(x)`` — the **derivative** :math:`Dm(x)`, for Newton-type
  steps. They coincide only when ``F`` is constant; otherwise they differ by a
  rank-one term.

A ``ConstantFunctional`` weight collapses to an ordinary ``ScaledLinOp``, so the
linear case is never forced through the non-linear type.

Linear functionals
------------------

.. autosummary::
   :nosignatures:

   spacecore.functional.LinearFunctional
   spacecore.functional.InnerProductFunctional
   spacecore.functional.MatrixFreeLinearFunctional

* ``LinearFunctional`` is the base class for linear maps to scalars.
* ``InnerProductFunctional`` stores a Riesz representer ``c`` and evaluates ``<c, x>``.
* ``MatrixFreeLinearFunctional`` wraps callable value and gradient logic.

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
  functional onto a Jordan spectrum (Lewis): the Schatten ``p``-norm is
  ``spectralize(X, lambda s: LpNormFunctional(s, p))``, the von Neumann entropy
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

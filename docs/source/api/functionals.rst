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
   spacecore.functional.SpectralLpNormFunctional
   spacecore.functional.NuclearNormFunctional
   spacecore.functional.NegativeEntropyFunctional
   spacecore.functional.KLDivergenceFunctional
   spacecore.functional.HuberFunctional

* ``least_squares`` builds the ``scale ||A x - b||^2`` objective as a ``LinOpQuadraticForm``.
* ``SquaredL2NormFunctional`` is ``1/2 ||x||_X^2`` (gradient ``x``, clean shrinkage prox).
* ``LpNormFunctional`` / ``L1NormFunctional`` are coordinate ``p``-norms.
* ``SpectralLpNormFunctional`` / ``NuclearNormFunctional`` are the Schatten ``p``-norm
  and nuclear norm of a Jordan spectrum (e.g. Hermitian eigenvalues).
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

.. autoclass:: spacecore.functional.SpectralLpNormFunctional
   :members:

.. autofunction:: spacecore.functional.NuclearNormFunctional

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

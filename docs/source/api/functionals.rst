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

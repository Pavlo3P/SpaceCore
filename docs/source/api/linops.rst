Linear Operators API
====================

Linear operators represent typed maps :math:`A : X \to Y`. ``apply`` evaluates
:math:`A x`; ``rapply`` evaluates the metric adjoint :math:`A^\sharp y`, which
matches the coordinate conjugate transpose only when both spaces are Euclidean.

Matrix-backed operators
-----------------------

.. autosummary::
   :nosignatures:

   spacecore.linop.DenseLinOp
   spacecore.linop.SparseLinOp
   spacecore.linop.DiagonalLinOp

* ``DenseLinOp`` stores a dense matrix or tensor interpreted in domain and codomain coordinates.
* ``SparseLinOp`` stores a backend sparse matrix for sparse forward and adjoint products.
* ``DiagonalLinOp`` stores coordinate diagonal entries for a map ``X -> X``.

Matrix-free and canonical operators
-----------------------------------

.. autosummary::
   :nosignatures:

   spacecore.linop.LinOp
   spacecore.linop.MatrixFreeLinOp
   spacecore.linop.IdentityLinOp
   spacecore.linop.ZeroLinOp

* ``LinOp`` is the abstract base contract.
* ``MatrixFreeLinOp`` wraps forward and metric-adjoint callables exactly as
  supplied; it does not derive or Riesz-correct matrix-free adjoints.
* ``IdentityLinOp`` represents ``I : X -> X``.
* ``ZeroLinOp`` represents ``0 : X -> Y``.

Algebraic operators
-------------------

.. autosummary::
   :nosignatures:

   spacecore.linop.SumLinOp
   spacecore.linop.ComposedLinOp
   spacecore.linop.ScaledLinOp
   spacecore.linop.make_sum
   spacecore.linop.make_composed
   spacecore.linop.make_scaled

* ``SumLinOp`` represents ``A + B`` for maps with the same domain and codomain.
* ``ComposedLinOp`` represents ``A @ B`` where ``B.codomain == A.domain``.
* ``ScaledLinOp`` represents ``alpha * A``.
* Helper constructors perform the same simplifications used by Python operator overloads.

Product and block operators
---------------------------

.. autosummary::
   :nosignatures:

   spacecore.linop.ProductLinOp
   spacecore.linop.BlockDiagonalLinOp
   spacecore.linop.StackedLinOp
   spacecore.linop.SumToSingleLinOp

* ``BlockDiagonalLinOp`` maps product components independently.
* ``StackedLinOp`` maps one domain into a product codomain by stacking outputs.
* ``SumToSingleLinOp`` maps a product domain into one codomain by summing component outputs.

Autodoc
-------

.. autoclass:: spacecore.linop.LinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.DenseLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.SparseLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.DiagonalLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.MatrixFreeLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.IdentityLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.ZeroLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.SumLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.ComposedLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.ScaledLinOp
   :members:
   :inherited-members:

.. autofunction:: spacecore.linop.make_sum
.. autofunction:: spacecore.linop.make_composed
.. autofunction:: spacecore.linop.make_scaled

.. autoclass:: spacecore.linop.ProductLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.BlockDiagonalLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.StackedLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.SumToSingleLinOp
   :members:
   :inherited-members:

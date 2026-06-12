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

Tree and block operators
------------------------

.. autosummary::
   :nosignatures:

   spacecore.linop.TreeLinOp
   spacecore.linop.BlockDiagonalLinOp
   spacecore.linop.BlockMatrixLinOp
   spacecore.linop.StackedLinOp
   spacecore.linop.SumToSingleLinOp

* ``TreeLinOp`` is the base for operators with a ``TreeSpace`` domain or codomain.
* ``BlockDiagonalLinOp(blocks)`` maps corresponding tree leaves independently
  and infers both ``TreeSpace`` endpoints from the block domains and codomains.
* ``BlockMatrixLinOp(block_rows)`` computes row sums for a rectangular matrix
  of compatible blocks and infers tuple-structured ``TreeSpace`` endpoints.
* ``StackedLinOp`` maps one domain into a tree codomain.
* ``SumToSingleLinOp`` maps a tree domain into one codomain by summing leaf outputs.

Both block classes represent operators over finite direct products. They are
not tensor-product or Kronecker-product operators. Their ``rapply`` methods use
the metric adjoint of every block, which differs from a coordinate conjugate
transpose when a leaf space has a non-Euclidean inner product.

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

.. autoclass:: spacecore.linop.TreeLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.BlockDiagonalLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.BlockMatrixLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.StackedLinOp
   :members:
   :inherited-members:

.. autoclass:: spacecore.linop.SumToSingleLinOp
   :members:
   :inherited-members:

Linear operators API
====================

Linear operators represent maps between spaces with forward and adjoint
actions.

.. autosummary::
   :nosignatures:

   spacecore.linop.LinOp
   spacecore.linop.ProductLinOp
   spacecore.linop.DenseLinOp
   spacecore.linop.DiagonalLinOp
   spacecore.linop.SparseLinOp
   spacecore.linop.MatrixFreeLinOp
   spacecore.linop.IdentityLinOp
   spacecore.linop.ZeroLinOp
   spacecore.linop.ScaledLinOp
   spacecore.linop.SumLinOp
   spacecore.linop.ComposedLinOp
   spacecore.linop.BlockDiagonalLinOp
   spacecore.linop.StackedLinOp
   spacecore.linop.SumToSingleLinOp
   spacecore.linop.make_scaled
   spacecore.linop.make_sum
   spacecore.linop.make_composed

LinOp
-----

.. autoclass:: spacecore.linop.LinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

ProductLinOp
------------

.. autoclass:: spacecore.linop.ProductLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

DenseLinOp
----------

.. autoclass:: spacecore.linop.DenseLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

DiagonalLinOp
-------------

.. autoclass:: spacecore.linop.DiagonalLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

SparseLinOp
-----------

.. autoclass:: spacecore.linop.SparseLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

MatrixFreeLinOp
---------------

.. autoclass:: spacecore.linop.MatrixFreeLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

IdentityLinOp
-------------

.. autoclass:: spacecore.linop.IdentityLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

ZeroLinOp
---------

.. autoclass:: spacecore.linop.ZeroLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Algebraic operators
-------------------

.. autoclass:: spacecore.linop.ScaledLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spacecore.linop.SumLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spacecore.linop.ComposedLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autofunction:: spacecore.linop.make_scaled

.. autofunction:: spacecore.linop.make_sum

.. autofunction:: spacecore.linop.make_composed

Product-structured operators
----------------------------

.. autoclass:: spacecore.linop.BlockDiagonalLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spacecore.linop.StackedLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spacecore.linop.SumToSingleLinOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

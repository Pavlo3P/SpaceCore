Batching and Vectorization
==========================

A space describes one element; a batch is a leading axis handled by the
``v``-methods, never a different space. For :math:`A : X \to Y`,
``A.vapply(xs)`` takes shape ``(B,) + X.shape`` and returns
``(B,) + Y.shape``.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   A = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)
   print(A.vapply(ctx.asarray([[1.0, 1.0], [2.0, 4.0]])))

.. code-block:: text

   [[ 2.  3.]
    [ 4. 12.]]

Batched methods
---------------

==========================  ================================================
Method                      Batched form of
==========================  ================================================
``LinOp.vapply``            ``apply``
``LinOp.rvapply``           ``rapply`` (metric adjoint)
``Functional.vvalue``       ``value``; returns shape ``(B,)``
``Functional.vgrad``        ``grad``, where the functional implements it
==========================  ================================================

* Implement ``_vapply_core`` / ``_rvapply_core`` for a vectorized path. Without
  one, the base class maps ``apply`` over the batch with the backend's ``vmap``:
  native on JAX and Torch, a Python loop on NumPy, which emits a
  ``RuntimeWarning`` once per class for batches larger than 32.
* Only the leading axis batches; use ``moveaxis`` for other layouts. Every leaf
  of a ``TreeSpace`` batch has the same batch size.
* When the stack itself is the mathematical object, use ``StackedSpace(base,
  count)``: a space of ``count`` copies of ``base``.
* The solvers (``cg``, ``lsqr``, ``lanczos_smallest``, ``power_iteration``,
  ``expm_multiply``) are unbatched; a batched right-hand side fails the
  membership check. Loop over the batch.

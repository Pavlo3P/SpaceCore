Batching and Vectorization
==========================

A SpaceCore space describes one element type. Batched computation is handled by
vectorized application mechanisms, not by changing the meaning of a space.

For a map :math:`A : X \to Y`, ``A.vapply(xs)`` accepts a leading-axis batch
with shape ``(B,) + X.shape`` and returns shape ``(B,) + Y.shape``. Likewise,
``A.rvapply(ys)`` vectorizes the metric adjoint over a leading axis.

Current mechanisms
------------------

* ``LinOp.vapply`` and ``LinOp.rvapply`` lift operators over a leading axis.
* ``Functional.vvalue`` and ``Functional.vgrad`` lift scalar functionals and
  gradients where implemented.
* Backends may implement native vectorization, for example JAX ``vmap``.
* Structured operators may provide specialized batched paths.
* ``StackedSpace`` is an explicit space for a fixed number of leading-axis
  copies when the stack itself is the mathematical object.

Batching does not change the mathematical domain or codomain of an ordinary
operator. ``A : X -> Y`` remains a map between one element of ``X`` and one
element of ``Y``; ``vapply`` is a vectorized evaluation of that same map.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   A = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)
   xs = ctx.asarray([[1.0, 1.0], [2.0, 4.0]])
   print(A.vapply(xs))

Expected output:

.. code-block:: text

   [[ 2.  3.]
    [ 4. 12.]]

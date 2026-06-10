BackendOps Tutorial
===================

Goal
   Write a small helper that works through ``BackendOps`` instead of direct
   NumPy calls.
Prerequisites
   Basic NumPy and :doc:`context`.
Estimated time
   10 minutes.
Backends used
   NumPy only in executable blocks.

1. Get backend operations from a context
----------------------------------------

``BackendOps`` is the operation surface used below spaces and operators.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   ops = ctx.ops

   x = ops.asarray([1.0, 2.0, 3.0], dtype=ctx.dtype)
   print(ops.family)
   print(ops.sum(x))

Expected output:

.. code-block:: text

   numpy
   6.0

Checkpoint: ``ops.is_dense(x)`` should be true.

2. Write a portable helper
--------------------------

The helper below depends on ``ops.matmul`` and ``ops.conj`` rather than on
``numpy.matmul`` directly.

.. code-block:: python

   def gram(ops, A):
       return ops.matmul(ops.conj(ops.transpose(A)), A)

   A = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
   print(gram(ops, A))

Expected output:

.. code-block:: text

   [[35. 44.]
    [44. 56.]]

Checkpoint: the result shape should be ``(2, 2)``.

3. Decide when to use spaces instead
------------------------------------

Use ``BackendOps`` for low-level backend-agnostic helpers. Use spaces and
operators when the code depends on mathematical domains, codomains, inner
products, or adjoints.

.. code-block:: python

   X = sc.DenseCoordinateSpace((2,), ctx)
   D = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)
   print(D.apply(ctx.asarray([4.0, 5.0])))

Expected output:

.. code-block:: text

   [ 8. 15.]

Checkpoint: if you need ``rapply`` or ``inner``, use ``LinOp`` and ``Space``.

What can go wrong
-----------------

.. admonition:: Reaching into backend modules too early

   ``ops.np``, ``ops.torch``, or ``ops.jax`` are explicit escape hatches. Code
   that uses them is no longer backend-portable unless you branch per backend.

Recap
-----

* ``BackendOps`` is SpaceCore's low-level numerical interface.
* Contexts carry an ``ops`` object plus dtype and checks.
* Portable helpers should call ``ops`` methods.
* Mathematical code should prefer spaces and operators.

Next steps
----------

Read :doc:`conversion_policy` for moving context-bound objects. See
:doc:`../api/backend` for the backend API index.

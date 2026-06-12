LinOps Tutorial
===============

Goal
   Build linear maps :math:`A : X \to Y`, apply them, inspect metric adjoints,
   and vectorize application over leading-axis batches.
Prerequisites
   Complete :doc:`spaces`; know basic matrix-vector multiplication.
Estimated time
   15 minutes.
Backends used
   NumPy only.

1. Set up spaces and a dense operator
-------------------------------------

A ``LinOp`` is a typed map between spaces. The dense matrix below represents
:math:`A : X \to Y`, where ``X`` has shape ``(2,)`` and ``Y`` has shape ``(3,)``.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
   X = sc.DenseCoordinateSpace((2,), ctx)
   Y = sc.DenseCoordinateSpace((3,), ctx)
   matrix = ctx.asarray([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
   A = sc.DenseLinOp(matrix, X, Y, ctx)

   print(A.domain.shape)
   print(A.codomain.shape)

Expected output:

.. code-block:: text

   (2,)
   (3,)

Checkpoint: ``A.domain == X`` and ``A.codomain == Y`` should both be true.

2. Apply the forward map and adjoint
------------------------------------

``apply`` computes :math:`A x`. ``rapply`` computes the metric adjoint
:math:`A^\sharp y`, satisfying
:math:`\langle A x, y \rangle_Y = \langle x, A^\sharp y \rangle_X`.
In Euclidean geometry this is the coordinate conjugate transpose.

.. code-block:: python

   x = ctx.asarray([1.0, 2.0])
   y = ctx.asarray([1.0, 1.0, 1.0])

   print(A.apply(x))
   print(A.rapply(y))

Expected output:

.. code-block:: text

   [5.  -2.   5.]
   [4. 2.]

Checkpoint: ``Y.inner(A.apply(x), y) == X.inner(x, A.rapply(y))`` should be true.

3. Use operator algebra
-----------------------

Algebra builds lazy operators. ``A.H @ A`` is a map :math:`X \to X`.

.. code-block:: python

   normal = A.H @ A
   print(normal.apply(x))

Expected output:

.. code-block:: text

   [20.  17.]

Checkpoint: ``normal.domain == X`` and ``normal.codomain == X`` should be true.

4. Vectorize over a leading batch
---------------------------------

A space still describes one element. ``vapply`` evaluates the same operator for
each row of a leading-axis batch.

.. code-block:: python

   xs = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
   print(A.vapply(xs))

Expected output:

.. code-block:: text

   [[ 5. -2.  5.]
    [11. -4. 13.]]

Checkpoint: ``A.vapply(xs)[0]`` matches ``A.apply(xs[0])``.

What can go wrong
-----------------

.. admonition:: Wrong input shape

   With checks enabled, applying ``A : X -> Y`` to an element outside ``X``
   raises a membership error before backend multiplication.

   .. code-block:: python

      try:
          A.apply(ctx.asarray([1.0, 2.0, 3.0]))
      except Exception as exc:
          print(type(exc).__name__)

   Expected output:

   .. code-block:: text

      SpaceValidationError

Recap
-----

* ``LinOp`` stores a domain, codomain, and context.
* ``apply`` is the forward map; ``rapply`` is the metric adjoint.
* Operator algebra keeps maps typed by their spaces.
* ``vapply`` handles leading-axis batches without changing ``A : X -> Y``.

Next steps
----------

Continue with :doc:`context` for backend ownership. Read :doc:`../api/linops`
for the complete operator index.

Spaces Tutorial
===============

Goal
   Build a dense vector space, verify its geometry, and see how weighted inner
   products differ from Euclidean dot products.
Prerequisites
   Basic Python and NumPy; no previous SpaceCore experience.
Estimated time
   10 minutes.
Backends used
   NumPy only.

1. Create a context
-------------------

A context owns the backend operations, dtype, and runtime checking policy used
by the objects you create.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
   print(ctx.ops.family)
   print(ctx.dtype == np.dtype("float64"))

Expected output:

.. code-block:: text

   numpy
   True

Checkpoint: ``ctx.asarray([1.0]).dtype`` should be ``float64``.

2. Build a Euclidean coordinate space
-------------------------------------

``DenseCoordinateSpace((2,), ctx)`` represents one element of
:math:`\mathbb{R}^2` stored as a dense backend array with shape ``(2,)``.

.. code-block:: python

   X = sc.DenseCoordinateSpace((2,), ctx)
   x = ctx.asarray([3.0, 4.0])
   y = ctx.asarray([1.0, 2.0])

   print(X.shape)
   print(X.inner(x, y))
   print(X.norm(x))

Expected output:

.. code-block:: text

   (2,)
   11.0
   5.0

Checkpoint: ``X.check_member(x)`` should run without raising.

3. Add weighted geometry
------------------------

A weighted space has the same element representation but a different inner
product. For weights ``w``, SpaceCore computes
:math:`\langle x, y \rangle_X = \operatorname{vdot}(x, w y)`.

.. code-block:: python

   weights = ctx.asarray([2.0, 5.0])
   Xw = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(weights))

   print(Xw.inner(x, y))
   print(Xw.riesz(x))
   print(Xw.is_euclidean)

Expected output:

.. code-block:: text

   46.0
   [ 6. 20.]
   False

Checkpoint: ``X.inner(x, y)`` is ``11.0`` but ``Xw.inner(x, y)`` is ``46.0``.

What can go wrong
-----------------

.. admonition:: Wrong weight shape

   ``WeightedInnerProduct`` weights must match the coordinate shape exactly.

   .. code-block:: python

      try:
          sc.DenseCoordinateSpace(
              (3,), ctx, geometry=sc.WeightedInnerProduct(weights)
          )
      except ValueError as exc:
          print(exc)

   Expected output:

   .. code-block:: text

      WeightedInnerProduct weights must have exactly the coordinate shape (3,); got (2,).

Recap
-----

* A ``Context`` fixes backend, dtype, and checking policy.
* A ``DenseCoordinateSpace`` describes one element shape, not a batch.
* Geometry changes ``inner``, ``norm``, and adjoint semantics.
* ``riesz`` maps coordinate elements to dual coordinates for the metric.

Next steps
----------

Continue with :doc:`linops` to build maps between spaces. Read
:doc:`../design/geometry` for metric adjoint details.

Capability Dispatch
===================

A space has an operation only if its class carries the capability that defines
it; test with ``isinstance(X, sc.StarSpace)`` and so on.

===============================  ===============================================
Capability                       Adds
===============================  ===============================================
``Space``                        context, ``check_member``
``VectorSpace``                  ``add``, ``scale``, ``zeros``, ...
``CoordinateSpace``              ``shape``, ``flatten``, ``unflatten``
``InnerProductSpace``            ``inner``, ``norm``, ``riesz``,
                                 ``riesz_inverse``
``StarSpace``                    ``star`` (involution)
``JordanAlgebraSpace``           ``jordan``, ``spectrum``,
                                 ``spectral_decompose``, ``spectral_apply``
``EuclideanJordanAlgebraSpace``  marker: Jordan structure compatible with the
                                 Euclidean inner product
===============================  ===============================================

* ``DenseCoordinateSpace``: inner product, no star. ``DenseVectorSpace`` adds
  star, no Jordan. ``ElementwiseJordanSpace`` adds Jordan.
* ``ElementwiseJordanSpace(shape, ctx)`` returns an
  ``EuclideanElementwiseJordanSpace`` when the dtype is real and the geometry
  Euclidean; constructing the Euclidean class directly with any other geometry
  raises ``TypeError``.
* ``TreeSpace`` and ``StackedSpace`` carry exactly the capabilities shared by
  all their leaves (or their base), and re-select them after ``convert``.
* A missing capability is a missing attribute (``AttributeError``).

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   V = sc.DenseVectorSpace((2,), ctx)
   D = sc.DenseCoordinateSpace((2,), ctx)

   assert isinstance(sc.TreeSpace.from_leaf_spaces((V, V)), sc.StarSpace)
   assert not isinstance(sc.TreeSpace.from_leaf_spaces((V, D)), sc.StarSpace)
   assert isinstance(sc.ElementwiseJordanSpace((2,), ctx),
                     sc.EuclideanElementwiseJordanSpace)

   try:
       sc.EuclideanElementwiseJordanSpace(
           (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([1.0, 2.0]))
       )
   except TypeError as exc:
       print(exc)  # EuclideanElementwiseJordanSpace requires EuclideanInnerProduct.

   try:
       D.star(ctx.asarray([1.0, 2.0]))
   except AttributeError as exc:
       print(exc)  # 'DenseCoordinateSpace' object has no attribute 'star'

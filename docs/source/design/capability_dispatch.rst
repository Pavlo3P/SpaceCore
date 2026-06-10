Capability Dispatch
===================

SpaceCore represents structure with Python capabilities. A concrete object may
be a plain coordinate space, an inner-product space, a star space, a Jordan
algebra space, or a Euclidean Jordan algebra space.

Implemented capabilities
------------------------

* ``Space`` owns context and membership checks.
* ``VectorSpace`` adds linear operations.
* ``CoordinateSpace`` adds shape, flattening, and unflattening.
* ``InnerProductSpace`` adds ``inner``, ``norm``, ``riesz``, and
  ``riesz_inverse``.
* ``StarSpace`` adds ``star`` / involution.
* ``JordanAlgebraSpace`` adds ``jordan``, ``spectrum``,
  ``spectral_decompose``, and ``spectral_apply``.
* ``EuclideanJordanAlgebraSpace`` marks Jordan structure compatible with the
  Euclidean inner product.

Concrete dispatch
-----------------

``ProductSpace`` and ``StackedSpace`` inspect component capabilities and return
an internal implementation that preserves only capabilities shared by the
components. Conversion reconstructs components in the target context and then
re-runs this capability selection.

Capabilities may be intentionally absent. ``DenseCoordinateSpace`` has an inner
product but no star operation. ``DenseVectorSpace`` adds star but no Jordan
capability by default. ``ElementwiseJordanSpace`` has Jordan operations;
``EuclideanElementwiseJordanSpace`` is used only for real Euclidean elementwise
geometry.

Refusal examples
----------------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   weights = ctx.asarray([1.0, 2.0])
   try:
       sc.EuclideanElementwiseJordanSpace(
           (2,), ctx, geometry=sc.WeightedInnerProduct(weights)
       )
   except TypeError as exc:
       print(exc)

Expected output:

.. code-block:: text

   EuclideanElementwiseJordanSpace requires EuclideanInnerProduct.

A missing capability also remains missing at runtime:

.. code-block:: python

   X = sc.DenseCoordinateSpace((2,), ctx)
   try:
       X.star(ctx.asarray([1.0, 2.0]))
   except AttributeError as exc:
       print(exc)

Expected output:

.. code-block:: text

   'DenseCoordinateSpace' object has no attribute 'star'

Current versus planned
----------------------

The capabilities listed above are implemented in ``0.3.x``. Additional
specialized structures or solver dispatch rules should be documented as planned
only when they are not present in the public API.

Spaces API
==========

Spaces describe valid elements, array representation, geometry, flattening, and
validation.

Coordinate and vector spaces
----------------------------

.. autosummary::
   :nosignatures:

   spacecore.space.Space
   spacecore.space.VectorSpace
   spacecore.space.CoordinateSpace
   spacecore.space.DenseCoordinateSpace
   spacecore.space.DenseVectorSpace

* ``Space`` owns context and membership checking.
* ``VectorSpace`` adds linear operations on elements.
* ``CoordinateSpace`` adds a finite coordinate shape and flattening.
* ``DenseCoordinateSpace`` represents dense arrays of fixed shape with an inner product.
* ``DenseVectorSpace`` represents one-dimensional dense vectors with star but no Jordan capability by default.

Structured spaces
-----------------

.. autosummary::
   :nosignatures:

   spacecore.space.HermitianSpace
   spacecore.space.TreeSpace
   spacecore.space.TreeElement
   spacecore.space.TreeSpectralDecomposition
   spacecore.space.StackedSpace

* ``HermitianSpace`` represents dense Hermitian or symmetric matrices with Frobenius geometry.
* ``TreeSpace`` is the single structured finite direct-product abstraction. Its
  Python tree records representation, not a tensor product.
* ``TreeElement`` binds ordered leaves to a ``TreeSpace`` and reconstructs the Python value.
* ``StackedSpace`` represents a fixed number of leading-axis copies of one coordinate leaf space.

Inner products and geometries
-----------------------------

.. autosummary::
   :nosignatures:

   spacecore.space.InnerProduct
   spacecore.space.EuclideanInnerProduct
   spacecore.space.WeightedInnerProduct
   spacecore.space.InnerProductSpace

* ``InnerProduct`` defines ``inner`` and Riesz maps.
* ``EuclideanInnerProduct`` is the standard coordinate ``vdot`` geometry.
* ``WeightedInnerProduct`` is a positive diagonal metric with weights stored as a backend array.
* ``InnerProductSpace`` exposes ``inner``, ``norm``, ``riesz``, and ``riesz_inverse``.

Star and Jordan-capable spaces
------------------------------

.. autosummary::
   :nosignatures:

   spacecore.space.StarSpace
   spacecore.space.JordanAlgebraSpace
   spacecore.space.EuclideanJordanAlgebraSpace
   spacecore.space.ElementwiseJordanSpace
   spacecore.space.EuclideanElementwiseJordanSpace

* ``StarSpace`` exposes a canonical involution.
* ``JordanAlgebraSpace`` exposes Jordan products and spectral calculus.
* ``EuclideanJordanAlgebraSpace`` marks Jordan structure compatible with Euclidean geometry.
* ``ElementwiseJordanSpace`` represents coordinatewise multiplication and spectral application.
* ``EuclideanElementwiseJordanSpace`` is the real Euclidean specialization.

Validation
----------

.. autosummary::
   :nosignatures:

   spacecore.space.SpaceCheck
   spacecore.space.SpaceValidationError
   spacecore.space.BackendCheck
   spacecore.space.ShapeCheck
   spacecore.space.FieldCheck
   spacecore.space.DTypeCheck
   spacecore.space.SquareMatrixCheck
   spacecore.space.HermitianCheck

Autodoc
-------

.. autoclass:: spacecore.space.Space
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.VectorSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.CoordinateSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.InnerProduct
   :members:

.. autoclass:: spacecore.space.EuclideanInnerProduct
   :members:

.. autoclass:: spacecore.space.WeightedInnerProduct
   :members:

.. autoclass:: spacecore.space.InnerProductSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.StarSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.JordanAlgebraSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.EuclideanJordanAlgebraSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.DenseCoordinateSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.DenseVectorSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.ElementwiseJordanSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.EuclideanElementwiseJordanSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.HermitianSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.TreeSpace
   :members:
   :inherited-members:
   :exclude-members: treedef, leaf_paths, shape

.. autoclass:: spacecore.space.TreeElement
   :members:
   :exclude-members: space, leaves

.. autoclass:: spacecore.space.TreeSpectralDecomposition
   :members:

.. autoclass:: spacecore.space.StackedSpace
   :members:
   :inherited-members:

.. autoclass:: spacecore.space.SpaceValidationError

.. autoclass:: spacecore.space.SpaceCheck
   :members:

.. autoclass:: spacecore.space.BackendCheck
   :members:

.. autoclass:: spacecore.space.ShapeCheck
   :members:

.. autoclass:: spacecore.space.FieldCheck
   :members:

.. autoclass:: spacecore.space.DTypeCheck
   :members:

.. autoclass:: spacecore.space.SquareMatrixCheck
   :members:

.. autoclass:: spacecore.space.HermitianCheck
   :members:

Spaces tutorial
===============

This tutorial follows ``tutorials/3_Space.ipynb``. It introduces ``Space`` as
the abstraction for the geometry of admissible numerical objects.

Current implemented concrete spaces are:

* ``DenseCoordinateSpace`` for dense arrays with an inner product;
* ``ElementwiseJordanSpace`` and ``DenseVectorSpace`` for Euclidean dense arrays
  with elementwise star, Jordan, and spectral operations;
* ``HermitianSpace`` for Hermitian or symmetric matrices;
* ``ProductSpace`` for Cartesian products of spaces;
* ``StackedSpace`` for repeated copies of a leaf space.

What a Space signifies
----------------------

A ``Space`` represents numerical objects together with their mathematical
structure. It encodes the operations that are valid for elements of that space,
not just the storage format.

A space may encode:

* shape;
* linear structure;
* inner product and norm;
* star operations;
* Jordan algebra operations and spectral calculus;
* projections;
* structural constraints such as Hermitian symmetry;
* product or stacked structure.

Thus a space captures geometry, not merely storage format.

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

Every space stores a Context
----------------------------

Each space carries a ``Context``. This means a space knows under which backend,
dtype, and checking policy its elements live. If ``X`` is a space, then
``X.ctx`` determines which backend arrays are expected, which dtype is targeted,
and whether runtime validation checks are enabled.

Core operations
---------------

.. dropdown:: Common Space methods

   * ``zeros()``
   * ``add(x, y)``
   * ``scale(a, x)``
   * ``axpy(a, x, y)``
   * ``inner(x, y)``
   * ``norm(x)``
   * ``flatten(x)``
   * ``unflatten(v)``

Concrete spaces differ, but they share the same idea: define valid elements,
provide linear operations, define geometry, and create or validate elements in a
backend-aware way. Capability-specific methods such as ``star``, ``jordan``,
``spectrum``, and ``spectral_apply`` are present only on spaces that implement
the corresponding algebraic interfaces.

DenseCoordinateSpace
--------------------

``DenseCoordinateSpace`` represents dense backend arrays with a fixed shape and
an inner-product geometry. If

.. math::

   X = \texttt{DenseCoordinateSpace}(\texttt{shape}=(n_1,\dots,n_k)),

then elements of ``X`` are dense arrays of shape ``(n_1, ..., n_k)`` compatible
with the stored context.

.. code-block:: python

   import numpy as np
   from spacecore.backend import Context, NumpyOps
   from spacecore.space import DenseCoordinateSpace

   ctx = Context(NumpyOps(), dtype=np.float64, enable_checks=True)
   X = DenseCoordinateSpace((2, 3), ctx=ctx)

   x = X.zeros()
   y = ctx.asarray([[1, 2, 3], [4, 5, 6]])

``DenseCoordinateSpace`` supports linear operations and inner products:

.. math::

   x + y,\qquad \alpha x,\qquad \langle x,y\rangle,\qquad \|x\|.

Its default geometry is Euclidean. It also accepts compatible custom inner
products such as ``WeightedInnerProduct``:

.. code-block:: python

   from spacecore.space import WeightedInnerProduct

   weights = ctx.asarray([2.0, 5.0, 11.0])
   X_weighted = DenseCoordinateSpace(
       (3,), ctx=ctx, geometry=WeightedInnerProduct(weights)
   )

Membership is contextual and geometric: correct shape, correct backend
representation, and correct dtype when checks are enabled.

ElementwiseJordanSpace and DenseVectorSpace
-------------------------------------------

``ElementwiseJordanSpace`` is the dense coordinate space whose algebra is
coordinatewise multiplication. It supports star, Jordan products, and spectral
calculus:

.. code-block:: python

   from spacecore.space import ElementwiseJordanSpace

   J = ElementwiseJordanSpace((3,), ctx=ctx)
   x = ctx.asarray([1.0, 2.0, 3.0])
   y = ctx.asarray([4.0, 5.0, 6.0])

   z = J.jordan(x, y)      # coordinatewise product
   s = J.spectrum(x)       # x itself for the elementwise algebra

Elementwise Jordan operations are compatible only with Euclidean inner-product
geometry. Passing a non-Euclidean geometry such as ``WeightedInnerProduct`` to
``ElementwiseJordanSpace`` or ``DenseVectorSpace`` raises ``TypeError``. Use
``DenseCoordinateSpace`` when weighted or otherwise non-Euclidean inner products
matter but star/Jordan operations are not required.

``DenseVectorSpace`` is a backward-compatible one-dimensional alias for the
same Euclidean elementwise Jordan capability. New code should prefer
``DenseCoordinateSpace`` for generic vectors and ``ElementwiseJordanSpace`` when
it needs elementwise algebra explicitly.

HermitianSpace
--------------

``HermitianSpace`` represents Hermitian matrices of fixed square shape. If
``H = HermitianSpace(n)``, then its elements satisfy

.. math::

   A \in \mathbb{F}^{n \times n},
   \qquad
   A = A^*.

This space encodes Hermitian symmetry in addition to shape. Because of that, it
provides geometry-specific operations such as symmetrization, eigendecomposition,
and projection onto the positive semidefinite cone.

Given a square matrix :math:`M`, symmetrization computes

.. math::

   \frac{M + M^*}{2}.

Hermitian eigendecomposition has the form

.. math::

   A = U \operatorname{diag}(\lambda) U^*.

Projection onto the PSD cone clips negative eigenvalues:

.. math::

   U \operatorname{diag}(\lambda) U^*
   \mapsto
   U \operatorname{diag}(\max(\lambda,0)) U^*.

.. code-block:: python

   import numpy as np
   from spacecore.backend import Context, NumpyOps
   from spacecore.space import HermitianSpace

   ctx = Context(NumpyOps(), dtype=np.complex128, enable_checks=True)
   H = HermitianSpace(3, ctx=ctx)

   A = ctx.asarray([
       [1, 1 + 2j, 0],
       [1 - 2j, 3, 4j],
       [0, -4j, 2],
   ])
   H.check_member(A)

ProductSpace
------------

``ProductSpace`` represents a Cartesian product. Tuple elements are the default
representation, with one component per factor space. If

.. math::

   X = X_1 \times \cdots \times X_k,

then the default element representation is

.. math::

   (x_1,\dots,x_k), \qquad x_i \in X_i.

This is useful whenever a variable is naturally block-structured.

Addition, scaling, and supported capability methods are componentwise:

.. math::

   (x_1,\dots,x_k) + (y_1,\dots,y_k)
   =
   (x_1+y_1,\dots,x_k+y_k).

When constructed through ``ProductSpace(...)``, SpaceCore returns the most
specific product subclass supported by all components:

* ``ProductInnerProductSpace`` when every component has an inner product;
* ``ProductStarSpace`` when every component has a star operation;
* ``ProductJordanAlgebraSpace`` when every component has Jordan operations;
* ``ProductEuclideanJordanAlgebraSpace`` when every component is a Euclidean
  Jordan algebra and has a star operation.

If any component lacks a capability, the resulting product does not advertise
that capability. This keeps ``isinstance`` checks truthful.

Flattening moves between any supported product element representation and flat
coordinates:

.. math::

   (x_1,\dots,x_k)
   \mapsto
   \operatorname{concat}(\operatorname{flatten}(x_1),\dots,\operatorname{flatten}(x_k)).

The flat vector depends only on the ordered components, not on whether the
user-facing product element is a tuple or a registered pytree/dataclass.

.. code-block:: python

   from spacecore.space import ProductSpace, DenseCoordinateSpace

   X1 = DenseCoordinateSpace((2,), ctx=ctx)
   X2 = DenseCoordinateSpace((3,), ctx=ctx)
   X_prod = ProductSpace((X1, X2), ctx=ctx)

   x = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0]))
   flat = X_prod.flatten(x)
   x_back = X_prod.unflatten(flat)

Structured product elements are available by building from a registered pytree
template. For dataclasses used with JAX tree utilities, register the dataclass
as a pytree; bare dataclasses are opaque leaves and are not automatically
treated as product components.

.. code-block:: python

   from dataclasses import dataclass
   import jax
   from spacecore.space import ProductSpace, DenseCoordinateSpace

   @jax.tree_util.register_pytree_node_class
   @dataclass(frozen=True)
   class State:
       position: object
       velocity: object

       def tree_flatten(self):
           return (self.position, self.velocity), None

       @classmethod
       def tree_unflatten(cls, aux, children):
           return cls(*children)

   X1 = DenseCoordinateSpace((2,), ctx=ctx)
   X2 = DenseCoordinateSpace((2,), ctx=ctx)
   template = State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0]))
   X_state = ProductSpace.from_template((X1, X2), template, ctx=ctx)

   x = State(ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0]))
   y = X_state.scale(2.0, x)  # returns State, not a tuple
   flat = X_state.flatten(x)

StackedSpace
------------

``StackedSpace(base, count)`` represents ``count`` independent copies of a leaf
space stacked on a leading axis. Like products, construction returns the most
specific stacked subclass supported by the base space, for example
``StackedInnerProductSpace`` or ``StackedEuclideanJordanAlgebraSpace``.

For product spaces, call ``ProductSpace(...).stacked(count)``. This stacks each
component instead of wrapping the product as a single leaf.

Choosing the right space
------------------------

Use ``DenseCoordinateSpace`` for generic dense arrays, including weighted or
custom inner-product geometry. Use ``ElementwiseJordanSpace`` when dense arrays
need coordinatewise star, Jordan, or spectral operations. Use
``HermitianSpace`` when Hermitian structure matters, ``ProductSpace`` when
variables are naturally block-structured, and ``StackedSpace`` for repeated
copies of a leaf space.

Summary
-------

``Space`` turns raw arrays into explicit geometric objects. It stores a context,
defines valid elements, and provides the operations algorithms should use. The
public hierarchy advertises only the capabilities each concrete space actually
supports.

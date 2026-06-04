Spaces tutorial
===============

This tutorial follows ``tutorials/3_Space.ipynb``. It introduces ``Space`` as
the abstraction for the geometry of admissible numerical objects.

Current implemented spaces are:

* ``DenseCoordinateSpace``
* ``HermitianSpace``
* ``ProductSpace``

What a Space signifies
----------------------

A ``Space`` represents a Hilbert space of numerical objects. It encodes the
geometric structure of its elements together with operations that are natural on
them.

A space may encode:

* shape;
* linear structure;
* inner product and norm;
* projections;
* structural constraints such as Hermitian symmetry;
* product structure.

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
   * ``apply(x, f)``

Concrete spaces differ, but they share the same idea: define valid elements,
provide linear operations, define geometry, and create or validate elements in a
backend-aware way.

DenseCoordinateSpace
--------------------

``DenseCoordinateSpace`` represents dense backend arrays with a fixed shape and
Euclidean geometry. If

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

A vector space supports the standard operations

.. math::

   x + y,\qquad \alpha x,\qquad \langle x,y\rangle,\qquad \|x\|.

Membership is contextual and geometric: correct shape, correct backend
representation, and correct dtype when checks are enabled.

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

Addition, scaling, and inner products are componentwise:

.. math::

   (x_1,\dots,x_k) + (y_1,\dots,y_k)
   =
   (x_1+y_1,\dots,x_k+y_k).

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

Choosing the right space
------------------------

Use ``DenseCoordinateSpace`` for generic dense arrays, ``HermitianSpace`` when Hermitian
structure matters, and ``ProductSpace`` when variables are naturally
block-structured.

Summary
-------

``Space`` turns raw arrays into explicit geometric objects. It stores a context,
defines valid elements, and provides the operations algorithms should use.

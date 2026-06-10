LinOp tutorial
==============

This tutorial follows ``tutorials/4_LinOp.ipynb``. It introduces ``LinOp`` as
the abstraction for linear maps between spaces.

Current implemented operator types are:

* ``DenseLinOp``
* ``DiagonalLinOp``
* ``SparseLinOp``
* ``MatrixFreeLinOp``
* ``IdentityLinOp``
* ``ZeroLinOp``
* ``ScaledLinOp``
* ``SumLinOp``
* ``ComposedLinOp``
* ``BlockDiagonalLinOp``
* ``StackedLinOp``
* ``SumToSingleLinOp``

What a LinOp signifies
----------------------

Let :math:`A` be a linear map

.. math::

   A : X \to Y.

``LinOp`` encodes this map together with the numerical policy under which it is
applied. It is not just an array or matrix. It is a map equipped with:

* a domain space;
* a codomain space;
* a context;
* methods for forward and adjoint application.

The geometry is explicit:

.. math::

   x \in X \mapsto Ax \in Y.

Core ingredients
----------------

A linear operator is organized around ``(X, Y, A)``:

* ``op.dom`` is the domain :math:`X`;
* ``op.cod`` is the codomain :math:`Y`;
* ``apply(x)`` computes the forward action;
* ``rapply(y)`` computes the adjoint action.

The expected geometry is:

.. math::

   \texttt{apply}: X \to Y,
   \qquad
   \texttt{rapply}: Y \to X.

In finite-dimensional real or complex spaces, ``rapply`` represents the adjoint
operator :math:`A^* : Y \to X`, satisfying

.. math::

   \langle Ax, y\rangle_Y = \langle x, A^*y\rangle_X.

Vectorized lifting
------------------

For a leading-axis batch of elements, use ``vapply`` or ``rvapply`` to lift the
operator:

.. math::

   A : X \to Y,
   \qquad
   A^{(B)} : X^B \to Y^B.

.. code-block:: python

   B = 8
   xs = ctx.asarray(np.ones((B,) + X.shape))
   ys = op.vapply(xs)
   xs_back = op.rvapply(ys)

This is equivalent to stacking scalar applications:

.. code-block:: python

   ys_ref = ctx.ops.stack(tuple(op.apply(x) for x in xs), axis=0)

The base fallback uses backend ``vmap``. Structured operators override this
path when they can use matrix multiplication, sparse multi-vector products,
broadcasting, or componentwise product-space vectorization.

DenseLinOp
----------

``DenseLinOp`` is the standard dense linear operator. If :math:`A : X \to Y`,
then dense storage is a tensor compatible with domain and codomain shapes.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)

   X = sc.DenseCoordinateSpace((2,), ctx=ctx)
   Y = sc.DenseCoordinateSpace((3,), ctx=ctx)

   A = np.array([
       [1.0, 2.0],
       [0.0, -1.0],
       [3.0, 1.0],
   ])

   op = sc.DenseLinOp(A, X, Y, ctx=ctx)

Forward application computes :math:`x \mapsto Ax`; adjoint application computes
:math:`y \mapsto A^*y`.

SparseLinOp
-----------

``SparseLinOp`` stores a sparse backend matrix and uses it for forward and
adjoint actions. It represents the same kind of map, :math:`A : X \to Y`, but
the internal storage is sparse rather than dense. Use it when sparsity is part
of the operator structure.

.. code-block:: python

   import scipy.sparse as sp

   A_sparse = sp.csr_matrix(np.array([
       [1.0, 0.0],
       [0.0, -1.0],
       [3.0, 0.0],
   ]))

   op_sparse = sc.SparseLinOp(A_sparse, X, Y, ctx=ctx)

MatrixFreeLinOp
---------------

``MatrixFreeLinOp`` stores callables for forward and adjoint actions instead
of matrix entries. Use it when a linear map has a fast procedural
implementation or when materializing a matrix is too expensive.

.. code-block:: python

   def apply(x):
       return ctx.asarray([x[0] + x[1], x[0] - x[1]])

   def rapply(y):
       return ctx.asarray([y[0] + y[1], y[0] - y[1]])

   op_free = sc.MatrixFreeLinOp(apply, rapply, X, X, ctx=ctx)

Canonical and algebraic operators
---------------------------------

``IdentityLinOp`` and ``ZeroLinOp`` represent the canonical identity and zero
maps on spaces. Operator algebra creates lazy operators without immediately
materializing dense storage:

.. code-block:: python

   I = sc.IdentityLinOp(X, ctx=ctx)
   Z = sc.ZeroLinOp(X, Y, ctx=ctx)

   scaled = 2.0 * I                 # ScaledLinOp
   summed = I + scaled              # SumLinOp
   composed = summed @ I            # ComposedLinOp

The helper constructors ``make_scaled``, ``make_sum``, and ``make_composed``
perform the same simplifications used by the Python operators.

Product operators
-----------------

Product operators compose linear maps over product spaces:

.. dropdown:: Product LinOp classes

   * ``BlockDiagonalLinOp`` applies independent component operators.
   * ``StackedLinOp`` maps one domain into a product codomain.
   * ``SumToSingleLinOp`` maps a product domain into one codomain by summing
     component outputs.

``BlockDiagonalLinOp`` is built from operators :math:`A_i : X_i \to Y_i` and
acts componentwise:

.. math::

   (x_1,\dots,x_k)
   \mapsto
   (A_1x_1,\dots,A_kx_k).

``StackedLinOp`` stacks operators with the same domain,
:math:`A_i : X \to Y_i`, into one operator
:math:`X \to Y_1 \times \cdots \times Y_k`:

.. math::

   x \mapsto (A_1x,\dots,A_kx).

The result is a product element whose representation follows the codomain
``ProductSpace``. This is a tuple for the default ``TupleStructure`` and the
registered pytree/dataclass type when the codomain was built with
``ProductSpace.from_template(...)`` or an explicit ``PytreeStructure``.

Its adjoint combines contributions:

.. math::

   (y_1,\dots,y_k) \mapsto A_1^*y_1 + \cdots + A_k^*y_k.

``SumToSingleLinOp`` combines operators with a common codomain,
:math:`A_i : X_i \to Y`, into one operator
:math:`X_1 \times \cdots \times X_k \to Y`:

.. math::

   (x_1,\dots,x_k) \mapsto A_1x_1 + \cdots + A_kx_k.

Its adjoint sends one output element back to a product element:

.. math::

   y \mapsto (A_1^*y,\dots,A_k^*y).

As with spaces, tuple is only the default product representation; structured
domain products preserve their registered pytree/dataclass representation.

.. code-block:: python

   from dataclasses import dataclass
   import jax
   from spacecore.linop import IdentityLinOp
   from spacecore.linop.product import StackedLinOp
   from spacecore.space import ProductSpace, DenseCoordinateSpace

   @jax.tree_util.register_pytree_node_class
   @dataclass(frozen=True)
   class Pair:
       left: object
       right: object

       def tree_flatten(self):
           return (self.left, self.right), None

       @classmethod
       def tree_unflatten(cls, aux, children):
           return cls(*children)

   X = DenseCoordinateSpace((2,), ctx=ctx)
   template = Pair(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0]))
   Y = ProductSpace.from_template((X, X), template, ctx=ctx)
   op = StackedLinOp(X, Y, (IdentityLinOp(X), IdentityLinOp(X)), ctx=ctx)

   y = op.apply(ctx.asarray([1.0, 2.0]))  # returns Pair

Defining a custom operator
--------------------------

Use existing concrete operator classes when you have dense or sparse storage.
Subclass ``LinOp`` when the map has special structure and you do not want to
materialize a matrix.

.. code-block:: python

   from spacecore.linop import LinOp

   class ScaleLinOp(LinOp):
       def __init__(self, alpha, dom, cod=None, ctx=None):
           if cod is None:
               cod = dom
           super().__init__(dom=dom, cod=cod, ctx=ctx)
           self.alpha = alpha

       def apply(self, x):
           return self.cod.scale(self.alpha, x)

       def rapply(self, y):
           return self.dom.scale(self.alpha, y)

Summary
-------

``LinOp`` turns raw matrix-like storage and structured block maps into explicit
geometric linear operators. It stores a domain, codomain, context, and forward
and adjoint actions.

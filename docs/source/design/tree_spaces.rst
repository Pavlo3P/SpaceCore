Tree-structured spaces
======================

``TreeSpace(treedef, leaf_spaces, ctx=None, check_level=None)`` is the finite
direct product :math:`X = \prod_{\ell} X_\ell` of coordinate spaces; an element
is a Python tree with one array per leaf. Coordinates of different leaves are
concatenated, not multiplied: this is not a tensor or Kronecker product.

* ``treedef`` is an ``optree.PyTreeSpec`` or an example element (see below).
  Leaves follow optree order, so dictionary keys are sorted. ``optree`` is a
  required dependency.
* ``leaf_spaces`` must be coordinate spaces (``TypeError``), one per leaf
  (``ValueError``).
* ``TreeSpace.from_leaf_spaces((X, S), ctx)`` builds the flat tuple case.
* ``ctx`` defaults to the context resolved from the leaves; every leaf is
  converted to it, so the tree has one ``dtype`` and ``field``. ``convert``
  rebuilds every leaf and keeps the structure.
* ``check_level`` defaults to the least strict leaf level. ``"cheap"`` checks the
  tree structure and each leaf's backend, shape, field and dtype; higher levels
  add the leaves' own checks. Errors name the leaf path.
* Vector operations, ``flatten``/``unflatten`` and batching act leaf by leaf. An
  inner-product, star, Jordan or Euclidean-Jordan capability is offered only when
  every leaf has it; the inner product is the sum of the leaf inner products.
* ``T.element(value)`` wraps a tree as a ``TreeElement`` bound to ``T``;
  ``.value`` returns the plain tree. Plain trees are the normal representation.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   S = sc.DenseCoordinateSpace((1,), ctx)

   T = sc.TreeSpace({"model": (0, {"bias": 0})}, (X, S), ctx=ctx)
   x = {"model": (ctx.asarray([1.0, 2.0]), {"bias": ctx.asarray([3.0])})}
   print(T.scale(2.0, x))
   print(T.leaf_paths)

   pair = sc.TreeSpace.from_leaf_spaces((X, S), ctx)
   pair.check_member((ctx.asarray([1.0, 2.0]), ctx.asarray([3.0])))

The template is an example element
----------------------------------

An element of a tree space is a plain Python container — tuple, dict,
``NamedTuple``, or any nesting of them — with one array per leaf. You describe the
space by handing it one such container, ``TreeSpace.from_template(UV(0, 0), (U, V))``
or ``TreeSpace(UV(0, 0), (U, V))``. SpaceCore reads only its structure; the leaf
values are placeholders and are discarded.

The leaf spaces already fix each leaf's shape, dtype and geometry. The template
supplies the one thing they cannot: the container.

* **Results keep the type.** ``zeros``, ``add``, ``scale``, gradients and operator
  outputs come back as ``UV``, so code reads ``z.u`` and ``z.v`` instead of
  ``z[0]`` and ``z[1]``.
* **Leaves pair with leaf spaces in order.** The k-th leaf of the template, in
  optree order, is an element of ``leaf_spaces[k]``; ``T.leaf_paths`` lists the
  order.
* **Membership checks the container first.** At ``"cheap"`` and above, a value
  with another structure is rejected before any leaf is looked at: a plain tuple
  is not a member of a ``UV`` space, even with the right arrays.

``T.element(value)`` wraps a tree as a ``TreeElement`` bound to ``T``, and
``.value`` returns the plain tree. It has no arithmetic; plain trees are the
normal representation, and the template is how the space knows what they look
like.

.. code-block:: python

   from typing import NamedTuple

   class UV(NamedTuple):
       u: object
       v: object

   U = sc.DenseCoordinateSpace((2,), ctx)
   V = sc.DenseCoordinateSpace((3,), ctx)
   P = sc.TreeSpace.from_template(UV(0, 0), (U, V), ctx=ctx)

   z = P.zeros()
   assert isinstance(z, UV) and z.u.shape == (2,) and z.v.shape == (3,)
   print(P.leaf_paths)                            # ((0,), (1,)): u, then v

   for wrong in [(z.u, z.v),                      # right arrays, wrong container
                 UV(z.v, z.u)]:                   # right container, swapped leaves
       try:
           P.check_member(wrong)
       except sc.SpaceValidationError as err:
           print(err)

Block operators
---------------

Block operators map between tree spaces they infer from their blocks; all blocks
must share one context and one check level.

* ``BlockDiagonalLinOp(blocks)``: ``blocks`` is a tree of ``LinOp``; the domain
  tree comes from the block domains, the codomain tree from the block codomains.
* ``BlockMatrixLinOp(block_rows)``: a nonempty rectangular sequence of rows,
  ``y_i = sum_j A_ij x_j`` for ``A_ij : X_j -> Y_i``. Blocks in a row share a
  codomain and blocks in a column share a domain (``ValueError`` otherwise). A
  side given as a one-element list or tuple is a single space, not a one-leaf
  tree: ``[[A1], [A2]]`` maps ``x -> (A1 x, A2 x)``.
* The adjoint transposes the layout and replaces each block by its ``.H``, so
  ``rapply`` uses each leaf's metric adjoint.

.. code-block:: python

   Y = sc.DenseCoordinateSpace((3,), ctx)
   A = sc.DenseLinOp(ctx.asarray(np.ones((3, 2))), X, Y, ctx)
   B = sc.DenseLinOp(ctx.asarray(np.ones((3, 1))), S, Y, ctx)
   C = sc.DenseLinOp(ctx.asarray(np.eye(2)), X, X, ctx)
   D = sc.DenseLinOp(ctx.asarray(np.zeros((2, 1))), S, X, ctx)
   x0, x1 = ctx.asarray([1.0, 2.0]), ctx.asarray([3.0])

   print(sc.BlockDiagonalLinOp((A, C)).apply((x0, x0)))   # (A x0, C x0)
   y0, y1 = sc.BlockMatrixLinOp(((A, B), (C, D))).apply((x0, x1))
   print(y0, y1)                                            # A x0 + B x1, C x0 + D x1

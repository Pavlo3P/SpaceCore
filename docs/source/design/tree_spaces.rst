Tree-structured spaces
======================

``TreeSpace`` represents a finite direct product

.. math::

   X = \prod_{\ell \in L} X_\ell.

Each leaf is an ordinary finite-coordinate SpaceCore space. The tree records how
the corresponding element is organized in Python. This is a Cartesian/direct
product, not a tensor product: coordinates from different leaves are concatenated,
not multiplied together.

SpaceCore uses ``optree`` as a required dependency for deterministic traversal,
structure comparison, reconstruction, and leaf paths. Mathematical operations,
context conversion, validation, geometry, and batching remain SpaceCore-owned.

Creating tree spaces
--------------------

The first argument may be an ``optree.PyTreeSpec`` or an example tree whose
leaves define the structure. Dictionary leaves use deterministic sorted-key
order.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   S = sc.DenseCoordinateSpace((1,), ctx)

   tuple_tree = sc.TreeSpace((0, 0), (X, S), ctx=ctx)
   dict_tree = sc.TreeSpace({"point": 0, "weight": 0}, (X, S), ctx=ctx)
   nested_tree = sc.TreeSpace(
       {"model": (0, {"bias": 0})},
       (X, S),
       ctx=ctx,
   )

Tree elements
-------------

Plain Python trees are the normal element representation. ``TreeElement`` is an
optional wrapper that explicitly binds ordered leaves to their ``TreeSpace``.
Use ``element`` to create that wrapper and ``value`` to reconstruct its Python
tree.

.. code-block:: python

   x = {"model": (ctx.asarray([1.0, 2.0]), {"bias": ctx.asarray([3.0])})}
   y = nested_tree.scale(2.0, x)

   print(y)
   print(nested_tree.leaf_paths)

``zero``/``zeros``, ``add``, ``scale``, dense coordinate
``flatten``/``unflatten``, and structure-of-batched-leaves operations are
leafwise. If every leaf is an ``InnerProductSpace``, the tree also advertises
that capability, sums leaf inner products, and uses the induced product norm.
The same intersection rule applies to star, Jordan, and Euclidean-Jordan
capabilities.

Validation and dtype behavior
-----------------------------

``check_level="cheap"`` validates tree structure and each leaf's backend,
shape, field, and exact dtype contract. Higher levels also run the corresponding
leaf-space mathematical checks. Errors identify the deterministic leaf path.

``TreeSpace`` resolves one context for all leaves. Its public ``dtype`` and
``field`` are inherited uniformly from those converted leaf spaces, while each
leaf remains responsible for exact membership checks. Conversion rebuilds every
leaf space in the target context and preserves the optree structure.

Tuple-style direct products
---------------------------

Use ``TreeSpace.from_leaf_spaces`` for the common flat tuple case. This is still
a finite Cartesian/direct product, not a tensor product.

.. code-block:: python

   pair = sc.TreeSpace.from_leaf_spaces((X, S), ctx)
   value = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0]))
   pair.check_member(value)

Block-structured linear operators
---------------------------------

``BlockDiagonalLinOp`` infers a ``TreeSpace`` domain from the domains of its
blocks and a ``TreeSpace`` codomain from their codomains. The block tree also
defines the Python structure of input and output elements.

.. code-block:: python

   A_diag = sc.BlockDiagonalLinOp((A, D))
   y0, y1 = A_diag.apply((x0, x1))

``BlockMatrixLinOp`` accepts a nonempty rectangular sequence of block rows. If

.. math::

   A : X_0 \to Y_0,\quad B : X_1 \to Y_0,\quad
   C : X_0 \to Y_1,\quad D : X_1 \to Y_1,

then the following operator maps ``X_0 x X_1`` to ``Y_0 x Y_1`` and computes
``(A x_0 + B x_1, C x_0 + D x_1)``.

.. code-block:: python

   block = sc.BlockMatrixLinOp(((A, B), (C, D)))
   y0, y1 = block.apply((x0, x1))

Rows must share compatible codomains and columns must share compatible domains.
The adjoint transposes the block layout and replaces each block by ``.H``.
Consequently, ``rapply`` and ``.H.apply`` use each leaf space's metric adjoint,
not merely the coordinate conjugate transpose when weighted or other
non-Euclidean inner products are present.

These constructions are finite direct-product block operators. They do not
construct tensor products, Kronecker products, or a ``ProductSpace``.

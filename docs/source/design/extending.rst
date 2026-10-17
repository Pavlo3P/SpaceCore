Extending SpaceCore
===================

A new operator, functional, space or set needs three things: its check-free
cores, a declaration of its constructor fields, and nothing else unless it holds
state its fields cannot rebuild.

Cores, not public methods
-------------------------

``LinOp`` owns the checked public methods ``apply``, ``rapply``, ``vapply`` and
``rvapply``. Each validates its input at the object's ``check_level`` and calls a
check-free core: ``_apply_core``, ``_rapply_core``, ``_vapply_core`` and
``_rvapply_core``. ``Functional`` does the same for ``value`` and ``vvalue``
through ``_value_core`` and ``_vvalue_core``. Spaces follow the same split:
``VectorSpace`` owns the checked ``add`` and ``scale`` over the abstract
``_add_core`` and ``_scale_core``, and ``InnerProductSpace`` owns the checked
``inner`` and ``real_inner`` over ``_inner_core``, which defaults to the space's
geometry.

A subclass writes the cores. Composite operators, the tree operators and the
``linalg`` solvers call the cores of their parts directly, so validation runs once
at the outer boundary; a core must therefore not validate. The batched cores
default to ``vmap`` of the single-element core, with a one-time warning, so a
subclass writes them only when it has a genuinely batched implementation.

A subclass may still override a public method instead; its core then calls that
method back. A class that defines neither raises ``NotImplementedError`` on use.

Requirements that hold at every check level, such as "an adjoint needs inner
products", are capability requirements, not validation: raise them in the core
with ``require(...)`` (``MatrixFreeLinOp`` does this), or they disappear at
``check_level="none"``.

Fields
------

A class lists the constructor keywords that rebuild it:

.. code-block:: python

   _fields = ("A", "dom", "cod", "ctx")   # "param:attr" when the attribute is named differently
   _dynamic = ("A",)                       # the pytree children; the rest is static aux

From that declaration the bases derive three things:

* **pytree** — ``tree_flatten`` returns the ``_dynamic`` fields as children and the
  rest as ``(field, value)`` pairs; ``tree_unflatten`` calls the constructor with
  them, keeping the exact class so that a traced rebuild re-runs its invariants;
* **conversion** — ``convert(new_ctx)`` rebuilds from the fields, replacing ``ctx``,
  converting every value that has a ``convert`` method (spaces, geometries, other
  operators, tuples of them) and moving dense arrays with ``new_ctx.asarray``; other
  values pass through. A class whose ``__new__`` chooses a subclass (``StackedSpace``,
  ``TreeSpace``, ``ElementwiseJordanSpace``) is rebuilt through that factory, so a
  real Euclidean space converted to complex becomes the general one;
* **equality** (operators and functionals) — two objects of the same type and
  mathematical context are equal when their fields are: arrays by shape and
  ``allclose`` (NaN matching NaN), user callables by identity, everything else by
  ``scalar_eq``. ``ctx`` and ``check_level`` are ignored.

A ``PyTreeNode`` that declares no fields and does not implement ``tree_flatten``
refuses construction; that is what keeps ``Functional`` and ``QuadraticForm``
abstract.

When to write the hooks yourself
--------------------------------

Override ``tree_flatten``/``tree_unflatten``, ``_convert`` or ``__eq__`` only when
the fields cannot express the object:

* **derived or cached state** — precomputed matrices, factorizations, a chosen
  computation mode. Keep it out of ``_fields`` and rebuild it in ``__init__``;
  flattening it would let a traced rebuild resurrect a stale cache.
* **arrays that need more than** ``asarray`` — ``SparseLinOp`` converts its matrix
  with ``assparse`` and a dtype cast.
* **a context that is not the target** — ``RealifiedFunctional`` (in
  ``spacecore-optimize``) converts its base functional to the complex partner of
  the target dtype.
* **a structure that is not a constructor argument** — ``BlockMatrixLinOp`` and
  ``BlockDiagonalLinOp`` flatten their blocks and keep the tree definitions as aux.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   class ScalarLinOp(sc.LinOp):
       """x -> c x on one space, for a real scalar c."""

       _fields = ("c", "space:domain", "ctx")

       def __init__(self, c, space, ctx=None, check_level=None):
           super().__init__(space, space, ctx, check_level=check_level)
           self.c = c

       def _apply_core(self, x):
           return self.c * x

       def _rapply_core(self, y):
           return self.c * y           # real c: the metric adjoint is c itself


   class HalfSquaredNorm(sc.Functional):
       """f(x) = <x, x> / 2, whose gradient in the space's own geometry is x."""

       _fields = ("dom", "ctx")

       def __init__(self, dom, ctx=None, check_level=None):
           super().__init__(dom, ctx, check_level=check_level)

       def _value_core(self, x):
           return 0.5 * self.domain.inner(x, x)

       def grad(self, x):
           return x


   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((3,), ctx)
   A = ScalarLinOp(2.0, X)
   f = HalfSquaredNorm(X)
   x = ctx.asarray([1.0, -2.0, 0.5])

   A(x), A.H(x)            # checked public methods, inherited
   (f @ A)(x)              # composition calls the cores
   A == ScalarLinOp(2.0, X)                    # True: equal fields
   A.convert(sc.Context(sc.JaxOps(), dtype=np.float32))   # rebuilt from the fields

Both classes work under ``jax.jit`` as pytrees without further code.

Related: :doc:`context_dtype_conversion`, :doc:`checking_policy`, :doc:`performance_and_jit`.

Extending SpaceCore
===================

A new operator, functional or space needs its check-free cores and a ``_fields``
declaration; it needs hand-written hooks only for state its fields cannot rebuild.
A custom ``Set`` has no cores: it overrides ``contains``/``project`` directly
(:doc:`proximal_contract`).

Cores, not public methods
-------------------------

**You implement the cores; SpaceCore owns the public methods.** Each public method
validates at the object's ``check_level``, then calls its check-free core.

==========================  ==============================================  ===========================================
Base class                  Public (checked, inherited)                     Core (you write)
==========================  ==============================================  ===========================================
``LinOp``                   ``apply``, ``rapply``                           ``_apply_core``, ``_rapply_core``
                            ``vapply``, ``rvapply``                         ``_vapply_core``, ``_rvapply_core``
``Functional``              ``value``, ``grad``                             ``_value_core``, ``_grad_core``
                            ``value_and_grad``                              ``_value_and_grad_core``
                            ``differential``, ``derivative_at``             ``_differential_core``, ``_derivative_core``
                            ``value_and_derivative``                        ``_value_and_derivative_core``
                            ``vvalue``, ``vgrad``                           ``_vvalue_core``, ``_vgrad_core``
``VectorSpace``             ``add``, ``scale``, ``axpy``                    ``_add_core``, ``_scale_core``, ``_axpy_core``
``InnerProductSpace``       ``inner``, ``real_inner``, ``norm``             ``_inner_core``, ``_real_inner_core``, ``_norm_core``
                            ``riesz``, ``riesz_inverse``                    ``_riesz_core``, ``_riesz_inverse_core``
==========================  ==============================================  ===========================================

Write only the cores your object needs; the rest have defaults. A functional
needs ``_value_core`` and, if it is differentiable, ``_grad_core``;
``_value_and_grad_core`` defaults to calling both, and the differential and
derivative cores are built from ``_grad_core``. A ``LinOp`` needs ``_apply_core``
and, for an adjoint, ``_rapply_core``. The batched cores default to ``vmap`` of
the single-element core, with a one-time warning; write them only for a genuinely
batched implementation. ``InnerProductSpace`` cores default to
the space's geometry.

``_grad_core`` returns the **Riesz gradient** in the domain's own geometry, not
the coordinate derivative. When you compute the coordinate derivative, finish
with ``self.domain.riesz_inverse(...)``; on a Euclidean space that is the
identity.

A core receives valid inputs and must not validate them. Composites, tree
operators and ``linalg`` solvers call their parts' cores directly, so validation
runs once, at the outer boundary.

Do not override a public method. An undecorated override is treated as that
class's core and wrapped in the base's checks; an override that calls
``super().<name>`` is left alone. A class that
defines neither the core nor a default raises ``NotImplementedError`` on use.

Capability requirements (e.g. "an adjoint needs inner products") must hold at every
check level, so raise them in the core, not as validation: on a ``LinOp`` call
``self._require(capability, name)``, as ``MatrixFreeLinOp._rapply_core`` does.

Fields
------

A class lists the constructor keywords that rebuild it:

.. code-block:: python

   _fields = ("A", "dom", "cod", "ctx")   # "param:attr" when the attribute is named differently
   _dynamic = ("A",)                       # the pytree children; the rest is static aux

From it the bases derive:

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

A ``PyTreeNode`` with no ``_fields`` and no ``tree_flatten`` refuses construction
(this keeps ``Functional`` and ``QuadraticForm`` abstract).

When to write the hooks yourself
--------------------------------

Override ``tree_flatten``/``tree_unflatten``, ``_convert`` or ``__eq__`` only when
the fields cannot express the object:

* **derived or cached state** — precomputed matrices, factorizations, a chosen
  computation mode. Keep it out of ``_fields`` and rebuild it in ``__init__``, so a
  traced rebuild cannot resurrect a stale cache.
* **arrays that need more than** ``asarray`` — ``SparseLinOp`` converts its matrix
  with ``assparse`` and a dtype cast.
* **a structure that is not a constructor argument** — ``BlockMatrixLinOp`` and
  ``BlockDiagonalLinOp`` flatten their blocks and keep the tree definitions as aux.

Custom space, checks and geometry
---------------------------------

A space is the minimum a value must satisfy, so custom validation lives in the
space as ``SpaceCheck`` objects; the geometry is a separate ``InnerProduct``.

* **A check** subclasses the frozen dataclass ``sc.SpaceCheck``: ``is_valid(space, x)``
  returns a bool, ``error_message(space, x)`` the diagnostic, and the class attribute
  ``minimum_level`` (``"cheap"``, ``"standard"`` or ``"strict"``) picks the levels it
  runs at. Every checked public method of the space (``add``, ``inner``, ``riesz``,
  ...) and every operator or functional on it runs the check on its inputs.
* **A space** attaches checks with a class attribute ``checks = (...)`` or, when they
  depend on instance state, a ``_local_checks(self)`` method. Each class in the MRO
  contributes its own, so a subclass adds to its parent's checks rather than
  replacing them. List the constructor keywords in ``_fields`` and extend
  ``_eq_algebra`` when a new field changes the space.
* **A geometry** subclasses ``sc.InnerProduct`` and implements ``inner``, ``riesz``
  (vector to covector) and ``riesz_inverse``, all taking ``ops`` first and
  broadcasting over leading batch axes. It is not a checked object: the space's
  public methods check, then call it. Give it ``__eq__`` (spaces compare their
  geometries), ``convert`` when it stores arrays, and optionally
  ``validate_for(space)``. See :doc:`geometry`.

The example below is the subspace of zero-mean vectors in R^3 with a weighted
inner product.

.. code-block:: python

   from dataclasses import dataclass
   import numpy as np
   import spacecore as sc


   @dataclass(frozen=True)
   class ZeroMeanCheck(sc.SpaceCheck):
       """Membership: the entries sum to zero within atol."""

       name: str = "zero-mean"
       atol: float = 1e-12
       minimum_level = "standard"          # runs at "standard" and "strict"

       def is_valid(self, space, x):
           return bool(abs(space.ops.sum(x)) <= self.atol)

       def error_message(self, space, x):
           return f"entries must sum to 0 (|sum| <= {self.atol})"


   class ZeroMeanSpace(sc.DenseCoordinateSpace):
       """Vectors in R^n whose entries sum to zero: a subspace, so add/scale stay inside."""

       _fields = ("shape", "atol", "ctx", "geometry")

       def __init__(self, shape, ctx=None, geometry=None, atol=1e-12, check_level=None):
           self.atol = atol                 # before super(): checks may be built during init
           super().__init__(shape, ctx, geometry=geometry, check_level=check_level)

       def _local_checks(self):             # added to DenseCoordinateSpace's shape/dtype checks
           return (ZeroMeanCheck(atol=self.atol),)

       def _eq_algebra(self, other):        # equal spaces: equal shape, geometry and atol
           return super()._eq_algebra(other) and self.atol == other.atol


   class Weighted(sc.InnerProduct):
       """<x, y> = sum_i w_i x_i y_i with positive weights w."""

       def __init__(self, w):
           self.w = w

       def inner(self, ops, x, y):
           return ops.sum(self.w * x * y)

       def riesz(self, ops, x):             # vector -> covector; broadcasts over batch axes
           return self.w * x

       def riesz_inverse(self, ops, x):     # covector -> vector
           return x / self.w

       def validate_for(self, space):
           if tuple(np.shape(self.w)) != space.shape:
               raise ValueError("weights must have the space's shape")

       def __eq__(self, other):             # spaces compare their geometries
           return isinstance(other, Weighted) and np.allclose(self.w, other.w)

       __hash__ = None

       def convert(self, ctx):              # move the stored array with the space
           return Weighted(ctx.asarray(self.w))


   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   w = ctx.asarray([1.0, 2.0, 4.0])
   Z = ZeroMeanSpace((3,), ctx, geometry=Weighted(w), check_level="standard")

   x = ctx.asarray([1.0, -2.0, 1.0])
   Z.check_member(x)                                 # passes
   try:
       Z.add(x, ctx.asarray([1.0, 0.0, 0.0]))        # second argument is not zero-mean
   except sc.SpaceValidationError as err:
       print(err)

   cheap = ZeroMeanSpace((3,), ctx, check_level="cheap")
   cheap.check_member(ctx.asarray([1.0, 0.0, 0.0]))  # "cheap" skips the standard-level check

   print(Z.inner(x, x), Z.riesz(x))                  # 1 + 8 + 4 = 13, w * x
   g = sc.SquaredL2NormFunctional(Z).grad(x)         # Riesz gradient in Z's geometry
   Z2 = Z.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))
   assert isinstance(Z2, ZeroMeanSpace) and Z2.atol == Z.atol
   assert Z == ZeroMeanSpace((3,), ctx, geometry=Weighted(w))
   assert Z != ZeroMeanSpace((3,), ctx, geometry=Weighted(w), atol=1e-6)

The check runs at ``"standard"`` and above, after ``DenseCoordinateSpace``'s
backend, shape, field and dtype checks. ``SquaredL2NormFunctional(Z).grad`` and
``DenseLinOp`` adjoints on ``Z`` use ``Weighted`` without further code.

Example: operator and functional
--------------------------------

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

       def _grad_core(self, x):
           return x                    # Riesz gradient of <x, x>/2 in any geometry


   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((3,), ctx)
   A = ScalarLinOp(2.0, X)
   f = HalfSquaredNorm(X)
   x = ctx.asarray([1.0, -2.0, 0.5])

   A(x), A.H(x)            # checked public methods, inherited
   f.grad(x)               # checked public grad, calls _grad_core
   (f @ A)(x), (f @ A).grad(x)   # composition calls the cores
   A == ScalarLinOp(2.0, X)                    # True: equal fields
   A.convert(sc.Context(sc.JaxOps(), dtype=np.float32))   # rebuilt from the fields

Both classes work under ``jax.jit`` as pytrees without further code.

A new array library is a backend, not a space: see *Adding a backend* in
:doc:`backend_conformance`.

Related: :doc:`context_dtype_conversion`, :doc:`checking_policy`, :doc:`performance_and_jit`.

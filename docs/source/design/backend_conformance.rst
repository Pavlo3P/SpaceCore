Backend conformance
===================

Every ``BackendOps`` method must match the NumPy reference within its per-op
tolerance on every backend; the matrix below records what ``tests/backend/``
checks.

How it is checked
-----------------

* **Harness.** ``tests/_conformance.py`` (re-exported by
  ``tests/backend/_conformance.py``) holds ``TOLERANCE_TABLE`` and
  ``assert_matches_reference``. Tolerance is keyed by ``(op, dtype kind)``;
  the default is ``rtol=1e-6, atol=1e-8``, looser for ``eigh``, ``svd``,
  complex ``sqrt``, and float32 ``logsumexp`` / ``cholesky`` / ``solve``.
* **Backends.** The ``backend_ops`` fixture (``tests/backend/conftest.py``)
  runs each test on NumPy and on each of JAX, Torch, CuPy that is importable.
* **dtypes.** ``float32``, ``float64``, ``complex64``, ``complex128``. A dtype
  the backend does not honor is skipped (``backend_supports_dtype``): JAX
  64-bit without ``jax_enable_x64``, Torch according to its default dtype.
* **Not checked.** Performance, device placement, sharding, autograd.
* **Differences.** A result that passes but differs from NumPy in a way users
  notice (dtype promotion, default precision) is listed in
  :doc:`backend_deviations`.

Adding a backend
----------------

A backend is a ``BackendOps`` subclass registered under a family name. Everything
the Array API standard covers delegates to the class attribute ``xp``; you write
only what it does not.

1. **Subclass** ``sc.BackendOps`` and set:

   * ``xp`` — an Array API namespace (``array_api_compat.<library>``);
   * ``_family`` — the registry key, any lowercase string;
   * ``_allow_sparse`` — whether the backend has sparse arrays.

2. **Implement the abstract methods:** ``dense_array`` and ``sparse_array`` (the
   array types), ``sanitize_dtype`` (``None`` means the default dtype),
   ``logsumexp``, ``ix_``, ``assparse``, ``sparse_matmul`` and
   ``allclose_sparse`` (raise ``CapabilityError`` without sparse support), and
   ``fori_loop`` / ``while_loop`` / ``scan`` / ``cond`` (see *Control flow* below:
   an eager library mixes in ``spacecore.backend.EagerControlFlowMixin``).
3. **Implement** ``_copy`` and ``_scatter_add_inplace`` if ``index_set`` /
   ``index_add`` should work; override any inherited method whose ``xp`` behaviour
   differs from NumPy's (``TorchOps`` overrides ``where``, ``concatenate``, ...).
4. **Register** by decorating the class with ``@sc.register_ops`` (or calling
   ``sc.register_ops(cls)``). A family can be registered once per
   process (``ContextConflictError`` otherwise). ``Context("<family>")`` and
   ``Context(cls())`` then both work, and ``convert`` moves spaces, operators and
   functionals onto it.
5. **Test it:** add the family to ``_all_families`` and ``backend_ops`` in
   ``tests/backend/conftest.py`` so the conformance suite runs against the NumPy
   reference, and add its column to the matrix below.

A complete dense-only backend over NumPy:

.. code-block:: python

   import numpy as np
   import scipy.special
   import array_api_compat.numpy
   import spacecore as sc
   from spacecore.backend import EagerControlFlowMixin


   @sc.register_ops                     # registers the class and returns it
   class DenseNumpyOps(EagerControlFlowMixin, sc.BackendOps):
       """NumPy arrays, no sparse support: the smallest complete backend."""

       xp = array_api_compat.numpy          # every Array API method delegates here
       _family = "npdense"                  # registry key; Context("npdense") finds it
       _allow_sparse = False

       # --- array types and dtypes ------------------------------------------
       @property
       def dense_array(self):
           return np.ndarray

       @property
       def sparse_array(self):
           return ()

       def sanitize_dtype(self, dtype):
           return np.float64 if dtype is None else np.dtype(dtype)

       # --- what the Array API standard does not cover ------------------------
       def logsumexp(self, a, axis=None, b=None, keepdims=False, return_sign=False):
           return scipy.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims,
                                          return_sign=return_sign)

       def ix_(self, *args):
           return np.ix_(*args)

       def _copy(self, x):                  # used by index_set / index_add
           return x.copy()

       def _scatter_add_inplace(self, y, index, values):
           np.add.at(y, index, values)

       # --- sparse: refuse, since _allow_sparse is False ----------------------
       def assparse(self, x, *, format="csr", dtype=None):
           raise sc.CapabilityError("npdense has no sparse arrays")

       def sparse_matmul(self, a, b):
           raise sc.CapabilityError("npdense has no sparse arrays")

       def allclose_sparse(self, a, b, rtol=1e-5, atol=1e-8):
           raise sc.CapabilityError("npdense has no sparse arrays")


   ctx = sc.Context("npdense", dtype=np.float64)
   X = sc.DenseCoordinateSpace((3,), ctx)
   A = sc.DenseLinOp(ctx.asarray(np.diag([1.0, 2.0, 4.0])), X, X, ctx)
   b = ctx.asarray([1.0, 1.0, 1.0])
   x = sc.cg(A, b).x
   assert ctx.ops.family == "npdense"
   np.testing.assert_allclose(x, [1.0, 0.5, 0.25])

   A_np = A.convert(sc.Context(sc.NumpyOps(), dtype=np.float64))   # move to another backend
   np.testing.assert_allclose(A_np.apply(b), A.apply(b))

Control flow
------------

**Library loops go through** ``ops``, **so one solver runs eagerly and compiles under
``jax.jit``.** ``cg``, ``power_iteration``, ``lanczos_smallest`` and
``expm_multiply`` never write a Python ``while``; they call ``ops.while_loop``,
``ops.fori_loop``, ``ops.cond`` and ``ops.scan``. Each backend decides what those
mean.

========================================================  =====================================================
Method (``jax.lax`` semantics)                            Eager meaning
========================================================  =====================================================
``fori_loop(lower, upper, body(i, carry), init)``         ``for i in range(lower, upper): carry = body(i, carry)``
``while_loop(cond(carry), body(carry), init)``            ``while cond(carry): carry = body(carry)``
``cond(pred, true_fun, false_fun, *operands)``            ``true_fun(*operands) if pred else false_fun(*operands)``
``scan(f(carry, x) -> (carry, y), init, xs)``             loop over the leading axis of ``xs``; stack the ``y``
========================================================  =====================================================

The two kinds of backend need different implementations:

* **Tracing** (``JaxOps``): a Python ``while`` cannot test a traced value, a Python
  ``if`` cannot branch on one, and a Python ``for`` unrolls into the compiled graph.
  ``JaxOps`` maps the four methods to ``jax.lax.while_loop`` / ``fori_loop`` /
  ``cond`` / ``scan`` and does not use the mixin.
* **Eager** (``NumpyOps``, ``TorchOps``, ``CuPyOps`` through ``NumpyOps``): the
  values are concrete, so plain Python loops are correct. ``EagerControlFlowMixin``
  is that one implementation, including the tree stacking ``scan`` needs; list it
  before ``BackendOps`` in the bases so its methods win.

A new eager backend therefore mixes it in and writes no control flow. A new
tracing backend implements the four with its own staged primitives, keeping the
carry structure fixed across iterations and both ``cond`` branches returning the
same structure, as ``jax.lax`` requires. Code written against ``ops`` follows the
same rules:

.. code-block:: python

   def newton_sqrt(ops, a, tol=1e-12):
       """sqrt(a) by Newton's method, written once against ops."""
       def keep_going(carry):
           x, step = carry
           return step > tol

       def update(carry):
           x, _ = carry
           x_new = 0.5 * (x + a / x)
           return x_new, ops.abs(x_new - x)

       x, _ = ops.while_loop(keep_going, update, (a, ops.asarray(1.0)))
       return x

   np_ops = sc.NumpyOps()
   print(newton_sqrt(np_ops, np_ops.asarray(2.0)))          # eager Python loop

   import jax
   jax_ops = sc.JaxOps()
   print(jax.jit(lambda a: newton_sqrt(jax_ops, a, tol=1e-6))(jax_ops.asarray(2.0)))  # lax.while_loop

Matrix
------

``N J T C`` = NumPy, JAX, Torch, CuPy. ``y`` = checked against the reference;
``s`` = checked, skipped when the library or platform lacks the operation
(e.g. CuPy without a GPU).

Metadata and predicates
~~~~~~~~~~~~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``family`` / ``allow_sparse`` / ``has_native_vmap``                             constant                              y y y y
``dense_array`` / ``sparse_array``                                              array type                            y y y s
``is_dense`` / ``is_sparse`` / ``is_array``                                     direct                                y y y y
``get_dtype`` / ``shape`` / ``ndim`` / ``size``                                 ``x.dtype`` / ``x.shape`` / ...       y y y y
``is_complex_dtype`` / ``real_dtype`` / ``complex_dtype`` / ``sanitize_dtype``  dtype kind                            y y y y
==============================================================================  ====================================  =======

Construction
~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``asarray`` / ``astype``                                                        ``np.asarray`` / ``x.astype``         y y y y
``empty`` / ``zeros`` / ``ones`` / ``full`` and their ``*_like``                ``np.zeros`` / ...                    y y y y
``arange`` / ``eye``                                                            ``np.arange`` / ``np.eye``            y y y y
==============================================================================  ====================================  =======

Shape and layout
~~~~~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``ravel`` / ``reshape`` / ``transpose`` / ``swapaxes`` / ``moveaxis``           direct                                y y y y
``broadcast_to`` / ``expand_dims`` / ``squeeze``                                direct                                y y y y
``stack`` / ``concatenate`` / ``hstack`` / ``vstack``                           direct                                y y y y
``dstack`` / ``column_stack``                                                   direct                                y y y y
==============================================================================  ====================================  =======

Elementwise and reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``conj`` / ``real`` / ``imag`` / ``abs`` / ``sign``                             direct                                y y y y
``sqrt`` / ``exp`` / ``log``                                                    direct                                y y y y
``sum`` / ``mean`` / ``min`` / ``max`` / ``prod``                               direct                                y y y y
``argmin`` / ``argmax`` / ``argsort`` / ``sort``                                direct                                y y y y
``maximum`` / ``minimum`` / ``where`` / ``clip``                                direct                                y y y y
``isfinite`` / ``isnan``                                                        direct                                y y y y
``logsumexp``                                                                   ``scipy.special.logsumexp``           y y y y
==============================================================================  ====================================  =======

Linear algebra
~~~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``vdot`` / ``matmul`` / ``kron`` / ``einsum``                                   ``np.vdot`` / ``np.matmul`` / ...     y y y y
``norm`` / ``solve`` / ``eigvalsh`` / ``svd`` / ``cholesky``                    ``np.linalg.*``                       y y y y
``eigh``                                                                        identity ``A v = λ v``                y y y s
``trace`` / ``diag`` / ``diagonal`` / ``tril`` / ``triu``                       direct                                y y y y
==============================================================================  ====================================  =======

Indexing, control flow, mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``take`` / ``ix_``                                                              ``np.take`` / ``np.ix_``              y y y y
``index_set`` / ``index_add``                                                   ``x[i] = v`` / ``x[i] += v``          y y y y
``fori_loop`` / ``while_loop`` / ``scan`` / ``cond``                            sequential Python                     y y y y
``vmap`` / ``vectorize``                                                        stacked Python loop                   y y y y
==============================================================================  ====================================  =======

Sparse
~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``assparse`` round-trip / ``sparse_matmul`` / ``allclose_sparse``               SciPy sparse                          y y y s
==============================================================================  ====================================  =======

Conversion and constants
~~~~~~~~~~~~~~~~~~~~~~~~

==============================================================================  ====================================  =======
Method                                                                          Reference                             N J T C
==============================================================================  ====================================  =======
``Context.asarray`` / ``Context.convert`` across backends                       round-trip equality                   y y y y
``allclose``                                                                    ``np.allclose``                       y y y y
``inf`` / ``nan`` / ``pi`` / ``e`` / ``eps(dtype)``                             ``np.inf`` / ``np.finfo(dtype).eps``  y y y y
==============================================================================  ====================================  =======

``trace`` sums the diagonal over the **trailing two axes**, so a
``(batch, n, n)`` stack returns one trace per matrix.

``is_tracing`` and ``install_pytree_protocol`` have no NumPy reference and are
not in the matrix.

See :doc:`context_dtype_conversion` for the dtype and conversion contract.

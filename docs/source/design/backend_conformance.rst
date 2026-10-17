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

1. Subclass ``BackendOps``, set ``_family``, and implement the abstract
   methods: ``dense_array``, ``sparse_array``, ``sanitize_dtype``,
   ``assparse``, ``sparse_matmul``, ``allclose_sparse``, ``logsumexp``,
   ``ix_``, and ``fori_loop`` / ``while_loop`` / ``scan`` / ``cond`` (or mix in
   ``EagerControlFlowMixin`` for those four).
2. Register it with ``spacecore.register_ops``.
3. Add the family to ``_all_families`` and ``backend_ops`` in
   ``tests/backend/conftest.py``.
4. Add its column to the matrix.

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

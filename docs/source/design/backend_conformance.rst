Backend conformance
===================

This page is the normative conformance matrix for ``BackendOps``. Every
public method on ``spacecore.backend.BackendOps`` appears as a row. Columns
record which backends implement it within tolerance, which dtypes each
backend honors, optional-argument coverage, sparse support, JIT
compatibility, native ``vmap``, and conversion behavior.

Tests live under ``tests/backend/``. The shared harness
(``tests/backend/_conformance.py``) defines per-op tolerances, the
``backend_ops`` fixture, and the ``assert_matches_reference`` helper. The
matrix is the contract; tests instantiate it.

Scope and policy
----------------

* **In scope.** NumPy is the reference. JAX, Torch, and CuPy are compared
  against NumPy through the harness. Sparse paths exist where the backend
  exposes a sparse array type.
* **Out of scope.** Performance, device placement, JAX sharding, autograd,
  and downstream library compatibility.
* **Tolerance.** Per-op + per-dtype, set in
  ``tests/backend/_conformance.py:TOLERANCE_TABLE``. The default is
  ``rtol=1e-6, atol=1e-8``. Spectral and complex sqrt entries are looser.
* **Skip vs deviation.** A backend that genuinely lacks an op is skipped
  with a justification. A backend that returns a different result inside
  tolerance is a *deviation* and appears in :doc:`backend_deviations`.
* **Adding a backend.** Implement ``BackendOps``, register the family,
  add a fixture branch in ``tests/backend/conftest.py``, and update the
  ``Backends`` column of every row this implementation supports.

How to read a row
-----------------

* **Method** — ``BackendOps`` method name.
* **NumPy reference** — the ``numpy`` or ``numpy.linalg`` callable used as
  ground truth. ``—`` means no direct NumPy spelling; the row is
  compared against another invariant (identity, round-trip, etc.).
* **Backends** — ``N J T C`` columns; ``y`` = covered, ``s`` = covered
  with a skip on absent platforms, ``d`` = deviation (see
  :doc:`backend_deviations`), ``—`` = not implemented for this backend.
* **dtypes** — ``r32 r64 c64 c128``; ``y`` covered, ``—`` not covered
  by the backend natively.

Matrix
------

Metadata and predicates
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``family``
     - identity
     - y y y y
     - n/a
   * - ``allow_sparse``
     - identity
     - y y y y
     - n/a
   * - ``has_native_vmap``
     - constant
     - y y y y
     - n/a
   * - ``dense_array``
     - identity
     - y y y y
     - n/a
   * - ``sparse_array``
     - identity
     - y y y s
     - n/a
   * - ``is_dense`` / ``is_sparse`` / ``is_array``
     - direct
     - y y y y
     - n/a
   * - ``get_dtype``
     - ``x.dtype``
     - y y y y
     - y y y y
   * - ``shape`` / ``ndim`` / ``size``
     - ``x.shape`` / ``x.ndim`` / ``x.size``
     - y y y y
     - y y y y
   * - ``is_complex_dtype`` / ``real_dtype`` / ``sanitize_dtype``
     - dtype kind
     - y y y y
     - y y y y

Construction and dtype
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``asarray``
     - ``np.asarray``
     - y y y y
     - y y y y
   * - ``astype``
     - ``x.astype``
     - y y y y
     - y y y y
   * - ``empty`` / ``zeros`` / ``ones`` / ``full``
     - ``np.empty`` / ...
     - y y y y
     - y y y y
   * - ``zeros_like`` / ``ones_like`` / ``full_like``
     - direct
     - y y y y
     - y y y y
   * - ``arange``
     - ``np.arange``
     - y y y y
     - y y y y
   * - ``eye``
     - ``np.eye``
     - y y y y
     - y y y y
   * - ``assparse``
     - SciPy sparse
     - y s s s
     - y y y y
   * - ``sparse_matmul``
     - SciPy sparse dot
     - y s s s
     - y y y y

Shape and layout
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``ravel`` / ``reshape``
     - direct
     - y y y y
     - y y y y
   * - ``transpose`` / ``swapaxes`` / ``moveaxis``
     - direct
     - y y y y
     - y y y y
   * - ``broadcast_to`` / ``expand_dims`` / ``squeeze``
     - direct
     - y y y y
     - y y y y
   * - ``stack`` / ``concatenate``
     - direct
     - y y y y
     - y y y y

Elementwise and reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``conj`` / ``real`` / ``imag`` / ``abs`` / ``sign``
     - direct
     - y y y y
     - y y y y
   * - ``sqrt`` / ``exp`` / ``log``
     - direct
     - y y y y
     - y y y y
   * - ``sum`` / ``mean`` / ``min`` / ``max`` / ``prod``
     - direct
     - y y y y
     - y y y y
   * - ``argmin`` / ``argmax`` / ``argsort`` / ``sort``
     - direct
     - y y y y
     - y y y y
   * - ``maximum`` / ``minimum`` / ``where`` / ``clip``
     - direct
     - y y y y
     - y y y y
   * - ``isfinite`` / ``isnan``
     - direct
     - y y y y
     - y y y y
   * - ``logsumexp``
     - ``scipy.special.logsumexp``
     - y y y y
     - y y y y

Linear algebra
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``vdot``
     - ``np.vdot``
     - y y y y
     - y y y y
   * - ``matmul``
     - ``np.matmul``
     - y y y y
     - y y y y
   * - ``kron`` / ``einsum``
     - direct
     - y y y y
     - y y y y
   * - ``norm``
     - ``np.linalg.norm``
     - y y y y
     - y y y y
   * - ``solve``
     - ``np.linalg.solve``
     - y y y y
     - y y y y
   * - ``eigh``
     - ``A v = λ v`` identity
     - y y y s
     - y y y y
   * - ``eigvalsh``
     - ``np.linalg.eigvalsh``
     - y y y y
     - y y y y
   * - ``svd``
     - ``np.linalg.svd``
     - y y y y
     - y y y y
   * - ``cholesky``
     - ``np.linalg.cholesky``
     - y y y y
     - y y y y
   * - ``trace`` / ``diag`` / ``diagonal`` / ``tril`` / ``triu``
     - direct
     - y y y y
     - y y y y

Indexing and updates
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``take``
     - ``np.take``
     - y y y y
     - y y y y
   * - ``index_set`` / ``index_add``
     - functional indexing
     - y y y y
     - y y y y
   * - ``ix_``
     - ``np.ix_``
     - y y y y
     - n/a

Control flow
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``fori_loop`` / ``while_loop`` / ``scan`` / ``cond``
     - sequential Python
     - y y y y
     - n/a
   * - ``vmap``
     - sequential stacked loop
     - y y y y
     - y y y y

Conversion and inspection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``Context.asarray`` (cross-backend)
     - round-trip equality
     - y y y y
     - y y y y
   * - ``Context.to_numpy``
     - identity
     - y y y y
     - y y y y
   * - ``allclose`` / ``allclose_sparse``
     - ``np.allclose``
     - y y y y
     - y y y y

Sparse paths
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``sparse_matmul``
     - SciPy
     - y — — s
     - y y y y
   * - ``assparse`` round-trip
     - SciPy
     - y — — s
     - y y y y

Constants
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
   * - ``inf`` / ``nan`` / ``pi`` / ``e``
     - ``np.inf`` / ...
     - y y y y
     - y y y y
   * - ``eps``
     - ``np.finfo(dtype).eps``
     - y y y y
     - y y y y

Reading the result columns
--------------------------

* ``s`` — skipped at runtime because the backend is not installed in the
  current environment, or because the underlying library does not expose
  the operation on this platform (e.g. CuPy sparse without a GPU).
* ``d`` — the backend returns a numerically valid result that differs from
  NumPy within the per-op tolerance; the deviation is documented in
  :doc:`backend_deviations`. Tests still pass.
* ``—`` — not implemented for this backend. ``BackendOps`` raises
  ``NotImplementedError`` (not a soft skip), and the tests assert it.

Cross-references
----------------

* :doc:`backend_deviations` — user-facing list of accepted differences.
* :doc:`context_dtype_conversion` — dtype/scalar-field
  contract underlying the conformance promises.

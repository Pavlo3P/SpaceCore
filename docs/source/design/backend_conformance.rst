Backend conformance
===================

This page is the normative conformance matrix for ``BackendOps``. Every
public method on ``spacecore.backend.BackendOps`` appears as a row. Columns
record which backends implement it within tolerance, which dtypes each
backend honors, optional-argument coverage, sparse support, JIT
compatibility, native ``vmap``, and conversion behavior. Each row links to
the test(s) that pin it.

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
* **Tests** — link(s) to the test file(s) that pin this row.

Matrix
------

Metadata and predicates
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``family``
     - identity
     - y y y y
     - n/a
     - ``test_backend_registry.py``
   * - ``allow_sparse``
     - identity
     - y y y y
     - n/a
     - ``test_backend_registry.py``
   * - ``has_native_vmap``
     - constant
     - y y y y
     - n/a
     - ``test_backend_vmap.py``
   * - ``dense_array``
     - identity
     - y y y y
     - n/a
     - ``test_backend_registry.py``
   * - ``sparse_array``
     - identity
     - y y y s
     - n/a
     - ``test_backend_registry.py``, ``test_backend_optional_args.py``
   * - ``is_dense`` / ``is_sparse`` / ``is_array``
     - direct
     - y y y y
     - n/a
     - ``test_conformance_numpy.py``
   * - ``get_dtype``
     - ``x.dtype``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``shape`` / ``ndim`` / ``size``
     - ``x.shape`` / ``x.ndim`` / ``x.size``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``is_complex_dtype`` / ``real_dtype`` / ``sanitize_dtype``
     - dtype kind
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``

Construction and dtype
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``asarray``
     - ``np.asarray``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_conversion.py``
   * - ``astype``
     - ``x.astype``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_dtype_promotion.py``
   * - ``empty`` / ``zeros`` / ``ones`` / ``full``
     - ``np.empty`` / ...
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``zeros_like`` / ``ones_like`` / ``full_like``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``arange``
     - ``np.arange``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``eye``
     - ``np.eye``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``assparse``
     - SciPy sparse
     - y s s s
     - y y y y
     - ``test_backend_conversion.py``
   * - ``sparse_matmul``
     - SciPy sparse dot
     - y s s s
     - y y y y
     - ``test_backend_conversion.py``

Shape and layout
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``ravel`` / ``reshape``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_consistency.py``
   * - ``transpose`` / ``swapaxes`` / ``moveaxis``
     - direct
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``broadcast_to`` / ``expand_dims`` / ``squeeze``
     - direct
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``stack`` / ``concatenate``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``

Elementwise and reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``conj`` / ``real`` / ``imag`` / ``abs`` / ``sign``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_field_consistency.py``
   * - ``sqrt`` / ``exp`` / ``log``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``sum`` / ``mean`` / ``min`` / ``max`` / ``prod``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_optional_args.py``
   * - ``argmin`` / ``argmax`` / ``argsort`` / ``sort``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``maximum`` / ``minimum`` / ``where`` / ``clip``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``isfinite`` / ``isnan``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``logsumexp``
     - ``scipy.special.logsumexp``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``

Linear algebra
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``vdot``
     - ``np.vdot``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_field_consistency.py``
   * - ``matmul``
     - ``np.matmul``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_consistency.py``
   * - ``kron`` / ``einsum``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``norm``
     - ``np.linalg.norm``
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``solve``
     - ``np.linalg.solve``
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``eigh``
     - ``A v = λ v`` identity
     - y y y s
     - y y y y
     - ``test_conformance_numpy.py``, ``test_backend_consistency.py``
   * - ``eigvalsh``
     - ``np.linalg.eigvalsh``
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``svd``
     - ``np.linalg.svd``
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``cholesky``
     - ``np.linalg.cholesky``
     - y y y y
     - y y y y
     - ``test_backend_consistency.py``
   * - ``trace`` / ``diag`` / ``diagonal`` / ``tril`` / ``triu``
     - direct
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``

Indexing and updates
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``take``
     - ``np.take``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``index_set`` / ``index_add``
     - functional indexing
     - y y y y
     - y y y y
     - ``test_backend_optional_args.py``
   * - ``ix_``
     - ``np.ix_``
     - y y y y
     - n/a
     - ``test_conformance_numpy.py``

Control flow
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``fori_loop`` / ``while_loop`` / ``scan`` / ``cond``
     - sequential Python
     - y y y y
     - n/a
     - ``test_backend_loops.py``
   * - ``vmap``
     - sequential stacked loop
     - y y y y
     - y y y y
     - ``test_backend_vmap.py``

Conversion and inspection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``Context.asarray`` (cross-backend)
     - round-trip equality
     - y y y y
     - y y y y
     - ``test_backend_conversion.py``
   * - ``Context.to_numpy``
     - identity
     - y y y y
     - y y y y
     - ``test_backend_conversion.py``
   * - ``allclose`` / ``allclose_sparse``
     - ``np.allclose``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``

Sparse paths
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``sparse_matmul``
     - SciPy
     - y — — s
     - y y y y
     - ``test_backend_conversion.py``
   * - ``assparse`` round-trip
     - SciPy
     - y — — s
     - y y y y
     - ``test_backend_conversion.py``

Constants
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 12 18

   * - Method
     - NumPy reference
     - Backends (N J T C)
     - dtypes (r32 r64 c64 c128)
     - Tests
   * - ``inf`` / ``nan`` / ``pi`` / ``e``
     - ``np.inf`` / ...
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``
   * - ``eps``
     - ``np.finfo(dtype).eps``
     - y y y y
     - y y y y
     - ``test_conformance_numpy.py``

Reading the result columns
--------------------------

* ``s`` — skipped at runtime because the backend is not installed in the
  current environment, or because the underlying library does not expose
  the operation on this platform (e.g. CuPy sparse without a GPU).
* ``d`` — the backend returns a numerically valid result that differs from
  NumPy within the per-op tolerance; the deviation is documented in
  :doc:`backend_deviations`. Tests still pass.
* ``—`` — not implemented for this backend. ``BackendOps`` raises
  ``NotImplementedError`` (not a soft skip) and the test asserts that
  behavior in ``test_backend_optional_args.py``.

Cross-references
----------------

* :doc:`backend_deviations` — user-facing list of accepted differences.
* :doc:`batching_test_policy` — what is and isn't tested for batched paths.
* :doc:`dtype_policy` and :doc:`conversion_policy` — dtype/scalar-field
  contract underlying the conformance promises.
* ``docs/dev/0.4.0-phase-a-inventory.md`` section A6 — the gap inventory
  that seeded this matrix.

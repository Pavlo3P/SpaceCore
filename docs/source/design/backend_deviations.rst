Backend deviations
==================

This page lists accepted differences between backends and the NumPy
reference defined in :doc:`backend_conformance`. A deviation is a backend
result that is numerically valid but disagrees with NumPy in a way that
matters to users — typically a dtype-promotion choice or a default
precision. Tolerance-only differences (eigenvector sign, complex sqrt
branch within the per-op tolerance) are *not* deviations.

Each row gives the op, the affected backend(s), what NumPy does, what the
backend does, the workaround, and a link to the test that pins the current
behavior. When a deviation is removed, the entry stays as a historical
note for one minor release.

Accepted deviations
-------------------

.. list-table::
   :header-rows: 1
   :widths: 14 16 22 22 26

   * - Op
     - Backend(s)
     - NumPy behavior
     - Backend behavior
     - Workaround / test
   * - ``asarray`` default dtype
     - JAX (default)
     - Python floats → ``float64``.
     - Python floats → ``float32`` when ``jax_enable_x64`` is unset
       (JAX's documented default).
     - Set ``jax_enable_x64=True`` for ``float64`` parity, or pass
       ``dtype=...`` explicitly. Pinned in
       ``test_backend_dtype_promotion.py``.
   * - ``asarray`` default dtype
     - Torch
     - Python floats → ``float64``.
     - Python floats → ``torch.get_default_dtype()`` (``float32`` by
       default).
     - Call ``torch.set_default_dtype(torch.float64)`` for parity, or pass
       ``dtype=...``. Pinned in
       ``test_backend_dtype_promotion.py``.
   * - ``matmul`` promotion (``float32 @ float64``)
     - JAX (default)
     - NEP 50: promote to ``float64``.
     - Keep ``float32`` in default mode (JAX weak-type promotion).
     - Pre-promote operands with ``ops.astype(..., float64)`` if you
       want NumPy-style promotion. Pinned in
       ``test_backend_dtype_promotion.py``.
   * - ``matmul`` (real × complex)
     - Torch
     - Auto-promotes the real operand to complex; produces a complex
       result.
     - Raises ``RuntimeError`` ("expected m1 and m2 to have the same
       dtype") rather than promoting.
     - Pre-promote the real operand with
       ``ops.astype(x, complex_dtype)`` before calling ``matmul``.
       Pinned in ``test_backend_dtype_promotion.py``.
   * - ``imag`` on a real tensor
     - Torch
     - Returns a zero array of the matching real dtype.
     - Raises ``RuntimeError`` ("imag is not implemented for tensors
       with non-complex dtypes").
     - Guard with ``ops.is_complex_dtype(ops.get_dtype(x))`` or convert
       to complex before calling ``imag``. Pinned in
       ``test_backend_field_consistency.py``.
   * - ``eigh`` eigenvalue order
     - All backends (within tolerance)
     - Ascending by eigenvalue.
     - Ascending by eigenvalue across NumPy/JAX/Torch/CuPy, but
       eigenvector signs and complex phases differ.
     - Tests compare via the eigenvalue identity ``A v = λ v`` rather
       than direct vector equality. Pinned in
       ``test_conformance_numpy.py`` and
       ``test_conformance_cross_backend.py``.
   * - ``sqrt`` complex branch
     - All backends (within tolerance)
     - Principal branch.
     - Same principal branch; output may differ by ULPs near the
       branch cut.
     - Per-op tolerance loosened to ``rtol=1e-5, atol=1e-6`` for
       complex ``sqrt``. Pinned in ``test_conformance_numpy.py``.
   * - ``logsumexp`` (float32)
     - JAX, Torch
     - SciPy ``logsumexp`` in ``float32``.
     - Numerically valid result; small drift near extreme dynamic
       ranges.
     - Per-op tolerance ``rtol=1e-4, atol=1e-5`` for float32
       ``logsumexp``. Pinned in ``test_conformance_numpy.py``.
   * - ``sparse_matmul`` / ``assparse``
     - JAX, Torch
     - SciPy CSR/CSC/COO; full support.
     - Best-effort or unsupported; ``BackendOps.allow_sparse`` reports
       ``False`` and the op raises ``NotImplementedError``.
     - Convert to a dense path or do sparse work in NumPy. Pinned in
       ``test_backend_conversion.py``.
   * - ``vmap``
     - NumPy, Torch, CuPy
     - n/a (not native).
     - Fallback Python loop; semantics match JAX's ``vmap``.
     - Parity vs JAX's native ``vmap`` is pinned in
       ``test_backend_vmap.py``. The fallback respects
       ``in_axes=None`` and structured outputs.
   * - ``jit``
     - NumPy, Torch, CuPy
     - n/a.
     - ``BackendOps`` does not expose a JIT — ``jax.jit`` over a
       ``NumpyOps`` callable raises ``TracerArrayConversionError``.
     - Use ``JaxOps`` for JIT-compiled paths. Pinned in
       ``test_backend_jit.py``.

Closed deviations
-----------------

None for ``0.4.0``.

Reading the workaround column
-----------------------------

The workaround is a code change inside SpaceCore or in user code that
restores NumPy-equivalent behavior. It is not an apology for the
deviation: each row's deviation is a documented backend choice, and
removing it would require changing the backend, not SpaceCore. If a
deviation surprises you, file an issue with the failing test so the
matrix entry in :doc:`backend_conformance` can be tightened or a
new row added.

Cross-references
----------------

* :doc:`backend_conformance` — the normative matrix every deviation
  refers back to.
* :doc:`dtype_policy` — dtype promotion contract.
* :doc:`conversion_policy` — cross-backend conversion rules and the
  complex-to-real refusal.
* ``docs/dev/0.4.0-phase-a-inventory.md`` section A6 — the gap inventory
  this page closes.

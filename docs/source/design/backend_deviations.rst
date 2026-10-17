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
     - Workaround
   * - ``asarray`` default dtype
     - JAX (default)
     - Python floats → ``float64``.
     - Python floats → ``float32`` when ``jax_enable_x64`` is unset
       (JAX's documented default).
     - Set ``jax_enable_x64=True`` for ``float64`` parity, or pass
       ``dtype=...`` explicitly.
   * - ``asarray`` default dtype
     - Torch
     - Python floats → ``float64``.
     - Python floats → ``torch.get_default_dtype()`` (``float32`` by
       default).
     - Call ``torch.set_default_dtype(torch.float64)`` for parity, or pass
       ``dtype=...``.
   * - ``matmul`` promotion (``float32 @ float64``)
     - JAX (default)
     - NEP 50: promote to ``float64``.
     - Keep ``float32`` in default mode (JAX weak-type promotion).
     - Pre-promote operands with ``ops.astype(..., float64)`` if you
       want NumPy-style promotion.
   * - ``matmul`` (real × complex)
     - Torch
     - Auto-promotes the real operand to complex; produces a complex
       result.
     - Raises ``RuntimeError`` ("expected m1 and m2 to have the same
       dtype") rather than promoting.
     - Pre-promote the real operand with
       ``ops.astype(x, complex_dtype)`` before calling ``matmul``.
   * - ``imag`` on a real tensor
     - Torch
     - Returns a zero array of the matching real dtype.
     - Raises ``RuntimeError`` ("imag is not implemented for tensors
       with non-complex dtypes").
     - Guard with ``ops.is_complex_dtype(ops.get_dtype(x))`` or convert
       to complex before calling ``imag``.
   * - ``logsumexp`` (float32)
     - JAX, Torch
     - SciPy ``logsumexp`` in ``float32``.
     - Numerically valid result; small drift near extreme dynamic
       ranges.
     - Per-op tolerance ``rtol=1e-4, atol=1e-5`` for float32
       ``logsumexp``.
   * - ``vmap``
     - NumPy, CuPy
     - n/a (not native).
     - Fallback Python loop; semantics match JAX's ``vmap``.
     - None needed; the fallback respects ``in_axes=None`` and structured
       outputs. JAX and Torch use their native ``vmap``.
   * - ``jit``
     - NumPy, Torch, CuPy
     - n/a.
     - ``BackendOps`` does not expose a JIT — ``jax.jit`` over a
       ``NumpyOps`` callable raises ``TracerArrayConversionError``.
     - Use ``JaxOps`` for JIT-compiled paths.

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
* :doc:`context_dtype_conversion` — dtype promotion and cross-backend conversion.

Backend deviations
==================

These are the places where a backend passes :doc:`backend_conformance` but
behaves differently from NumPy in a way your code can notice; each row gives
the workaround. Differences inside the per-op tolerance (eigenvector sign,
float32 round-off) are not listed.

.. list-table::
   :header-rows: 1
   :widths: 18 14 34 34

   * - Op
     - Backend
     - Behavior (NumPy: ``float64`` default, NEP 50 promotion)
     - Workaround
   * - ``asarray`` default dtype; any ``float64`` input
     - JAX
     - With ``jax_enable_x64`` unset, Python floats become ``float32`` and
       ``float64`` inputs are cast to ``float32`` (so ``float32 @ float64``
       is ``float32``).
     - Set ``jax.config.update("jax_enable_x64", True)`` before creating
       arrays.
   * - ``asarray`` default dtype
     - Torch
     - Python floats become ``torch.get_default_dtype()`` (``float32`` unless
       changed).
     - Pass ``dtype=`` or call ``torch.set_default_dtype(torch.float64)``.
   * - ``matmul`` with mixed dtypes
     - Torch
     - ``float32 @ float64`` and real ``@`` complex raise ``RuntimeError``
       instead of promoting.
     - Cast first: ``ops.astype(a, ops.get_dtype(b))``.
   * - ``imag`` of a real array
     - Torch
     - Raises ``RuntimeError`` instead of returning zeros.
     - Check ``ops.is_complex_dtype(ops.get_dtype(x))`` first.
   * - ``vmap``
     - NumPy, CuPy
     - Python-loop fallback (``has_native_vmap`` is ``False``). It maps and
       stacks through nested tuples only; dicts and lists are not walked.
     - Return tuples from the mapped function, or use JAX/Torch native
       ``vmap``.
   * - ``jax.jit``
     - NumPy, Torch, CuPy
     - Tracing a function built on these backends fails (NumPy:
       ``TracerArrayConversionError``).
     - JIT only ``JaxOps`` code.

The Torch workarounds together:

.. code-block:: python

   import numpy as np
   from spacecore import TorchOps

   ops = TorchOps()
   a = ops.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)  # not float32
   z = ops.asarray(np.eye(2), dtype=np.complex128)

   out = ops.matmul(ops.astype(a, ops.get_dtype(z)), z)  # promote before mixing
   im = ops.imag(a) if ops.is_complex_dtype(ops.get_dtype(a)) else ops.zeros_like(a)
   assert ops.get_dtype(out) == ops.get_dtype(z) and float(ops.sum(im)) == 0.0

See :doc:`context_dtype_conversion` for dtype promotion and cross-backend
conversion.

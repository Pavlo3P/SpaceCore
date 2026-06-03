Performance Notes
=================

Matrix-backed operators have a small fixed Python wrapper cost around
``apply``, ``rapply``, ``vapply``, and ``rvapply``. In eager NumPy this is
typically on the order of 0.5-1 microsecond per call, depending on the Python
version and machine. That overhead is visible when repeatedly applying tiny
operators inside a Python loop.

For arrays above a few thousand elements, the wrapper cost is usually
sub-percent relative to BLAS, sparse-matrix, or backend execution time. Batched
methods amortize the wrapper cost further: ``vapply`` and ``rvapply`` perform
one Python call for the whole leading-axis batch instead of one call per
element.

Under ``jax.jit``, the wrapper and mode-dispatch logic is trace-time constant
and compiles away from the executed computation. If eager NumPy on very small
operands is performance-critical, prefer batching with ``vapply``/``rvapply``
or moving the tight loop into a JIT-compatible backend.

Performance and JIT
===================

Wrapper cost
------------

Every public method (``apply``, ``rapply``, ``vapply``, ``rvapply``, ``value``,
``grad``, space operations) is a thin checked wrapper over a check-free core. In
eager NumPy the wrapper costs on the order of 0.5-1 microsecond per call,
depending on the Python version and machine; that is visible only when tiny
operators are applied repeatedly inside a Python loop.

For arrays above a few thousand elements the wrapper is usually sub-percent of
the BLAS, sparse-matrix or backend time. Batched methods amortize it further:
``vapply`` and ``rvapply`` make one Python call for the whole leading-axis batch
instead of one per element. If eager NumPy on very small operands is
performance-critical, batch with ``vapply``/``rvapply`` or move the tight loop to
a JIT-compatible backend.

Under ``jax.jit``
-----------------

Numerical kernels run under ``jax.jit`` when values live in a JAX-backed
``Context``. The object model stays ordinary Python: spaces, operators and
functionals are assembled before the kernel is traced and passed into the jitted
function. Wrapper logic is a trace-time constant and compiles away. While a
backend traces, checks run at most at ``get_traced_check_level()``, which
defaults to ``"none"`` (:doc:`checking_policy`).

Operator algebra such as ``A @ B`` and ``A + B`` runs Python-level
simplification rules at construction time, so:

* construct operator expressions outside the jitted function;
* pass the assembled operator as an argument to it;
* avoid calling ``make_sum`` or ``make_composed`` inside a ``jax.jit`` body.

This is a trace-time concern, not a correctness one: composing inside ``jax.jit``
gives the same result but reruns the simplification on every trace.

.. code-block:: python

   import jax
   import spacecore as sc

   ctx = sc.Context(sc.JaxOps(), dtype="float32")
   X = sc.DenseCoordinateSpace((128,), ctx)
   A = build_operator(X)
   B = build_preconditioner(X)

   # Build algebra outside the JIT boundary.
   system = B.H @ A @ B + 0.01 * sc.IdentityLinOp(X, ctx)

   @jax.jit
   def solve(op, rhs):
       return sc.cg(op, rhs, maxiter=50).x

   x = solve(system, rhs)

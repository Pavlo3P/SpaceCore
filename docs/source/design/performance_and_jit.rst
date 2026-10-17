Performance and JIT
===================

Wrapper cost
------------

Each public method call pays a fixed checking cost that does not grow with the
array size; when it dominates, lower the level or batch.

* Order of magnitude on NumPy: under a microsecond over the core at
  ``"none"``, around ten microseconds at ``"cheap"`` or ``"standard"``. Time
  your own hot loop with ``timeit`` before lowering the level.
* ``vapply``, ``rvapply``, ``vvalue`` and ``vgrad`` validate the whole batch in
  one call (:doc:`batching`).
* Under tracing, checks run at trace time only, capped by
  :func:`spacecore.get_traced_check_level` (default ``"none"``); the compiled
  program carries no checks (:doc:`checking_policy`).

Under ``jax.jit``
-----------------

Build operator expressions outside the jitted function and pass them in as
arguments; operators are pytrees. ``A @ B`` and ``A + B`` simplify in Python at
construction, so building them inside the jitted body repeats that work on
every trace.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import spacecore as sc

   ctx = sc.Context(sc.JaxOps(), dtype="float32")
   X = sc.DenseCoordinateSpace((128,), ctx)
   A = sc.DiagonalLinOp(ctx.asarray(jnp.linspace(1.0, 2.0, 128)), X, ctx)

   system = A.H @ A + 0.01 * sc.IdentityLinOp(X, ctx)  # outside the JIT

   @jax.jit
   def solve(op, rhs):
       return sc.cg(op, rhs, maxiter=50).x

   x = solve(system, ctx.asarray(jnp.ones(128)))

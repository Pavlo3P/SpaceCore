JAX integration
===============

JIT usage notes
---------------

SpaceCore's numerical kernels are written to run under ``jax.jit`` when values
live in a JAX-backed ``Context``. The object model remains ordinary Python:
spaces, operators, and functionals are assembled before the numerical kernel is
traced, then passed into the jitted function.

Operator algebra such as ``A @ B`` and ``A + B`` executes Python-level
simplification rules at construction time. For maximum JIT efficiency:

* construct operator expressions outside the JIT-decorated function;
* pass the assembled operator as an argument to the jitted function;
* avoid calling ``make_sum`` or ``make_composed`` from inside a ``jax.jit``
  body.

This is a trace-time concern rather than a correctness concern. The algebra is
correct either way, but composing inside ``jax.jit`` means the simplification
runs once per trace. For repeatedly invoked code with stable operator
structure, build the expression once outside the jitted function.

Example:

.. code-block:: python

   import jax
   import spacecore as sc

   ctx = sc.Context(sc.JaxOps(), dtype="float32")
   X = sc.VectorSpace((128,), ctx)
   A = build_operator(X)
   B = build_preconditioner(X)

   # Build algebra outside the JIT boundary.
   system = B.H @ A @ B + 0.01 * sc.IdentityLinOp(X, ctx)

   @jax.jit
   def solve(op, rhs):
       return sc.cg(op, rhs, maxiter=50).x

   x = solve(system, rhs)

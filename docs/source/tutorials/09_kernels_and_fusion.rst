9 · Lazy operator algebra and fusion
====================================

The first eight tutorials are about *expressing* mathematics. This one
is about making operator expressions run fast **without changing how you
write them**: lazy operator algebra and fusion (design record ADR-021,
``docs/dev/adr/021_lazy_operator_algebra_and_simplification.md`` in the repository).
Cheap rewrites happen automatically at construction; materializing
fusions, such as collapsing ``A @ B @ C`` of dense operators into one
matrix, are an **explicit** ``fuse()`` call. Neither touches the
matrix-free contract: a matrix-free operator is never silently turned
into a dense one.

.. code:: python

    import numpy as np
    import spacecore as sc
    
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    rng = np.random.default_rng(0)

.. code:: python

    n = 6
    X = sc.DenseCoordinateSpace((n,), ctx)

1 · Lazy operator algebra and ``fuse()``
----------------------------------------

Operator expressions are **lazy**: ``A @ B``, ``A + B``, ``c * A``,
``A.H`` build a small tree, evaluating nothing until you apply it.
Simplification has two tiers.

**Tier 1 — automatic, at construction.** Cheap, local rewrites happen
the moment you build the expression: an identity factor disappears, a
zero map collapses the product, nested scalars fold. These cost nothing
and are always applied.

.. code:: python

    M1 = sc.DenseLinOp(ctx.asarray(rng.standard_normal((n, n))), X, X, ctx)
    I  = sc.IdentityLinOp(X, ctx)
    
    print("M1 @ I  is  M1 :", (M1 @ I) is M1)          # identity elided at construction
    print("type of 2*(3*M1):", type(2.0 * (3.0 * M1)).__name__,
          "-> one ScaledLinOp with scalar", (2.0 * (3.0 * M1)).scalar)


.. parsed-literal::

    M1 @ I  is  M1 : True
    type of 2*(3*M1): ScaledLinOp -> one ScaledLinOp with scalar 6.0


**Tier 2 — explicit, with ``fuse()``.** *Multiplying operators together*
is a materializing rewrite, so it is something you ask for. ``fuse()``
collapses each maximal run of dense operators into a single
``DenseLinOp`` holding the matrix product — the whole chain becomes one
matrix.

Fusing reassociates the arithmetic (a matrix product, then apply,
differs at the last bit from applying in sequence), so the result is
equal **up to rounding**, not bit-for-bit. It is adjoint-consistent on
any geometry — including non-Euclidean — because the shared middle-space
Riesz maps cancel: ``R_X^-1 B^H R_Y R_Y^-1 A^H R_Z = R_X^-1 (A B)^H R_Z``.
Trefethen & Bau, *Numerical Linear Algebra*, Eq. (2.4), gives the conjugate-transpose
product identity; :doc:`../design/geometry` supplies the Gram factors.

.. code:: python

    M2 = sc.DenseLinOp(ctx.asarray(rng.standard_normal((n, n))), X, X, ctx)
    M3 = sc.DenseLinOp(ctx.asarray(rng.standard_normal((n, n))), X, X, ctx)
    
    expr  = M1 @ M2 @ M3                 # a lazy composition tree
    fused = expr.fuse()                  # one DenseLinOp: M1 @ M2 @ M3 as a single matrix
    
    xv = ctx.asarray(rng.standard_normal(n))
    print("before fuse :", type(expr).__name__)
    print("after  fuse :", type(fused).__name__)
    print("same action :", np.allclose(np.asarray(fused.apply(xv)), np.asarray(expr.apply(xv))))


.. parsed-literal::

    before fuse : ComposedLinOp
    after  fuse : DenseLinOp
    same action : True


The matrix-free rail
~~~~~~~~~~~~~~~~~~~~

``fuse()`` only multiplies operators that already hold a cheap matrix. A
**matrix-free** operator — defined by callables, with no stored matrix —
is left untouched: it stays a lazy leaf and simply breaks the fusible
run, so its matrix-free contract is preserved.

If you genuinely want to densify a matrix-free operand — accepting the
cost and giving up the matrix-free property — that is an **explicit
opt-in**: ``fuse(materialize=True)``.

.. code:: python

    Mf = sc.MatrixFreeLinOp(lambda v: 2.0 * v, lambda v: 2.0 * v, X, X, ctx)  # matrix-free
    
    print("(M1 @ Mf).fuse()                 ->", type((M1 @ Mf).fuse()).__name__,
          " (matrix-free kept lazy)")
    print("(M1 @ Mf).fuse(materialize=True) ->", type((M1 @ Mf).fuse(materialize=True)).__name__,
          " (explicitly densified)")


.. parsed-literal::

    (M1 @ Mf).fuse()                 -> ComposedLinOp  (matrix-free kept lazy)
    (M1 @ Mf).fuse(materialize=True) -> DenseLinOp  (explicitly densified)


2 · The performance payoff
--------------------------

Here is *why you reach for* ``fuse()`` — measured on this machine, on
the NumPy backend.

.. code:: python

    import timeit
    
    def per_apply_us(fn, number=200, repeat=9):
        """Average per-apply time in microseconds.
    
        Times ``number`` back-to-back calls in each of ``repeat`` batches and returns
        the **mean** over all ``repeat x number`` timed calls (after a warm-up), so the
        figure is averaged over many runs rather than read off a single measurement.
        """
        fn()                                    # warm up (allocator, caches, any JIT)
        batches = timeit.repeat(fn, number=number, repeat=repeat)
        return float(np.mean(batches)) / number * 1e6
    
    n_big, depth = 384, 16
    Xb = sc.DenseCoordinateSpace((n_big,), ctx)
    factors = [sc.DenseLinOp(ctx.asarray(rng.standard_normal((n_big, n_big))), Xb, Xb, ctx)
               for _ in range(depth)]
    chain = factors[0]
    for f in factors[1:]:
        chain = chain @ f                       # a lazy depth-16 composition
    fused = chain.fuse()                        # one DenseLinOp: the product matrix
    xb = ctx.asarray(rng.standard_normal(n_big))
    
    t_lazy  = per_apply_us(lambda: chain.apply(xb))
    t_fused = per_apply_us(lambda: fused.apply(xb))
    print(f"lazy chain (depth {depth}) : {t_lazy:7.1f} us / apply   ({depth} matvecs)")
    print(f"fused (one matmul)      : {t_fused:7.1f} us / apply   (1 matvec)")
    print(f"speedup per apply       : {t_lazy / t_fused:5.1f}x")


.. parsed-literal::

    lazy chain (depth 16) :   218.9 us / apply   (16 matvecs)
    fused (one matmul)      :    19.1 us / apply   (1 matvec)
    speedup per apply       :  11.5x


**Fusion amortizes a deep composition.** Applying the lazy chain walks
every factor on *every* call — ``depth`` matrix–vector products.
``fuse()`` precomputes the single product matrix once, so each later
apply is **one** matvec. The wall-clock gain tracks that ``depth → 1``
reduction once the operator is large enough for the matrix multiply —
not the fixed per-call validation — to dominate.

This is the classic *fuse once, apply many* win: in a solver that
applies the same operator hundreds of times, you pay the product once
and collect the speedup on every iteration. It holds on **any** backend
— it is simply fewer operations per apply.

Takeaways
---------

-  **Fusion (ADR-021)** simplifies the *operator expression*. Cheap
   identities fold automatically at construction; ``fuse()`` multiplies
   dense operators into one matrix (equal up to rounding,
   adjoint-consistent on any geometry). It never densifies a matrix-free
   operand unless you pass ``materialize=True``.
-  **Performance:** fusion is a structural win on *any* backend — fewer
   operations per apply, a multi-× speedup for a deep chain applied
   repeatedly.
-  Under ``jax.jit`` or ``torch.compile`` the compiler fuses and batches
   the work of a lazy expression on its own; ``fuse()`` remains the way
   to precompute a product matrix once and reuse it across calls.

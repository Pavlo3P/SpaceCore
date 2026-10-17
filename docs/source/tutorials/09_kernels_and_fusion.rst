9 · Lazy operator algebra and fusion
====================================

**Goal:** build a smoothing operator from a dense blur and a custom
difference operator with SpaceCore's lazy algebra, collapse its dense part
with ``fuse()``, and run power iteration on both forms.

**Before you start:** read :doc:`02_linear_algebra` and the custom
functional in :doc:`03_functionals`. Run the blocks in order in a fresh
session; see :doc:`index` for setup.

1 · A custom operator from two cores
------------------------------------

The forward difference :math:`D:\mathbb R^n\to\mathbb R^{n-1}` is

.. math::

   (Dx)_i = x_{i+1}-x_i,\qquad
   (D^\# y)_j = y_{j-1}-y_j\quad(y_0=y_n=0).

The adjoint follows from summation by parts:
:math:`\sum_i y_i(x_{i+1}-x_i)=\sum_j x_j(y_{j-1}-y_j)`.

.. code:: python

    import time
    import numpy as np
    import spacecore as sc

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    n = 120
    X = sc.DenseCoordinateSpace((n,), ctx)
    Y = sc.DenseCoordinateSpace((n - 1,), ctx)

    class ForwardDifference(sc.LinOp):
        """(D x)_i = x[i+1] - x[i]; no matrix is stored."""

        _fields = ("dom", "cod", "ctx")

        def __init__(self, dom, cod, ctx=None, check_level=None):
            super().__init__(dom, cod, ctx, check_level=check_level)

        # Indexing the last axis makes each core work on one vector or a batch.
        def _apply_core(self, x):
            return x[..., 1:] - x[..., :-1]

        def _rapply_core(self, y):
            return self.ops.concatenate(
                [-y[..., :1], y[..., :-1] - y[..., 1:], y[..., -1:]], axis=-1)

        _vapply_core, _rvapply_core = _apply_core, _rapply_core

    D = ForwardDifference(X, Y, ctx)

    rng = np.random.default_rng(0)
    x = ctx.asarray(rng.standard_normal(n))
    y = ctx.asarray(rng.standard_normal(n - 1))
    D_dense = np.diff(np.eye(n), axis=0)          # (n-1) x n reference matrix
    np.testing.assert_allclose(D.apply(x), D_dense @ x, atol=1e-12)
    np.testing.assert_allclose(Y.inner(D.apply(x), y), X.inner(x, D.H.apply(y)),
                               atol=1e-12)
    np.testing.assert_allclose(D.to_dense(), D_dense, atol=1e-12)

The class supplies only the check-free cores. The public ``apply`` and
``rapply`` it inherits check membership and then call them; ``vapply`` and
``rvapply`` do the same over a leading batch axis, which is why the cores
index from the end. ``_fields``
lists the constructor arguments, which gives the operator equality,
pytree flattening and ``convert``. ``to_dense`` probes ``D`` with basis
vectors; it exists for tests like this one, not for solvers.

2 · Build the operator lazily
-----------------------------

Let K be a Gaussian blur matrix, :math:`K_{ij}=e^{-(i-j)^2/(2s^2)}`. It is
symmetric positive semidefinite. Combine it with D:

.. math::

   T = K^2 + \tfrac12 K + \alpha\,D^\# D .

Each term is symmetric positive semidefinite, so the largest eigenvalue of T
is real and nonnegative. :math:`D^\#D` is the discrete Neumann Laplacian.

.. code:: python

    s, alpha = 4.0, 0.3
    idx = np.arange(n)
    K_mat = np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2 * s**2))
    K = sc.DenseLinOp(ctx.asarray(K_mat), X, X, ctx)

    T = K @ K + 0.5 * K + alpha * (D.H @ D)
    T_dense = K_mat @ K_mat + 0.5 * K_mat + alpha * D_dense.T @ D_dense

    print(T)
    assert isinstance(T, sc.SumLinOp) and len(T.parts) == 3
    square, half_K, laplacian = T.parts
    assert isinstance(square, sc.ComposedLinOp)
    assert isinstance(half_K, sc.ScaledLinOp) and half_K.scalar == 0.5 and half_K.op is K
    assert isinstance(laplacian, sc.ScaledLinOp) and laplacian.scalar == alpha
    assert isinstance(laplacian.op, sc.ComposedLinOp)

    np.testing.assert_allclose(T.apply(x), T_dense @ x, rtol=1e-12, atol=1e-12)
    x2 = ctx.asarray(rng.standard_normal(n))
    np.testing.assert_allclose(X.inner(T.apply(x), x2), X.inner(x, T.H.apply(x2)),
                               rtol=1e-12)

Building ``T`` computed nothing: it is a tree whose three summands are a
composition, a scaled dense leaf and a scaled composition. Its ``apply``
checks ``x`` once at the root and then calls the children's cores. The
adjoint of the sum is the sum of the children's adjoints, so ``T.H`` needs
no extra code.

Some rewrites happen at construction because they are free:

.. code:: python

    I = sc.IdentityLinOp(X, ctx)
    assert (K @ I) is K and (I @ K) is K
    folded = 2.0 * (3.0 * K)
    assert isinstance(folded, sc.ScaledLinOp) and folded.scalar == 6.0 and folded.op is K
    assert isinstance(0.0 * K, sc.ZeroLinOp)
    assert len(((K + K) + K).parts) == 3

An identity factor disappears, nested scalars multiply, a zero scalar gives
the zero map, and nested sums flatten. None of these multiplies matrices.

3 · Fuse the dense part
-----------------------

``fuse()`` is the explicit, materializing rewrite. It replaces every
composition of two dense operators by one ``DenseLinOp`` holding the
product, folds scalars into dense matrices, and adds dense summands:

.. math::

   \operatorname{fuse}(T) = \underbrace{(K^2+\tfrac12K)}_{\text{one matrix}}
   + \alpha\,D^\# D .

The custom D holds no matrix, so the Laplacian term stays lazy.

.. code:: python

    T_fused = T.fuse()
    print(T_fused)
    assert isinstance(T_fused, sc.SumLinOp) and len(T_fused.parts) == 2
    dense_part, lazy_part = T_fused.parts
    assert isinstance(dense_part, sc.DenseLinOp)
    np.testing.assert_allclose(dense_part.to_dense(), K_mat @ K_mat + 0.5 * K_mat,
                               rtol=1e-12, atol=1e-12)
    assert lazy_part == laplacian

    for v in (x, x2):
        np.testing.assert_allclose(T_fused.apply(v), T.apply(v), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(T_fused.rapply(v), T.rapply(v), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(X.inner(T_fused.apply(x), x2),
                               X.inner(x, T_fused.H.apply(x2)), rtol=1e-12)
    np.testing.assert_allclose(T_fused.to_dense(), T_dense, rtol=1e-10, atol=1e-10)

The fused operator agrees with the lazy one only up to rounding, because
the matrix product reassociates the arithmetic. Its adjoint is still the
metric adjoint: in the composition the Riesz maps of the middle space
cancel, :math:`R_X^{-1}K^\dagger R_X R_X^{-1}K^\dagger R_X
=R_X^{-1}(KK)^\dagger R_X`. ``lazy_part == laplacian`` compares by
``_fields``, so the unfused term is the same operator.

4 · Power iteration with both forms
-----------------------------------

Power iteration repeats :math:`v\leftarrow Tv/\|Tv\|` and reads the
Rayleigh quotient :math:`\lambda=\langle v,Tv\rangle`. For a symmetric T
it converges to the largest eigenvalue when the start has a component
along its eigenvector.

.. code:: python

    def power_iteration(op, v0, iterations=300):
        v = X.scale(1.0 / X.norm(v0), v0)
        for _ in range(iterations):
            w = op.apply(v)
            v = X.scale(1.0 / X.norm(w), w)
        return float(X.inner(v, op.apply(v))), v

    v0 = ctx.asarray(rng.standard_normal(n))
    start = time.perf_counter()
    lam_lazy, v_lazy = power_iteration(T, v0)
    t_lazy = time.perf_counter() - start
    start = time.perf_counter()
    lam_fused, v_fused = power_iteration(T_fused, v0)
    t_fused = time.perf_counter() - start

    lam_exact = np.linalg.eigvalsh(T_dense)[-1]
    np.testing.assert_allclose(lam_lazy, lam_fused, rtol=1e-12)
    np.testing.assert_allclose(lam_lazy, lam_exact, rtol=1e-8)
    np.testing.assert_allclose(v_lazy, v_fused, atol=1e-8)
    residual = T_dense @ v_lazy - lam_lazy * v_lazy
    assert np.linalg.norm(residual) < 1e-4 * lam_exact
    print(f"largest eigenvalue: {lam_lazy:.10f}  (eigvalsh {lam_exact:.10f})")
    print(f"lazy  loop: {t_lazy * 1e3:6.1f} ms   fused loop: {t_fused * 1e3:6.1f} ms")

Both loops return the same eigenvalue and eigenvector, and both match
``eigvalsh``. Per apply, the lazy ``T`` does three dense matrix-vector
products plus the difference cores; the fused ``T`` does one plus the
same cores. The printed timings depend on the machine and are not
checked. Fusing pays off only after enough applications to amortize the
one matrix-matrix product it costs.

Check your understanding
------------------------

* Why does ``T.fuse()`` keep two summands rather than one? ``D`` stores
  no matrix, so its term is not densely fusible.
* Rewrite ``T`` as ``0.5 * K + K @ K + alpha * (D.H @ D)``. Is the fused
  structure the same? Check with the assertions of section 3.
* Time ``T.fuse()`` itself. If it costs S seconds and the fused apply
  saves d seconds, fusion pays off after about S/d applications.
* Drop the two end terms from ``_rapply_core``, keeping ``y[..., :-1] - y[..., 1:]``.
  Which assertion of section 1 fails first?

This is the last tutorial; :doc:`index` lists the full sequence.

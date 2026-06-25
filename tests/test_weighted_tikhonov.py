"""Integration regression for the weighted Tikhonov pattern.

This pins the SpaceCore behaviour that the former ``examples/weighted_tikhonov.py``
demo used to guard (the demo is now superseded by
``tutorials/05_weighted_tikhonov.ipynb``): an operator between weighted spaces must
expose the *metric* adjoint rather than the coordinate transpose, and the regularised
normal equations assembled with operator algebra must match an independent dense solve.
"""

import numpy as np

import spacecore as sc


def _problem(n=16, m=24, lam=1e-2, seed=3):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(m, n)) / np.sqrt(n)
    x_weights = 0.7 + np.linspace(0.0, 1.3, n)
    y_weights = 1.1 + np.linspace(0.0, 1.7, m)
    x_true = rng.normal(size=n)
    b = M @ x_true + 0.03 * rng.normal(size=m)
    return M, x_weights, y_weights, b, lam


def _spaces(M, x_weights, y_weights):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
    X = sc.DenseVectorSpace(
        (M.shape[1],), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(x_weights))
    )
    Y = sc.DenseVectorSpace(
        (M.shape[0],), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(y_weights))
    )
    A = sc.DenseLinOp(ctx.asarray(M), X, Y, ctx)
    return ctx, X, Y, A


def test_spacecore_weighted_solve_matches_dense_reference():
    M, x_weights, y_weights, b, lam = _problem()
    ctx, X, _Y, A = _spaces(M, x_weights, y_weights)
    Gx, Gy = np.diag(x_weights), np.diag(y_weights)
    x_ref = np.linalg.solve(M.T @ Gy @ M + lam * Gx, M.T @ Gy @ b)

    normal = A.H @ A + lam * sc.IdentityLinOp(X)
    res = sc.cg(
        normal, A.H.apply(ctx.asarray(b)), tol=1e-12, atol=0.0,
        maxiter=8 * M.shape[1], check_every=1,
    )
    assert res.converged
    np.testing.assert_allclose(np.asarray(res.x), x_ref, rtol=1e-7, atol=1e-9)


def test_metric_adjoint_holds_and_coordinate_transpose_fails():
    M, x_weights, y_weights, _b, _lam = _problem(n=12, m=18, seed=5)
    ctx, X, Y, A = _spaces(M, x_weights, y_weights)
    rng = np.random.default_rng(17)
    x = ctx.asarray(rng.normal(size=M.shape[1]))
    y = ctx.asarray(rng.normal(size=M.shape[0]))

    lhs = float(Y.inner(A.apply(x), y))
    metric = float(X.inner(x, A.H.apply(y)))
    transpose = float(X.inner(x, ctx.asarray(M.T @ np.asarray(y))))

    assert abs(lhs - metric) <= 1e-10
    assert abs(lhs - transpose) >= 1e-2


def test_normal_operator_algebra_matches_dense_first_order_residual():
    M, x_weights, y_weights, b, lam = _problem(n=10, m=14, seed=8)
    ctx, X, _Y, A = _spaces(M, x_weights, y_weights)
    Gx, Gy = np.diag(x_weights), np.diag(y_weights)
    x = ctx.asarray(np.linspace(-0.4, 0.6, M.shape[1]))

    normal = A.H @ A + lam * sc.IdentityLinOp(X)
    rhs = A.H.apply(ctx.asarray(b))
    # SpaceCore writes the normal equation in X-coordinates; multiplying by Gx
    # recovers the independent dense first-order residual.
    spacecore_residual = Gx @ np.asarray(normal.apply(x) - rhs)
    dense_residual = (M.T @ Gy @ M + lam * Gx) @ np.asarray(x) - M.T @ Gy @ b

    np.testing.assert_allclose(spacecore_residual, dense_residual, rtol=1e-12, atol=1e-12)

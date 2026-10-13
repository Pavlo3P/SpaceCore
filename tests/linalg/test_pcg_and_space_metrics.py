"""A metric lives only in a space: preconditioned CG, and generalized problems as spaces."""
import numpy as np
import pytest

import spacecore as sc

N = 6


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _spd(seed, shift=1.0):
    G = np.random.default_rng(seed).standard_normal((N, N))
    return G @ G.T + shift * np.eye(N)


def _metric_space(ctx, M_np):
    E = sc.DenseCoordinateSpace((N,), ctx)
    M = sc.DenseLinOp(ctx.asarray(M_np), E, E, ctx)
    Minv = sc.DenseLinOp(ctx.asarray(np.linalg.inv(M_np)), E, E, ctx)
    return sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(M, Minv))


@pytest.mark.parametrize("metric", [False, True], ids=["euclidean", "operator-metric"])
def test_exact_preconditioner_converges_at_once(ctx, metric):
    W = _spd(9)
    X = _metric_space(ctx, W) if metric else sc.DenseCoordinateSpace((N,), ctx)
    # A = W^{-1} S is self-adjoint in the W metric; with W = I it is S.
    S = _spd(0) @ np.diag(np.logspace(0, 4, N)) @ _spd(0).T + np.eye(N)
    A_np = np.linalg.solve(W, S) if metric else S
    A = sc.DenseLinOp(ctx.asarray(A_np), X, X, ctx)
    K = sc.DenseLinOp(ctx.asarray(np.linalg.inv(A_np)), X, X, ctx)
    b = np.arange(1.0, N + 1)
    pre = sc.cg(A, b, preconditioner=K, tol=1e-10, maxiter=50, check_every=1)
    plain = sc.cg(A, b, tol=1e-10, maxiter=50, check_every=1)
    np.testing.assert_allclose(pre.x, np.linalg.solve(A_np, b), rtol=1e-8)
    assert bool(pre.converged) and int(pre.num_iters) <= 2 < int(plain.num_iters)
    np.testing.assert_allclose(pre.residual_norm, X.norm(b - A_np @ pre.x), atol=1e-8)


def test_preconditioner_must_live_on_the_domain(ctx):
    X = sc.DenseCoordinateSpace((N,), ctx)
    A = sc.DenseLinOp(ctx.asarray(_spd(1)), X, X, ctx)
    other = _metric_space(ctx, _spd(2))
    with pytest.raises(ValueError, match="preconditioner"):
        sc.cg(A, np.ones(N), preconditioner=sc.DenseLinOp(ctx.asarray(np.eye(N)), other, other, ctx))


def test_generalized_eigenproblem_is_an_operator_on_a_metric_space(ctx):
    K, M = _spd(5, shift=0.1), _spd(6)
    XM = _metric_space(ctx, M)
    A = sc.DenseLinOp(ctx.asarray(np.linalg.solve(M, K)), XM, XM, ctx, check_level="standard")
    eigenvalues = np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real)
    small = sc.lanczos_smallest(A, np.ones(N), max_iter=N, tol=1e-10)
    np.testing.assert_allclose(small.eigenvalue, eigenvalues[0], rtol=1e-8)
    large = sc.power_iteration(A, maxiter=3000, tol=1e-9)
    np.testing.assert_allclose(large.eigenvalue, eigenvalues[-1], rtol=1e-6)

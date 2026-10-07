"""``OperatorInnerProduct``: an operator-valued metric <x, y> = <x, M y>_E."""
import numpy as np
import pytest

import spacecore as sc

N = 4


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _spd(seed=0, n=N):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    return G @ G.T + n * np.eye(n)


def _metric_space(ctx, M_np, inverse=True):
    E = sc.DenseCoordinateSpace((N,), ctx)
    M = sc.DenseLinOp(ctx.asarray(M_np), E, E, ctx)
    M_inv = sc.DenseLinOp(ctx.asarray(np.linalg.inv(M_np)), E, E, ctx) if inverse else None
    return sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(M, M_inv))


def test_inner_and_riesz_maps(ctx):
    M_np = _spd()
    X = _metric_space(ctx, M_np)
    x, y = np.arange(1., N + 1), np.linspace(-1, 2, N)
    np.testing.assert_allclose(X.inner(x, y), x @ M_np @ y)
    np.testing.assert_allclose(X.riesz(x), M_np @ x)
    np.testing.assert_allclose(X.riesz_inverse(X.riesz(x)), x)
    np.testing.assert_allclose(X.norm(x), np.sqrt(x @ M_np @ x))


def test_diagonal_metric_matches_weighted_geometry(ctx):
    w = np.array([2., 5., 11., 3.])
    E = sc.DenseCoordinateSpace((N,), ctx)
    D = sc.DiagonalLinOp(ctx.asarray(w), E, ctx)
    Dinv = sc.DiagonalLinOp(ctx.asarray(1 / w), E, ctx)
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(D, Dinv))
    W = sc.DenseCoordinateSpace((N,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(w)))
    A = np.random.default_rng(1).standard_normal((N, N))
    y = np.linspace(1, 2, N)
    via_operator = sc.DenseLinOp(ctx.asarray(A), X, X, ctx).rapply(y)
    via_weights = sc.DenseLinOp(ctx.asarray(A), W, W, ctx).rapply(y)
    np.testing.assert_allclose(via_operator, via_weights)


def test_metric_adjoint_identity(ctx):
    X = _metric_space(ctx, _spd(0))
    Y = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([1., 4., 9.])))
    A = sc.DenseLinOp(ctx.asarray(np.random.default_rng(2).standard_normal((3, N))), X, Y, ctx)
    x, y = np.arange(1., N + 1), np.array([1., -1., 2.])
    np.testing.assert_allclose(Y.inner(A.apply(x), y), X.inner(x, A.rapply(y)))


def test_batched_riesz_and_inner_match_rows(ctx):
    M_np = _spd()
    X = _metric_space(ctx, M_np)
    xs = np.random.default_rng(3).standard_normal((5, N))
    np.testing.assert_allclose(X.riesz(xs), xs @ M_np.T)
    np.testing.assert_allclose(X.riesz_inverse(X.riesz(xs)), xs)


def test_cg_in_the_metric_is_preconditioned_cg(ctx):
    # A = M^{-1} B is self-adjoint in the M-inner product; solving A x = M^{-1} b solves B x = b.
    M_np, B = _spd(0), _spd(5)
    X = _metric_space(ctx, M_np)
    Minv = np.linalg.inv(M_np)
    A = sc.DenseLinOp(ctx.asarray(Minv @ B), X, X, ctx)
    b = np.arange(1., N + 1)
    result = sc.cg(A, ctx.asarray(Minv @ b), tol=1e-12)
    np.testing.assert_allclose(result.x, np.linalg.solve(B, b), rtol=1e-8)


def test_inexact_inverse_through_cg(ctx):
    M_np = _spd()
    E = sc.DenseCoordinateSpace((N,), ctx)
    M = sc.DenseLinOp(ctx.asarray(M_np), E, E, ctx)
    solve = lambda r: sc.cg(M, r, tol=1e-12).x  # noqa: E731
    M_inv = sc.MatrixFreeLinOp(solve, solve, E, E, ctx)
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(M, M_inv))
    x = np.arange(1., N + 1)
    np.testing.assert_allclose(X.riesz_inverse(X.riesz(x)), x, rtol=1e-8)


def test_missing_inverse_is_caught_when_needed(ctx):
    X = _metric_space(ctx, _spd(), inverse=False)
    x = np.arange(1., N + 1)
    np.testing.assert_allclose(X.riesz(x), _spd() @ x)
    with pytest.raises(sc.CapabilityError, match="M_inverse"):
        X.riesz_inverse(x)
    A = sc.DenseLinOp(ctx.asarray(np.eye(N)), X, X, ctx)
    with pytest.raises(sc.CapabilityError, match="M_inverse"):
        A.rapply(x)


def test_invalid_metric_operators_are_rejected(ctx):
    E = sc.DenseCoordinateSpace((N,), ctx)
    skew = np.triu(np.ones((N, N)))
    with pytest.raises(ValueError, match="self-adjoint"):
        sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(
            sc.DenseLinOp(ctx.asarray(skew), E, E, ctx)))
    small = sc.DenseCoordinateSpace((2,), ctx)
    with pytest.raises(ValueError, match="shape"):
        sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(
            sc.IdentityLinOp(small, ctx)))
    weighted = sc.DenseCoordinateSpace((N,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(np.ones(N))))
    with pytest.raises(ValueError, match="Euclidean"):
        sc.DenseCoordinateSpace((N,), ctx, geometry=sc.OperatorInnerProduct(sc.IdentityLinOp(weighted, ctx)))


def test_conversion_and_equality(ctx):
    X = _metric_space(ctx, _spd())
    Y = X.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))
    assert Y.dtype == np.dtype("float32")
    assert Y.geometry.M.dtype == np.dtype("float32")
    assert X == _metric_space(ctx, _spd())
    assert X != _metric_space(ctx, _spd(seed=7))


def test_jit_through_an_operator_metric():
    jax = pytest.importorskip("jax")
    ctx = sc.Context(sc.JaxOps(), dtype=np.float32)
    M_np = _spd().astype(np.float32)
    X = _metric_space(ctx, M_np)
    A = sc.DenseLinOp(ctx.asarray(np.eye(N, dtype=np.float32)), X, X, ctx, check_level="none")
    y = ctx.asarray(np.arange(1., N + 1, dtype=np.float32))
    np.testing.assert_allclose(jax.jit(lambda op, v: op.rapply(v))(A, y), y, rtol=1e-5)

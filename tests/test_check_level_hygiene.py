"""S1: checks run only when on, and self-adjointness is decided structurally where possible."""
import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def test_weight_values_are_checked_only_when_checks_are_on(ctx):
    negative = sc.WeightedInnerProduct(ctx.asarray([1.0, -1.0, 2.0]))
    sc.DenseCoordinateSpace((3,), ctx, geometry=negative, check_level="none")
    with pytest.raises(ValueError, match="positive"):
        sc.DenseCoordinateSpace((3,), ctx, geometry=negative, check_level="standard")
    with pytest.raises(ValueError, match="broadcast"):
        sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([1.0, 2.0])))


def test_scalar_weights_broadcast(ctx):
    scalar = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(2.0)))
    full = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 2.0, 2.0])))
    x, y = np.arange(1.0, 4.0), np.array([1.0, -1.0, 0.5])
    np.testing.assert_allclose(scalar.inner(x, y), 2 * x @ y)
    A = np.random.default_rng(0).standard_normal((3, 3))
    np.testing.assert_allclose(sc.DenseLinOp(ctx.asarray(A), scalar, scalar, ctx).rapply(y),
                               sc.DenseLinOp(ctx.asarray(A), full, full, ctx).rapply(y))
    assert scalar == full


def test_traced_weights_under_jit():
    jax = pytest.importorskip("jax")
    ctx = sc.Context(sc.JaxOps(), dtype=np.float32)
    x = ctx.asarray([1.0, 2.0, 3.0])

    def norm(w):
        X = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(w), check_level="none")
        return X.norm(x)

    w = ctx.asarray([1.0, 4.0, 9.0])
    np.testing.assert_allclose(jax.jit(norm)(w), np.sqrt(1 + 16 + 81), rtol=1e-5)


def test_solver_self_adjointness_probe_follows_check_level(ctx):
    X = sc.DenseCoordinateSpace((2,), ctx)
    upper = ctx.asarray([[2.0, 1.0], [0.0, 3.0]])
    with pytest.raises(ValueError, match="Hermitian"):
        sc.cg(sc.DenseLinOp(upper, X, X, ctx, check_level="standard"), ctx.asarray([1.0, 1.0]))
    sc.cg(sc.DenseLinOp(upper, X, X, ctx, check_level="none"), ctx.asarray([1.0, 1.0]), maxiter=2)


def test_structural_self_adjointness(ctx):
    rng = np.random.default_rng(1)
    X = sc.DenseCoordinateSpace((3,), ctx)
    Y = sc.DenseCoordinateSpace((2,), ctx)
    L = sc.DenseLinOp(ctx.asarray(rng.standard_normal((2, 3))), X, Y, ctx)
    V = sc.DenseLinOp(ctx.asarray(np.diag([1.0, 2.0])), Y, Y, ctx)
    assert (L.H @ V @ L).is_hermitian() is True
    assert (L.H @ L).is_hermitian() is True
    H = sc.DenseLinOp(ctx.asarray(np.eye(3)), X, X, ctx)
    block = sc.BlockMatrixLinOp([[H, L.H], [L, V]])
    assert block.is_hermitian() is True
    assert sc.BlockMatrixLinOp([[H, L.H], [L, L @ L.H + V]]).is_hermitian() is True
    W = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([1.0, 2.0, 3.0])))
    assert sc.DiagonalLinOp(ctx.asarray([4.0, 5.0, 6.0]), W, ctx).is_hermitian() is True


def test_declared_hermitian_matrix_free(ctx):
    X = sc.DenseCoordinateSpace((2,), ctx)
    S = np.array([[2.0, 1.0], [1.0, 3.0]])
    op = sc.MatrixFreeLinOp(lambda x: S @ x, lambda y: S @ y, X, X, ctx, hermitian=True)
    assert op.is_hermitian() is True
    assert op.convert(sc.Context(sc.NumpyOps(), dtype=np.float32)).is_hermitian() is True
    U = np.array([[2.0, 1.0], [0.0, 3.0]])
    with pytest.raises(ValueError):
        sc.MatrixFreeLinOp(lambda x: U @ x, lambda y: U.T @ y, X, X, ctx, hermitian=True, check_level="strict")


def test_strict_operator_metric_probes_the_inverse(ctx):
    E = sc.DenseCoordinateSpace((3,), ctx)
    M = sc.DenseLinOp(ctx.asarray(np.diag([1.0, 2.0, 3.0])), E, E, ctx)
    wrong = sc.DenseLinOp(ctx.asarray(np.eye(3)), E, E, ctx)
    with pytest.raises(ValueError, match="does not invert"):
        sc.DenseCoordinateSpace((3,), ctx, geometry=sc.OperatorInnerProduct(M, wrong), check_level="strict")
    sc.DenseCoordinateSpace((3,), ctx, geometry=sc.OperatorInnerProduct(M, wrong), check_level="standard")

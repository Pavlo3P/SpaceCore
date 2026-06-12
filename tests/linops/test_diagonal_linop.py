import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, to_numpy


def _ctx(dtype=np.float64, enable_checks=True):
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


def _weighted_space(weights, ctx):
    sc = importlib.import_module("spacecore")
    return sc.DenseCoordinateSpace(
        tuple(np.asarray(weights).shape), ctx, sc.WeightedInnerProduct(ctx.asarray(weights))
    )


def test_apply_and_rapply_flat_diagonal():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    x = ctx.asarray([4.0, -1.0, 0.5])

    np.testing.assert_allclose(op.apply(x), [4.0, -2.0, 1.5])
    np.testing.assert_allclose(op.rapply(x), [4.0, -2.0, 1.5])


def test_apply_and_rapply_tensor_shaped_diagonal():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2, 2), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), space, ctx)
    x = ctx.asarray([[2.0, -1.0], [0.5, 3.0]])

    np.testing.assert_allclose(op.apply(x), [[2.0, -2.0], [1.5, 12.0]])
    np.testing.assert_allclose(op.rapply(x), [[2.0, -2.0], [1.5, 12.0]])


def test_complex_diagonal_satisfies_adjoint_identity_and_hermitian_predicate():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(np.complex128)
    space = sc.DenseCoordinateSpace((2,), ctx)
    hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 0.0j, 2.0 + 0.0j]), space, ctx)
    non_hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 2.0j, 3.0 - 1.0j]), space, ctx)
    u = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
    v = ctx.asarray([1.5 + 0.5j, -2.0j])

    lhs = space.inner(non_hermitian.apply(u), v)
    rhs = space.inner(u, non_hermitian.rapply(v))

    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))
    assert hermitian.is_hermitian() is True
    assert non_hermitian.is_hermitian() is False
    np.testing.assert_allclose(
        non_hermitian.rapply(v), np.conj(np.asarray(non_hermitian.diagonal)) * np.asarray(v)
    )


def test_vapply_and_rvapply_with_leading_batch():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    xs = ctx.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])

    expected = ctx.asarray([[1.0, 4.0, 9.0], [4.0, -2.0, 1.5]])
    np.testing.assert_allclose(op.vapply(xs), expected)
    np.testing.assert_allclose(op.rvapply(xs), expected)


def test_to_dense_matches_numpy_diagonal_for_tensor_space():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2, 2), ctx)
    diagonal = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    op = sc.DiagonalLinOp(diagonal, space, ctx)

    expected = np.diag(np.asarray(diagonal).reshape((4,))).reshape((2, 2, 2, 2))

    np.testing.assert_allclose(op.to_matrix(), np.diag(np.asarray(diagonal).reshape((4,))))
    np.testing.assert_allclose(op.to_dense(), expected)


def test_weighted_diagonal_satisfies_metric_adjoint_identity_and_batches():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = _weighted_space([2.0, 5.0, 7.0], ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, -2.0, 0.5]), space, ctx)
    x = ctx.asarray([0.25, -1.5, 2.0])
    y = ctx.asarray([2.0, -0.5, 1.25])
    xs = ctx.asarray([[0.25, -1.5, 2.0], [3.0, 0.5, -1.0]])
    ys = ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 4.0, 0.75]])

    np.testing.assert_allclose(space.inner(op.apply(x), y), space.inner(x, op.rapply(y)))
    np.testing.assert_allclose(
        to_numpy(op.vapply(xs)), np.stack([to_numpy(op.apply(xi)) for xi in xs])
    )
    np.testing.assert_allclose(
        to_numpy(op.rvapply(ys)), np.stack([to_numpy(op.rapply(yi)) for yi in ys])
    )


def test_product_space_diagonal_uses_flatten_unflatten_paths():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    x1 = sc.DenseCoordinateSpace((2,), ctx)
    x2 = sc.DenseCoordinateSpace((1,), ctx)
    space = sc.TreeSpace.from_leaf_spaces((x1, x2), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0, 0.5]), space, ctx)
    x = (ctx.asarray([1.0, 3.0]), ctx.asarray([-2.0]))
    xs = (ctx.asarray([[1.0, 3.0], [-1.0, 4.0]]), ctx.asarray([[-2.0], [0.5]]))
    ys = (ctx.asarray([[2.0, -1.0], [0.25, 3.0]]), ctx.asarray([[4.0], [-2.0]]))

    y = op.apply(x)
    np.testing.assert_allclose(y[0], [2.0, -3.0])
    np.testing.assert_allclose(y[1], [-1.0])
    np.testing.assert_allclose(op.rapply(x)[0], [2.0, -3.0])
    np.testing.assert_allclose(op.rapply(x)[1], [-1.0])

    actual_v = op.vapply(xs)
    expected_v_rows = tuple(op.apply((xs[0][i], xs[1][i])) for i in range(xs[0].shape[0]))
    np.testing.assert_allclose(actual_v[0], np.stack([row[0] for row in expected_v_rows]))
    np.testing.assert_allclose(actual_v[1], np.stack([row[1] for row in expected_v_rows]))

    actual_rv = op.rvapply(ys)
    expected_rv_rows = tuple(op.rapply((ys[0][i], ys[1][i])) for i in range(ys[0].shape[0]))
    np.testing.assert_allclose(actual_rv[0], np.stack([row[0] for row in expected_rv_rows]))
    np.testing.assert_allclose(actual_rv[1], np.stack([row[1] for row in expected_rv_rows]))


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_pytree_flatten_unflatten_round_trip():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)

    leaves, treedef = jax.tree_util.tree_flatten(op)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored == op
    np.testing.assert_allclose(restored.apply(ctx.asarray([1.0, 1.0, 1.0])), [1.0, 2.0, 3.0])


def test_convert_changes_context_dtype():
    sc = importlib.import_module("spacecore")
    ctx64 = _ctx(np.float64)
    ctx32 = _ctx(np.float32)
    space = sc.DenseCoordinateSpace((3,), ctx64)
    op = sc.DiagonalLinOp(ctx64.asarray([1.0, 2.0, 3.0]), space, ctx64)

    converted = op._convert(ctx32)

    assert converted.ctx == ctx32
    assert converted.domain.ctx == ctx32
    assert converted.diagonal.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(converted.apply(ctx32.asarray([1.0, 1.0, 1.0])), [1.0, 2.0, 3.0])

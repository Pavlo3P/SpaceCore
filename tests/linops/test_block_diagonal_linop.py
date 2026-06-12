import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy


def _ctx(*, check_level="standard"):
    return sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)


def _weighted_space(weights, ctx):
    return sc.DenseCoordinateSpace(
        tuple(np.asarray(weights).shape), ctx, sc.WeightedInnerProduct(ctx.asarray(weights))
    )


def _assert_tree_allclose(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_allclose(actual[key], expected[key])
    elif isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected):
            _assert_tree_allclose(actual_leaf, expected_leaf)
    else:
        np.testing.assert_allclose(to_numpy(actual), expected)


def test_infers_tree_spaces_and_preserves_block_tree_structure():
    ctx = _ctx()
    x0 = sc.DenseCoordinateSpace((2,), ctx)
    x1 = sc.DenseCoordinateSpace((1,), ctx)
    y0 = sc.DenseCoordinateSpace((1,), ctx)
    y1 = sc.DenseCoordinateSpace((2,), ctx)
    blocks = {
        "left": sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x0, y0, ctx),
        "right": sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x1, y1, ctx),
    }

    op = sc.BlockDiagonalLinOp(blocks)

    assert isinstance(op.domain, sc.TreeSpace)
    assert isinstance(op.codomain, sc.TreeSpace)
    assert op.domain.treedef == op.codomain.treedef
    result = op.apply({"left": ctx.asarray([2.0, 4.0]), "right": ctx.asarray([5.0])})
    _assert_tree_allclose(result, {"left": [10.0], "right": [15.0, -5.0]})


def test_metric_adjoint_and_structural_double_adjoint():
    ctx = _ctx()
    x0 = _weighted_space([2.0, 5.0], ctx)
    x1 = _weighted_space([3.0], ctx)
    y0 = _weighted_space([7.0], ctx)
    y1 = _weighted_space([11.0, 13.0], ctx)
    blocks = (
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x0, y0, ctx),
        sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x1, y1, ctx),
    )
    op = sc.BlockDiagonalLinOp(blocks)
    x = (ctx.asarray([2.0, -1.0]), ctx.asarray([4.0]))
    y = (ctx.asarray([3.0]), ctx.asarray([1.0, -2.0]))

    np.testing.assert_allclose(
        to_numpy(op.codomain.inner(op.apply(x), y)),
        to_numpy(op.domain.inner(x, op.rapply(y))),
    )
    assert isinstance(op.H, sc.BlockDiagonalLinOp)
    assert op.H.H is op
    _assert_tree_allclose(op.H.apply(y), op.rapply(y))


def test_batched_apply_and_rapply_match_leafwise_loops():
    ctx = _ctx()
    x0 = sc.DenseCoordinateSpace((2,), ctx)
    x1 = sc.DenseCoordinateSpace((1,), ctx)
    y0 = sc.DenseCoordinateSpace((1,), ctx)
    y1 = sc.DenseCoordinateSpace((2,), ctx)
    op = sc.BlockDiagonalLinOp(
        (
            sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x0, y0, ctx),
            sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x1, y1, ctx),
        )
    )
    xs = (ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), ctx.asarray([[5.0], [6.0]]))
    ys = (ctx.asarray([[2.0], [-1.0]]), ctx.asarray([[3.0, 4.0], [5.0, 6.0]]))

    expected_y = tuple(np.stack([to_numpy(op.apply((xs[0][i], xs[1][i]))[j]) for i in range(2)]) for j in range(2))
    expected_x = tuple(np.stack([to_numpy(op.rapply((ys[0][i], ys[1][i]))[j]) for i in range(2)]) for j in range(2))
    _assert_tree_allclose(op.vapply(xs), expected_y)
    _assert_tree_allclose(op.rvapply(ys), expected_x)


def test_algebra_and_block_validation():
    ctx = _ctx()
    x = sc.DenseCoordinateSpace((1,), ctx)
    block = sc.DenseLinOp(ctx.asarray([[2.0]]), x, x, ctx)
    op = sc.BlockDiagonalLinOp((block, block))
    value = (ctx.asarray([3.0]), ctx.asarray([4.0]))

    _assert_tree_allclose((op + op).apply(value), ([12.0], [16.0]))
    _assert_tree_allclose((3.0 * op).apply(value), ([18.0], [24.0]))
    _assert_tree_allclose((op @ op).apply(value), ([12.0], [16.0]))

    with pytest.raises(TypeError, match="every block"):
        sc.BlockDiagonalLinOp((block, object()))
    other_ctx = _ctx(check_level="cheap")
    other_x = sc.DenseCoordinateSpace((1,), other_ctx)
    other = sc.IdentityLinOp(other_x)
    with pytest.raises(ValueError, match="check policy"):
        sc.BlockDiagonalLinOp((block, other))

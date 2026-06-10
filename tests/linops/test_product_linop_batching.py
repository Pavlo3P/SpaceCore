import importlib

import numpy as np
import pytest

from tests._helpers import to_numpy


sc = importlib.import_module("spacecore")


def _weighted_space(weights, ctx):
    return sc.DenseCoordinateSpace(
        tuple(np.asarray(weights).shape), ctx, sc.WeightedInnerProduct(ctx.asarray(weights))
    )


def _assert_product_allclose(actual, expected):
    actual_np = to_numpy(actual)
    expected_np = to_numpy(expected)
    if isinstance(actual_np, tuple):
        assert isinstance(expected_np, tuple)
        assert len(actual_np) == len(expected_np)
        for a, e in zip(actual_np, expected_np):
            _assert_product_allclose(a, e)
    else:
        np.testing.assert_allclose(actual_np, expected_np, rtol=1e-7, atol=1e-7)


def _slice_product_batch(xs, i):
    return tuple(_slice_product_batch(xi, i) if isinstance(xi, tuple) else xi[i] for xi in xs)


def _stack_product_rows(rows):
    if isinstance(rows[0], tuple):
        return tuple(
            _stack_product_rows(tuple(row[i] for row in rows)) for i in range(len(rows[0]))
        )
    return np.stack([to_numpy(row) for row in rows], axis=0)


def _assert_vapply_loop(op, xs):
    rows = tuple(op.apply(_slice_product_batch(xs, i)) for i in range(xs[0].shape[0]))
    _assert_product_allclose(op.vapply(xs), _stack_product_rows(rows))


def _assert_rvapply_loop(op, ys):
    rows = tuple(op.rapply(_slice_product_batch(ys, i)) for i in range(ys[0].shape[0]))
    _assert_product_allclose(op.rvapply(ys), _stack_product_rows(rows))


def _inner(space, x, y):
    return space.inner(x, y)


def test_space_add_batch_for_vector_and_recursive_product_space():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    vector = sc.DenseCoordinateSpace((2,), ctx)
    product = sc.ProductSpace(
        (vector, sc.ProductSpace((sc.DenseCoordinateSpace((1,), ctx), vector), ctx)), ctx
    )

    x = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    y = ctx.asarray([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose(vector.add_batch(x, y), [[6.0, 8.0], [10.0, 12.0]])

    px = (x, (ctx.asarray([[1.0], [2.0]]), y))
    py = (y, (ctx.asarray([[3.0], [4.0]]), x))
    actual = product.add_batch(px, py)
    _assert_product_allclose(
        actual,
        (
            np.asarray([[6.0, 8.0], [10.0, 12.0]]),
            (
                np.asarray([[4.0], [6.0]]),
                np.asarray([[6.0, 8.0], [10.0, 12.0]]),
            ),
        ),
    )


def test_product_linops_use_space_add_batch_for_accumulation():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class CountingVectorSpace(sc.DenseCoordinateSpace):
        def __init__(self, shape, ctx, counter):
            self.counter = counter
            super().__init__(shape, ctx)

        def add_batch(self, x, y):
            self.counter["calls"] += 1
            return super().add_batch(x, y)

        def _convert(self, new_ctx):
            return CountingVectorSpace(self.shape, new_ctx, self.counter)

    counter = {"calls": 0}
    shared = CountingVectorSpace((2,), ctx, counter)
    cod1, cod2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), shared, cod1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[3.0, -1.0]]), shared, cod2, ctx)
    stacked = sc.StackedLinOp.from_operators((A1, A2))
    ys = (ctx.asarray([[1.0], [2.0]]), ctx.asarray([[3.0], [4.0]]))

    stacked.rvapply(ys)
    assert counter["calls"] == 1

    counter["calls"] = 0
    dom1, dom2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    B1 = sc.DenseLinOp(ctx.asarray([[1.0], [2.0]]), dom1, shared, ctx)
    B2 = sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), dom2, shared, ctx)
    summed = sc.SumToSingleLinOp.from_operators((B1, B2))
    xs = (ctx.asarray([[1.0], [2.0]]), ctx.asarray([[3.0], [4.0]]))

    summed.vapply(xs)
    assert counter["calls"] == 1


def test_block_diagonal_vapply_and_rvapply_match_loops_for_two_and_three_components():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x1, x2, x3 = (
        sc.DenseCoordinateSpace((2,), ctx),
        sc.DenseCoordinateSpace((1,), ctx),
        sc.DenseCoordinateSpace((2,), ctx),
    )
    y1, y2, y3 = (
        sc.DenseCoordinateSpace((1,), ctx),
        sc.DenseCoordinateSpace((2,), ctx),
        sc.DenseCoordinateSpace((2,), ctx),
    )
    parts = (
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x1, y1, ctx),
        sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x2, y2, ctx),
        sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.5, 4.0]]), x3, y3, ctx),
    )

    op2 = sc.BlockDiagonalLinOp.from_operators(parts[:2])
    xs2 = (ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]), ctx.asarray([[3.0], [4.0]]))
    ys2 = (ctx.asarray([[5.0], [-2.0]]), ctx.asarray([[1.0, 2.0], [0.5, -1.0]]))
    _assert_vapply_loop(op2, xs2)
    _assert_rvapply_loop(op2, ys2)

    op3 = sc.BlockDiagonalLinOp.from_operators(parts)
    xs3 = xs2 + (ctx.asarray([[2.0, -3.0], [0.25, 1.5]]),)
    ys3 = ys2 + (ctx.asarray([[1.0, -1.0], [2.0, 0.5]]),)
    _assert_vapply_loop(op3, xs3)
    _assert_rvapply_loop(op3, ys3)


def test_stacked_adjoint_identity_and_batched_rvapply_with_weighted_space():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = _weighted_space([2.0, 5.0], ctx)
    cod1 = _weighted_space([3.0, 7.0], ctx)
    cod2 = _weighted_space([11.0], ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), domain, cod1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[0.5, 4.0]]), domain, cod2, ctx)
    op = sc.StackedLinOp.from_operators((A1, A2))
    x = ctx.asarray([1.0, -2.0])
    y = (ctx.asarray([0.5, 3.0]), ctx.asarray([-1.0]))

    np.testing.assert_allclose(
        _inner(op.codomain, op.apply(x), y), _inner(op.domain, x, op.rapply(y))
    )

    ys = (ctx.asarray([[0.5, 3.0], [1.0, -2.0]]), ctx.asarray([[-1.0], [4.0]]))
    rows = tuple(op.rapply(_slice_product_batch(ys, i)) for i in range(2))
    np.testing.assert_allclose(to_numpy(op.rvapply(ys)), _stack_product_rows(rows))


def test_sum_to_single_adjoint_identity_and_batched_vapply_with_weighted_space():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom1 = _weighted_space([2.0, 5.0], ctx)
    dom2 = _weighted_space([3.0], ctx)
    cod = _weighted_space([7.0, 11.0], ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), dom1, cod, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[0.5], [4.0]]), dom2, cod, ctx)
    op = sc.SumToSingleLinOp.from_operators((A1, A2))
    x = (ctx.asarray([1.0, -2.0]), ctx.asarray([3.0]))
    y = ctx.asarray([0.5, 3.0])

    np.testing.assert_allclose(
        _inner(op.codomain, op.apply(x), y), _inner(op.domain, x, op.rapply(y))
    )

    xs = (ctx.asarray([[1.0, -2.0], [0.5, 4.0]]), ctx.asarray([[3.0], [-1.0]]))
    rows = tuple(op.apply(_slice_product_batch(xs, i)) for i in range(2))
    np.testing.assert_allclose(to_numpy(op.vapply(xs)), _stack_product_rows(rows))


def test_product_linop_batch_checks_reject_wrong_tuple_layout():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
    x1, x2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    y1, y2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x1, y1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x2, y2, ctx)
    op = sc.BlockDiagonalLinOp.from_operators((A1, A2))

    with pytest.raises(ValueError, match="tuple of length"):
        op.vapply((ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]),))
    with pytest.raises(ValueError, match="trailing shape"):
        op.vapply((ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]), ctx.asarray([[3.0, 4.0]])))

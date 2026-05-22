import importlib

import numpy as np
import scipy.sparse as sps


def _ctx():
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _unchecked_ctx():
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)


def _stack_apply(ctx, op, xs):
    return ctx.ops.stack(tuple(op.apply(x) for x in xs), axis=0)


def _stack_rapply(ctx, op, ys):
    return ctx.ops.stack(tuple(op.rapply(y) for y in ys), axis=0)


def test_dense_linop_vapply_and_rvapply_match_stacked_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(matrix, dom, cod, ctx)
    xs = ctx.asarray([[7.0, 8.0], [1.0, -1.0], [0.5, 2.0]])
    ys = ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

    assert np.allclose(op.vapply(xs), _stack_apply(ctx, op, xs))
    assert np.allclose(op.rvapply(ys), _stack_rapply(ctx, op, ys))


def test_sparse_linop_vapply_and_rvapply_match_stacked_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = np.array([[1.0, 0.0], [0.0, 4.0], [5.0, 6.0]])
    op = sc.SparseLinOp(ctx.assparse(sps.csr_matrix(dense)), dom, cod, ctx)
    xs = ctx.asarray([[7.0, 8.0], [1.0, -1.0], [0.5, 2.0]])
    ys = ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

    assert np.allclose(op.vapply(xs), _stack_apply(ctx, op, xs))
    assert np.allclose(op.rvapply(ys), _stack_rapply(ctx, op, ys))


def test_dense_and_sparse_batched_lifting_fast_paths_without_checks():
    sc = importlib.import_module("spacecore")
    ctx = _unchecked_ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sparse = ctx.assparse(sps.csr_matrix([[1.0, 0.0], [0.0, 4.0], [5.0, 6.0]]))
    dense_op = sc.DenseLinOp(matrix, dom, cod, ctx)
    sparse_op = sc.SparseLinOp(sparse, dom, cod, ctx)
    batch_dom = dom.batch((3,), (0,))
    batch_cod = cod.batch((2,), (0,))
    xs = ctx.asarray([[7.0, 8.0], [1.0, -1.0], [0.5, 2.0]])
    ys = ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

    assert np.allclose(dense_op.vapply(xs, batch_dom), np.asarray(xs) @ np.asarray(matrix).T)
    assert np.allclose(dense_op.rvapply(ys, batch_cod), np.asarray(ys) @ np.asarray(matrix))
    assert np.allclose(sparse_op.vapply(xs, batch_dom), (sparse @ np.asarray(xs).T).T)
    assert np.allclose(sparse_op.rvapply(ys, batch_cod), (sparse.T @ np.asarray(ys).T).T)


def test_diagonal_identity_zero_sum_composed_and_adjoint_batched_lifting():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((3,), ctx)
    d1 = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    d2 = sc.DiagonalLinOp(ctx.asarray([-1.0, 0.5, 4.0]), space, ctx)
    identity = sc.IdentityLinOp(space, ctx)
    zero = sc.ZeroLinOp(space, space, ctx)
    summed = d1 + d2 + zero
    composed = d1 @ (d2 + identity)
    adjoint = composed.H
    xs = ctx.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 0.0]])

    for op in (d1, identity, zero, summed, composed):
        assert np.allclose(op.vapply(xs), _stack_apply(ctx, op, xs))
        assert np.allclose(op.rvapply(xs), _stack_rapply(ctx, op, xs))
    assert np.allclose(adjoint.vapply(xs), _stack_apply(ctx, adjoint, xs))
    assert np.allclose(adjoint.rvapply(xs), _stack_rapply(ctx, adjoint, xs))


def test_matrix_free_vapply_uses_callback_when_supplied_and_fallback_when_absent():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    calls = {"vapply": 0, "rvapply": 0}

    def apply(x):
        return ctx.asarray(matrix @ np.asarray(x))

    def rapply(y):
        return ctx.asarray(matrix.T @ np.asarray(y))

    def vapply(xs):
        calls["vapply"] += 1
        return ctx.asarray(np.asarray(xs) @ matrix.T)

    def rvapply(ys):
        calls["rvapply"] += 1
        return ctx.asarray(np.asarray(ys) @ matrix)

    with_callbacks = sc.MatrixFreeLinOp(apply, rapply, dom, cod, ctx, vapply, rvapply)
    fallback = sc.MatrixFreeLinOp(apply, rapply, dom, cod, ctx)
    xs = ctx.asarray([[7.0, 8.0], [1.0, -1.0]])
    ys = ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

    assert np.allclose(with_callbacks.vapply(xs), _stack_apply(ctx, with_callbacks, xs))
    assert np.allclose(with_callbacks.rvapply(ys), _stack_rapply(ctx, with_callbacks, ys))
    assert calls == {"vapply": 1, "rvapply": 1}
    assert np.allclose(fallback.vapply(xs), _stack_apply(ctx, fallback, xs))
    assert np.allclose(fallback.rvapply(ys), _stack_rapply(ctx, fallback, ys))


def test_product_linops_batched_lifting_matches_stacked_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    x0 = sc.VectorSpace((2,), ctx)
    x1 = sc.VectorSpace((3,), ctx)
    y0 = sc.VectorSpace((3,), ctx)
    y1 = sc.VectorSpace((2,), ctx)
    a0 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), x0, y0, ctx)
    a1 = sc.DenseLinOp(ctx.asarray([[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]]), x1, y1, ctx)
    s1 = sc.DenseLinOp(ctx.asarray([[2.0, -1.0], [0.0, 3.0]]), x0, y1, ctx)
    block = sc.BlockDiagonalLinOp.from_operators((a0, a1))
    stacked = sc.StackedLinOp.from_operators((a0, s1))
    sum_to_single = sc.SumToSingleLinOp.from_operators((a0.H, s1.H))

    xb = (ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), ctx.asarray([[5.0, 6.0, 7.0], [1.0, -1.0, 0.5]]))
    yb = (ctx.asarray([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]), ctx.asarray([[2.0, 1.0], [3.0, -2.0]]))
    single_x = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    single_y = ctx.asarray([[1.0, 2.0], [3.0, -2.0]])

    block_v = block.vapply(xb)
    block_expected = tuple(ctx.ops.stack(tuple(block.apply((xb[0][i], xb[1][i]))[j] for i in range(2))) for j in range(2))
    assert np.allclose(block_v[0], block_expected[0])
    assert np.allclose(block_v[1], block_expected[1])

    block_rv = block.rvapply(yb)
    block_r_expected = tuple(ctx.ops.stack(tuple(block.rapply((yb[0][i], yb[1][i]))[j] for i in range(2))) for j in range(2))
    assert np.allclose(block_rv[0], block_r_expected[0])
    assert np.allclose(block_rv[1], block_r_expected[1])

    stacked_v = stacked.vapply(single_x)
    stacked_expected = tuple(ctx.ops.stack(tuple(stacked.apply(single_x[i])[j] for i in range(2))) for j in range(2))
    assert np.allclose(stacked_v[0], stacked_expected[0])
    assert np.allclose(stacked_v[1], stacked_expected[1])

    assert np.allclose(
        stacked.rvapply(yb),
        ctx.ops.stack(tuple(stacked.rapply((yb[0][i], yb[1][i])) for i in range(2))),
    )
    assert np.allclose(
        sum_to_single.vapply(yb),
        ctx.ops.stack(tuple(sum_to_single.apply((yb[0][i], yb[1][i])) for i in range(2))),
    )
    sum_rv = sum_to_single.rvapply(single_y)
    sum_r_expected = tuple(
        ctx.ops.stack(tuple(sum_to_single.rapply(single_y[i])[j] for i in range(2)))
        for j in range(2)
    )
    assert np.allclose(sum_rv[0], sum_r_expected[0])
    assert np.allclose(sum_rv[1], sum_r_expected[1])

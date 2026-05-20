import importlib

import numpy as np
import scipy.sparse as sps


def _ctx():
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _assert_to_dense_matches_apply(op, x):
    dense = op.to_dense()
    matrix = dense.reshape((np.prod(op.codomain.shape), np.prod(op.domain.shape)))
    y_from_dense = matrix @ op.domain.flatten(x)
    y_from_apply = op.codomain.flatten(op.apply(x))
    assert np.allclose(y_from_dense, y_from_apply)


def test_dense_linop_to_dense_returns_stored_matrix_and_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    A = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(A, dom, cod, ctx)

    assert op.to_dense() is A
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))


def test_sparse_linop_to_dense_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.SparseLinOp(sps.csr_matrix(dense), dom, cod, ctx)

    assert np.allclose(op.to_dense(), dense)
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))


def test_identity_linop_to_dense_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2, 2), ctx)
    op = sc.IdentityLinOp(space, ctx)

    assert np.allclose(op.to_dense().reshape((4, 4)), np.eye(4))
    _assert_to_dense_matches_apply(op, ctx.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_zero_linop_to_dense_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    op = sc.ZeroLinOp(dom, cod, ctx)

    assert np.allclose(op.to_dense(), np.zeros((3, 2)))
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))


def test_matrix_free_linop_to_dense_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.MatrixFreeLinOp(
        lambda x: ctx.asarray(dense @ np.asarray(x)),
        lambda y: ctx.asarray(dense.T @ np.asarray(y)),
        dom,
        cod,
        ctx,
    )

    assert np.allclose(op.to_dense(), dense)
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))


def test_sum_linop_to_dense_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dom, cod, ctx)
    B = sc.DenseLinOp(ctx.asarray([[0.5, 1.0], [-1.0, 2.0], [3.0, -0.5]]), dom, cod, ctx)
    op = A + B

    assert np.allclose(op.to_dense(), A.to_dense() + B.to_dense())
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))

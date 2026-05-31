import importlib

import numpy as np
import pytest
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


def test_dense_linop_A_returns_stored_dense_representation():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    A = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(A, dom, cod, ctx)

    assert op.A is A


def test_sparse_linop_to_dense_matches_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.SparseLinOp(sps.csr_matrix(dense), dom, cod, ctx)

    assert np.allclose(op.to_dense(), dense)
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))


def test_sparse_linop_A_returns_stored_sparse_representation():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    A = sps.csr_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.SparseLinOp(A, dom, cod, ctx)

    assert op.A is A


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
    assert op.to_matrix().shape == (3, 2)
    assert np.allclose(op.to_matrix(), dense)
    _assert_to_dense_matches_apply(op, ctx.asarray([7.0, 8.0]))


def test_default_to_matrix_uses_batched_vapply_path():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    class BatchedMaterializationLinOp(sc.LinOp):
        def __init__(self):
            super().__init__(dom, cod, ctx)
            self.apply_calls = 0
            self.vapply_calls = 0

        def apply(self, x):
            self.apply_calls += 1
            return dense @ x

        def rapply(self, y):
            return dense.T @ y

        def vapply(self, xs):
            self.vapply_calls += 1
            return xs @ dense.T

        def tree_flatten(self):
            return (), (self.domain, self.codomain, self.ctx)

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls()

    op = BatchedMaterializationLinOp()

    matrix = op.to_matrix()

    assert op.vapply_calls == 1
    assert op.apply_calls == 0
    assert matrix.shape == (3, 2)
    assert np.allclose(matrix, dense)
    assert np.allclose(op.to_dense(), dense)


def test_default_to_matrix_falls_back_when_batch_helpers_unavailable():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()

    class NoBatchVectorSpace(sc.VectorSpace):
        def flatten_batch(self, xs):
            raise NotImplementedError

        def unflatten_batch(self, vs):
            raise NotImplementedError

        def _convert(self, new_ctx):
            if new_ctx == self.ctx:
                return self
            return NoBatchVectorSpace(self.shape, new_ctx)

    dom = NoBatchVectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    class FallbackMaterializationLinOp(sc.LinOp):
        def __init__(self):
            super().__init__(dom, cod, ctx)
            self.apply_calls = 0
            self.vapply_calls = 0

        def apply(self, x):
            self.apply_calls += 1
            return dense @ x

        def rapply(self, y):
            return dense.T @ y

        def vapply(self, xs):
            self.vapply_calls += 1
            return super().vapply(xs)

        def tree_flatten(self):
            return (), (self.domain, self.codomain, self.ctx)

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls()

    op = FallbackMaterializationLinOp()

    matrix = op.to_matrix()

    assert op.vapply_calls == 0
    assert op.apply_calls == 2
    assert matrix.shape == (3, 2)
    assert np.allclose(matrix, dense)
    assert np.allclose(op.to_dense(), dense)


def test_matrix_free_linop_A_is_not_implemented_by_default():
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

    with pytest.raises(NotImplementedError, match="native numerical representation"):
        _ = op.A


def test_custom_linop_can_define_A_representation():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    class CustomLinOp(sc.LinOp):
        @property
        def A(self):
            return {"backend": "custom", "data": dense}

        def apply(self, x):
            return ctx.asarray(np.asarray(dense) @ np.asarray(x))

        def rapply(self, y):
            return ctx.asarray(np.asarray(dense).T @ np.asarray(y))

        def tree_flatten(self):
            return (), (self.domain, self.codomain, self.ctx)

        @classmethod
        def tree_unflatten(cls, aux, children):
            domain, codomain, ctx = aux
            return cls(domain, codomain, ctx)

    op = CustomLinOp(dom, cod, ctx)

    assert op.A["backend"] == "custom"
    assert op.A["data"] is dense


def test_diagonal_linop_A_is_cached():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)

    A = op.A

    assert op.A is A
    assert np.allclose(A, np.diag([1.0, 2.0, 3.0]))


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

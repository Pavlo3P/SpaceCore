import importlib

import numpy as np
import scipy.sparse as sps


class TransposeCountingCSR(sps.csr_matrix):
    def __init__(self, *args, counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter

    @property
    def T(self):
        if self.counter is not None:
            self.counter["calls"] += 1
        return super().T


def test_sparse_linop_construct_apply_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = np.array([[1., 2.], [3., 4.], [5., 6.]])
    op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)

    x = ctx.asarray([7., 8.])
    y = ctx.asarray([1., -1., 2.])

    assert np.allclose(op.apply(x), dense @ np.asarray(x))
    assert np.allclose(op.rapply(y), dense.T @ np.asarray(y))


def test_sparse_linop_reuses_cached_transpose():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    counter = {"calls": 0}
    A = TransposeCountingCSR([[1., 2.], [3., 4.], [5., 6.]], counter=counter)

    op = sc.SparseLinOp(A, dom, cod, ctx)
    transpose_calls = counter["calls"]

    op.rapply(ctx.asarray([1., -1., 2.]))
    op.rapply(ctx.asarray([3., -2., 1.]))

    assert transpose_calls == 1
    assert counter["calls"] == transpose_calls

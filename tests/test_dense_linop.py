import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype, to_numpy


class ReshapeCountingArray(np.ndarray):
    def __new__(cls, data, counter):
        obj = np.asarray(data).view(cls)
        obj.counter = counter
        obj._track_reshape = True
        return obj

    def __array_finalize__(self, obj):
        self.counter = getattr(obj, "counter", None)
        self._track_reshape = False

    def reshape(self, *shape, **kwargs):
        if self.counter is not None and self._track_reshape:
            self.counter["calls"] += 1
        return super().reshape(*shape, **kwargs)


def test_dense_linop_construct_apply_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    A = ctx.asarray([[1.,2.],[3.,4.],[5.,6.]])
    op = sc.DenseLinOp(A, dom, cod, ctx)
    x = ctx.asarray([7.,8.])
    y = ctx.asarray([1.,-1.,2.])
    assert np.allclose(op.apply(x), np.array([[1.,2.],[3.,4.],[5.,6.]]) @ np.array([7.,8.]))
    assert np.allclose(op.rapply(y), np.array([[1.,2.],[3.,4.],[5.,6.]]).T @ np.array([1.,-1.,2.]))


def test_dense_linop_reuses_cached_matrix_reshape():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    counter = {"calls": 0}
    A = ReshapeCountingArray([[1., 2.], [3., 4.], [5., 6.]], counter)

    op = sc.DenseLinOp(A, dom, cod, ctx)
    matrix_reshape_calls = counter["calls"]

    op.apply(ctx.asarray([7., 8.]))
    op.rapply(ctx.asarray([1., -1., 2.]))
    op.apply(ctx.asarray([9., 10.]))
    op.rapply(ctx.asarray([3., -2., 1.]))

    assert matrix_reshape_calls == 1
    assert counter["calls"] == matrix_reshape_calls


def test_dense_linop_bad_shape_raises():
    sc = importlib.import_module("spacecore")
    import pytest
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    with pytest.raises(Exception):
        sc.DenseLinOp(ctx.asarray([[1.,2.],[3.,4.]]), sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx), ctx)


def test_dense_linop_convert_preserves_action():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    op = sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.],[5.,6.]]), sc.VectorSpace((2,), src), sc.VectorSpace((3,), src), src)
    op2 = op.convert(dst)
    x = op2.ctx.asarray([7.,8.])
    assert np.allclose(to_numpy(op2.apply(x)), [23.,53.,83.])


def test_dense_linop_convert_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    src = sc.Context(sc.NumpyOps(), dtype=dt)
    dst = sc.Context(sc.JaxOps(), dtype=dt)
    op = sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.],[5.,6.]]), sc.VectorSpace((2,), src), sc.VectorSpace((3,), src), src)
    op2 = op.convert(dst)
    assert op2.ctx.ops.family == "jax"

import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype, to_numpy


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

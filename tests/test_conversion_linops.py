import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype, to_numpy


def test_dense_linop_conversion_preserves_action():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    op = sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.],[5.,6.]]), sc.VectorSpace((2,),src), sc.VectorSpace((3,),src), src)
    op2 = op.convert(dst)
    y = op2.apply(op2.ctx.asarray([7.,8.]))
    assert np.allclose(to_numpy(y), [23.,53.,83.])


def test_product_linop_conversion_same_backend_preserves_action():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((2,),src)
    Y1,Y2 = sc.VectorSpace((2,),src), sc.VectorSpace((1,),src)
    op = sc.StackedLinOp.from_operators((
        sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.]]), X, Y1, src),
        sc.DenseLinOp(src.asarray([[5.,6.]]), X, Y2, src),
    ))
    op2 = op.convert(dst)
    y = op2.apply(op2.ctx.asarray([10.,20.]))
    assert np.allclose(to_numpy(y[0]), [50.,110.])


def test_linop_conversion_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    src = sc.Context(sc.NumpyOps(), dtype=dt)
    dst = sc.Context(sc.JaxOps(), dtype=dt)
    op = sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.],[5.,6.]]), sc.VectorSpace((2,),src), sc.VectorSpace((3,),src), src)
    op2 = op.convert(dst)
    assert op2.ctx.ops.family == 'jax'

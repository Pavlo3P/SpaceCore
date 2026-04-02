import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype, to_numpy


def test_smoke_numpy_workflow():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((2,), ctx)
    Y = sc.VectorSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.,2.],[3.,4.],[5.,6.]]), X, Y, ctx)
    x = ctx.asarray([7.,8.])
    y = op.apply(x)
    assert np.allclose(to_numpy(y), [23.,53.,83.])
    P = sc.ProductSpace((X,Y), ctx)
    z = P.unflatten(P.flatten((x,y)))
    assert np.allclose(to_numpy(z[0]), to_numpy(x))
    assert np.allclose(to_numpy(z[1]), to_numpy(y))


def test_smoke_jax_conversion_workflow_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ctx = sc.Context(sc.NumpyOps(), dtype=dt)
    jx_ctx = sc.Context(sc.JaxOps(), dtype=dt)
    X = sc.VectorSpace((2,), np_ctx)
    Y = sc.VectorSpace((3,), np_ctx)
    op = sc.DenseLinOp(np_ctx.asarray([[1.,2.],[3.,4.],[5.,6.]]), X, Y, np_ctx)
    op2 = op.convert(jx_ctx)
    y = op2.apply(jx_ctx.asarray([7.,8.]))
    assert np.allclose(to_numpy(y), [23.,53.,83.])

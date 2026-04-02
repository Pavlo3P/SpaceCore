import importlib
import numpy as np
from ._helpers import to_numpy


def test_stacked_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((2,),ctx)
    Y1,Y2 = sc.VectorSpace((2,),ctx), sc.VectorSpace((1,),ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.,2.],[3.,4.]]), X, Y1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[5.,6.]]), X, Y2, ctx)
    op = sc.StackedLinOp.from_operators((A1,A2))
    y = op.apply(ctx.asarray([10.,20.]))
    assert np.allclose(to_numpy(y[0]), [50.,110.])
    assert np.allclose(to_numpy(y[1]), [170.])
    x = op.rapply((ctx.asarray([2.,-1.]), ctx.asarray([3.])))
    assert np.allclose(to_numpy(x), [14.,18.])

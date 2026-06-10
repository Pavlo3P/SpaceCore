import importlib
import numpy as np
from tests._helpers import to_numpy


def test_stacked_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y1, Y2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X, Y1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[5.0, 6.0]]), X, Y2, ctx)
    op = sc.StackedLinOp.from_operators((A1, A2))
    y = op.apply(ctx.asarray([10.0, 20.0]))
    assert np.allclose(to_numpy(y[0]), [50.0, 110.0])
    assert np.allclose(to_numpy(y[1]), [170.0])
    x = op.rapply((ctx.asarray([2.0, -1.0]), ctx.asarray([3.0])))
    assert np.allclose(to_numpy(x), [14.0, 18.0])

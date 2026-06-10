import importlib
import numpy as np
from tests._helpers import to_numpy


def test_sum_to_single_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X1, X2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)
    Y = sc.DenseCoordinateSpace((2,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X1, Y, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), X2, Y, ctx)
    op = sc.SumToSingleLinOp.from_operators((A1, A2))
    y = op.apply((ctx.asarray([10.0, 20.0]), ctx.asarray([1.0, 2.0, 3.0])))
    assert np.allclose(to_numpy(y), [88.0, 166.0])
    x = op.rapply(ctx.asarray([2.0, -1.0]))
    assert np.allclose(to_numpy(x[0]), [-1.0, 0.0])
    assert np.allclose(to_numpy(x[1]), [2.0, 3.0, 4.0])

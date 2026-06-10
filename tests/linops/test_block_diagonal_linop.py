import importlib
import numpy as np
from tests._helpers import to_numpy


def test_block_diagonal_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X1, X2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)
    Y1, Y2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X1, Y1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[5.0, 6.0, 7.0]]), X2, Y2, ctx)
    op = sc.BlockDiagonalLinOp.from_operators((A1, A2))
    y = op.apply((ctx.asarray([10.0, 20.0]), ctx.asarray([1.0, 2.0, 3.0])))
    assert np.allclose(to_numpy(y[0]), [50.0, 110.0])
    assert np.allclose(to_numpy(y[1]), [38.0])
    x = op.rapply((ctx.asarray([2.0, -1.0]), ctx.asarray([3.0])))
    assert np.allclose(to_numpy(x[0]), [-1.0, 0.0])
    assert np.allclose(to_numpy(x[1]), [15.0, 18.0, 21.0])


def test_block_diagonal_empty_raises():
    sc = importlib.import_module("spacecore")
    import pytest

    with pytest.raises(Exception):
        sc.BlockDiagonalLinOp.from_operators(())

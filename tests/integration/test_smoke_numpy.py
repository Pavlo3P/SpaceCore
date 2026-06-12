import importlib

import numpy as np

from tests._helpers import to_numpy


def test_smoke_numpy_workflow():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x_space = sc.DenseCoordinateSpace((2,), ctx)
    y_space = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), x_space, y_space, ctx)

    x = ctx.asarray([7.0, 8.0])
    y = op.apply(x)

    assert np.allclose(to_numpy(y), [23.0, 53.0, 83.0])
    product = sc.TreeSpace.from_leaf_spaces((x_space, y_space), ctx)
    roundtrip = product.unflatten(product.flatten((x, y)))
    assert np.allclose(to_numpy(roundtrip[0]), to_numpy(x))
    assert np.allclose(to_numpy(roundtrip[1]), to_numpy(y))

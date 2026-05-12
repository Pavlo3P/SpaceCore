import importlib

import numpy as np

from tests._helpers import to_numpy


def test_smoke_numpy_workflow():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x_space = sc.VectorSpace((2,), ctx)
    y_space = sc.VectorSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1., 2.], [3., 4.], [5., 6.]]), x_space, y_space, ctx)

    x = ctx.asarray([7., 8.])
    y = op.apply(x)

    assert np.allclose(to_numpy(y), [23., 53., 83.])
    product = sc.ProductSpace((x_space, y_space), ctx)
    roundtrip = product.unflatten(product.flatten((x, y)))
    assert np.allclose(to_numpy(roundtrip[0]), to_numpy(x))
    assert np.allclose(to_numpy(roundtrip[1]), to_numpy(y))

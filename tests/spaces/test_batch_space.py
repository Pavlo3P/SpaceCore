import importlib

import numpy as np


def test_vector_space_batch_wrapper_shape_and_membership():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x = sc.VectorSpace((2, 3), ctx)

    xb = x.batch(batch_shape=(4,), batch_axes=(0,))

    assert isinstance(xb, sc.BatchSpace)
    assert xb.base == x
    assert xb.batch_shape == (4,)
    assert xb.batch_axes == (0,)
    assert xb.shape == (4, 2, 3)
    xb.check_member(ctx.ops.zeros((4, 2, 3), dtype=ctx.dtype))


def test_product_space_batch_wrapper_validates_component_batches():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x0 = sc.VectorSpace((2,), ctx)
    x1 = sc.VectorSpace((3,), ctx)
    product = sc.ProductSpace((x0, x1), ctx)
    batched = product.batch((5,), (0,))

    value = (
        ctx.ops.zeros((5, 2), dtype=ctx.dtype),
        ctx.ops.zeros((5, 3), dtype=ctx.dtype),
    )

    assert batched.shape == (5, 5)
    batched.check_member(value)
    zeros = batched.zeros()
    assert np.allclose(zeros[0], np.zeros((5, 2)))
    assert np.allclose(zeros[1], np.zeros((5, 3)))

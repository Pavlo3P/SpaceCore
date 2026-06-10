import importlib
import numpy as np
from tests._helpers import has_jax, jax_real_dtype, to_numpy


def test_dense_linop_conversion_preserves_action():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    op = sc.DenseLinOp(
        src.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        sc.DenseCoordinateSpace((2,), src),
        sc.DenseCoordinateSpace((3,), src),
        src,
    )
    op2 = op.convert(dst)
    y = op2.apply(op2.ctx.asarray([7.0, 8.0]))
    assert np.allclose(to_numpy(y), [23.0, 53.0, 83.0])


def test_product_linop_conversion_same_backend_preserves_action():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.DenseCoordinateSpace((2,), src)
    Y1, Y2 = sc.DenseCoordinateSpace((2,), src), sc.DenseCoordinateSpace((1,), src)
    op = sc.StackedLinOp.from_operators(
        (
            sc.DenseLinOp(src.asarray([[1.0, 2.0], [3.0, 4.0]]), X, Y1, src),
            sc.DenseLinOp(src.asarray([[5.0, 6.0]]), X, Y2, src),
        )
    )
    op2 = op.convert(dst)
    y = op2.apply(op2.ctx.asarray([10.0, 20.0]))
    assert np.allclose(to_numpy(y[0]), [50.0, 110.0])


def test_linop_conversion_to_same_effective_context_returns_self():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, ctx)

    assert op.convert(ctx) is op


def test_linop_conversion_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    src = sc.Context(sc.NumpyOps(), dtype=dt)
    dst = sc.Context(sc.JaxOps(), dtype=dt)
    op = sc.DenseLinOp(
        src.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        sc.DenseCoordinateSpace((2,), src),
        sc.DenseCoordinateSpace((3,), src),
        src,
    )
    op2 = op.convert(dst)
    assert op2.ctx.ops.family == "jax"

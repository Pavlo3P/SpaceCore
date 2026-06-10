import importlib.util

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import jax_real_dtype, to_numpy


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax is not installed",
)


def _jax_ctx():
    return sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False)


def test_dense_linop_jit_apply_and_rapply_with_operator_argument():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    op = sc.DenseLinOp(
        ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        sc.DenseCoordinateSpace((2,), ctx),
        sc.DenseCoordinateSpace((3,), ctx),
        ctx,
    )
    x = ctx.asarray([7.0, 8.0])
    y = ctx.asarray([1.0, -1.0, 2.0])

    apply_jit = jax.jit(lambda A, z: A.apply(z))
    rapply_jit = jax.jit(lambda A, z: A.rapply(z))

    np.testing.assert_allclose(to_numpy(apply_jit(op, x)), [23.0, 53.0, 83.0])
    np.testing.assert_allclose(to_numpy(rapply_jit(op, y)), [8.0, 10.0])


def test_decorated_apply_rapply_value_and_grad_jit_compile():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 1.0], [1.0, 4.0]]), space, space, ctx)
    q = sc.LinOpQuadraticForm(op, ctx=ctx)
    x = ctx.asarray([3.0, -1.0])

    apply_jit = jax.jit(lambda A, z: A.apply(z))
    rapply_jit = jax.jit(lambda A, z: A.rapply(z))
    value_jit = jax.jit(lambda functional, z: functional.value(z))
    grad_jit = jax.jit(lambda functional, z: functional.grad(z))

    np.testing.assert_allclose(to_numpy(apply_jit(op, x)), to_numpy(op.apply(x)))
    np.testing.assert_allclose(to_numpy(rapply_jit(op, x)), to_numpy(op.rapply(x)))
    np.testing.assert_allclose(to_numpy(value_jit(q, x)), to_numpy(q.value(x)))
    np.testing.assert_allclose(to_numpy(grad_jit(q, x)), to_numpy(q.grad(x)))


def test_tensor_dense_linop_jit_preserves_shapes():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    dom = sc.DenseCoordinateSpace((2, 2), ctx)
    cod = sc.DenseCoordinateSpace((3, 1), ctx)
    A = ctx.asarray(np.arange(12, dtype=np.float32).reshape(cod.shape + dom.shape))
    op = sc.DenseLinOp(A, dom, cod, ctx)
    x = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])

    apply_jit = jax.jit(lambda Aop, z: Aop.apply(z))
    y = apply_jit(op, x)

    assert y.shape == cod.shape
    np.testing.assert_allclose(to_numpy(y), to_numpy(op.apply(x)))


def test_product_linops_jit_compile():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y1 = sc.DenseCoordinateSpace((2,), ctx)
    Y2 = sc.DenseCoordinateSpace((1,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X, Y1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[5.0, 6.0]]), X, Y2, ctx)
    stacked = sc.StackedLinOp.from_operators((A1, A2))
    x = ctx.asarray([7.0, 8.0])

    stacked_apply = jax.jit(lambda Aop, z: Aop.apply(z))
    stacked_rapply = jax.jit(lambda Aop, a, b: Aop.rapply((a, b)))
    y = stacked_apply(stacked, x)
    xr = stacked_rapply(stacked, ctx.asarray([1.0, -1.0]), ctx.asarray([2.0]))

    np.testing.assert_allclose(to_numpy(y[0]), [23.0, 53.0])
    np.testing.assert_allclose(to_numpy(y[1]), [83.0])
    np.testing.assert_allclose(to_numpy(xr), [8.0, 10.0])

    X1 = sc.DenseCoordinateSpace((2,), ctx)
    X2 = sc.DenseCoordinateSpace((3,), ctx)
    Z1 = sc.DenseCoordinateSpace((2,), ctx)
    Z2 = sc.DenseCoordinateSpace((1,), ctx)
    B1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X1, Z1, ctx)
    B2 = sc.DenseLinOp(ctx.asarray([[5.0, 6.0, 7.0]]), X2, Z2, ctx)
    block = sc.BlockDiagonalLinOp.from_operators((B1, B2))
    x1 = ctx.asarray([7.0, 8.0])
    x2 = ctx.asarray([1.0, 2.0, 3.0])

    block_apply = jax.jit(lambda Aop, a, b: Aop.apply((a, b)))
    block_rapply = jax.jit(lambda Aop, a, b: Aop.rapply((a, b)))
    z = block_apply(block, x1, x2)
    zr = block_rapply(block, ctx.asarray([1.0, -1.0]), ctx.asarray([2.0]))

    np.testing.assert_allclose(to_numpy(z[0]), [23.0, 53.0])
    np.testing.assert_allclose(to_numpy(z[1]), [38.0])
    np.testing.assert_allclose(to_numpy(zr[0]), [-2.0, -2.0])
    np.testing.assert_allclose(to_numpy(zr[1]), [10.0, 12.0, 14.0])

    sum_to_single = sc.SumToSingleLinOp.from_operators((A1, A1))
    sum_apply = jax.jit(lambda Aop, a, b: Aop.apply((a, b)))
    sum_rapply = jax.jit(lambda Aop, z: Aop.rapply(z))

    np.testing.assert_allclose(to_numpy(sum_apply(sum_to_single, x, x)), [46.0, 106.0])
    yr = sum_rapply(sum_to_single, ctx.asarray([1.0, -1.0]))
    np.testing.assert_allclose(to_numpy(yr[0]), [-2.0, -2.0])
    np.testing.assert_allclose(to_numpy(yr[1]), [-2.0, -2.0])


def test_sparse_linop_jit_apply_if_supported():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    op = sc.SparseLinOp(
        ctx.assparse(np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jax_real_dtype())),
        sc.DenseCoordinateSpace((2,), ctx),
        sc.DenseCoordinateSpace((3,), ctx),
        ctx,
    )
    x = ctx.asarray([7.0, 8.0])

    apply_jit = jax.jit(lambda Aop, z: Aop.apply(z))
    rapply_jit = jax.jit(lambda Aop, z: Aop.rapply(z))

    np.testing.assert_allclose(to_numpy(apply_jit(op, x)), [23.0, 53.0, 83.0])
    np.testing.assert_allclose(to_numpy(rapply_jit(op, ctx.asarray([1.0, -1.0, 2.0]))), [8.0, 10.0])


def test_product_space_flatten_unflatten_jit_compile():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    X1 = sc.DenseCoordinateSpace((2, 2), ctx)
    X2 = sc.DenseCoordinateSpace((3,), ctx)
    product = sc.ProductSpace((X1, X2), ctx)
    x1 = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    x2 = ctx.asarray([5.0, 6.0, 7.0])

    flatten_jit = jax.jit(lambda a, b: product.flatten((a, b)))
    unflatten_jit = jax.jit(lambda v: product.unflatten(v))

    flat = flatten_jit(x1, x2)
    roundtrip = unflatten_jit(flat)

    np.testing.assert_allclose(to_numpy(flat), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(to_numpy(roundtrip[0]), to_numpy(x1))
    np.testing.assert_allclose(to_numpy(roundtrip[1]), to_numpy(x2))

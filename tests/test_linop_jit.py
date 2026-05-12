import importlib.util

import numpy as np
import pytest

import spacecore as sc

from ._helpers import jax_real_dtype, to_numpy


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
        ctx.asarray([[1., 2.], [3., 4.], [5., 6.]]),
        sc.VectorSpace((2,), ctx),
        sc.VectorSpace((3,), ctx),
        ctx,
    )
    x = ctx.asarray([7., 8.])
    y = ctx.asarray([1., -1., 2.])

    apply_jit = jax.jit(lambda A, z: A.apply(z))
    rapply_jit = jax.jit(lambda A, z: A.rapply(z))

    np.testing.assert_allclose(to_numpy(apply_jit(op, x)), [23., 53., 83.])
    np.testing.assert_allclose(to_numpy(rapply_jit(op, y)), [8., 10.])


def test_tensor_dense_linop_jit_preserves_shapes():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    dom = sc.VectorSpace((2, 2), ctx)
    cod = sc.VectorSpace((3, 1), ctx)
    A = ctx.asarray(np.arange(12, dtype=np.float32).reshape(cod.shape + dom.shape))
    op = sc.DenseLinOp(A, dom, cod, ctx)
    x = ctx.asarray([[1., 2.], [3., 4.]])

    apply_jit = jax.jit(lambda Aop, z: Aop.apply(z))
    y = apply_jit(op, x)

    assert y.shape == cod.shape
    np.testing.assert_allclose(to_numpy(y), to_numpy(op.apply(x)))


def test_product_linops_jit_compile():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    X = sc.VectorSpace((2,), ctx)
    Y1 = sc.VectorSpace((2,), ctx)
    Y2 = sc.VectorSpace((1,), ctx)
    A1 = sc.DenseLinOp(ctx.asarray([[1., 2.], [3., 4.]]), X, Y1, ctx)
    A2 = sc.DenseLinOp(ctx.asarray([[5., 6.]]), X, Y2, ctx)
    stacked = sc.StackedLinOp.from_operators((A1, A2))
    x = ctx.asarray([7., 8.])

    stacked_apply = jax.jit(lambda Aop, z: Aop.apply(z))
    stacked_rapply = jax.jit(lambda Aop, a, b: Aop.rapply((a, b)))
    y = stacked_apply(stacked, x)
    xr = stacked_rapply(stacked, ctx.asarray([1., -1.]), ctx.asarray([2.]))

    np.testing.assert_allclose(to_numpy(y[0]), [23., 53.])
    np.testing.assert_allclose(to_numpy(y[1]), [83.])
    np.testing.assert_allclose(to_numpy(xr), [8., 10.])

    X1 = sc.VectorSpace((2,), ctx)
    X2 = sc.VectorSpace((3,), ctx)
    Z1 = sc.VectorSpace((2,), ctx)
    Z2 = sc.VectorSpace((1,), ctx)
    B1 = sc.DenseLinOp(ctx.asarray([[1., 2.], [3., 4.]]), X1, Z1, ctx)
    B2 = sc.DenseLinOp(ctx.asarray([[5., 6., 7.]]), X2, Z2, ctx)
    block = sc.BlockDiagonalLinOp.from_operators((B1, B2))
    x1 = ctx.asarray([7., 8.])
    x2 = ctx.asarray([1., 2., 3.])

    block_apply = jax.jit(lambda Aop, a, b: Aop.apply((a, b)))
    block_rapply = jax.jit(lambda Aop, a, b: Aop.rapply((a, b)))
    z = block_apply(block, x1, x2)
    zr = block_rapply(block, ctx.asarray([1., -1.]), ctx.asarray([2.]))

    np.testing.assert_allclose(to_numpy(z[0]), [23., 53.])
    np.testing.assert_allclose(to_numpy(z[1]), [38.])
    np.testing.assert_allclose(to_numpy(zr[0]), [-2., -2.])
    np.testing.assert_allclose(to_numpy(zr[1]), [10., 12., 14.])

    sum_to_single = sc.SumToSingleLinOp.from_operators((A1, A1))
    sum_apply = jax.jit(lambda Aop, a, b: Aop.apply((a, b)))
    sum_rapply = jax.jit(lambda Aop, z: Aop.rapply(z))

    np.testing.assert_allclose(to_numpy(sum_apply(sum_to_single, x, x)), [46., 106.])
    yr = sum_rapply(sum_to_single, ctx.asarray([1., -1.]))
    np.testing.assert_allclose(to_numpy(yr[0]), [-2., -2.])
    np.testing.assert_allclose(to_numpy(yr[1]), [-2., -2.])


def test_sparse_linop_jit_apply_if_supported():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    op = sc.SparseLinOp(
        ctx.assparse(np.asarray([[1., 2.], [3., 4.], [5., 6.]], dtype=jax_real_dtype())),
        sc.VectorSpace((2,), ctx),
        sc.VectorSpace((3,), ctx),
        ctx,
    )
    x = ctx.asarray([7., 8.])

    apply_jit = jax.jit(lambda Aop, z: Aop.apply(z))
    rapply_jit = jax.jit(lambda Aop, z: Aop.rapply(z))

    np.testing.assert_allclose(to_numpy(apply_jit(op, x)), [23., 53., 83.])
    np.testing.assert_allclose(to_numpy(rapply_jit(op, ctx.asarray([1., -1., 2.]))), [8., 10.])


def test_product_space_flatten_unflatten_jit_compile():
    jax = pytest.importorskip("jax")

    ctx = _jax_ctx()
    X1 = sc.VectorSpace((2, 2), ctx)
    X2 = sc.VectorSpace((3,), ctx)
    product = sc.ProductSpace((X1, X2), ctx)
    x1 = ctx.asarray([[1., 2.], [3., 4.]])
    x2 = ctx.asarray([5., 6., 7.])

    flatten_jit = jax.jit(lambda a, b: product.flatten((a, b)))
    unflatten_jit = jax.jit(lambda v: product.unflatten(v))

    flat = flatten_jit(x1, x2)
    roundtrip = unflatten_jit(flat)

    np.testing.assert_allclose(to_numpy(flat), [1., 2., 3., 4., 5., 6., 7.])
    np.testing.assert_allclose(to_numpy(roundtrip[0]), to_numpy(x1))
    np.testing.assert_allclose(to_numpy(roundtrip[1]), to_numpy(x2))

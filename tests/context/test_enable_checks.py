import numpy as np
import pytest

import spacecore as sc
from spacecore._contextual.contextual import ContextConversionError

from tests._helpers import has_jax, jax_real_dtype


def _checked_ctx(dtype=np.float64):
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=True)


def _unchecked_ctx(dtype=np.float64):
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=False)


def test_enable_checks_accepts_valid_space_and_linop_inputs():
    ctx = _checked_ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1., 2.], [3., 4.], [5., 6.]]), dom, cod, ctx)

    x = ctx.asarray([7., 8.])
    y = op.apply(x)

    dom.check_member(x)
    cod.check_member(y)
    np.testing.assert_allclose(y, [23., 53., 83.])


def test_enable_checks_rejects_vector_shape_mismatch():
    ctx = _checked_ctx(np.float32)
    space = sc.VectorSpace((2,), ctx)

    with pytest.raises(TypeError, match=r"Expected shape \(2,\), got \(3,\)"):
        space.check_member(np.asarray([1., 2., 3.], dtype=np.float32))


def test_enable_checks_rejects_vector_dtype_mismatch():
    ctx = _checked_ctx(np.float32)
    space = sc.VectorSpace((2,), ctx)

    with pytest.raises(TypeError, match=r"Expected dtype float32, got float64"):
        space.check_member(np.asarray([1., 2.], dtype=np.float64))


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_enable_checks_rejects_cross_backend_dense_array():
    np_ctx = _checked_ctx(jax_real_dtype())
    jx_ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=True)
    space = sc.VectorSpace((2,), np_ctx)

    with pytest.raises(TypeError, match="Expected dense array for numpy"):
        space.check_member(jx_ctx.asarray([1., 2.]))


def test_enable_checks_rejects_non_hermitian_matrix():
    ctx = _checked_ctx()
    space = sc.HermitianSpace(2, ctx=ctx)

    with pytest.raises(TypeError, match="not Hermitian"):
        space.check_member(ctx.asarray([[1., 2.], [0., 1.]]))


def test_enable_checks_rejects_invalid_product_structure():
    ctx = _checked_ctx()
    product = sc.ProductSpace(
        (sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx)),
        ctx,
    )

    with pytest.raises(TypeError, match="ProductSpace element must be a tuple"):
        product.check_member([ctx.asarray([1., 2.]), ctx.asarray([3., 4., 5.])])

    with pytest.raises(ValueError, match="Expected tuple of length 2, got 1"):
        product.check_member((ctx.asarray([1., 2.]),))

    with pytest.raises(TypeError, match=r"Invalid component 1.*Expected shape \(3,\)"):
        product.check_member((ctx.asarray([1., 2.]), ctx.asarray([3., 4.])))


def test_enable_checks_rejects_dense_linop_matrix_and_vector_dimensions():
    ctx = _checked_ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)

    with pytest.raises(TypeError, match=r"Expected A\.shape == cod\.shape \+ dom\.shape"):
        sc.DenseLinOp(ctx.asarray([[1., 2.], [3., 4.]]), dom, cod, ctx)

    op = sc.DenseLinOp(ctx.asarray([[1., 2.], [3., 4.], [5., 6.]]), dom, cod, ctx)
    with pytest.raises(TypeError, match=r"Expected shape \(2,\), got \(3,\)"):
        op.apply(ctx.asarray([1., 2., 3.]))

    with pytest.raises(TypeError, match=r"Expected shape \(3,\), got \(2,\)"):
        op.rapply(ctx.asarray([1., 2.]))


def test_enable_checks_rejects_product_linop_domain_codomain_mismatch():
    ctx = _checked_ctx()
    dom2 = sc.VectorSpace((2,), ctx)
    dom3 = sc.VectorSpace((3,), ctx)
    cod1 = sc.VectorSpace((1,), ctx)
    first = sc.DenseLinOp(ctx.asarray([[1., 2.]]), dom2, cod1, ctx)
    second = sc.DenseLinOp(ctx.asarray([[1., 2., 3.]]), dom3, cod1, ctx)

    with pytest.raises(TypeError, match=r"Component op 1 must map dom -> cod\.spaces\[1\]"):
        sc.StackedLinOp.from_operators((first, second))


def test_enable_checks_rejects_invalid_conversion_target():
    ctx = _checked_ctx()
    space = sc.VectorSpace((2,), ctx)

    with pytest.raises(TypeError, match="Expected Context, BackendFamily, str, or None"):
        space.convert(object())


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_enable_checks_rejects_forbidden_cross_backend_conversion():
    original_policy = sc.get_resolution_policy()
    try:
        sc.set_resolution_policy("error")
        ctx = _checked_ctx(jax_real_dtype())
        space = sc.VectorSpace((2,), ctx)

        with pytest.raises(ContextConversionError, match="Conversion from .* is forbidden"):
            space.convert(sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=True))
    finally:
        sc.set_resolution_policy(original_policy)


def test_disabled_checks_skip_space_membership_validations():
    ctx = _unchecked_ctx()
    vector = sc.VectorSpace((2,), ctx)
    hermitian = sc.HermitianSpace(2, ctx=ctx)
    product = sc.ProductSpace((vector, vector), ctx)

    vector.check_member(np.asarray([1., 2., 3.], dtype=np.float32))
    hermitian.check_member(ctx.asarray([[1., 2.], [0., 1.]]))
    product.check_member([ctx.asarray([1.]), ctx.asarray([2., 3., 4.])])

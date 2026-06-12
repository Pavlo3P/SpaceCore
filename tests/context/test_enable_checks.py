import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype


def _checked_ctx(dtype=np.float64):
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=True)


def _unchecked_ctx(dtype=np.float64):
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=False)


def test_enable_checks_accepts_valid_space_and_linop_inputs():
    ctx = _checked_ctx()
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dom, cod, ctx)

    x = ctx.asarray([7.0, 8.0])
    y = op.apply(x)

    dom.check_member(x)
    cod.check_member(y)
    np.testing.assert_allclose(y, [23.0, 53.0, 83.0])


def test_enable_checks_rejects_vector_shape_mismatch():
    ctx = _checked_ctx(np.float32)
    space = sc.DenseCoordinateSpace((2,), ctx)

    with pytest.raises(TypeError, match=r"Expected shape \(2,\), got \(3,\)"):
        space.check_member(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_enable_checks_rejects_vector_dtype_mismatch():
    ctx = _checked_ctx(np.float32)
    space = sc.DenseCoordinateSpace((2,), ctx)

    with pytest.raises(TypeError, match=r"Expected dtype float32, got float64"):
        space.check_member(np.asarray([1.0, 2.0], dtype=np.float64))


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_enable_checks_rejects_cross_backend_dense_array():
    np_ctx = _checked_ctx(jax_real_dtype())
    jx_ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=True)
    space = sc.DenseCoordinateSpace((2,), np_ctx)

    with pytest.raises(TypeError, match="Expected dense array for numpy"):
        space.check_member(jx_ctx.asarray([1.0, 2.0]))


def test_enable_checks_rejects_non_hermitian_matrix():
    ctx = _checked_ctx()
    space = sc.HermitianSpace(2, ctx=ctx)

    with pytest.raises(TypeError, match="not Hermitian"):
        space.check_member(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))


def test_enable_checks_rejects_invalid_tree_structure():
    ctx = _checked_ctx()
    product = sc.TreeSpace.from_leaf_spaces(
        (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)),
        ctx,
    )

    with pytest.raises(TypeError, match="structure mismatch"):
        product.check_member([ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0])])

    with pytest.raises(TypeError, match="structure mismatch"):
        product.check_member((ctx.asarray([1.0, 2.0]),))

    with pytest.raises(TypeError, match=r"\$\[1\].*Expected shape \(3,\)"):
        product.check_member((ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0])))


def test_enable_checks_rejects_dense_linop_matrix_and_vector_dimensions():
    ctx = _checked_ctx()
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)

    with pytest.raises(TypeError, match=r"Expected A\.shape == cod\.shape \+ dom\.shape"):
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), dom, cod, ctx)

    op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dom, cod, ctx)
    with pytest.raises(TypeError, match=r"Expected shape \(2,\), got \(3,\)"):
        op.apply(ctx.asarray([1.0, 2.0, 3.0]))

    with pytest.raises(TypeError, match=r"Expected shape \(3,\), got \(2,\)"):
        op.rapply(ctx.asarray([1.0, 2.0]))


def test_enable_checks_rejects_tree_linop_domain_codomain_mismatch():
    ctx = _checked_ctx()
    dom2 = sc.DenseCoordinateSpace((2,), ctx)
    dom3 = sc.DenseCoordinateSpace((3,), ctx)
    cod1 = sc.DenseCoordinateSpace((1,), ctx)
    first = sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), dom2, cod1, ctx)
    second = sc.DenseLinOp(ctx.asarray([[1.0, 2.0, 3.0]]), dom3, cod1, ctx)

    with pytest.raises(
        TypeError, match=r"Component op 1 must map dom -> cod\.leaf_spaces\[1\]"
    ):
        sc.StackedLinOp.from_operators((first, second))


def test_enable_checks_rejects_invalid_conversion_target():
    ctx = _checked_ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)

    with pytest.raises(TypeError, match="Expected Context, BackendFamily, str, or None"):
        space.convert(object())


def test_disabled_checks_skip_space_membership_validations():
    ctx = _unchecked_ctx()
    vector = sc.DenseCoordinateSpace((2,), ctx)
    hermitian = sc.HermitianSpace(2, ctx=ctx)
    product = sc.TreeSpace.from_leaf_spaces((vector, vector), ctx)

    vector.check_member(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    hermitian.check_member(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))
    product.check_member([ctx.asarray([1.0]), ctx.asarray([2.0, 3.0, 4.0])])

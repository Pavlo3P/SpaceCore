from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore._batching import _check_batched
from tests._helpers import has_jax, jax_complex_dtype, jax_real_dtype, to_numpy


def _np_ctx(dtype=np.float64, enable_checks=True):
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


@pytest.mark.parametrize("leading_shape", [(), (8,), (2, 3)])
def test_shape_check_matches_trailing_core_axes(leading_shape):
    ctx = _np_ctx()
    space = sc.DenseCoordinateSpace((2, 3), ctx)
    x = ctx.asarray(np.ones(leading_shape + space.shape))

    assert sc.ShapeCheck().is_valid(space, x)
    _check_batched(space, x)

    if leading_shape:
        with pytest.raises(ValueError, match=r"Expected shape \(2, 3\)"):
            space.check_member(x)
    else:
        space.check_member(x)


def test_direct_shape_check_uses_member_semantics_not_batched_semantics():
    ctx = _np_ctx()
    space = sc.DenseCoordinateSpace((2, 3), ctx)
    x = ctx.asarray(np.ones(space.shape))
    xs = ctx.asarray(np.ones((5,) + space.shape))

    sc.ShapeCheck()(space, x)
    with pytest.raises(ValueError, match=r"Expected shape \(2, 3\), got \(5, 2, 3\)"):
        sc.ShapeCheck()(space, xs)
    _check_batched(space, xs)


@pytest.mark.parametrize("bad_shape", [(2, 4), (8, 2, 4), (2, 3, 2)])
def test_shape_check_rejects_wrong_trailing_core_axes(bad_shape):
    ctx = _np_ctx()
    space = sc.DenseCoordinateSpace((2, 3), ctx)
    x = ctx.asarray(np.ones(bad_shape))

    assert not sc.ShapeCheck().is_valid(space, x)
    with pytest.raises(ValueError, match="trailing shape"):
        _check_batched(space, x)


@pytest.mark.parametrize("leading_shape", [(), (8,), (2, 3)])
def test_square_and_hermitian_checks_match_trailing_matrix_axes(leading_shape):
    ctx = _np_ctx(np.complex128)
    space = sc.HermitianSpace(4, ctx=ctx)
    eye = np.eye(4, dtype=np.complex128)
    x = ctx.asarray(np.broadcast_to(eye, leading_shape + eye.shape).copy())

    assert sc.SquareMatrixCheck().is_valid(space, x)
    assert sc.HermitianCheck().is_valid(space, x)
    _check_batched(space, x)

    if leading_shape:
        with pytest.raises(ValueError, match=r"Expected shape \(4, 4\)"):
            space.check_member(x)
    else:
        space.check_member(x)


def test_hermitian_check_rejects_one_bad_batched_slice():
    ctx = _np_ctx(np.complex128)
    space = sc.HermitianSpace(4, ctx=ctx)
    xs = np.broadcast_to(np.eye(4, dtype=np.complex128), (8, 4, 4)).copy()
    xs[3, 0, 1] = 2.0
    xs[3, 1, 0] = 0.0
    x = ctx.asarray(xs)

    assert not sc.HermitianCheck().is_valid(space, x)
    with pytest.raises(ValueError, match="not Hermitian"):
        _check_batched(space, x)


def test_direct_hermitian_check_uses_member_semantics_not_batched_semantics():
    ctx = _np_ctx(np.complex128)
    space = sc.HermitianSpace(2, ctx=ctx)
    x = ctx.asarray(np.eye(2, dtype=np.complex128))
    xs = ctx.asarray(np.broadcast_to(np.eye(2, dtype=np.complex128), (3, 2, 2)).copy())

    sc.HermitianCheck()(space, x)
    with pytest.raises(ValueError, match=r"Expected Hermitian matrix"):
        sc.HermitianCheck()(space, xs)
    _check_batched(space, xs)


def test_backend_and_dtype_checks_are_rank_agnostic_for_batches():
    ctx = _np_ctx(np.float32)
    space = sc.DenseCoordinateSpace((2,), ctx)
    x = ctx.asarray(np.ones((5, 2), dtype=np.float32))

    assert sc.BackendCheck().is_valid(space, x)
    assert sc.DTypeCheck().is_valid(space, x)
    _check_batched(space, x)

    with pytest.raises(ValueError, match="Expected dtype"):
        _check_batched(space, np.ones((5, 2), dtype=np.float64))


def test_product_component_checks_recurse_with_batched_mode():
    ctx = _np_ctx()
    product = sc.ProductSpace(
        (sc.DenseCoordinateSpace((2,), ctx), sc.HermitianSpace(2, ctx=ctx)),
        ctx,
    )
    xs = (
        ctx.asarray(np.ones((3, 2))),
        ctx.asarray(np.broadcast_to(np.eye(2), (3, 2, 2)).copy()),
    )

    _check_batched(product, xs)
    with pytest.raises(ValueError, match="Invalid component 0"):
        product.check_member(xs)
    with pytest.raises(ValueError, match="Invalid component 0"):
        sc.ProductComponentCheck()(product, xs)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_vectorized_checks_work_eager_and_jit_with_checks_disabled():
    import jax

    eager_ctx = sc.Context(sc.JaxOps(), dtype=jax_complex_dtype(), enable_checks=True)
    eager_space = sc.HermitianSpace(2, ctx=eager_ctx)
    x = eager_ctx.asarray(np.broadcast_to(np.eye(2), (4, 2, 2)).copy())
    _check_batched(eager_space, x)

    jit_ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False)
    jit_space = sc.DenseCoordinateSpace((2,), ctx=jit_ctx)

    @jax.jit
    def add_batch(xs):
        return jit_space.add_batch(xs, xs)

    xs = jit_ctx.asarray(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jax_real_dtype()))
    np.testing.assert_allclose(to_numpy(add_batch(xs)), [[2.0, 4.0], [6.0, 8.0]])

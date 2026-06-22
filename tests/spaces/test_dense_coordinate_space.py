"""Tests for :class:`spacecore.DenseCoordinateSpace`.

Checklist item 11:

* Construction with shape + ctx + optional geometry.
* ``__eq__`` includes shape + ctx + geometry (different geometry → not equal).
* ``_local_checks`` returns the per-instance check list.
* ``zeros`` / ``add`` / ``scale`` / ``inner`` on real and complex.
* ``flatten`` ↔ ``unflatten`` round-trip.
* Batched ``add_batch`` / ``scale_batch`` / ``flatten_batch`` / ``unflatten_batch``.
* ``_convert`` across contexts preserves shape and recasts geometry weights.

Cross-references:

* ``test_inner_product.py`` — WeightedInnerProduct validation
* ``test_coordinate_space_base.py`` — size / stacked
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# Construction and basic capability inference
# ===========================================================================
class TestConstruction:
    def test_default_construction_is_euclidean_inner_product(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        assert isinstance(space, sc.InnerProductSpace)
        assert space.is_euclidean is True

    def test_dense_coordinate_is_not_star_not_jordan(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        assert isinstance(space, sc.CoordinateSpace)
        assert not isinstance(space, sc.StarSpace)
        assert not isinstance(space, sc.JordanAlgebraSpace)
        assert not hasattr(space, "spectrum")

    @pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 1, 4)])
    def test_shape_preserved_after_construction(self, numpy_ctx, shape):
        assert sc.DenseCoordinateSpace(shape, numpy_ctx).shape == shape


# ===========================================================================
# __eq__: shape + ctx + geometry
# ===========================================================================
class TestEquality:
    def test_same_shape_and_ctx_default_geometry(self, numpy_ctx):
        a = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        b = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        assert a == b

    def test_different_shape_not_equal(self, numpy_ctx):
        a = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        b = sc.DenseCoordinateSpace((3, 2), numpy_ctx)
        assert a != b

    def test_different_geometry_not_equal(self, numpy_ctx):
        euclidean = sc.DenseCoordinateSpace((2,), numpy_ctx)
        weighted = sc.DenseCoordinateSpace(
            (2,), numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        assert euclidean != weighted

    def test_different_weights_not_equal(self, numpy_ctx):
        a = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        b = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 4.0])),
        )
        assert a != b

    def test_eq_symmetric_against_hermitian(self, numpy_ctx):
        """DenseCoordinateSpace != HermitianSpace from either side."""
        a = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        h = sc.HermitianSpace(2, ctx=numpy_ctx)
        assert (a == h) is False
        assert (h == a) is False


# ===========================================================================
# _local_checks
# ===========================================================================
class TestLocalChecks:
    def test_local_checks_include_shape_and_backend(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        kinds = [type(c).__name__ for c in space._local_checks()]
        assert "ShapeCheck" in kinds
        assert "BackendCheck" in kinds


# ===========================================================================
# zeros / add / scale / inner
# ===========================================================================
class TestVectorOperations:
    def test_zeros_is_zero_array(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        z = space.zeros()
        np.testing.assert_allclose(to_numpy(z), np.zeros((2, 3)))

    def test_add_is_elementwise(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        y = numpy_ctx.asarray([4.0, 5.0, 6.0])
        np.testing.assert_allclose(space.add(x, y), [5.0, 7.0, 9.0])

    def test_scale_is_elementwise(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(space.scale(2.0, x), [2.0, 4.0, 6.0])

    def test_inner_euclidean_real(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        y = numpy_ctx.asarray([4.0, 5.0, 6.0])
        np.testing.assert_allclose(space.inner(x, y), 1*4 + 2*5 + 3*6)


# ===========================================================================
# Batched variants
# ===========================================================================
class TestBatched:
    def test_add_batch(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        xs = numpy_ctx.asarray(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        ys = numpy_ctx.asarray(np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))
        out = space.add_batch(xs, ys)
        np.testing.assert_allclose(out, np.asarray([[11, 22, 33], [44, 55, 66]]))

    def test_scale_batch(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        xs = numpy_ctx.asarray(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        out = space.scale_batch(2.0, xs)
        np.testing.assert_allclose(out, np.asarray([[2, 4, 6], [8, 10, 12]]))

    def test_flatten_batch_round_trip(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        xs = numpy_ctx.asarray(np.arange(12.0).reshape(2, 2, 3))
        flat = space.flatten_batch(xs)
        assert flat.shape == (2, 6)
        np.testing.assert_allclose(space.unflatten_batch(flat), xs)


# ===========================================================================
# _convert across contexts
# ===========================================================================
class TestConvert:
    def test_convert_same_ctx_returns_self(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        assert space.convert(numpy_ctx) is space

    def test_convert_dtype_change(self, numpy_ctx, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        out = space.convert(numpy_f32_ctx)
        assert out.ctx == numpy_f32_ctx
        assert out.shape == space.shape

    def test_convert_recasts_weighted_geometry(self, numpy_ctx, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace(
            (2,), numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        out = space.convert(numpy_f32_ctx)
        assert isinstance(out.geometry, sc.WeightedInnerProduct)
        assert out.geometry.weights.dtype == numpy_f32_ctx.dtype

    def test_convert_round_trip_preserves_equality(self, numpy_ctx, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        roundtrip = space.convert(numpy_f32_ctx).convert(numpy_ctx)
        assert roundtrip == space

    def test_convert_to_jax_backend(self):
        from tests._helpers import has_jax, jax_real_dtype
        if not has_jax():
            pytest.skip("jax not installed")
        dt = jax_real_dtype()
        src = sc.Context(sc.NumpyOps(), dtype=dt)
        dst = sc.Context(sc.JaxOps(), dtype=dt, check_level="none")
        space = sc.DenseCoordinateSpace((2, 3), src)
        out = space.convert(dst)
        assert out.ctx == dst
        assert out.shape == space.shape

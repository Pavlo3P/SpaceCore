"""Tests for :class:`spacecore.CoordinateSpace` — the dense-array vector space base.

Checklist item 3:

* ``size`` property — product of ``shape``.
* ``flatten`` / ``unflatten`` round-trip.
* ``flatten_batch`` / ``unflatten_batch`` for batched leading-axis arrays.
* ``stacked(count)`` returns a ``StackedSpace`` with the right leading axis.
* ``__eq__`` includes ``shape`` (different shape → not equal).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


# ===========================================================================
# size: product of shape
# ===========================================================================
class TestSize:
    @pytest.mark.parametrize("shape, expected", [
        ((3,), 3),
        ((2, 3), 6),
        ((2, 3, 4), 24),
        ((1,), 1),
    ])
    def test_size_is_product_of_shape(self, numpy_ctx, shape, expected):
        space = sc.DenseCoordinateSpace(shape, numpy_ctx)
        assert space.size == expected


# ===========================================================================
# flatten / unflatten round-trip
# ===========================================================================
class TestFlattenUnflatten:
    @pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)])
    def test_unflatten_inverts_flatten(self, numpy_ctx, shape):
        space = sc.DenseCoordinateSpace(shape, numpy_ctx)
        x = numpy_ctx.asarray(np.arange(space.size, dtype=float).reshape(shape))
        flat = space.flatten(x)
        assert tuple(flat.shape) == (space.size,)
        np.testing.assert_allclose(space.unflatten(flat), x)

    def test_flatten_returns_1d_array(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        x = numpy_ctx.asarray(np.arange(6.0).reshape(2, 3))
        flat = space.flatten(x)
        assert flat.ndim == 1


# ===========================================================================
# flatten_batch / unflatten_batch
# ===========================================================================
class TestBatchedFlatten:
    def test_flatten_batch_collapses_trailing_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        xs = numpy_ctx.asarray(np.arange(12.0).reshape(2, 2, 3))
        out = space.flatten_batch(xs)
        # Leading axis preserved; trailing collapsed to size.
        assert out.shape == (2, 6)
        np.testing.assert_allclose(out, np.asarray(xs).reshape(2, 6))

    def test_unflatten_batch_restores_trailing_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        vs = numpy_ctx.asarray(np.arange(12.0).reshape(2, 6))
        out = space.unflatten_batch(vs)
        assert out.shape == (2, 2, 3)
        # Round-trip
        np.testing.assert_allclose(space.flatten_batch(out), vs)


# ===========================================================================
# stacked(count): returns a StackedSpace
# ===========================================================================
class TestStacked:
    @pytest.mark.parametrize("base_shape, count, expected", [
        ((3,), 2, (2, 3)),
        ((2, 3), 4, (4, 2, 3)),
        ((1,), 1, (1, 1)),
    ])
    def test_stacked_inserts_leading_axis(self, numpy_ctx, base_shape, count, expected):
        space = sc.DenseCoordinateSpace(base_shape, numpy_ctx)
        stacked = space.stacked(count)
        assert isinstance(stacked, sc.StackedSpace)
        assert stacked.shape == expected

    def test_stacked_count_zero_is_allowed(self, numpy_ctx):
        """Zero-count stacking yields a zero-leading-axis StackedSpace."""
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        stacked = space.stacked(0)
        assert stacked.shape == (0, 3)


# ===========================================================================
# __eq__: shape matters
# ===========================================================================
class TestEquality:
    def test_same_shape_and_ctx_is_equal(self, numpy_ctx):
        a = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        b = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        assert a == b

    def test_different_shape_is_not_equal(self, numpy_ctx):
        a = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        b = sc.DenseCoordinateSpace((3, 2), numpy_ctx)
        assert a != b

    def test_different_ctx_dtype_is_not_equal(self, numpy_ctx, numpy_f32_ctx):
        a = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        b = sc.DenseCoordinateSpace((2, 3), numpy_f32_ctx)
        assert a != b

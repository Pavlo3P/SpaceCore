"""Tests for :class:`spacecore.DenseVectorSpace`.

Checklist item 12:

* 1-dimensional constraint — multi-dim shapes raise.
* Capability set: ``CoordinateSpace``, ``InnerProductSpace``, ``StarSpace``;
  NOT ``JordanAlgebraSpace``.
* ``star`` is per-element conjugation on complex, identity on real.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# 1-D shape constraint
# ===========================================================================
class TestShapeConstraint:
    def test_construct_1d_works(self, numpy_ctx):
        space = sc.DenseVectorSpace((4,), numpy_ctx)
        assert space.shape == (4,)

    @pytest.mark.parametrize("shape", [(2, 2), (1, 3), (2, 2, 2)])
    def test_construct_higher_dim_raises(self, numpy_ctx, shape):
        with pytest.raises(ValueError, match="one-dimensional"):
            sc.DenseVectorSpace(shape, numpy_ctx)


# ===========================================================================
# Capability inference
# ===========================================================================
class TestCapabilities:
    def test_is_coordinate_inner_star_not_jordan(self, numpy_ctx):
        space = sc.DenseVectorSpace((4,), numpy_ctx)
        assert isinstance(space, sc.CoordinateSpace)
        assert isinstance(space, sc.InnerProductSpace)
        assert isinstance(space, sc.StarSpace)
        assert not isinstance(space, sc.JordanAlgebraSpace)
        assert not hasattr(space, "jordan")


# ===========================================================================
# star: conj on complex, identity on real
# ===========================================================================
class TestStar:
    def test_star_real_is_identity(self, numpy_ctx):
        space = sc.DenseVectorSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        np.testing.assert_allclose(space.star(x), x)

    def test_star_complex_is_conjugate(self, numpy_complex_ctx):
        space = sc.DenseVectorSpace((3,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1 + 2j, -3 + 0.5j, 4 + 0j])
        np.testing.assert_allclose(space.star(x), np.conj(to_numpy(x)))


# ===========================================================================
# Conversion preserves the shape constraint
# ===========================================================================
class TestConvert:
    def test_convert_dtype_preserves_1d(self, numpy_ctx, numpy_f32_ctx):
        space = sc.DenseVectorSpace((4,), numpy_ctx)
        converted = space.convert(numpy_f32_ctx)
        assert converted.shape == (4,)
        assert isinstance(converted, sc.DenseVectorSpace)

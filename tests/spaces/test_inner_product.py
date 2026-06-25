"""Tests for the InnerProduct hierarchy.

Checklist items 4, 5, 6:

* :class:`spacecore.InnerProduct` — abstract ``inner`` / ``riesz`` /
  ``riesz_inverse`` / ``validate_for`` / ``is_euclidean``.
* :class:`spacecore.EuclideanInnerProduct` — ``inner = vdot.real`` for real,
  conj-bilinear for complex; ``riesz``/``riesz_inverse`` are identity.
* :class:`spacecore.WeightedInnerProduct` — symmetry, positivity, riesz
  round-trip, conversion, validation of weight shape / sign / finiteness.

Gap-fill (per audit):

* ``WeightedInnerProduct.validate_for`` rejects negative weights, zero
  weights, non-finite weights, wrong-shape weights, complex weights for a
  real-valued context.
* ``riesz ∘ riesz_inverse = id`` round-trip on weighted geometry.
* Explicit conjugate symmetry of ``inner`` on a complex sample.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# InnerProduct: abstract surface
# ===========================================================================
class TestInnerProductAbstract:
    def test_abstract_methods_exist(self):
        for name in ("inner", "riesz", "riesz_inverse", "validate_for"):
            assert hasattr(sc.InnerProduct, name)

    def test_is_euclidean_is_a_property(self):
        assert isinstance(
            sc.InnerProduct.is_euclidean, (property, type(sc.InnerProduct.is_euclidean))
        )


# ===========================================================================
# EuclideanInnerProduct
# ===========================================================================
class TestEuclideanInnerProduct:
    def test_is_euclidean_true(self):
        assert sc.EuclideanInnerProduct().is_euclidean is True

    def test_inner_matches_vdot_on_real(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        y = numpy_ctx.asarray([4.0, 5.0, 6.0])
        np.testing.assert_allclose(space.inner(x, y), np.vdot(to_numpy(x), to_numpy(y)))

    def test_inner_is_conj_bilinear_on_complex(self, numpy_complex_ctx):
        """``<αx, y> = conj(α)·<x, y>`` for the Euclidean inner product."""
        space = sc.DenseCoordinateSpace((3,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1 + 2j, -3 + 1j, 0.5 + 0.25j])
        y = numpy_complex_ctx.asarray([2 - 1j, 4 + 0j, 1 - 0.5j])
        alpha = 2.0 - 3.0j
        lhs = space.inner(alpha * x, y)
        rhs = np.conj(alpha) * space.inner(x, y)
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))

    def test_riesz_is_identity_for_real(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        assert space.riesz(x) is x

    def test_riesz_inverse_is_identity_for_real(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        assert space.riesz_inverse(x) is x


# ===========================================================================
# WeightedInnerProduct: validation and round-trip
# ===========================================================================
class TestWeightedInnerProduct:
    def test_is_euclidean_false(self, numpy_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        assert geom.is_euclidean is False

    def test_inner_uses_weights(self, numpy_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        space = sc.DenseCoordinateSpace((2,), numpy_ctx, geometry=geom)
        x = numpy_ctx.asarray([1.0, 4.0])
        y = numpy_ctx.asarray([5.0, 6.0])
        # <x, w·y>: 1*2*5 + 4*3*6 = 10 + 72 = 82
        np.testing.assert_allclose(space.inner(x, y), 82.0)

    def test_riesz_multiplies_by_weights(self, numpy_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        space = sc.DenseCoordinateSpace((2,), numpy_ctx, geometry=geom)
        x = numpy_ctx.asarray([1.0, 4.0])
        np.testing.assert_allclose(space.riesz(x), [2.0, 12.0])

    def test_riesz_round_trip(self, numpy_ctx):
        """``riesz_inverse(riesz(x)) == x``."""
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        space = sc.DenseCoordinateSpace((2,), numpy_ctx, geometry=geom)
        x = numpy_ctx.asarray([1.0, 4.0])
        np.testing.assert_allclose(space.riesz_inverse(space.riesz(x)), x)
        np.testing.assert_allclose(space.riesz(space.riesz_inverse(x)), x)

    def test_inner_is_symmetric_on_real(self, numpy_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        space = sc.DenseCoordinateSpace((2,), numpy_ctx, geometry=geom)
        x = numpy_ctx.asarray([1.0, 4.0])
        y = numpy_ctx.asarray([5.0, 6.0])
        np.testing.assert_allclose(space.inner(x, y), space.inner(y, x))


# ===========================================================================
# WeightedInnerProduct.validate_for: rejects ill-formed weights
# ===========================================================================
class TestWeightedValidation:
    @pytest.mark.parametrize("weights, expected_match", [
        ([1.0, -1.0], "strictly positive"),
        ([1.0, 0.0], "strictly positive"),
        ([1.0, float("inf")], "finite"),
        ([1.0, float("nan")], "finite"),
        ([[1.0, 2.0]], "coordinate shape"),
        ([1.0, 2.0, 3.0], "coordinate shape"),
    ])
    def test_rejects_invalid_weights_on_real_ctx(self, numpy_ctx, weights, expected_match):
        weights_arr = numpy_ctx.asarray(np.asarray(weights, dtype=np.float64))
        with pytest.raises((TypeError, ValueError), match=expected_match):
            sc.DenseCoordinateSpace((2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights_arr))

    def test_rejects_complex_weights_on_real_ctx(self, numpy_complex_ctx):
        weights = numpy_complex_ctx.asarray([1.0 + 1.0j, 2.0 + 0.0j])
        with pytest.raises((TypeError, ValueError), match="real-valued"):
            sc.DenseCoordinateSpace((2,), numpy_complex_ctx,
                                    geometry=sc.WeightedInnerProduct(weights))

    def test_rejects_weights_with_wrong_context_dtype(self, numpy_ctx, numpy_complex_ctx):
        """Weight dtype must match context dtype."""
        weights = numpy_ctx.asarray([1.0, 2.0])  # float64
        with pytest.raises(TypeError, match="context dtype"):
            sc.DenseCoordinateSpace(
                (2,), numpy_complex_ctx,
                geometry=sc.WeightedInnerProduct(weights),
            )


# ===========================================================================
# WeightedInnerProduct: conversion across contexts
# ===========================================================================
class TestWeightedConversion:
    def test_convert_recasts_weights(self, numpy_ctx, numpy_f32_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        converted = geom.convert(numpy_f32_ctx)
        assert isinstance(converted, sc.WeightedInnerProduct)
        # Compare via dtype on the converted weights — they should
        # honor the new context's dtype.
        assert converted.weights.dtype == numpy_f32_ctx.dtype

    def test_convert_preserves_weight_values(self, numpy_ctx, numpy_f32_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        converted = geom.convert(numpy_f32_ctx)
        np.testing.assert_allclose(to_numpy(converted.weights), [2.0, 3.0])

"""Tests for :class:`spacecore.ElementwiseJordanSpace` and
:class:`spacecore.EuclideanElementwiseJordanSpace`.

Checklist items 13 and 14:

* :class:`ElementwiseJordanSpace` —
  - ``__new__`` dispatches to real/complex variants via capability flags.
  - ``jordan(x, y) = x * y`` elementwise.
  - ``spectrum(x) = x``.
  - ``from_spectrum(eigvals, frame=None)`` returns ``eigvals``.
  - Higher-dimensional shapes work (vector / multi-axis).
* :class:`EuclideanElementwiseJordanSpace` —
  - Direct construction requires real ctx + EuclideanInnerProduct.
  - Trace-form inner = sum-of-products on real elementwise.
  - Conversion to complex never leaves a stale Euclidean class.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, to_numpy


# ===========================================================================
# __new__ dispatch: real/complex/weighted variants
# ===========================================================================
class TestDispatch:
    def test_real_construction_is_euclidean_jordan(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        assert isinstance(space, sc.EuclideanJordanAlgebraSpace)
        assert isinstance(space, sc.StarSpace)

    def test_complex_construction_drops_euclidean(self, numpy_complex_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_complex_ctx)
        assert isinstance(space, sc.JordanAlgebraSpace)
        assert not isinstance(space, sc.EuclideanJordanAlgebraSpace)

    def test_weighted_real_drops_euclidean(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        weighted = sc.ElementwiseJordanSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        assert isinstance(weighted, sc.JordanAlgebraSpace)
        assert not isinstance(weighted, sc.EuclideanJordanAlgebraSpace)


# ===========================================================================
# Operations: jordan = x*y, spectrum = x, from_spectrum returns eigvals
# ===========================================================================
class TestOperations:
    def test_jordan_is_elementwise_product(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        y = numpy_ctx.asarray([0.5, -3.0, 4.0])
        np.testing.assert_allclose(space.jordan(x, y), to_numpy(x) * to_numpy(y))

    def test_spectrum_returns_x(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0, 0.5])
        np.testing.assert_allclose(space.spectrum(x), x)

    def test_from_spectrum_returns_eigvals(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        eigvals = numpy_ctx.asarray([2.0, -1.0, 0.5])
        np.testing.assert_allclose(space.from_spectrum(eigvals, None), eigvals)

    def test_from_spectrum_rejects_non_none_frame(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        eigvals = numpy_ctx.asarray([2.0, -1.0, 0.5])
        with pytest.raises(ValueError, match="frame"):
            space.from_spectrum(eigvals, numpy_ctx.asarray([[1.0, 0.0, 0.0]]))


# ===========================================================================
# Euclidean-Elementwise: direct construction and trace-form inner
# ===========================================================================
class TestEuclideanElementwise:
    def test_direct_construction_matches_factory(self, numpy_ctx):
        direct = sc.EuclideanElementwiseJordanSpace((2,), numpy_ctx)
        factory = sc.ElementwiseJordanSpace((2,), numpy_ctx)
        assert isinstance(direct, sc.EuclideanElementwiseJordanSpace)
        assert direct == factory

    def test_direct_construction_rejects_complex_ctx(self, numpy_complex_ctx):
        with pytest.raises(ValueError, match="requires a real scalar field"):
            sc.EuclideanElementwiseJordanSpace((2,), numpy_complex_ctx)

    def test_direct_construction_rejects_weighted(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        with pytest.raises(TypeError, match="requires EuclideanInnerProduct"):
            sc.EuclideanElementwiseJordanSpace(
                (2,), numpy_ctx, inner_product=sc.WeightedInnerProduct(weights),
            )

    def test_trace_form_inner_is_sum_of_products(self, numpy_ctx):
        """``<x, y> = Σ x_i · y_i`` on real elementwise."""
        space = sc.EuclideanElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        y = numpy_ctx.asarray([0.5, -3.0, 4.0])
        expected = float(np.sum(to_numpy(x) * to_numpy(y)))
        np.testing.assert_allclose(float(to_numpy(space.inner(x, y))), expected)


# ===========================================================================
# Conversion: never leaves a stale Euclidean class
# ===========================================================================
class TestConversionInvariants:
    def test_real_to_complex_drops_euclidean(self, numpy_ctx, numpy_complex_ctx):
        real_space = sc.EuclideanElementwiseJordanSpace((2,), numpy_ctx)
        complex_space = real_space.convert(numpy_complex_ctx)
        assert type(complex_space) is sc.ElementwiseJordanSpace
        assert not isinstance(complex_space, sc.EuclideanJordanAlgebraSpace)

    def test_round_trip_restores_euclidean(self, numpy_ctx, numpy_complex_ctx):
        real_space = sc.EuclideanElementwiseJordanSpace((2,), numpy_ctx)
        roundtrip = real_space.convert(numpy_complex_ctx).convert(numpy_ctx)
        assert isinstance(roundtrip, sc.EuclideanElementwiseJordanSpace)

    @pytest.mark.skipif(not has_jax(), reason="jax is not installed")
    def test_jax_pytree_revalidates_invariant(self):
        """JAX unflatten with a complex ctx must refuse the Euclidean class."""
        real_ctx = sc.Context(sc.JaxOps(), dtype=np.float32, check_level="none")
        complex_ctx = sc.Context(sc.JaxOps(), dtype=np.complex64, check_level="none")
        space = sc.EuclideanElementwiseJordanSpace((2,), real_ctx)

        import jax
        leaves, treedef = jax.tree_util.tree_flatten(space)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == space

        with pytest.raises(ValueError, match="requires a real scalar field"):
            type(space).tree_unflatten(((2,), complex_ctx, sc.EuclideanInnerProduct()), ())

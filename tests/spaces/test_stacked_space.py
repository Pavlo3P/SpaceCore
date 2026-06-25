"""Tests for :class:`spacecore.StackedSpace`.

Checklist item 16:

* Core contract: ``shape = (count,) + base.shape``, ``count``, ``base``.
* Leafwise ``zeros`` / ``ones`` / ``add`` / ``scale`` / ``inner`` / ``norm``
  / ``flatten`` / ``unflatten`` over the leading axis.
* Weighted inner product is lifted from the base elementwise.
* ``__new__`` capability dispatch: every concrete ``_Stacked*`` mixin
  combination is reachable by constructing a base with the matching
  capability and stacking it (gap-fill).
* TreeSpace bases reject ``StackedSpace(tree, ...)`` directly — must use
  ``tree.stacked(...)``.
* JAX pytree round-trip.

Cross-references:

* ``test_inner_product.py`` — WeightedInnerProduct validation
* ``test_coordinate_space_base.py`` — ``stacked(count)`` from the base
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy


# ===========================================================================
# Capability dispatch — gap-fill
# ===========================================================================
# Toy concrete subclasses adapted from the deleted test_space_hierarchy.py.
# Used to exercise the StackedSpace.__new__ dispatch for every
# capability-mixin combination, plus the negative cases.


class _PairCoordinateSpace(sc.CoordinateSpace):
    def __init__(self, ctx=None):
        super().__init__((2,), ctx)

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def flatten(self, x):
        return self.ctx.asarray([x[0], x[1]])

    def unflatten(self, v):
        return (float(v[0]), float(v[1]))

    def _convert(self, new_ctx):
        return type(self)(new_ctx)


class _InnerPair(_PairCoordinateSpace, sc.InnerProductSpace):
    def __init__(self, ctx=None):
        _PairCoordinateSpace.__init__(self, ctx)
        self.geometry = sc.EuclideanInnerProduct()

    def inner(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def riesz(self, x):
        return x

    def riesz_inverse(self, x):
        return x

    @property
    def is_euclidean(self):
        return True


class _StarPair(_PairCoordinateSpace, sc.StarSpace):
    def star(self, x):
        return self.ops.conj(x)


class _JordanPair(_PairCoordinateSpace, sc.JordanAlgebraSpace):
    def jordan(self, x, y):
        return x * y

    def spectrum(self, x):
        return x

    def spectral_decompose(self, x):
        return x, None

    def from_spectrum(self, eigvals, frame):
        if frame is not None:
            raise ValueError("frame must be None")
        return eigvals


# ===========================================================================
# Core contract
# ===========================================================================
class TestCoreContract:
    def test_count_and_shape(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((3,), numpy_ctx)
        space = sc.StackedSpace(base, 4, numpy_ctx)
        assert space.count == 4
        assert space.shape == (4, 3)
        assert space.base == base

    def test_leading_axis_inner_norm(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((3,), numpy_ctx)
        space = sc.StackedSpace(base, 4, numpy_ctx)
        x = numpy_ctx.asarray(np.arange(12.0).reshape(4, 3))
        y = numpy_ctx.asarray(np.ones((4, 3)))
        # Inner aggregates over both axes.
        np.testing.assert_allclose(
            to_numpy(space.inner(x, y)),
            np.vdot(to_numpy(x), to_numpy(y)),
        )
        np.testing.assert_allclose(
            to_numpy(space.norm(x)),
            np.linalg.norm(to_numpy(x).reshape(-1)),
        )

    def test_leafwise_add_scale_zeros_ones(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((3,), numpy_ctx)
        space = sc.StackedSpace(base, 4, numpy_ctx)
        x = numpy_ctx.asarray(np.arange(12.0).reshape(4, 3))
        y = numpy_ctx.asarray(np.ones((4, 3)))
        np.testing.assert_allclose(to_numpy(space.add(x, y)), to_numpy(x + y))
        np.testing.assert_allclose(to_numpy(space.scale(2.0, x)), to_numpy(2.0 * x))
        np.testing.assert_allclose(to_numpy(space.zeros()), np.zeros((4, 3)))
        np.testing.assert_allclose(to_numpy(space.ones()), np.ones((4, 3)))

    def test_flatten_unflatten_round_trip(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((3,), numpy_ctx)
        space = sc.StackedSpace(base, 4, numpy_ctx)
        x = numpy_ctx.asarray(np.arange(12.0).reshape(4, 3))
        np.testing.assert_allclose(to_numpy(space.unflatten(space.flatten(x))), to_numpy(x))


# ===========================================================================
# Weighted geometry — leafwise riesz
# ===========================================================================
class TestWeightedGeometry:
    def test_inner_uses_lifted_weights(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 5.0, 11.0])
        base = sc.DenseCoordinateSpace((3,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights))
        space = base.stacked(2)
        x = numpy_ctx.asarray([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]])
        y = numpy_ctx.asarray([[0.25, 2.0, -1.0], [-2.0, 0.5, 4.0]])
        expected = np.vdot(to_numpy(x), to_numpy(weights) * to_numpy(y))
        np.testing.assert_allclose(to_numpy(space.inner(x, y)), expected)

    def test_riesz_is_leafwise_and_round_trips(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 5.0, 11.0])
        base = sc.DenseCoordinateSpace((3,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights))
        space = base.stacked(2)
        x = numpy_ctx.asarray([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]])
        np.testing.assert_allclose(to_numpy(space.riesz(x)), to_numpy(weights) * to_numpy(x))
        np.testing.assert_allclose(to_numpy(space.riesz_inverse(space.riesz(x))), to_numpy(x))
        assert space.is_euclidean is False


# ===========================================================================
# Capability-dispatch enumeration (gap-fill: every _Stacked* mixin reachable)
# ===========================================================================
def _capability_set(space):
    return {
        cls
        for cls in (sc.InnerProductSpace, sc.StarSpace, sc.JordanAlgebraSpace,
                    sc.EuclideanJordanAlgebraSpace)
        if isinstance(space, cls)
    }


class TestCapabilityDispatch:
    @pytest.mark.parametrize("factory, expected_caps", [
        (_PairCoordinateSpace, set()),
        (_InnerPair, {sc.InnerProductSpace}),
        (_StarPair, {sc.StarSpace}),
        (_JordanPair, {sc.JordanAlgebraSpace}),
    ])
    def test_single_capability_lifts_through_stacked(self, numpy_ctx, factory, expected_caps):
        base = factory(numpy_ctx)
        stacked = base.stacked(2)
        assert isinstance(stacked, sc.StackedSpace)
        assert isinstance(stacked, sc.CoordinateSpace)
        assert _capability_set(stacked) == expected_caps

    def test_inner_star_combination_lifts(self, numpy_ctx):
        class _InnerStarPair(_InnerPair, sc.StarSpace):
            def star(self, x):
                return self.ops.conj(x)

        stacked = _InnerStarPair(numpy_ctx).stacked(2)
        assert _capability_set(stacked) == {sc.InnerProductSpace, sc.StarSpace}

    def test_inner_jordan_combination_lifts(self, numpy_ctx):
        class _InnerJordanPair(_InnerPair, sc.JordanAlgebraSpace):
            def jordan(self, x, y):
                return x * y

            def spectrum(self, x):
                return x

            def spectral_decompose(self, x):
                return x, None

            def from_spectrum(self, eigvals, frame):
                if frame is not None:
                    raise ValueError("frame must be None")
                return eigvals

        stacked = _InnerJordanPair(numpy_ctx).stacked(2)
        assert _capability_set(stacked) == {sc.InnerProductSpace, sc.JordanAlgebraSpace}

    def test_star_jordan_combination_lifts(self, numpy_ctx):
        class _StarJordanPair(_StarPair, sc.JordanAlgebraSpace):
            def jordan(self, x, y):
                return x * y

            def spectrum(self, x):
                return x

            def spectral_decompose(self, x):
                return x, None

            def from_spectrum(self, eigvals, frame):
                if frame is not None:
                    raise ValueError("frame must be None")
                return eigvals

        stacked = _StarJordanPair(numpy_ctx).stacked(2)
        assert _capability_set(stacked) == {sc.StarSpace, sc.JordanAlgebraSpace}

    def test_all_capabilities_via_elementwise_jordan(self, numpy_ctx):
        """``ElementwiseJordanSpace`` (real) on real ctx → all four caps."""
        stacked = sc.ElementwiseJordanSpace((2,), numpy_ctx).stacked(2)
        assert _capability_set(stacked) == {
            sc.InnerProductSpace, sc.StarSpace,
            sc.JordanAlgebraSpace, sc.EuclideanJordanAlgebraSpace,
        }

    def test_baseline_stacked_has_no_capability_methods(self, numpy_ctx):
        """Non-inner/non-star/non-jordan base must not expose those methods."""
        stacked = _PairCoordinateSpace(numpy_ctx).stacked(2)
        for name in ("inner", "riesz", "riesz_inverse", "norm",
                     "is_euclidean", "star", "jordan",
                     "spectrum", "spectral_decompose", "from_spectrum",
                     "spectral_apply"):
            assert not hasattr(stacked, name), (type(stacked).__name__, name)

    def test_capability_recomputed_on_convert(self, numpy_ctx, numpy_complex_ctx):
        """Convert real → complex on EuclideanElementwise drops Euclidean cap."""
        base = sc.EuclideanElementwiseJordanSpace((2,), numpy_ctx)
        stacked = sc.StackedSpace(base, 2, numpy_complex_ctx)
        assert type(stacked.base) is sc.ElementwiseJordanSpace
        assert not isinstance(stacked, sc.EuclideanJordanAlgebraSpace)
        assert isinstance(stacked, sc.JordanAlgebraSpace)


# ===========================================================================
# Equality, conversion, composition
# ===========================================================================
class TestEqualityAndConversion:
    def test_same_base_same_count_equal(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((2,), numpy_ctx)
        a = base.stacked(3)
        b = sc.StackedSpace(sc.DenseCoordinateSpace((2,), numpy_ctx), 3, numpy_ctx)
        assert a == b

    def test_different_count_not_equal(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((2,), numpy_ctx)
        assert base.stacked(3) != base.stacked(4)

    def test_convert_dtype(self, numpy_ctx, numpy_f32_ctx):
        base = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        space = base.stacked(3)
        converted = space.convert(numpy_ctx)
        assert converted.ctx == numpy_ctx
        assert converted.base.ctx == numpy_ctx
        assert converted.shape == (3, 2)

    def test_stacked_of_stacked_multiplies_count(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((2,), numpy_ctx)
        out = base.stacked(3).stacked(5)
        assert out.count == 15


# ===========================================================================
# TreeSpace base rejection — must use TreeSpace.stacked instead
# ===========================================================================
class TestTreeSpaceBaseRejected:
    def test_stacked_of_tree_space_raises(self, numpy_ctx):
        x = sc.DenseCoordinateSpace((2,), numpy_ctx)
        y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        product = sc.TreeSpace.from_leaf_spaces((x, y), numpy_ctx)
        with pytest.raises(TypeError, match="TreeSpace"):
            sc.StackedSpace(product, 4, numpy_ctx)

    def test_tree_space_stacked_pushes_into_leaves(self, numpy_ctx):
        x = sc.DenseCoordinateSpace((2,), numpy_ctx)
        y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        product = sc.TreeSpace.from_leaf_spaces((x, y), numpy_ctx)
        stacked = product.stacked(4)
        assert isinstance(stacked, sc.TreeSpace)
        assert all(isinstance(s, sc.StackedSpace) for s in stacked.leaf_spaces)


# ===========================================================================
# Negative cases
# ===========================================================================
class TestConstructorValidation:
    def test_negative_count_raises(self, numpy_ctx):
        base = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(ValueError, match="nonnegative"):
            sc.StackedSpace(base, -1, numpy_ctx)


# ===========================================================================
# JAX pytree round-trip
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJaxPytree:
    def test_pytree_round_trip(self):
        import jax
        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        space = sc.DenseCoordinateSpace((2,), ctx).stacked(3)
        leaves, treedef = jax.tree_util.tree_flatten(space)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == space

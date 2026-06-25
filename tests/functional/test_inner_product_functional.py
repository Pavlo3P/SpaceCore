"""Tests for :class:`spacecore.InnerProductFunctional`.

Checklist section 7, ``InnerProductFunctional``:

* ``value(x) == domain.inner(representer, x)`` (Euclidean and weighted).
* ``representer`` property returns the stored, context-converted element.
* ``vvalue`` matches an element-wise loop without a Python ``for`` loop
  (Euclidean fast path and the weighted/Riesz path).
* Construction rejects complex data on a real space (ADR-015 Stage 1).
* Domain conversion + membership checks under an explicit context.
* ``__eq__`` compares representers; ``_convert`` preserves behaviour.
* ``tree_flatten`` / ``tree_unflatten`` round-trip.

Numerical value/gradient references against analytic formulae are covered by
:mod:`tests.functional.test_generated_functionals`; this file pins the
per-object API contract.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# value == inner(representer, x)
# ===========================================================================
class TestValue:
    def test_value_matches_domain_inner_euclidean(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0, 0.5])
        x = numpy_ctx.asarray([3.0, 4.0, -1.0])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        np.testing.assert_allclose(to_numpy(f.value(x)), to_numpy(space.inner(c, x)))
        np.testing.assert_allclose(to_numpy(f.value(x)), to_numpy(f(x)))

    def test_value_matches_domain_inner_weighted(self, weighted_space, numpy_ctx):
        c = numpy_ctx.asarray([0.25, -1.5, 2.0])
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        f = sc.InnerProductFunctional(c, weighted_space, numpy_ctx)
        np.testing.assert_allclose(
            to_numpy(f.value(x)), to_numpy(weighted_space.inner(c, x))
        )

    def test_value_is_conjugate_linear_in_representer_for_complex(
        self, numpy_complex_ctx
    ):
        space = sc.DenseCoordinateSpace((3,), numpy_complex_ctx)
        c = numpy_complex_ctx.asarray([1.5 - 0.5j, -0.25 + 1.0j, 0.75 + 0.25j])
        x = numpy_complex_ctx.asarray([0.5 + 0.25j, -1.0 + 0.75j, 2.0 - 0.5j])
        f = sc.InnerProductFunctional(c, space, numpy_complex_ctx)
        # Euclidean Hermitian inner product conjugates the first argument.
        np.testing.assert_allclose(
            to_numpy(f.value(x)), np.vdot(to_numpy(c), to_numpy(x))
        )


# ===========================================================================
# representer
# ===========================================================================
class TestRepresenter:
    def test_representer_returns_stored_element(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0, 0.5])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        np.testing.assert_allclose(to_numpy(f.representer), to_numpy(c))

    def test_representer_is_converted_to_functional_context(
        self, numpy_f32_ctx, numpy_ctx
    ):
        space = sc.DenseCoordinateSpace((3,), numpy_f32_ctx)
        c = numpy_f32_ctx.asarray([1.0, -2.0, 0.5])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        assert f.ops.get_dtype(f.representer) == np.dtype(np.float64)


# ===========================================================================
# vvalue (no Python loop)
# ===========================================================================
class TestVValue:
    def test_vvalue_matches_elementwise_euclidean(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([0.25, -1.5, 2.0])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        xs = numpy_ctx.asarray([[0.5, -1.0, 2.0], [1.25, 0.75, -0.5], [-2.0, 0.25, 1.5]])
        expected = f.ops.stack(tuple(f.value(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(f.vvalue(xs)), to_numpy(expected))

    def test_vvalue_matches_elementwise_weighted(self, weighted_space, numpy_ctx):
        c = numpy_ctx.asarray([0.25, -1.5, 2.0])
        f = sc.InnerProductFunctional(c, weighted_space, numpy_ctx)
        xs = numpy_ctx.asarray([[0.5, -1.0, 2.0], [1.25, 0.75, -0.5], [-2.0, 0.25, 1.5]])
        expected = f.ops.stack(tuple(f.value(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(f.vvalue(xs)), to_numpy(expected))


# ===========================================================================
# Construction guards
# ===========================================================================
class TestConstruction:
    def test_rejects_complex_data_on_real_space(self, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        representer = np.asarray([1.0 + 1.0j, 2.0], dtype=np.complex64)
        with pytest.raises(TypeError, match="rejected complex-valued input.*x.real"):
            sc.InnerProductFunctional(representer, space, numpy_f32_ctx)

    def test_explicit_context_converts_domain_and_representer(
        self, numpy_f32_ctx, numpy_ctx
    ):
        space = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        f = sc.InnerProductFunctional(numpy_f32_ctx.asarray([1.0, 2.0]), space, numpy_ctx)
        assert f.ctx == numpy_ctx
        assert f.dtype == np.dtype(np.float64)
        assert f.domain.ctx == numpy_ctx
        np.testing.assert_allclose(
            to_numpy(f.value(numpy_ctx.asarray([3.0, 4.0]))), 11.0
        )

    def test_membership_check_rejects_wrong_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        f = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 2.0]), space, numpy_ctx)
        with pytest.raises(sc.SpaceValidationError, match="Expected shape"):
            f.value(numpy_ctx.asarray([1.0, 2.0, 3.0]))


# ===========================================================================
# __eq__
# ===========================================================================
class TestEquality:
    def test_equal_when_same_representer_and_domain(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0, 0.5])
        assert sc.InnerProductFunctional(c, space, numpy_ctx) == (
            sc.InnerProductFunctional(c, space, numpy_ctx)
        )

    def test_not_equal_when_representer_differs(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        a = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 0.0, 0.0]), space, numpy_ctx)
        b = sc.InnerProductFunctional(numpy_ctx.asarray([0.0, 1.0, 0.0]), space, numpy_ctx)
        assert a != b

    def test_not_equal_to_other_type(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 0.0, 0.0]), space, numpy_ctx)
        assert (f == "functional") is False
        assert (f == 42) is False


# ===========================================================================
# Pytree round-trip
# ===========================================================================
class TestPytree:
    def test_tree_flatten_unflatten_round_trip(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0, 0.5])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        children, aux = f.tree_flatten()
        restored = sc.InnerProductFunctional.tree_unflatten(aux, children)
        assert restored == f
        x = numpy_ctx.asarray([3.0, 4.0, -1.0])
        np.testing.assert_allclose(to_numpy(restored.value(x)), to_numpy(f.value(x)))


# ===========================================================================
# _convert
# ===========================================================================
class TestConvert:
    def test_convert_preserves_value_across_dtype(self, numpy_f32_ctx, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_f32_ctx)
        c = numpy_f32_ctx.asarray([1.0, -2.0, 0.5])
        f = sc.InnerProductFunctional(c, space, numpy_f32_ctx)
        g = f.convert(numpy_ctx)
        assert g.ctx == numpy_ctx
        assert g.domain.dtype == np.dtype(np.float64)
        x = numpy_ctx.asarray([3.0, 4.0, -1.0])
        np.testing.assert_allclose(
            to_numpy(g.value(x)), to_numpy(f.value(numpy_f32_ctx.asarray([3.0, 4.0, -1.0])))
        )

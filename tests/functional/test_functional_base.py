"""Tests for :class:`spacecore.Functional` — the scalar-map base.

Checklist section 7, ``Functional`` base:

* ``domain`` property returns the context-converted domain.
* ``value(x)`` / ``__call__(x)`` alias.
* ``compose(A)`` returns a pull-back and validates ``A.codomain == domain``.
* ``vvalue(xs)`` default ``vmap`` fallback matches an element-wise loop.
* ``assert_domain`` raises on a domain mismatch.
* Module-private helpers ``_check_scalar_shape``, ``_leading_batch_size`` and
  ``_warn_vmap_fallback_once`` (called by every ``Functional``).
* ``tree_flatten`` / ``tree_unflatten`` round-trip.
* Explicit context wins over the context inferred from the domain.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spacecore as sc

from spacecore.functional._base import (
    _check_scalar_shape,
    _leading_batch_size,
    _warn_vmap_fallback_once,
    _VMAP_FALLBACK_WARNED,
)
from tests._helpers import to_numpy


class _SumSquares(sc.Functional):
    """Minimal concrete ``Functional``: ``F(x) = sum(x * x)``.

    Defines only the abstract surface (``value``, pytree hooks, ``_convert``)
    so the base-class behaviours can be exercised directly.
    """

    def value(self, x):
        return self.ops.sum(x * x)

    def tree_flatten(self):
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx):
        return _SumSquares(self.domain.convert(new_ctx), new_ctx)


# ===========================================================================
# Abstract enforcement
# ===========================================================================
class TestAbstract:
    def test_functional_is_not_directly_instantiable(self, numpy_ctx):
        """``Functional`` is abstract — ``value`` and pytree hooks are required."""
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(TypeError):
            sc.Functional(space, numpy_ctx)


# ===========================================================================
# domain / value / __call__
# ===========================================================================
class TestDomainAndValue:
    def test_domain_property_returns_converted_domain(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        assert f.domain is f.dom
        assert f.domain == space
        assert f.domain.ctx == numpy_ctx

    def test_call_is_alias_for_value(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(to_numpy(f(x)), to_numpy(f.value(x)))
        np.testing.assert_allclose(to_numpy(f.value(x)), 14.0)


# ===========================================================================
# Explicit context priority
# ===========================================================================
class TestContextPriority:
    def test_explicit_context_overrides_domain_inferred_context(
        self, numpy_ctx, numpy_f32_ctx
    ):
        # Domain is built in float32, functional is asked for float64.
        space = sc.DenseCoordinateSpace((3,), numpy_f32_ctx)
        f = _SumSquares(space, numpy_ctx)
        assert f.ctx == numpy_ctx
        assert f.dtype == np.dtype(np.float64)
        assert f.domain.ctx == numpy_ctx

    def test_context_inferred_from_domain_when_unspecified(self, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_f32_ctx)
        f = _SumSquares(space)
        assert f.ctx == numpy_f32_ctx


# ===========================================================================
# compose: delegates to make_functional_composed and validates the join
# ===========================================================================
class TestCompose:
    def test_compose_returns_composed_functional(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -1.0]), X, numpy_ctx)
        # _SumSquares is generic, so composition stays generic.
        F = _SumSquares(Y, numpy_ctx)
        pullback = F.compose(A)
        x = numpy_ctx.asarray([3.0, 4.0])

        assert isinstance(pullback, sc.ComposedFunctional)
        np.testing.assert_allclose(
            to_numpy(pullback.value(x)), to_numpy(F.value(A.apply(x)))
        )

    def test_compose_rejects_codomain_mismatch(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.IdentityLinOp(X, numpy_ctx)
        F = _SumSquares(Y, numpy_ctx)
        with pytest.raises(ValueError, match="A.codomain == F.domain"):
            F.compose(A)


# ===========================================================================
# vvalue default fallback (ops.vmap)
# ===========================================================================
class TestVValueFallback:
    def test_default_vvalue_matches_elementwise_loop(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        xs = numpy_ctx.asarray([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0], [4.0, 0.5, -2.0]])
        expected = f.ops.stack(tuple(f.value(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(f.vvalue(xs)), to_numpy(expected))


# ===========================================================================
# assert_domain
# ===========================================================================
class TestAssertDomain:
    def test_assert_domain_accepts_valid_element(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        f.assert_domain(numpy_ctx.asarray([1.0, 2.0, 3.0]))  # no raise

    def test_assert_domain_rejects_wrong_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        with pytest.raises(Exception):
            f.assert_domain(numpy_ctx.asarray([1.0, 2.0]))


# ===========================================================================
# tree_flatten / tree_unflatten round-trip
# ===========================================================================
class TestPytree:
    def test_tree_flatten_unflatten_round_trip(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        children, aux = f.tree_flatten()
        restored = _SumSquares.tree_unflatten(aux, children)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        assert restored.domain == f.domain
        np.testing.assert_allclose(to_numpy(restored.value(x)), to_numpy(f.value(x)))


# ===========================================================================
# convert
# ===========================================================================
class TestConvert:
    def test_convert_to_same_context_returns_self(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = _SumSquares(space, numpy_ctx)
        assert f.convert(numpy_ctx) is f

    def test_convert_preserves_value_across_dtype(self, numpy_f32_ctx, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_f32_ctx)
        f = _SumSquares(space, numpy_f32_ctx)
        g = f.convert(numpy_ctx)
        assert g.ctx == numpy_ctx
        assert g.dtype == np.dtype(np.float64)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(to_numpy(g.value(x)), 14.0)


# ===========================================================================
# Private helper: _check_scalar_shape
# ===========================================================================
class TestCheckScalarShape:
    def test_accepts_matching_shape(self):
        _check_scalar_shape(np.asarray(3.0), ())  # scalar, no raise
        _check_scalar_shape(np.zeros((4,)), (4,))  # batch, no raise

    def test_rejects_mismatched_shape(self):
        with pytest.raises(ValueError, match="Expected scalar batch output with shape"):
            _check_scalar_shape(np.zeros((2,)), ())

    def test_treats_objects_without_shape_as_scalar(self):
        _check_scalar_shape(3.0, ())  # Python float has no ``shape`` -> ()


# ===========================================================================
# Private helper: _leading_batch_size
# ===========================================================================
class TestLeadingBatchSize:
    def test_returns_leading_axis_for_dense_batch(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        xs = numpy_ctx.asarray(np.zeros((5, 3)))
        assert _leading_batch_size(space, xs) == 5

    def test_returns_zero_for_unshaped_input(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        assert _leading_batch_size(space, 3.0) == 0

    def test_recurses_into_tuple_first_leaf(self, numpy_ctx):
        left = sc.DenseCoordinateSpace((2,), numpy_ctx)
        right = sc.DenseCoordinateSpace((1,), numpy_ctx)
        space = sc.TreeSpace.from_leaf_spaces((left, right), ctx=numpy_ctx)
        xs = (numpy_ctx.asarray(np.zeros((7, 2))), numpy_ctx.asarray(np.zeros((7, 1))))
        assert _leading_batch_size(space, xs) == 7


# ===========================================================================
# Private helper: _warn_vmap_fallback_once
# ===========================================================================
class TestWarnVmapFallbackOnce:
    def test_warns_once_per_class_and_method_on_python_loop_backend(self, numpy_ctx):
        _VMAP_FALLBACK_WARNED.clear()
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        obj = _SumSquares(space, numpy_ctx)
        assert obj.ops.has_native_vmap is False

        with pytest.warns(RuntimeWarning, match="falls back to a Python loop"):
            _warn_vmap_fallback_once(obj, "vvalue", 64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_vmap_fallback_once(obj, "vvalue", 64)
        assert caught == []

    def test_does_not_warn_for_small_batches(self, numpy_ctx):
        _VMAP_FALLBACK_WARNED.clear()
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        obj = _SumSquares(space, numpy_ctx)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_vmap_fallback_once(obj, "vvalue", 8)
        assert caught == []

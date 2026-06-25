"""Tests for :class:`spacecore.MatrixFreeLinearFunctional`.

Checklist section 7, ``MatrixFreeLinearFunctional``:

* The supplied ``value`` callable is used verbatim.
* ``representer`` raises ``NotImplementedError`` (no stored dual vector).
* Construction rejects non-callable ``value`` / ``vvalue``.
* ``value`` enforces a scalar output under ``standard`` checks.
* ``vvalue`` uses the supplied batched callable when given, and otherwise
  falls back to the base ``vmap`` path (with a one-time Python-loop warning
  on backends without native ``vmap``).
* ``__eq__`` compares callables by identity; ``_convert`` keeps them.
* ``tree_flatten`` / ``tree_unflatten`` round-trip preserves the callables.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spacecore as sc

from spacecore.functional._base import _VMAP_FALLBACK_WARNED
from tests._helpers import to_numpy


# ===========================================================================
# Supplied callable is used verbatim
# ===========================================================================
class TestValue:
    def test_value_calls_supplied_callable(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        c = numpy_ctx.asarray([2.0, 3.0])
        f = sc.MatrixFreeLinearFunctional(lambda y: space.inner(c, y), space, numpy_ctx)
        x = numpy_ctx.asarray([4.0, 5.0])
        np.testing.assert_allclose(to_numpy(f.value(x)), 23.0)

    def test_representer_raises(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        f = sc.MatrixFreeLinearFunctional(lambda y: space.inner(space.zeros(), y), space, numpy_ctx)
        with pytest.raises(NotImplementedError):
            f.representer


# ===========================================================================
# Construction guards
# ===========================================================================
class TestConstruction:
    def test_rejects_non_callable_value(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(TypeError, match="value must be callable"):
            sc.MatrixFreeLinearFunctional(123, space, numpy_ctx)

    def test_rejects_non_callable_vvalue(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(TypeError, match="vvalue must be callable"):
            sc.MatrixFreeLinearFunctional(lambda y: y, space, numpy_ctx, vvalue=123)

    def test_value_enforces_scalar_output_under_standard_checks(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        # Callable returns a vector, not a scalar.
        f = sc.MatrixFreeLinearFunctional(lambda x: x, space, numpy_ctx)
        with pytest.raises(ValueError, match="Expected scalar batch output"):
            f.value(numpy_ctx.asarray([1.0, 2.0]))

    def test_value_skips_scalar_check_at_none_level(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        space = sc.DenseCoordinateSpace((2,), ctx)
        f = sc.MatrixFreeLinearFunctional(lambda x: x, space, ctx)
        out = f.value(ctx.asarray([1.0, 2.0]))  # no raise
        np.testing.assert_allclose(to_numpy(out), [1.0, 2.0])


# ===========================================================================
# vvalue: supplied callable vs base fallback
# ===========================================================================
class TestVValue:
    def test_uses_supplied_batched_callable(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0])
        sentinel = {"called": 0}

        def vvalue_fn(xs):
            sentinel["called"] += 1
            return xs @ numpy_ctx.asarray([1.0, -2.0])

        f = sc.MatrixFreeLinearFunctional(
            lambda y: space.inner(c, y), space, numpy_ctx, vvalue=vvalue_fn
        )
        xs = numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        out = f.vvalue(xs)
        assert sentinel["called"] == 1
        expected = f.ops.stack(tuple(f.value(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(out), to_numpy(expected))

    def test_falls_back_to_base_vmap_without_callable(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0])
        f = sc.MatrixFreeLinearFunctional(lambda y: space.inner(c, y), space, numpy_ctx)
        xs = numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        expected = f.ops.stack(tuple(f.value(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(f.vvalue(xs)), to_numpy(expected))

    def test_python_loop_fallback_warns_once_on_numpy(self):
        _VMAP_FALLBACK_WARNED.clear()
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        space = sc.DenseCoordinateSpace((2,), ctx)
        c = ctx.asarray([1.0, -2.0])
        f = sc.MatrixFreeLinearFunctional(lambda y: space.inner(c, y), space, ctx)
        xs = ctx.asarray(np.arange(80.0).reshape(40, 2))

        with pytest.warns(RuntimeWarning, match="falls back to a Python loop"):
            f.vvalue(xs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f.vvalue(xs)
        assert caught == []


# ===========================================================================
# __eq__
# ===========================================================================
class TestEquality:
    def test_equal_when_same_callables_and_domain(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        fn = lambda y: space.inner(numpy_ctx.asarray([1.0, 1.0]), y)  # noqa: E731
        assert sc.MatrixFreeLinearFunctional(fn, space, numpy_ctx) == (
            sc.MatrixFreeLinearFunctional(fn, space, numpy_ctx)
        )

    def test_not_equal_when_callables_differ(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        a = sc.MatrixFreeLinearFunctional(lambda y: y[0], space, numpy_ctx)
        b = sc.MatrixFreeLinearFunctional(lambda y: y[1], space, numpy_ctx)
        assert a != b

    def test_not_equal_to_other_type(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        f = sc.MatrixFreeLinearFunctional(lambda y: y[0], space, numpy_ctx)
        assert (f == 42) is False


# ===========================================================================
# Pytree round-trip + convert preserve callables
# ===========================================================================
class TestPytreeAndConvert:
    def test_tree_flatten_unflatten_preserves_callables(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        value_fn = lambda y: space.inner(numpy_ctx.asarray([1.0, 1.0]), y)  # noqa: E731
        vvalue_fn = lambda xs: xs @ numpy_ctx.asarray([1.0, 1.0])  # noqa: E731
        f = sc.MatrixFreeLinearFunctional(value_fn, space, numpy_ctx, vvalue=vvalue_fn)

        children, aux = f.tree_flatten()
        restored = sc.MatrixFreeLinearFunctional.tree_unflatten(aux, children)
        assert restored.value_fn is value_fn
        assert restored.vvalue_fn is vvalue_fn
        assert restored == f

    def test_convert_preserves_callable_and_converts_domain(
        self, numpy_f32_ctx, numpy_ctx
    ):
        space = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        value_fn = lambda y: y[0] + y[1]  # noqa: E731
        f = sc.MatrixFreeLinearFunctional(value_fn, space, numpy_f32_ctx)
        g = f.convert(numpy_ctx)
        assert g.value_fn is value_fn
        assert g.ctx == numpy_ctx
        assert g.domain.dtype == np.dtype(np.float64)
        np.testing.assert_allclose(to_numpy(g.value(numpy_ctx.asarray([3.0, 4.0]))), 7.0)

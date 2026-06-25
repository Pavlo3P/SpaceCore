"""Tests for :class:`spacecore.ComposedFunctional` and ``make_functional_composed``.

Checklist section 7, ``ComposedFunctional`` / ``make_functional_composed``:

* ``make_functional_composed`` specializes by type:
  ``InnerProductFunctional ∘ A`` -> ``InnerProductFunctional``,
  ``LinOpQuadraticForm ∘ A`` -> ``LinOpQuadraticForm``,
  generic ``Functional ∘ A`` -> ``ComposedFunctional``.
* ``_require_composable`` rejects non-``Functional`` ``F``, non-``LinOp`` ``A``,
  and a codomain/domain mismatch.
* ``ComposedFunctional.value(x) == F(A x)``.
* ``__eq__``, ``tree_flatten`` / ``tree_unflatten`` round-trip, ``_convert``.
* Private element helpers ``_convert_space_element`` / ``_broadcast_space_element``.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from spacecore.functional._composed import make_functional_composed, _require_composable
from spacecore.functional._linear import _convert_space_element
from spacecore.kernels.core.functional import _broadcast_space_element
from tests._helpers import to_numpy


class _SumSquares(sc.Functional):
    """Generic (non-specialized) functional: ``F(x) = sum(x * x)``."""

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
# make_functional_composed: type specialization
# ===========================================================================
class TestSpecialization:
    def test_inner_product_compose_specializes_to_inner_product(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        matrix_np = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]])
        A = sc.DenseLinOp(numpy_ctx.asarray(matrix_np), X, Y, numpy_ctx)
        c_np = np.array([2.0, -1.0, 0.5])
        c = numpy_ctx.asarray(c_np)
        F = sc.InnerProductFunctional(c, Y, numpy_ctx)
        pullback = F.compose(A)
        x = numpy_ctx.asarray([4.0, -2.0])

        assert isinstance(pullback, sc.InnerProductFunctional)
        # Independent reference: on Euclidean spaces the pulled-back representer is
        # Aᴴc, computed here with NumPy rather than A.H.apply(c) — which is the very
        # code path make_functional_composed uses, so comparing to it was circular.
        np.testing.assert_allclose(to_numpy(pullback.representer), matrix_np.conj().T @ c_np)
        np.testing.assert_allclose(
            to_numpy(pullback.value(x)), to_numpy(F.value(A.apply(x)))
        )

    def test_quadratic_compose_specializes_to_quadratic(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]]), X, Y, numpy_ctx
        )
        Q = sc.IdentityLinOp(Y, numpy_ctx)
        linear = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, -2.0, 0.5]), Y, numpy_ctx)
        F = sc.LinOpQuadraticForm(Q, linear, 1.25, numpy_ctx)
        pullback = F.compose(A)
        x = numpy_ctx.asarray([0.5, -1.5])

        assert isinstance(pullback, sc.LinOpQuadraticForm)
        np.testing.assert_allclose(
            to_numpy(pullback.value(x)), to_numpy(F.value(A.apply(x)))
        )
        np.testing.assert_allclose(
            to_numpy(pullback.grad(x)), to_numpy(A.H.apply(F.grad(A.apply(x))))
        )

    def test_generic_compose_falls_back_to_composed_functional(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -1.0]), X, numpy_ctx)
        F = _SumSquares(Y, numpy_ctx)
        pullback = make_functional_composed(F, A)
        x = numpy_ctx.asarray([3.0, 4.0])

        assert isinstance(pullback, sc.ComposedFunctional)
        np.testing.assert_allclose(
            to_numpy(pullback.value(x)), to_numpy(F.value(A.apply(x)))
        )


# ===========================================================================
# _require_composable
# ===========================================================================
class TestRequireComposable:
    def test_rejects_non_functional(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.IdentityLinOp(X, numpy_ctx)
        with pytest.raises(TypeError, match="F must be a Functional"):
            _require_composable("not-a-functional", A)

    def test_rejects_non_linop(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        F = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 2.0]), X, numpy_ctx)
        with pytest.raises(TypeError, match="A must be a LinOp"):
            _require_composable(F, "not-a-linop")

    def test_rejects_codomain_mismatch(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.IdentityLinOp(X, numpy_ctx)
        F = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 2.0, 3.0]), Y, numpy_ctx)
        with pytest.raises(ValueError, match="A.codomain == F.domain"):
            _require_composable(F, A)


# ===========================================================================
# ComposedFunctional behaviour
# ===========================================================================
class TestComposedFunctional:
    def _make(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -1.0]), X, numpy_ctx)
        F = _SumSquares(Y, numpy_ctx)
        return sc.ComposedFunctional(F, A), F, A

    def test_value_is_pullback(self, numpy_ctx):
        composed, F, A = self._make(numpy_ctx)
        x = numpy_ctx.asarray([3.0, 4.0])
        np.testing.assert_allclose(
            to_numpy(composed.value(x)), to_numpy(F.value(A.apply(x)))
        )
        np.testing.assert_allclose(to_numpy(composed(x)), to_numpy(composed.value(x)))

    def test_domain_is_operator_domain(self, numpy_ctx):
        composed, _F, A = self._make(numpy_ctx)
        assert composed.domain == A.domain

    def test_equality(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -1.0]), X, numpy_ctx)
        F = _SumSquares(Y, numpy_ctx)
        # Equality compares operands; share F and A so both fields match.
        a = sc.ComposedFunctional(F, A)
        b = sc.ComposedFunctional(F, A)
        assert a == b
        assert (a == 42) is False

    def test_tree_flatten_unflatten_round_trip(self, numpy_ctx):
        composed, _F, _A = self._make(numpy_ctx)
        children, aux = composed.tree_flatten()
        restored = sc.ComposedFunctional.tree_unflatten(aux, children)
        x = numpy_ctx.asarray([3.0, 4.0])
        assert restored == composed
        np.testing.assert_allclose(
            to_numpy(restored.value(x)), to_numpy(composed.value(x))
        )

    def test_convert_preserves_value_across_dtype(self, numpy_f32_ctx, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        Y = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        A = sc.DiagonalLinOp(numpy_f32_ctx.asarray([2.0, -1.0]), X, numpy_f32_ctx)
        F = _SumSquares(Y, numpy_f32_ctx)
        composed = sc.ComposedFunctional(F, A)
        converted = composed.convert(numpy_ctx)
        assert converted.ctx == numpy_ctx
        x = numpy_ctx.asarray([3.0, 4.0])
        # F(A x) = sum((2*3, -1*4)^2) = 36 + 16 = 52.
        np.testing.assert_allclose(to_numpy(converted.value(x)), 52.0)


# ===========================================================================
# Private element helpers
# ===========================================================================
class TestElementHelpers:
    def test_convert_space_element_casts_dense_dtype(self, numpy_f32_ctx, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        value = numpy_f32_ctx.asarray([1.0, 2.0, 3.0])
        converted = _convert_space_element(space, value)
        assert space.ops.get_dtype(converted) == np.dtype(np.float64)
        np.testing.assert_allclose(to_numpy(converted), [1.0, 2.0, 3.0])

    def test_convert_space_element_handles_tree_space(self, numpy_f32_ctx, numpy_ctx):
        left = sc.DenseCoordinateSpace((2,), numpy_ctx)
        right = sc.DenseCoordinateSpace((1,), numpy_ctx)
        space = sc.TreeSpace.from_leaf_spaces((left, right), ctx=numpy_ctx)
        value = (numpy_f32_ctx.asarray([1.0, 2.0]), numpy_f32_ctx.asarray([3.0]))
        converted = _convert_space_element(space, value)
        leaves = space.flatten_tree(converted)
        np.testing.assert_allclose(to_numpy(leaves[0]), [1.0, 2.0])
        np.testing.assert_allclose(to_numpy(leaves[1]), [3.0])

    def test_broadcast_space_element_dense(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        value = numpy_ctx.asarray([1.0, 2.0, 3.0])
        out = _broadcast_space_element(space, value, 4)
        assert out.shape == (4, 3)
        np.testing.assert_allclose(to_numpy(out), np.broadcast_to([1.0, 2.0, 3.0], (4, 3)))

    def test_broadcast_space_element_tree(self, numpy_ctx):
        left = sc.DenseCoordinateSpace((2,), numpy_ctx)
        right = sc.DenseCoordinateSpace((1,), numpy_ctx)
        space = sc.TreeSpace.from_leaf_spaces((left, right), ctx=numpy_ctx)
        value = (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0]))
        out = _broadcast_space_element(space, value, 5)
        leaves = space.flatten_tree(out)
        assert leaves[0].shape == (5, 2)
        assert leaves[1].shape == (5, 1)

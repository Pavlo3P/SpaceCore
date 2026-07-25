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
# Chain rule: grad(F o A)(x) = A^#(grad F(A x))
# ===========================================================================
def _weighted(ctx, weights):
    return sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(np.asarray(weights)))
    )


_M = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]])


class TestChainRuleGradient:
    """``ComposedFunctional`` had no gradient at all; it raised ``NotImplementedError``.

    The load-bearing case is **different** non-Euclidean metrics on domain and
    codomain: there the metric adjoint ``rapply`` differs from the coordinate
    adjoint, so a Euclidean-only test would certify nothing. ``rapply`` *is*
    ``A^#``, so no extra Riesz map belongs in the chain rule — applying one
    would count the geometry twice.
    """

    def _setup(self, ctx, weighted):
        if weighted:
            X, Y = _weighted(ctx, [2.0, 5.0, 11.0]), _weighted(ctx, [3.0, 1.0, 7.0])
        else:
            X = Y = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray(_M), X, Y, ctx)
        return X, Y, A, sc.SquaredL2NormFunctional(Y)

    @pytest.mark.parametrize("weighted", [False, True], ids=["euclidean", "weighted"])
    def test_matches_the_explicit_formula(self, numpy_ctx, weighted):
        X, Y, A, F = self._setup(numpy_ctx, weighted)
        G = sc.ComposedFunctional(F, A)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        np.testing.assert_allclose(
            to_numpy(G.grad(x)), to_numpy(A.rapply(F.grad(A.apply(x))))
        )

    @pytest.mark.parametrize("weighted", [False, True], ids=["euclidean", "weighted"])
    def test_satisfies_the_riesz_defining_property(self, numpy_ctx, weighted):
        """``<grad G(x), h>_X == DG(x)[h]`` — the gradient's actual definition."""
        X, Y, A, F = self._setup(numpy_ctx, weighted)
        G = sc.ComposedFunctional(F, A)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        eps = 1e-6
        directional = (
            float(G.value(X.axpy(eps, h, x))) - float(G.value(X.axpy(-eps, h, x)))
        ) / (2.0 * eps)
        np.testing.assert_allclose(
            to_numpy(X.inner(G.grad(x), h)), directional, rtol=1e-6, atol=1e-6
        )

    def test_metric_case_rejects_the_coordinate_adjoint(self, numpy_ctx):
        """Without this the weighted test above could pass a wrong implementation."""
        X, Y, A, F = self._setup(numpy_ctx, weighted=True)
        G = sc.ComposedFunctional(F, A)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        coordinate_answer = _M.T @ to_numpy(F.grad(A.apply(x)))
        assert not np.allclose(to_numpy(G.grad(x)), coordinate_answer)

    def test_agrees_with_the_specialized_pullback(self, numpy_ctx):
        """Cross-check against a path that was already correct.

        ``make_functional_composed`` rewrites ``<c, ·> o A`` to
        ``<A^# c, ·>`` without ever building a ``ComposedFunctional``. Forcing
        the generic node on the same operands must give the same gradient.
        """
        X, Y = _weighted(numpy_ctx, [2.0, 5.0, 11.0]), _weighted(numpy_ctx, [3.0, 1.0, 7.0])
        A = sc.DenseLinOp(numpy_ctx.asarray(_M), X, Y, numpy_ctx)
        c = numpy_ctx.asarray([1.0, 0.5, -2.0])
        F = sc.InnerProductFunctional(c, Y, numpy_ctx)

        specialized = make_functional_composed(F, A)
        assert not isinstance(specialized, sc.ComposedFunctional)
        generic = sc.ComposedFunctional(F, A)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        np.testing.assert_allclose(to_numpy(generic.grad(x)), to_numpy(specialized.grad(x)))

    def test_value_and_grad_is_consistent(self, numpy_ctx):
        X, Y, A, F = self._setup(numpy_ctx, weighted=True)
        G = sc.ComposedFunctional(F, A)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        value, gradient = G.value_and_grad(x)
        np.testing.assert_allclose(to_numpy(value), to_numpy(G.value(x)))
        np.testing.assert_allclose(to_numpy(gradient), to_numpy(G.grad(x)))

    def test_value_and_grad_applies_the_operator_once(self, numpy_ctx):
        """The point of the fused path: the base default would apply ``A`` twice."""
        X, Y, A, F = self._setup(numpy_ctx, weighted=False)
        calls = []
        counting = sc.MatrixFreeLinOp(
            lambda v: calls.append("apply") or A.apply(v),
            lambda v: A.rapply(v),
            X, Y, numpy_ctx,
        )
        sc.ComposedFunctional(F, counting).value_and_grad(numpy_ctx.asarray([1.0, 2.0, -1.0]))
        assert calls == ["apply"]

    def test_batched_gradient(self, numpy_ctx):
        X, Y, A, F = self._setup(numpy_ctx, weighted=True)
        G = sc.ComposedFunctional(F, A)
        xs = numpy_ctx.asarray([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]])
        np.testing.assert_allclose(
            to_numpy(G.vgrad(xs)),
            np.stack([to_numpy(G.grad(numpy_ctx.asarray(row))) for row in to_numpy(xs)]),
        )

    def test_propagates_missing_inner_gradient(self, numpy_ctx):
        """A composition is differentiable only if its inner functional is."""
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.DenseLinOp(numpy_ctx.asarray(_M), X, X, numpy_ctx)
        G = sc.ComposedFunctional(_SumSquares(X, numpy_ctx), A)
        with pytest.raises(NotImplementedError, match="grad"):
            G.grad(numpy_ctx.asarray([1.0, 2.0, -1.0]))


class TestChainRuleComplex:
    """Complex domains, where the conjugation in ``rapply`` has to be right."""

    def _spaces(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
        X = _weighted(ctx, [2.0, 5.0, 11.0])
        Y = _weighted(ctx, [3.0, 1.0, 7.0])
        A = sc.DenseLinOp(
            ctx.asarray(np.array([[1 + 1j, 2, 0], [0, 1, 3 - 2j], [2, 0, 1j]])), X, Y, ctx
        )
        return ctx, X, Y, A

    def test_holomorphic_functional_satisfies_the_plain_identity(self):
        """A complex-*linear* inner: ``<grad G, h> == DG[h]`` exactly."""
        ctx, X, Y, A = self._spaces()
        c = ctx.asarray([1 + 1j, 2 - 0.5j, -1 + 0.3j])
        G = sc.ComposedFunctional(sc.InnerProductFunctional(c, Y, ctx), A)
        x = ctx.asarray([1 + 0j, 2 - 1j, -1 + 2j])
        h = ctx.asarray([0.5 + 1j, -1 + 0j, 2 + 0j])
        eps = 1e-6
        directional = (
            complex(G.value(X.axpy(eps, h, x))) - complex(G.value(X.axpy(-eps, h, x)))
        ) / (2.0 * eps)
        np.testing.assert_allclose(
            complex(X.inner(G.grad(x), h)), directional, rtol=1e-6, atol=1e-6
        )
        # ...and equals the adjoint pull-back of the representer.
        np.testing.assert_allclose(to_numpy(G.grad(x)), to_numpy(A.H.apply(c)))

    def test_real_valued_functional_uses_the_real_part_convention(self):
        """``1/2||y||^2`` is real-valued on a complex space, hence not holomorphic.

        The library's convention there is ``Re<grad, h> == DF[h]`` (CR calculus);
        that already holds for the inner functional, and composition preserves it.
        """
        ctx, X, Y, A = self._spaces()
        G = sc.ComposedFunctional(sc.SquaredL2NormFunctional(Y), A)
        x = ctx.asarray([1 + 0j, 2 - 1j, -1 + 2j])
        h = ctx.asarray([0.5 + 1j, -1 + 0j, 2 + 0j])
        eps = 1e-6
        directional = (
            complex(G.value(X.axpy(eps, h, x))) - complex(G.value(X.axpy(-eps, h, x)))
        ) / (2.0 * eps)
        np.testing.assert_allclose(
            complex(X.inner(G.grad(x), h)).real, directional.real, rtol=1e-6, atol=1e-6
        )


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

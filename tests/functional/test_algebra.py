"""Functional algebra: scalar multiples and sums (0.4.2 W4, mirrors LinOp algebra)."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, to_numpy
from tests.optimize._helpers import has_optax


class _SumSquares(sc.Functional):
    """Sum-of-squares test functional; ``grad = 2x`` as a domain element.

    ``grad``/``value`` go through the domain vector ops (``scale``/``flatten``)
    so the functional is valid on dense *and* tree domains.
    """

    def value(self, x, *args, **kwargs):
        return self.ops.sum(self.domain.flatten(x) ** 2)

    def grad(self, x, *args, **kwargs):
        return self.domain.scale(2.0, x)

    def tree_flatten(self):
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx):
        return _SumSquares(self.domain.convert(new_ctx), new_ctx)


def _dense(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    return X, _SumSquares(X, ctx), ctx.asarray([1.0, -2.0, 3.0])


def _tree(ctx):
    left = sc.DenseCoordinateSpace((2,), ctx)
    right = sc.DenseCoordinateSpace((1,), ctx)
    X = sc.TreeSpace.from_leaf_spaces((left, right), ctx=ctx)
    x = X.element((ctx.asarray([1.0, -2.0]), ctx.asarray([3.0])))
    return X, _SumSquares(X, ctx), x


# ===========================================================================
# Scalar multiples: a * F, F * a, -F
# ===========================================================================
class TestScaled:
    def test_value_and_grad_scale(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = 2.0 * F
        np.testing.assert_allclose(to_numpy(G.value(x)), 2.0 * to_numpy(F.value(x)))
        np.testing.assert_allclose(to_numpy(G.grad(x)), 2.0 * to_numpy(F.grad(x)))
        v, g = G.value_and_grad(x)
        np.testing.assert_allclose(to_numpy(v), 2.0 * to_numpy(F.value(x)))
        np.testing.assert_allclose(to_numpy(g), 2.0 * to_numpy(F.grad(x)))

    def test_left_and_right_mul_agree(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        np.testing.assert_allclose(to_numpy((F * 3.0).value(x)), to_numpy((3.0 * F).value(x)))

    def test_negation(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        np.testing.assert_allclose(to_numpy((-F).value(x)), -to_numpy(F.value(x)))
        np.testing.assert_allclose(to_numpy((-F).grad(x)), -to_numpy(F.grad(x)))

    def test_non_scalar_returns_notimplemented(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert F.__mul__("x") is NotImplemented
        # A Functional operand is *not* rejected — it builds a pointwise product
        # (see TestProduct). Only operands that are neither scalar-like nor
        # Functional defer to the reflected operation.
        assert F.__mul__(object()) is NotImplemented

    def test_type_and_scalar_guards(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        with pytest.raises(TypeError, match="scalar-like"):
            sc.ScaledFunctional("nope", F)
        with pytest.raises(TypeError, match="Functional"):
            sc.ScaledFunctional(2.0, "nope")

    def test_complex_scalar_conjugates_gradient(self):
        # Riesz gradient of a*F is conj(a)*grad(F): the inner product conjugates
        # its first argument, so <grad(aF), h> must recover a * <grad(F), h>.
        ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
        X = sc.DenseCoordinateSpace((3,), ctx, check_level="standard")
        c = ctx.asarray([1 + 1j, 2 - 0.5j, -1 + 0.3j])
        F = sc.InnerProductFunctional(c, X, ctx, check_level="standard")
        a = 2 + 3j
        x = ctx.asarray([0.5 - 1j, 1 + 0j, -2 + 0.5j])
        h = ctx.asarray([1 + 0j, 0 + 1j, 0.5 - 0.5j])
        lhs = complex(to_numpy(X.inner((a * F).grad(x), h)))
        rhs = a * complex(to_numpy(X.inner(c, h)))
        assert np.isclose(lhs, rhs)


class TestScaledFactory:
    def test_unit_scalar_passes_through(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert sc.make_scaled_functional(1.0, F) is F

    def test_nested_scaled_folds_to_one_scalar(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        folded = sc.make_scaled_functional(2.0, sc.make_scaled_functional(3.0, F))
        assert isinstance(folded, sc.ScaledFunctional)
        assert not isinstance(folded.functional, sc.ScaledFunctional)
        assert folded.scalar == 6.0


class TestScaledPytree:
    def test_round_trip(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = 2.5 * F
        children, aux = G.tree_flatten()
        restored = sc.ScaledFunctional.tree_unflatten(aux, children)
        assert restored == G
        np.testing.assert_allclose(to_numpy(restored.value(x)), to_numpy(G.value(x)))


# ===========================================================================
# Sums: F + G, F - G, sum(...)
# ===========================================================================
class TestSum:
    def test_value_grad_linearity(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        S = F + G
        np.testing.assert_allclose(
            to_numpy(S.value(x)), to_numpy(F.value(x)) + to_numpy(G.value(x))
        )
        np.testing.assert_allclose(to_numpy(S.grad(x)), to_numpy(X.add(F.grad(x), G.grad(x))))
        v, g = S.value_and_grad(x)
        np.testing.assert_allclose(to_numpy(v), to_numpy(F.value(x)) + to_numpy(G.value(x)))
        np.testing.assert_allclose(to_numpy(g), to_numpy(X.add(F.grad(x), G.grad(x))))

    def test_difference_is_zero_when_equal(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        np.testing.assert_allclose(to_numpy((F - G).value(x)), 0.0, atol=1e-12)
        np.testing.assert_allclose(to_numpy((F - G).grad(x)), 0.0, atol=1e-12)

    def test_sum_builtin(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        np.testing.assert_allclose(to_numpy(sum([F, G]).value(x)), to_numpy((F + G).value(x)))

    def test_mixed_scale_and_sum_grad(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        C = 2.0 * F + (-1.0) * G
        expected = X.add(X.scale(2.0, F.grad(x)), X.scale(-1.0, G.grad(x)))
        np.testing.assert_allclose(to_numpy(C.grad(x)), to_numpy(expected))

    def test_non_functional_returns_notimplemented(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert F.__add__(object()) is NotImplemented
        assert F.__sub__(object()) is NotImplemented


class TestSumFactory:
    def test_nested_sums_flatten(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        H = _SumSquares(X, numpy_ctx)
        s = sc.make_functional_sum([F, sc.make_functional_sum([G, H])])
        assert isinstance(s, sc.SumFunctional)
        assert len(s.parts) == 3

    def test_single_term_unwraps(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert sc.make_functional_sum([F]) is F


class TestSumDomainAndPytree:
    def test_domain_mismatch_raises(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        G = _SumSquares(sc.DenseCoordinateSpace((2,), numpy_ctx), numpy_ctx)
        with pytest.raises(ValueError, match="same domain"):
            F + G

    def test_pytree_round_trip(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        S = F + G
        children, aux = S.tree_flatten()
        restored = sc.SumFunctional.tree_unflatten(aux, children)
        assert restored == S
        np.testing.assert_allclose(to_numpy(restored.value(x)), to_numpy(S.value(x)))


# ===========================================================================
# Tree domain: gradients must combine via X.add / X.scale, not raw +/*
# ===========================================================================
class TestTreeDomainGrad:
    def test_sum_grad_via_domain_add(self, numpy_ctx):
        X, F, x = _tree(numpy_ctx)
        G = _SumSquares(X, numpy_ctx)
        np.testing.assert_allclose(
            to_numpy(X.flatten((F + G).grad(x))),
            to_numpy(X.flatten(X.add(F.grad(x), G.grad(x)))),
        )

    def test_scaled_grad_via_domain_scale(self, numpy_ctx):
        X, F, x = _tree(numpy_ctx)
        np.testing.assert_allclose(
            to_numpy(X.flatten((3.0 * F).grad(x))),
            to_numpy(X.flatten(X.scale(3.0, F.grad(x)))),
        )


# ===========================================================================
# Affine shift: F + c  (value shifted, gradient unchanged)
# ===========================================================================
class TestShifted:
    def test_affine_value_and_grad(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        A = F + 3.0
        np.testing.assert_allclose(to_numpy(A.value(x)), to_numpy(F.value(x)) + 3.0)
        np.testing.assert_allclose(to_numpy(A.grad(x)), to_numpy(F.grad(x)))  # shift ⇒ same grad

    def test_left_scalar_add_and_scalar_sub(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        np.testing.assert_allclose(to_numpy((3.0 + F).value(x)), to_numpy((F + 3.0).value(x)))
        np.testing.assert_allclose(to_numpy((F - 2.0).value(x)), to_numpy(F.value(x)) - 2.0)
        np.testing.assert_allclose(to_numpy((10.0 - F).value(x)), 10.0 - to_numpy(F.value(x)))
        np.testing.assert_allclose(to_numpy((10.0 - F).grad(x)), -to_numpy(F.grad(x)))

    def test_zero_offset_and_nested_fold(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert (F + 0.0) is F
        s = (F + 3.0) + 2.0
        assert isinstance(s, sc.ShiftedFunctional) and s.offset == 5.0

    def test_non_scalar_non_functional_notimplemented(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert F.__add__(object()) is NotImplemented


# ===========================================================================
# ZeroFunctional (additive identity) + factory zero handling
# ===========================================================================
class TestZero:
    def test_value_and_grad(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        Z = sc.ZeroFunctional(X, numpy_ctx)
        np.testing.assert_allclose(to_numpy(Z.value(x)), 0.0)
        np.testing.assert_allclose(to_numpy(Z.grad(x)), 0.0)

    def test_zero_scalar_makes_zero(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        assert isinstance(sc.make_scaled_functional(0.0, F), sc.ZeroFunctional)

    def test_zero_is_additive_identity(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        assert sc.make_functional_sum([F, sc.ZeroFunctional(X, numpy_ctx)]) is F

    def test_all_zero_sum_is_zero(self, numpy_ctx):
        X, _, _ = _dense(numpy_ctx)
        s = sc.make_functional_sum([sc.ZeroFunctional(X, numpy_ctx), sc.ZeroFunctional(X, numpy_ctx)])
        assert isinstance(s, sc.ZeroFunctional)

    def test_domain_mismatch_with_dropped_zero_raises(self, numpy_ctx):
        # A mismatched-domain Zero term must not be silently dropped.
        X, F, _ = _dense(numpy_ctx)
        other = sc.ZeroFunctional(sc.DenseCoordinateSpace((2,), numpy_ctx), numpy_ctx)
        with pytest.raises(ValueError, match="same domain"):
            sc.make_functional_sum([F, other])

    def test_pytree_round_trip(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        for obj, cls in [(F + 3.0, sc.ShiftedFunctional), (sc.ZeroFunctional(X, numpy_ctx), sc.ZeroFunctional)]:
            children, aux = obj.tree_flatten()
            assert cls.tree_unflatten(aux, children) == obj


# ===========================================================================
# ConstantFunctional (the scalar embedding) + factory zero collapse
# ===========================================================================
class TestConstant:
    def test_value_is_the_constant_everywhere(self, numpy_ctx):
        X, _, x = _dense(numpy_ctx)
        C = sc.ConstantFunctional(X, 2.5, numpy_ctx)
        np.testing.assert_allclose(to_numpy(C.value(x)), 2.5)
        np.testing.assert_allclose(
            to_numpy(C.value(numpy_ctx.asarray([9.0, 9.0, 9.0]))), 2.5
        )

    def test_gradient_is_zero(self, numpy_ctx):
        X, _, x = _dense(numpy_ctx)
        C = sc.ConstantFunctional(X, 2.5, numpy_ctx)
        np.testing.assert_allclose(to_numpy(C.grad(x)), 0.0)
        v, g = C.value_and_grad(x)
        np.testing.assert_allclose(to_numpy(v), 2.5)
        np.testing.assert_allclose(to_numpy(g), 0.0)

    def test_factory_collapses_zero_to_zero_functional(self, numpy_ctx):
        """The additive identity keeps a single representation."""
        X, _, _ = _dense(numpy_ctx)
        assert isinstance(sc.make_constant_functional(X, 0.0, numpy_ctx), sc.ZeroFunctional)
        assert isinstance(
            sc.make_constant_functional(X, 2.5, numpy_ctx), sc.ConstantFunctional
        )

    def test_rejects_non_scalar(self, numpy_ctx):
        X, _, _ = _dense(numpy_ctx)
        with pytest.raises(TypeError, match="scalar-like"):
            sc.ConstantFunctional(X, "nope", numpy_ctx)

    def test_equality_and_pytree_round_trip(self, numpy_ctx):
        X, _, _ = _dense(numpy_ctx)
        C = sc.ConstantFunctional(X, 2.5, numpy_ctx)
        assert C == sc.ConstantFunctional(X, 2.5, numpy_ctx)
        assert C != sc.ConstantFunctional(X, 3.5, numpy_ctx)
        children, aux = C.tree_flatten()
        assert sc.ConstantFunctional.tree_unflatten(aux, children) == C


# ===========================================================================
# ProductFunctional (pointwise product) + product-rule gradient
# ===========================================================================
class TestProduct:
    def test_value_is_the_pointwise_product(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        G = F * F
        assert isinstance(G, sc.ProductFunctional)
        np.testing.assert_allclose(
            to_numpy(G.value(x)), to_numpy(F.value(x)) ** 2
        )

    def test_gradient_follows_the_product_rule(self, numpy_ctx):
        """``grad(F·F) = 2 F(x) grad F(x)`` for a real-valued ``F``."""
        X, F, x = _dense(numpy_ctx)
        expected = 2.0 * to_numpy(F.value(x)) * to_numpy(F.grad(x))
        np.testing.assert_allclose(to_numpy((F * F).grad(x)), expected)

    def test_value_and_grad_matches_separate_calls(self, numpy_ctx):
        X, F, x = _dense(numpy_ctx)
        P = F * F
        v, g = P.value_and_grad(x)
        np.testing.assert_allclose(to_numpy(v), to_numpy(P.value(x)))
        np.testing.assert_allclose(to_numpy(g), to_numpy(P.grad(x)))

    def test_gradient_matches_finite_differences(self, numpy_ctx):
        """Independent check of the product rule on two *different* factors."""
        X, F, x = _dense(numpy_ctx)
        G = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 0.5, -2.0]), X, numpy_ctx)
        P = F * G
        direction = numpy_ctx.asarray([0.25, -0.5, 0.75])
        eps = 1e-6
        finite_difference = (
            P.value(X.axpy(eps, direction, x)) - P.value(X.axpy(-eps, direction, x))
        ) / (2.0 * eps)
        np.testing.assert_allclose(
            to_numpy(X.inner(P.grad(x), direction)),
            to_numpy(finite_difference),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_tree_domain_gradient_uses_domain_ops(self, numpy_ctx):
        """On a pytree domain the product rule must go through ``X.scale``/``X.add``.

        A raw ``*``/``+`` on the gradients would fail here, where an element is a
        tuple of leaves rather than an array.
        """
        X, F, x = _tree(numpy_ctx)
        np.testing.assert_allclose(
            to_numpy(X.flatten((F * F).grad(x))),
            2.0 * to_numpy(F.value(x)) * to_numpy(X.flatten(F.grad(x))),
        )

    def test_constant_factor_folds_to_a_scaled_functional(self, numpy_ctx):
        """Multiplying by a constant *is* scaling — the product node is not built."""
        X, F, x = _dense(numpy_ctx)
        C = sc.ConstantFunctional(X, 3.0, numpy_ctx)
        for product in (C * F, F * C):
            assert isinstance(product, sc.ScaledFunctional)
            np.testing.assert_allclose(
                to_numpy(product.value(x)), 3.0 * to_numpy(F.value(x))
            )

    def test_zero_factor_collapses_to_zero(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        Z = sc.ZeroFunctional(X, numpy_ctx)
        assert isinstance(F * Z, sc.ZeroFunctional)
        assert isinstance(Z * F, sc.ZeroFunctional)

    def test_no_value_based_simplification(self, numpy_ctx):
        """A functional that merely *happens* to be constant is not recognized.

        Canonicalization is structural: it reads node types, never values.
        """
        X, F, _ = _dense(numpy_ctx)
        constant_valued = sc.InnerProductFunctional(
            numpy_ctx.asarray([0.0, 0.0, 0.0]), X, numpy_ctx
        )
        assert isinstance(F * constant_valued, sc.ProductFunctional)

    def test_domain_mismatch_raises(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        other = sc.ZeroFunctional(sc.DenseCoordinateSpace((2,), numpy_ctx), numpy_ctx)
        with pytest.raises(ValueError, match="same domain"):
            sc.make_functional_product(F, other)

    def test_type_guard(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        with pytest.raises(TypeError, match="Functional"):
            sc.ProductFunctional(F, "nope")

    def test_equality_is_structural_and_ordered(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        G = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 0.5, -2.0]), X, numpy_ctx)
        assert (F * G) == (F * G)
        # Multiplication commutes, but this is expression-tree equality.
        assert (F * G) != (G * F)

    def test_pytree_round_trip(self, numpy_ctx):
        X, F, _ = _dense(numpy_ctx)
        G = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 0.5, -2.0]), X, numpy_ctx)
        P = F * G
        children, aux = P.tree_flatten()
        assert sc.ProductFunctional.tree_unflatten(aux, children) == P

    def test_complex_factors_conjugate_the_cofactor(self):
        """Riesz convention: coefficients enter conjugated, as in ``ScaledFunctional``.

        With ``F = <a,·>`` and ``G = <b,·>``, ``<grad(FG)(x), h>`` must recover
        ``G(x)<a,h> + F(x)<b,h>``.
        """
        ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
        X = sc.DenseCoordinateSpace((3,), ctx, check_level="standard")
        a = ctx.asarray([1 + 1j, 2 - 0.5j, -1 + 0.3j])
        b = ctx.asarray([0.5 - 1j, 1 + 0j, 0.25 + 0.75j])
        x = ctx.asarray([0.5 + 0.25j, -1.0 + 0.75j, 2.0 - 0.5j])
        h = ctx.asarray([1.0 + 0j, -0.5 + 0.25j, 0.75 - 1j])

        F = sc.InnerProductFunctional(a, X, ctx)
        G = sc.InnerProductFunctional(b, X, ctx)
        expected = G.value(x) * X.inner(a, h) + F.value(x) * X.inner(b, h)
        np.testing.assert_allclose(
            to_numpy(X.inner((F * G).grad(x), h)), to_numpy(expected)
        )


# ===========================================================================
# End-to-end: a composed functional drives minimize_optax
# ===========================================================================
@pytest.mark.skipif(not (has_jax() and has_optax()), reason="requires jax and optax")
class TestOptimizerIntegration:
    def test_minimize_sum_functional(self):
        import optax

        from tests.linalg._helpers import make_ctx
        from tests.optimize._helpers import euclidean_problem

        ctx = make_ctx("jax", np.float32)
        X, F, x_star = euclidean_problem(ctx)
        # A SumFunctional of two ScaledFunctionals; argmin(0.5F + 0.5F) == argmin F.
        composed = 0.5 * F + 0.5 * F
        assert isinstance(composed, sc.SumFunctional)
        res = sc.minimize_optax(composed, X.zeros(), optax.adam(1e-1), max_iter=2000, tol=1e-5, verbose=0)
        assert res.success
        np.testing.assert_allclose(to_numpy(res.x_element), x_star, atol=1e-2)

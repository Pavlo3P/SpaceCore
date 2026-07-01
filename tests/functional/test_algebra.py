"""Functional algebra: scalar multiples and sums (0.4.2 W4, mirrors LinOp algebra)."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


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
        assert F.__mul__(F) is NotImplemented  # a functional is not a scalar multiplier

    def test_type_and_scalar_guards(self, numpy_ctx):
        _, F, _ = _dense(numpy_ctx)
        with pytest.raises(TypeError, match="scalar-like"):
            sc.ScaledFunctional("nope", F)
        with pytest.raises(TypeError, match="Functional"):
            sc.ScaledFunctional(2.0, "nope")


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

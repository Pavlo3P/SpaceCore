"""Functional algebra: scalar multiples and sums (0.4.2 W4, mirrors LinOp algebra)."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


class _SumSquares(sc.Functional):
    """Differentiable test functional: ``F(x) = sum(x * x)``, ``grad = 2x``."""

    def value(self, x, *args, **kwargs):
        return self.ops.sum(x * x)

    def grad(self, x, *args, **kwargs):
        return 2.0 * x

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

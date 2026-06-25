"""Tests for the ADR-019 battery functionals.

Covers value/gradient correctness against NumPy references, the ADR-010 Riesz
gradient contract on non-Euclidean (weighted) metrics via a finite-difference
check, batched evaluation, validation, and pytree/convert round-trips.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


def _weighted_space(ctx, weights):
    return sc.DenseCoordinateSpace(
        (len(weights),), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(weights))
    )


def _fd_directional(f, x, h, eps=1e-6):
    """Central-difference directional derivative ``D f(x)[h]``."""
    xp = to_numpy(x)
    hp = to_numpy(h)
    plus = float(f.value(f.ctx.asarray(xp + eps * hp)))
    minus = float(f.value(f.ctx.asarray(xp - eps * hp)))
    return (plus - minus) / (2.0 * eps)


# ---------------------------------------------------------------------------
# SquaredL2NormFunctional
# ---------------------------------------------------------------------------
class TestSquaredL2NormFunctional:
    def test_value_is_half_squared_norm(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.SquaredL2NormFunctional(X)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        np.testing.assert_allclose(to_numpy(f.value(x)), 0.5 * (1 + 4 + 9))

    def test_gradient_is_identity(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.SquaredL2NormFunctional(X)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        np.testing.assert_allclose(to_numpy(f.grad(x)), to_numpy(x))

    def test_weighted_value_uses_metric(self, numpy_ctx):
        w = [2.0, 5.0, 11.0]
        X = _weighted_space(numpy_ctx, w)
        f = sc.SquaredL2NormFunctional(X)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        expected = 0.5 * np.sum(np.asarray(w) * to_numpy(x) ** 2)
        np.testing.assert_allclose(to_numpy(f.value(x)), expected)

    def test_weighted_gradient_still_identity(self, numpy_ctx):
        # grad of 1/2 <x,x>_X is x in any metric.
        X = _weighted_space(numpy_ctx, [2.0, 5.0, 11.0])
        f = sc.SquaredL2NormFunctional(X)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        np.testing.assert_allclose(to_numpy(f.grad(x)), to_numpy(x))


# ---------------------------------------------------------------------------
# LpNormFunctional / L1NormFunctional
# ---------------------------------------------------------------------------
class TestLpNormFunctional:
    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
    def test_value_matches_numpy(self, numpy_ctx, p):
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        f = sc.LpNormFunctional(X, p)
        x = numpy_ctx.asarray([1.0, -2.0, 0.5, 3.0])
        expected = np.sum(np.abs(to_numpy(x)) ** p) ** (1.0 / p)
        np.testing.assert_allclose(to_numpy(f.value(x)), expected)

    def test_l1_gradient_is_sign(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.L1NormFunctional(X)
        x = numpy_ctx.asarray([2.0, -3.0, 0.0])
        np.testing.assert_allclose(to_numpy(f.grad(x)), [1.0, -1.0, 0.0])

    def test_l1_is_lp_with_p_one(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.L1NormFunctional(X)
        assert isinstance(f, sc.LpNormFunctional)
        assert f.p == 1.0

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_gradient_zero_at_origin(self, numpy_ctx, p):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.LpNormFunctional(X, p)
        g = to_numpy(f.grad(numpy_ctx.asarray([0.0, 0.0, 0.0])))
        assert np.all(np.isfinite(g))
        np.testing.assert_allclose(g, 0.0)

    @pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
    def test_gradient_matches_finite_difference(self, numpy_ctx, p):
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        f = sc.LpNormFunctional(X, p)
        x = numpy_ctx.asarray([1.0, -2.0, 0.5, 3.0])
        h = numpy_ctx.asarray([0.3, -0.7, 1.1, -0.2])
        lhs = float(numpy_ctx.ops.real(X.inner(f.grad(x), h)))
        np.testing.assert_allclose(lhs, _fd_directional(f, x, h), rtol=1e-5)

    def test_rejects_p_below_one(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(ValueError):
            sc.LpNormFunctional(X, 0.5)

    def test_rejects_non_finite_p(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(ValueError):
            sc.LpNormFunctional(X, float("inf"))


# ---------------------------------------------------------------------------
# NegativeEntropyFunctional
# ---------------------------------------------------------------------------
class TestNegativeEntropyFunctional:
    def test_value_matches_numpy(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.NegativeEntropyFunctional(X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        expected = np.sum(to_numpy(x) * np.log(to_numpy(x)))
        np.testing.assert_allclose(to_numpy(f.value(x)), expected)

    def test_zero_coordinate_contributes_zero(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.NegativeEntropyFunctional(X)
        x = numpy_ctx.asarray([0.0, 1.0, 1.0])
        np.testing.assert_allclose(to_numpy(f.value(x)), 0.0)

    def test_gradient_matches_analytic(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.NegativeEntropyFunctional(X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        np.testing.assert_allclose(to_numpy(f.grad(x)), np.log(to_numpy(x)) + 1.0)

    def test_weighted_gradient_is_riesz_corrected(self, numpy_ctx):
        w = [2.0, 5.0, 11.0]
        X = _weighted_space(numpy_ctx, w)
        f = sc.NegativeEntropyFunctional(X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        expected = (np.log(to_numpy(x)) + 1.0) / np.asarray(w)
        np.testing.assert_allclose(to_numpy(f.grad(x)), expected)

    def test_weighted_gradient_satisfies_riesz_identity(self, numpy_ctx):
        X = _weighted_space(numpy_ctx, [2.0, 5.0, 11.0])
        f = sc.NegativeEntropyFunctional(X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        h = numpy_ctx.asarray([0.3, -0.7, 1.1])
        lhs = float(numpy_ctx.ops.real(X.inner(f.grad(x), h)))
        np.testing.assert_allclose(lhs, _fd_directional(f, x, h), rtol=1e-5)


# ---------------------------------------------------------------------------
# KLDivergenceFunctional
# ---------------------------------------------------------------------------
class TestKLDivergenceFunctional:
    def test_value_matches_numpy(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        target = numpy_ctx.asarray([1.0, 2.0, 0.5])
        f = sc.KLDivergenceFunctional(target, X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        expected = np.sum(to_numpy(x) * np.log(to_numpy(x) / to_numpy(target)))
        np.testing.assert_allclose(to_numpy(f.value(x)), expected)

    def test_reduces_to_negative_entropy_for_unit_target(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        target = numpy_ctx.asarray([1.0, 1.0, 1.0])
        kl = sc.KLDivergenceFunctional(target, X)
        ne = sc.NegativeEntropyFunctional(X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        np.testing.assert_allclose(to_numpy(kl.value(x)), to_numpy(ne.value(x)))
        np.testing.assert_allclose(to_numpy(kl.grad(x)), to_numpy(ne.grad(x)))

    def test_gradient_matches_analytic(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        target = numpy_ctx.asarray([1.0, 2.0, 0.5])
        f = sc.KLDivergenceFunctional(target, X)
        x = numpy_ctx.asarray([0.5, 1.0, 2.0])
        expected = np.log(to_numpy(x) / to_numpy(target)) + 1.0
        np.testing.assert_allclose(to_numpy(f.grad(x)), expected)

    def test_target_property(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        target = numpy_ctx.asarray([1.0, 2.0, 0.5])
        f = sc.KLDivergenceFunctional(target, X)
        np.testing.assert_allclose(to_numpy(f.target), to_numpy(target))


# ---------------------------------------------------------------------------
# HuberFunctional
# ---------------------------------------------------------------------------
class TestHuberFunctional:
    def test_value_matches_numpy(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        delta = 1.0
        f = sc.HuberFunctional(X, delta)
        x = numpy_ctx.asarray([0.5, -0.5, 2.0, -3.0])
        a = np.abs(to_numpy(x))
        expected = np.sum(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta)))
        np.testing.assert_allclose(to_numpy(f.value(x)), expected)

    def test_gradient_matches_analytic(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        delta = 1.0
        f = sc.HuberFunctional(X, delta)
        x = numpy_ctx.asarray([0.5, -0.5, 2.0, -3.0])
        a = np.abs(to_numpy(x))
        expected = np.where(a <= delta, to_numpy(x), delta * np.sign(to_numpy(x)))
        np.testing.assert_allclose(to_numpy(f.grad(x)), expected)

    def test_gradient_matches_finite_difference(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        f = sc.HuberFunctional(X, 1.0)
        x = numpy_ctx.asarray([0.4, -0.6, 2.0, -3.0])
        h = numpy_ctx.asarray([0.3, -0.7, 1.1, -0.2])
        lhs = float(numpy_ctx.ops.real(X.inner(f.grad(x), h)))
        np.testing.assert_allclose(lhs, _fd_directional(f, x, h), rtol=1e-5)

    def test_rejects_nonpositive_delta(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(ValueError):
            sc.HuberFunctional(X, 0.0)


# ---------------------------------------------------------------------------
# Batched evaluation and pytree/convert round-trips
# ---------------------------------------------------------------------------
class TestBatchingAndConversion:
    @pytest.mark.parametrize(
        "make",
        [
            lambda X: sc.SquaredL2NormFunctional(X),
            lambda X: sc.LpNormFunctional(X, 2.0),
            lambda X: sc.HuberFunctional(X, 1.0),
        ],
    )
    def test_vvalue_and_vgrad_match_loop(self, numpy_ctx, make):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = make(X)
        xs = numpy_ctx.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        values = to_numpy(f.vvalue(xs))
        grads = to_numpy(f.vgrad(xs))
        for i in range(2):
            xi = numpy_ctx.asarray(to_numpy(xs)[i])
            np.testing.assert_allclose(values[i], to_numpy(f.value(xi)))
            np.testing.assert_allclose(grads[i], to_numpy(f.grad(xi)))

    def test_convert_changes_dtype(self, numpy_ctx, numpy_f32_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.LpNormFunctional(X, 2.0)
        g = f.convert(numpy_f32_ctx)
        assert g.ctx == numpy_f32_ctx
        assert g.p == 2.0

    def test_kl_convert_preserves_target(self, numpy_ctx, numpy_f32_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        target = numpy_ctx.asarray([1.0, 2.0, 0.5])
        f = sc.KLDivergenceFunctional(target, X)
        g = f.convert(numpy_f32_ctx)
        np.testing.assert_allclose(to_numpy(g.target), to_numpy(target), rtol=1e-6)

    def test_domain_and_field(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        f = sc.HuberFunctional(X, 1.0)
        assert f.domain == X
        # __call__ aliases value
        x = numpy_ctx.asarray([0.5, -0.5, 2.0])
        np.testing.assert_allclose(to_numpy(f(x)), to_numpy(f.value(x)))

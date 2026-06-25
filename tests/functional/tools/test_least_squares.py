"""Tests for the ADR-019 :func:`least_squares` constructor."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


def _problem(ctx, m=4, n=3, seed=0):
    rng = np.random.default_rng(seed)
    Amat = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    X = sc.DenseCoordinateSpace((n,), ctx)
    Y = sc.DenseCoordinateSpace((m,), ctx)
    A = sc.DenseLinOp(ctx.asarray(Amat), X, Y, ctx)
    return A, ctx.asarray(b), Amat, b, X, Y


class TestLeastSquaresValueAndType:
    def test_returns_linop_quadratic_form(self, numpy_ctx):
        A, b, *_ = _problem(numpy_ctx)
        f = sc.least_squares(A, b)
        assert isinstance(f, sc.LinOpQuadraticForm)

    def test_value_equals_half_squared_residual(self, numpy_ctx):
        A, b, Amat, bn, X, Y = _problem(numpy_ctx)
        f = sc.least_squares(A, b)
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        r = Amat @ to_numpy(x) - bn
        np.testing.assert_allclose(to_numpy(f.value(x)), 0.5 * r @ r)

    def test_scale_one_drops_the_half(self, numpy_ctx):
        A, b, Amat, bn, X, Y = _problem(numpy_ctx)
        f = sc.least_squares(A, b, scale=1.0)
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        r = Amat @ to_numpy(x) - bn
        np.testing.assert_allclose(to_numpy(f.value(x)), r @ r)

    def test_gradient_is_normal_equation_residual(self, numpy_ctx):
        A, b, Amat, bn, X, Y = _problem(numpy_ctx)
        f = sc.least_squares(A, b)
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        expected = Amat.T @ (Amat @ to_numpy(x) - bn)
        np.testing.assert_allclose(to_numpy(f.grad(x)), expected)

    def test_minimizer_solves_normal_equations(self, numpy_ctx):
        A, b, Amat, bn, X, Y = _problem(numpy_ctx)
        f = sc.least_squares(A, b)
        x_star = np.linalg.lstsq(Amat, bn, rcond=None)[0]
        g = to_numpy(f.grad(numpy_ctx.asarray(x_star)))
        np.testing.assert_allclose(g, 0.0, atol=1e-10)


class TestWeightedLeastSquares:
    def test_value_uses_residual_weights(self, numpy_ctx):
        A, b, Amat, bn, X, Y = _problem(numpy_ctx)
        weights = numpy_ctx.asarray([1.0, 2.0, 3.0, 4.0])
        f = sc.least_squares(A, b, weights=weights)
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        r = Amat @ to_numpy(x) - bn
        expected = 0.5 * np.sum(to_numpy(weights) * r * r)
        np.testing.assert_allclose(to_numpy(f.value(x)), expected)

    def test_gradient_uses_weights(self, numpy_ctx):
        A, b, Amat, bn, X, Y = _problem(numpy_ctx)
        weights = numpy_ctx.asarray([1.0, 2.0, 3.0, 4.0])
        f = sc.least_squares(A, b, weights=weights)
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        W = np.diag(to_numpy(weights))
        expected = Amat.T @ W @ (Amat @ to_numpy(x) - bn)
        np.testing.assert_allclose(to_numpy(f.grad(x)), expected)

    def test_rejects_wrong_weight_shape(self, numpy_ctx):
        A, b, *_ = _problem(numpy_ctx)
        with pytest.raises(ValueError):
            sc.least_squares(A, b, weights=numpy_ctx.asarray([1.0, 2.0]))

    def test_rejects_nonpositive_weights(self, numpy_ctx):
        A, b, *_ = _problem(numpy_ctx)
        with pytest.raises(ValueError):
            sc.least_squares(A, b, weights=numpy_ctx.asarray([1.0, 0.0, 3.0, 4.0]))


class TestLeastSquaresMetric:
    def test_weighted_domain_gradient_is_metric_correct(self, numpy_ctx):
        # Non-Euclidean *domain*: the gradient must be the Riesz gradient.
        rng = np.random.default_rng(2)
        Amat = rng.standard_normal((4, 3))
        bn = rng.standard_normal(4)
        w = np.array([2.0, 5.0, 11.0])
        X = sc.DenseCoordinateSpace(
            (3,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray(w))
        )
        Y = sc.DenseCoordinateSpace((4,), numpy_ctx)
        A = sc.DenseLinOp(numpy_ctx.asarray(Amat), X, Y, numpy_ctx)
        f = sc.least_squares(A, numpy_ctx.asarray(bn))
        x = numpy_ctx.asarray([0.5, -1.0, 2.0])
        # Riesz gradient = G^{-1} * (Euclidean coordinate gradient).
        euclid = Amat.T @ (Amat @ to_numpy(x) - bn)
        np.testing.assert_allclose(to_numpy(f.grad(x)), euclid / w)

    def test_rejects_non_linop(self, numpy_ctx):
        with pytest.raises(TypeError):
            sc.least_squares(object(), numpy_ctx.asarray([1.0]))

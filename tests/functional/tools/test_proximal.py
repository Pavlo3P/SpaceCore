"""Tests for the ADR-019 proximal / projection primitive.

The centerpiece is the S04 regression (``TestMetricTrap``): on a weighted metric
a proximal step must threshold by ``lam / (2 eps w_i)``. A proximal-gradient
iteration using the metric-correct prox converges to the true minimizer, while
the Euclidean-threshold prox converges to a different, worse point -- exactly
the silent correctness trap ADR-019 was written to defuse.
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


def _soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


class _FullMetric(sc.InnerProduct):
    """A genuinely non-diagonal metric, for the raise-path tests."""

    def __init__(self, matrix):
        self.matrix = matrix

    def inner(self, ops, x, y):
        return ops.vdot(x, self.matrix @ y)

    def riesz(self, ops, x):
        return self.matrix @ x

    def riesz_inverse(self, ops, x):
        return np.linalg.solve(np.asarray(self.matrix), np.asarray(x))

    def validate_for(self, space):
        return None


# ---------------------------------------------------------------------------
# generalized_shrinkage closed form
# ---------------------------------------------------------------------------
class TestGeneralizedShrinkage:
    def test_data_step_weight_cancels(self, numpy_ctx):
        # With lam = 0 the minimizer is x0 - c / (2 eps), independent of weights.
        X = _weighted_space(numpy_ctx, [2.0, 5.0, 11.0])
        c = numpy_ctx.asarray([1.0, 2.0, 3.0])
        x0 = numpy_ctx.asarray([0.5, -1.0, 4.0])
        out = sc.generalized_shrinkage(X, c=c, x0=x0, eps=0.5, lam=0.0)
        expected = to_numpy(x0) - to_numpy(c) / (2 * 0.5)
        np.testing.assert_allclose(to_numpy(out), expected)

    def test_euclidean_soft_threshold(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x0 = numpy_ctx.asarray([-3.0, 0.4, 2.0])
        c = X.zeros()
        out = sc.generalized_shrinkage(X, c=c, x0=x0, eps=0.5, lam=1.0)
        # tau = lam / (2 eps) = 1.0
        np.testing.assert_allclose(to_numpy(out), _soft_threshold(to_numpy(x0), 1.0))

    def test_weighted_threshold_is_lam_over_2_eps_w(self, numpy_ctx):
        w = np.array([2.0, 5.0, 11.0])
        X = _weighted_space(numpy_ctx, w)
        x0 = numpy_ctx.asarray([-3.0, 0.4, 2.0])
        c = X.zeros()
        eps, lam = 0.5, 1.0
        out = sc.generalized_shrinkage(X, c=c, x0=x0, eps=eps, lam=lam)
        tau = lam / (2 * eps * w)
        np.testing.assert_allclose(to_numpy(out), _soft_threshold(to_numpy(x0), tau))

    def test_nonneg_projection(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x0 = numpy_ctx.asarray([-3.0, 0.4, 2.0])
        c = X.zeros()
        out = sc.generalized_shrinkage(X, c=c, x0=x0, eps=0.5, lam=0.0, nonneg=True)
        np.testing.assert_allclose(to_numpy(out), np.maximum(to_numpy(x0), 0.0))

    def test_nonneg_with_l1(self, numpy_ctx):
        w = np.array([2.0, 5.0, 11.0])
        X = _weighted_space(numpy_ctx, w)
        x0 = numpy_ctx.asarray([3.0, 0.05, -2.0])
        c = X.zeros()
        eps, lam = 0.5, 1.0
        out = sc.generalized_shrinkage(X, c=c, x0=x0, eps=eps, lam=lam, nonneg=True)
        tau = lam / (2 * eps * w)
        np.testing.assert_allclose(to_numpy(out), np.maximum(to_numpy(x0) - tau, 0.0))

    def test_rejects_nonpositive_eps(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(ValueError):
            sc.generalized_shrinkage(
                X, c=X.zeros(), x0=X.zeros(), eps=0.0, lam=1.0
            )

    def test_rejects_negative_lam(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(ValueError):
            sc.generalized_shrinkage(
                X, c=X.zeros(), x0=X.zeros(), eps=0.5, lam=-1.0
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_eps_and_lam(self, numpy_ctx, bad):
        # A NaN/inf weight must raise, not silently propagate NaN through the result.
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(ValueError):
            sc.generalized_shrinkage(X, c=X.zeros(), x0=X.zeros(), eps=bad, lam=1.0)
        with pytest.raises(ValueError):
            sc.generalized_shrinkage(X, c=X.zeros(), x0=X.zeros(), eps=0.5, lam=bad)

    def test_rejects_complex_nonneg(self, numpy_complex_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_complex_ctx)
        with pytest.raises(ValueError):
            sc.generalized_shrinkage(
                X, c=X.zeros(), x0=X.zeros(), eps=0.5, lam=0.0, nonneg=True
            )


# ---------------------------------------------------------------------------
# Non-diagonal metric must raise (ADR-019 / ADR-020 diagonal-metric rule)
# ---------------------------------------------------------------------------
class TestNonDiagonalRaises:
    def _full_metric_space(self, ctx):
        matrix = ctx.asarray(
            [[2.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 2.0]]
        )
        return sc.DenseCoordinateSpace((3,), ctx, geometry=_FullMetric(matrix))

    def test_generalized_shrinkage_raises(self, numpy_ctx):
        X = self._full_metric_space(numpy_ctx)
        with pytest.raises(ValueError, match="not diagonal"):
            sc.generalized_shrinkage(X, c=X.zeros(), x0=X.zeros(), eps=0.5, lam=0.1)

    def test_prox_l1_raises(self, numpy_ctx):
        X = self._full_metric_space(numpy_ctx)
        with pytest.raises(ValueError, match="not diagonal"):
            sc.prox_l1(numpy_ctx.asarray([1.0, 2.0, 3.0]), 0.5, X)

    def test_project_nonneg_raises(self, numpy_ctx):
        X = self._full_metric_space(numpy_ctx)
        with pytest.raises(ValueError, match="not diagonal"):
            sc.project_nonneg(numpy_ctx.asarray([1.0, 2.0, 3.0]), X)


# ---------------------------------------------------------------------------
# Named wrappers
# ---------------------------------------------------------------------------
class TestWrappers:
    def test_prox_l1_euclidean(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        v = numpy_ctx.asarray([-3.0, 0.5, 2.0])
        out = sc.prox_l1(v, 1.0, X)
        np.testing.assert_allclose(to_numpy(out), _soft_threshold(to_numpy(v), 1.0))

    def test_prox_l1_weighted_threshold(self, numpy_ctx):
        w = np.array([2.0, 5.0, 11.0])
        X = _weighted_space(numpy_ctx, w)
        v = numpy_ctx.asarray([-3.0, 0.5, 2.0])
        out = sc.prox_l1(v, 1.0, X)
        np.testing.assert_allclose(to_numpy(out), _soft_threshold(to_numpy(v), 1.0 / w))

    def test_prox_l1_is_actual_prox(self, numpy_ctx):
        # prox_l1(v, t) minimizes 1/2 ||x - v||^2_X + t ||x||_1.
        w = np.array([2.0, 5.0, 11.0])
        X = _weighted_space(numpy_ctx, w)
        v = numpy_ctx.asarray([-3.0, 0.5, 2.0])
        t = 1.3
        out = to_numpy(sc.prox_l1(v, t, X))

        def objective(x):
            diff = x - to_numpy(v)
            return 0.5 * np.sum(w * diff * diff) + t * np.sum(np.abs(x))

        base = objective(out)
        rng = np.random.default_rng(0)
        for _ in range(200):
            perturbed = out + 1e-3 * rng.standard_normal(3)
            assert objective(perturbed) >= base - 1e-9

    @pytest.mark.parametrize("t", [0.0, 0.5, 3.0])
    def test_prox_l2sq_is_shrinkage(self, numpy_ctx, t):
        X = _weighted_space(numpy_ctx, [2.0, 5.0, 11.0])
        v = numpy_ctx.asarray([-3.0, 0.5, 2.0])
        out = sc.prox_l2sq(v, t, X)
        np.testing.assert_allclose(to_numpy(out), to_numpy(v) / (1.0 + t))

    def test_project_nonneg(self, numpy_ctx):
        X = _weighted_space(numpy_ctx, [2.0, 5.0, 11.0])
        v = numpy_ctx.asarray([-3.0, 0.5, 2.0])
        out = sc.project_nonneg(v, X)
        np.testing.assert_allclose(to_numpy(out), np.maximum(to_numpy(v), 0.0))

    def test_wrappers_reject_negative_step(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        v = numpy_ctx.asarray([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            sc.prox_l1(v, -1.0, X)
        with pytest.raises(ValueError):
            sc.prox_l2sq(v, -1.0, X)

    def test_wrappers_reject_non_finite_step(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        v = numpy_ctx.asarray([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            sc.prox_l1(v, float("nan"), X)

    def test_wrong_shape_v_names_the_caller_argument(self, numpy_ctx):
        # The diagnostic should mention ``v`` (the wrapper argument), not an
        # internal ``c``/``x0`` forwarded into the primitive.
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        bad = numpy_ctx.asarray([1.0, 2.0])
        with pytest.raises(ValueError, match="prox_l2sq v"):
            sc.prox_l2sq(bad, 1.0, X)
        with pytest.raises(ValueError, match="prox_l1 v"):
            sc.prox_l1(bad, 1.0, X)


# ---------------------------------------------------------------------------
# S04 regression: metric-correct proximal gradient on a weighted space
# ---------------------------------------------------------------------------
class TestMetricTrap:
    def _setup(self, ctx):
        w = np.array([2.0, 5.0, 11.0])
        X = _weighted_space(ctx, w)
        y = ctx.asarray([3.0, -1.5, 0.2])
        lam = 1.0
        # On the weighted space the LASSO  min 1/2||x - y||^2_X + lam||x||_1
        # has the per-coordinate minimizer soft(y_i, lam / w_i).
        x_true = _soft_threshold(to_numpy(y), lam / w)
        return X, y, lam, w, x_true

    def test_ista_with_metric_prox_converges_to_true_optimum(self, numpy_ctx):
        X, y, lam, w, x_true = self._setup(numpy_ctx)
        # Riesz gradient of 1/2 ||x - y||^2_X is (x - y), independent of weights.
        x = X.zeros()
        alpha = 0.5
        for _ in range(500):
            grad = numpy_ctx.asarray(to_numpy(x) - to_numpy(y))
            v = numpy_ctx.asarray(to_numpy(x) - alpha * to_numpy(grad))
            x = sc.prox_l1(v, alpha * lam, X)
        np.testing.assert_allclose(to_numpy(x), x_true, atol=1e-8)

    def test_euclidean_threshold_converges_to_wrong_point(self, numpy_ctx):
        # The trap: using a Euclidean threshold (alpha*lam, not alpha*lam/w_i)
        # in a metric-gradient iteration converges to a different point with a
        # strictly worse objective.
        X, y, lam, w, x_true = self._setup(numpy_ctx)

        def objective(x):
            diff = x - to_numpy(y)
            return 0.5 * np.sum(w * diff * diff) + lam * np.sum(np.abs(x))

        x = np.zeros(3)
        alpha = 0.5
        for _ in range(500):
            grad = x - to_numpy(y)
            v = x - alpha * grad
            x = _soft_threshold(v, alpha * lam)  # WRONG: ignores the metric
        wrong = x
        assert not np.allclose(wrong, x_true, atol=1e-3)
        assert objective(wrong) > objective(x_true) + 1e-6

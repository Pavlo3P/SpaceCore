"""The central ADR-018 correctness claim: the metric -> coordinate handoff.

A SpaceCore ``F.grad`` is a metric (Riesz) gradient; an external optimizer wants
a coordinate gradient. The adapters convert with ``X.riesz(F.grad(x))``. On a
weighted space the two gradients genuinely differ, and only the riesz-converted
one matches the finite-difference gradient of the flat objective the optimizer
sees. These tests pin both halves of that claim.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize as sopt

import spacecore as sc
from spacecore.optimize._common import coordinate_gradient

from tests.linalg._helpers import make_ctx
from tests.optimize._helpers import flat_fun, weighted_problem


@pytest.fixture
def ctx():
    return make_ctx("numpy", np.float64)


def test_riesz_gradient_matches_finite_difference_on_weighted_space(ctx):
    """``X.riesz(F.grad(x))`` equals the finite-difference gradient of ``fun``."""
    X, F, _ = weighted_problem(ctx)
    x = ctx.asarray([0.5, -2.0, 1.5])

    x_flat = np.asarray(X.flatten(x))
    fd = sopt.approx_fprime(x_flat, flat_fun(F, X), 1e-6)
    coord_grad = np.asarray(X.flatten(X.riesz(F.grad(x))))

    np.testing.assert_allclose(coord_grad, fd, rtol=1e-4, atol=1e-4)


def test_raw_metric_gradient_is_the_trap_on_weighted_space(ctx):
    """The un-converted metric gradient does *not* match ``fun``'s gradient."""
    X, F, _ = weighted_problem(ctx)
    x = ctx.asarray([0.5, -2.0, 1.5])

    x_flat = np.asarray(X.flatten(x))
    fd = sopt.approx_fprime(x_flat, flat_fun(F, X), 1e-6)
    metric_grad = np.asarray(X.flatten(F.grad(x)))

    # Sanity: on a non-trivial metric they must differ, else the test proves
    # nothing. They are related by the diagonal weights.
    assert not np.allclose(metric_grad, fd, atol=1e-3)
    np.testing.assert_allclose(metric_grad * np.array([2.0, 5.0, 11.0]), fd, rtol=1e-4, atol=1e-4)


def test_euclidean_riesz_is_identity(ctx):
    """On a Euclidean space the handoff is the identity, not a branch."""
    X = sc.DenseCoordinateSpace((4,), ctx)
    F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
    x = ctx.asarray([1.0, -2.0, 3.0, 0.5])

    np.testing.assert_array_equal(
        np.asarray(coordinate_gradient(F, X, x)),
        np.asarray(F.grad(x)),
    )

"""Tests for :func:`spacecore.line_search_scipy` (ADR-018).

The line search must return a step that decreases the objective, with the slope
computed from the coordinate gradient so that it is correct even on a weighted
space. ``d`` is a coordinate displacement and is not transformed by the adapter.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy
from tests.linalg._helpers import make_ctx
from tests.optimize._helpers import euclidean_problem, flat_fun, weighted_problem


@pytest.fixture
def ctx():
    return make_ctx("numpy", np.float64)


def test_returns_scipy_six_tuple(ctx):
    X, F, _ = euclidean_problem(ctx)
    x = X.zeros()
    d = X.scale(-1.0, F.grad(x))

    result = sc.line_search_scipy(F, x, d)

    assert isinstance(result, tuple) and len(result) == 6


def test_descent_step_decreases_objective_euclidean(ctx):
    X, F, _ = euclidean_problem(ctx)
    x = ctx.asarray([0.0, 0.0])
    d = X.scale(-1.0, F.grad(x))
    fun = flat_fun(F, X)

    alpha = sc.line_search_scipy(F, x, d)[0]

    assert alpha is not None and alpha > 0.0
    x_new = X.add(x, X.scale(alpha, d))
    assert fun(to_numpy(X.flatten(x_new))) < fun(to_numpy(X.flatten(x)))


def test_descent_step_decreases_objective_weighted(ctx):
    """On a weighted space the riesz slope keeps the Wolfe test correct."""
    X, F, _ = weighted_problem(ctx)
    x = ctx.asarray([0.5, -2.0, 1.5])
    # natural (metric) steepest descent: d = -F.grad(x)
    d = X.scale(-1.0, F.grad(x))
    fun = flat_fun(F, X)

    alpha = sc.line_search_scipy(F, x, d)[0]

    assert alpha is not None and alpha > 0.0
    x_new = X.add(x, X.scale(alpha, d))
    assert fun(to_numpy(X.flatten(x_new))) < fun(to_numpy(X.flatten(x)))


def test_reported_values_match_the_objective(ctx):
    """The adapter wires ``F.value`` to SciPy: old/new fvals match the objective.

    (SciPy's 6th tuple element is implementation-defined -- a scalar slope in the
    Wolfe2 path, the gradient vector in the Wolfe1 path -- so this asserts the
    adapter's own contract rather than that internal field.)
    """
    X, F, _ = weighted_problem(ctx)
    x = ctx.asarray([0.5, -2.0, 1.5])
    d = X.scale(-1.0, F.grad(x))

    alpha, _fc, _gc, new_fval, old_fval, _new_slope = sc.line_search_scipy(F, x, d)

    np.testing.assert_allclose(old_fval, float(F.value(x)), rtol=1e-10)
    x_new = X.add(x, X.scale(alpha, d))
    np.testing.assert_allclose(new_fval, float(F.value(x_new)), rtol=1e-10)


@pytest.mark.filterwarnings("ignore:The line search algorithm did not converge")
def test_ascent_direction_returns_no_step(ctx):
    """An ascent direction fails the Wolfe conditions; alpha is None."""
    X, F, _ = euclidean_problem(ctx)
    x = ctx.asarray([0.0, 0.0])
    d = F.grad(x)  # +gradient is an ascent direction

    alpha = sc.line_search_scipy(F, x, d)[0]

    assert alpha is None

"""Shared fixtures for the ``spacecore.optimize`` adapter tests (ADR-018)."""
from __future__ import annotations

import importlib.util

import numpy as np

import spacecore as sc

# Symmetric positive-definite Hessian used by the weighted-metric problems.
# ``S = B / w[:, None]`` is self-adjoint in the ``diag(w)`` metric (M S = B is
# symmetric), so ``LinOpQuadraticForm`` accepts it; the same construction the cg
# tests use for metric-aware operators.
B_SPD = np.array([[6.0, 1.0, 0.5], [1.0, 5.0, -0.25], [0.5, -0.25, 4.0]])
WEIGHTS = np.array([2.0, 5.0, 11.0])
LINEAR = np.array([0.3, -1.1, 0.7])


def has_optax() -> bool:
    return importlib.util.find_spec("optax") is not None


def euclidean_problem(ctx):
    """Return ``(X, F, x_star)`` for ``f(x)=1/2 x^T diag(3,1) x - 3 x0 - 2 x1``."""
    X = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.DenseLinOp(ctx.asarray([[3.0, 0.0], [0.0, 1.0]]), X, X, ctx)
    linear = sc.InnerProductFunctional(ctx.asarray([-3.0, -2.0]), X)
    F = sc.LinOpQuadraticForm(Q, linear)
    return X, F, np.array([1.0, 2.0])


def weighted_problem(ctx):
    """Return ``(X, F, x_star)`` on a weighted space with a non-trivial metric.

    ``f(x) = 1/2 <x, Qx>_X + <c, x>_X`` reduces, in flat coordinates, to
    ``1/2 x^T B x + (M c)^T x`` so the true minimizer is ``-B^{-1} (w * c)``.
    """
    w = WEIGHTS
    X = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(w)))
    S = B_SPD / w[:, None]
    Q = sc.DenseLinOp(ctx.asarray(S), X, X, ctx)
    F = sc.LinOpQuadraticForm(Q, sc.InnerProductFunctional(ctx.asarray(LINEAR), X))
    x_star = -np.linalg.solve(B_SPD, w * LINEAR)
    return X, F, x_star


def flat_fun(F, X):
    """Return a flat-coordinate objective ``fun(v) = F.value(unflatten(v))``."""
    ctx = X.ctx

    def fun(v):
        return float(np.real(F.value(X.unflatten(ctx.asarray(np.asarray(v))))))

    return fun

"""
Closed-form proximal steps in the space metric (ADR-019).

A prox must be taken in the metric that the gradient uses: a Euclidean
soft-threshold paired with a metric gradient converges to the wrong point. Only
Euclidean or diagonal metrics separate; other metrics raise. See Beck,
*First-Order Methods in Optimization*, 2017, chapter 6.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, cast

from ...space import WeightedInnerProduct


def _diagonal_weights_or_raise(X: Any, fname: str) -> Any:
    """Return diagonal weights (``None`` for Euclidean) or raise on a non-diagonal metric."""
    # Euclidean gives None, diagonal gives its weights, anything else cannot separate.
    geometry = getattr(X, "geometry", None)
    if geometry is None:
        raise TypeError(
            f"{fname} requires an inner-product space with a geometry; "
            f"got {type(X).__name__}."
        )
    if geometry.is_euclidean:
        return None
    if isinstance(geometry, WeightedInnerProduct):
        return geometry.weights
    raise ValueError(
        f"{fname} is a separable closed form valid only for Euclidean or diagonal "
        f"(weighted) metrics; the metric {type(geometry).__name__} is not diagonal, "
        f"so the subproblem does not separate. Refusing to return a wrong "
        f"(separable) answer."
    )


def _require_member_shape(X: Any, arr: Any, name: str, fname: str) -> None:
    """Raise when ``arr`` does not have the coordinate shape of ``X``."""
    shape = tuple(getattr(arr, "shape", ()))
    if shape != tuple(X.shape):
        raise ValueError(
            f"{fname} {name} must have the space shape {tuple(X.shape)}, got {shape}."
        )


def generalized_shrinkage(
    X: Any,
    *,
    c: Any,
    x0: Any,
    eps: float,
    lam: float = 0.0,
    nonneg: bool = False,
) -> Any:
    r"""
    Solve ``min_x <c, x>_X + eps ||x - x0||_X^2 + lam ||x||_1`` coordinatewise, optionally with ``x >= 0``.

    The smooth minimizer is ``v = x0 - c / (2 eps)``; the l1 term soft-thresholds it
    with width ``lam / (2 eps w_i)`` for diagonal metric weights ``w``.

    Parameters
    ----------
    X : Space
        Euclidean or diagonal-metric space; other metrics raise.
    c : array-like
        Metric gradient in ``X``.
    x0 : array-like
        Proximal center in ``X``.
    eps : float
        Positive quadratic weight.
    lam : float, optional
        Non-negative l1 weight; ``0`` disables thresholding.
    nonneg : bool, optional
        Constrain to ``x >= 0`` (real spaces only).

    Returns
    -------
    Element
        The minimizer.
    """
    # Validate eps, lam, the metric, the shapes, and nonneg on real spaces only.
    if not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"generalized_shrinkage requires a finite eps > 0, got {eps!r}.")
    if not isinstance(lam, numbers.Real) or not math.isfinite(lam) or lam < 0:
        raise ValueError(f"generalized_shrinkage requires a finite lam >= 0, got {lam!r}.")

    weights = _diagonal_weights_or_raise(X, "generalized_shrinkage")

    ops = X.ops
    c = X.ctx.asarray(c)
    x0 = X.ctx.asarray(x0)
    _require_member_shape(X, c, "c", "generalized_shrinkage")
    _require_member_shape(X, x0, "x0", "generalized_shrinkage")
    if nonneg and ops.is_complex_dtype(ops.get_dtype(x0)):
        raise ValueError("generalized_shrinkage(nonneg=True) is defined for real spaces only.")

    # Data step: minimizer of <c, x>_X + eps ||x - x0||^2_X. The metric weight
    # cancels here precisely because c is a metric gradient.
    v = x0 - c / (2.0 * eps)

    # Threshold width lam / (2 eps w_i), with w_i = 1 on Euclidean spaces.
    if lam == 0:
        tau: Any = 0.0
    elif weights is None:
        tau = lam / (2.0 * eps)
    else:
        tau = lam / (2.0 * eps * weights)

    # On x >= 0 the l1 term is linear, so the step is a one-sided shift.
    if nonneg:
        return ops.maximum(v - tau, cast(Any, 0.0))
    # Soft-threshold; tau == 0 collapses to the identity v.
    return ops.sign(v) * ops.maximum(ops.abs(v) - tau, cast(Any, 0.0))


def _require_step(t: float, fname: str) -> float:
    """Validate and return a finite, nonnegative scalar proximal step ``t``."""
    if not isinstance(t, numbers.Real) or not math.isfinite(t) or t < 0:
        raise ValueError(f"{fname} requires a finite step t >= 0, got {t!r}.")
    return float(t)


def prox_l1(v: Any, t: Any, X: Any, *, nonneg: bool = False) -> Any:
    r"""
    Prox of ``t ||.||_1`` in the space metric: soft-threshold with width ``t / w_i``.

    Parameters
    ----------
    v : array-like
        Point in ``X``.
    t : float
        Non-negative threshold.
    X : Space
        Euclidean or diagonal-metric space.
    nonneg : bool, optional
        Constrain to ``x >= 0``, giving ``max(v_i - t / w_i, 0)``; at ``t = 0`` this is
        :func:`project_nonneg`.

    Returns
    -------
    Element
        The thresholded point.
    """
    # prox of t ||.||_1 = shrinkage with c = 0, x0 = v, eps = 1/2.
    t = _require_step(t, "prox_l1")
    v = X.ctx.asarray(v)
    _require_member_shape(X, v, "v", "prox_l1")
    return generalized_shrinkage(X, c=X.zeros(), x0=v, eps=0.5, lam=t, nonneg=nonneg)


def prox_l2sq(v: Any, t: Any, X: Any, *, nonneg: bool = False) -> Any:
    r"""
    Prox of ``(t / 2) ||.||_X^2``: the shrinkage ``v / (1 + t)``.

    Parameters
    ----------
    v : array-like
        Point in ``X``.
    t : float
        Non-negative step size.
    X : Space
        Euclidean or diagonal-metric space.
    nonneg : bool, optional
        Constrain to ``x >= 0``, giving ``max(v, 0) / (1 + t)``.

    Returns
    -------
    Element
        The shrunk point.
    """
    # prox of (t/2) ||.||^2 = shrinkage with c = -v, x0 = 0, eps = (1 + t) / 2.
    t = _require_step(t, "prox_l2sq")
    v = X.ctx.asarray(v)
    _require_member_shape(X, v, "v", "prox_l2sq")
    return generalized_shrinkage(
        X, c=-v, x0=X.zeros(), eps=0.5 * (1.0 + t), lam=0.0, nonneg=nonneg
    )


def project_nonneg(v: Any, X: Any) -> Any:
    r"""
    Project onto ``x >= 0`` in the space metric: coordinatewise ``max(v, 0)``.

    Parameters
    ----------
    v : array-like
        Point in ``X``.
    X : Space
        Euclidean or diagonal-metric space.

    Returns
    -------
    Element
        ``max(v, 0)``.
    """
    v = X.ctx.asarray(v)
    _require_member_shape(X, v, "v", "project_nonneg")
    return generalized_shrinkage(X, c=X.zeros(), x0=v, eps=0.5, lam=0.0, nonneg=True)

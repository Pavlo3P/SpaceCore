"""Closed-form proximal / projection primitive (ADR-019).

This module carries ADR-019's most important correctness finding. A proximal
step must be taken *in the space metric*: a hand-written Euclidean soft-threshold
paired with a metric gradient converges to the wrong fixed point. The fix is a
single separable solver of the forward--backward subproblem that folds the
metric into the threshold, plus three named wrappers.

:func:`generalized_shrinkage` solves, coordinatewise,

.. code-block:: text

    argmin_x  <c, x>_X + eps * ||x - x0||^2_X + lam * ||x||_1   (optionally x >= 0)

where ``c`` is a *metric gradient* (e.g. ``F.grad(x)``), consistent with
[ADR-010](010_functional_contract.md). Under a diagonal weighted metric
``G = diag(w)`` the per-coordinate soft-threshold is ``tau_i = lam / (2 eps w_i)``
-- the weight enters the threshold even though it cancels in the data step
``x0 - c / (2 eps)``. The closed form is separable only on Euclidean or diagonal
metrics; on a non-diagonal metric the subproblem does not separate, so the
primitive **raises** rather than returning a wrong (separable) answer
(ADR-019 / ADR-020 diagonal-metric rule).
"""
from __future__ import annotations

import math
import numbers
from typing import Any, cast

from ...space import WeightedInnerProduct


def _diagonal_weights_or_raise(X: Any, fname: str) -> Any:
    """Return diagonal weights (``None`` for Euclidean) or raise on a non-diagonal metric."""
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
    Solve the separable forward--backward subproblem in the space metric.

    Returns the coordinatewise minimizer of

    .. code-block:: text

        <c, x>_X + eps * ||x - x0||^2_X + lam * ||x||_1   (optionally x >= 0)

    The smooth part has unconstrained minimizer ``v = x0 - c / (2 eps)`` (the
    metric weight cancels, because ``c`` is a metric gradient). The ``l1`` term
    then applies a soft-threshold with the metric-aware width
    ``tau_i = lam / (2 eps w_i)`` for a diagonal weight ``w`` (``w_i = 1`` on a
    Euclidean space); with ``nonneg=True`` the nonnegative-orthant constraint
    turns the step into ``max(v - tau, 0)``.

    Parameters
    ----------
    X : Space
        Ambient inner-product space. Must be Euclidean or have a diagonal
        (weighted) metric; a non-diagonal metric raises.
    c : array-like
        Metric gradient (linear-term coefficient) in ``X``.
    x0 : array-like
        Proximal center in ``X``.
    eps : float
        Strictly positive, finite quadratic weight.
    lam : float, optional
        Nonnegative, finite ``l1`` weight. Default ``0.0`` (no thresholding).
    nonneg : bool, optional
        Constrain the solution to ``x >= 0`` (real spaces only). Default ``False``.

    Returns
    -------
    Element
        The minimizer, an element of ``X``.
    """
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

    if lam == 0:
        tau: Any = 0.0
    elif weights is None:
        tau = lam / (2.0 * eps)
    else:
        tau = lam / (2.0 * eps * weights)

    if nonneg:
        return ops.maximum(v - tau, cast(Any, 0.0))
    # Soft-threshold; tau == 0 collapses to the identity v.
    return ops.sign(v) * ops.maximum(ops.abs(v) - tau, cast(Any, 0.0))


def _require_step(t: float, fname: str) -> float:
    """Validate and return a finite, nonnegative scalar proximal step ``t``."""
    if not isinstance(t, numbers.Real) or not math.isfinite(t) or t < 0:
        raise ValueError(f"{fname} requires a finite step t >= 0, got {t!r}.")
    return float(t)


def prox_l1(v: Any, t: Any, X: Any) -> Any:
    r"""
    Proximal operator of ``t * ||.||_1`` in the space metric (soft-threshold).

    Returns ``argmin_x  1/2 ||x - v||^2_X + t ||x||_1``, i.e. the metric-aware
    soft-threshold with per-coordinate width ``t / w_i`` on a diagonal metric.

    Parameters
    ----------
    v : array-like
        Point in ``X`` to be thresholded.
    t : float
        Nonnegative threshold / step size.
    X : Space
        Ambient inner-product space (Euclidean or diagonal metric).

    Returns
    -------
    Element
        The soft-thresholded element.
    """
    t = _require_step(t, "prox_l1")
    v = X.ctx.asarray(v)
    _require_member_shape(X, v, "v", "prox_l1")
    return generalized_shrinkage(X, c=X.zeros(), x0=v, eps=0.5, lam=t)


def prox_l2sq(v: Any, t: Any, X: Any) -> Any:
    r"""
    Proximal operator of ``t * (1/2) ||.||_X^2`` (linear shrinkage ``v / (1 + t)``).

    Returns ``argmin_x  1/2 ||x - v||^2_X + t (1/2) ||x||^2_X = v / (1 + t)``.

    Completing the square casts this into the primitive's form: it is the
    minimizer of ``<-v, x>_X + ((1 + t)/2) ||x||^2_X``, i.e.
    ``generalized_shrinkage`` with ``c = -v``, ``x0 = 0`` and ``eps = (1 + t)/2``.

    Parameters
    ----------
    v : array-like
        Point in ``X`` to be shrunk.
    t : float
        Nonnegative step size.
    X : Space
        Ambient inner-product space (Euclidean or diagonal metric).

    Returns
    -------
    Element
        The shrunk element ``v / (1 + t)``.
    """
    t = _require_step(t, "prox_l2sq")
    v = X.ctx.asarray(v)
    _require_member_shape(X, v, "v", "prox_l2sq")
    return generalized_shrinkage(X, c=-v, x0=X.zeros(), eps=0.5 * (1.0 + t), lam=0.0)


def project_nonneg(v: Any, X: Any) -> Any:
    r"""
    Metric projection onto the nonnegative orthant: coordinatewise ``max(v, 0)``.

    This is the proximal operator of the indicator of ``{x : x >= 0}``. On a
    diagonal metric the projection is still coordinatewise (the orthant is
    separable), and a non-diagonal metric raises via the shared primitive.

    Parameters
    ----------
    v : array-like
        Point in ``X`` to project.
    X : Space
        Ambient inner-product space (Euclidean or diagonal metric).

    Returns
    -------
    Element
        The projected element ``max(v, 0)``.
    """
    v = X.ctx.asarray(v)
    _require_member_shape(X, v, "v", "project_nonneg")
    return generalized_shrinkage(X, c=X.zeros(), x0=v, eps=0.5, lam=0.0, nonneg=True)

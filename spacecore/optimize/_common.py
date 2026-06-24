"""Shared marshalling helpers for the external-optimizer adapters (ADR-018).

The adapters in this subpackage drive a mature external optimizer (SciPy, optax)
from a SpaceCore :class:`~spacecore.Functional`. They share one central
responsibility, recorded in ADR-018: a SpaceCore gradient ``F.grad(x)`` is a
*metric* (Riesz) gradient (see [ADR-009](009_metric_adjoint.md) and
[ADR-010](010_functional_contract.md)), but external NumPy/JAX optimizers consume
a *coordinate* gradient. The correct, geometry-aware handoff is
``X.riesz(F.grad(x))`` -- the identity on a Euclidean space, and mandatory on a
weighted/non-Euclidean one. :func:`coordinate_gradient` performs this conversion
once, centrally, so no adapter -- and no user wrapper -- re-derives it.
"""
from __future__ import annotations

from typing import Any

from ..functional import Functional
from ..space import InnerProductSpace


def require_functional(F: Any, fname: str) -> Functional:
    """Return ``F`` as a :class:`~spacecore.Functional` or raise a clear type error."""
    if not isinstance(F, Functional):
        raise TypeError(
            f"{fname} requires a spacecore Functional, got {type(F).__name__}."
        )
    return F


def domain_with_geometry(F: Functional, fname: str) -> InnerProductSpace:
    """Return ``F.domain`` after checking it carries an inner-product geometry.

    The metric-to-coordinate gradient handoff needs ``X.riesz``; a domain without
    inner-product geometry has no Riesz map and no well-defined metric gradient
    (ADR-010), so the adapter refuses it rather than silently assuming Euclidean.
    """
    X = F.domain
    if not isinstance(X, InnerProductSpace):
        raise TypeError(
            f"{fname} requires F.domain to be an InnerProductSpace so the metric "
            f"gradient can be converted to a coordinate gradient with X.riesz; "
            f"got {type(X).__name__}."
        )
    return X


def coordinate_gradient(F: Functional, X: InnerProductSpace, x: Any) -> Any:
    """Return the coordinate gradient ``X.riesz(F.grad(x))`` at ``x``.

    This is ADR-018's central conversion. ``F.grad`` returns a metric (Riesz)
    gradient; an external coordinate-gradient consumer wants ``X.riesz`` applied
    to it. On a Euclidean space ``X.riesz`` is the identity -- not a special-cased
    branch -- and on a weighted (diagonal) metric it multiplies by the weights.
    """
    return X.riesz(F.grad(x))


def require_real_field(X: InnerProductSpace, fname: str) -> None:
    """Raise when ``X`` is over the complex field.

    The SciPy adapters drive real-valued optimizers (``scipy.optimize`` works on
    real coordinate vectors). A complex domain must be split into real and
    imaginary coordinates by the caller; refusing here is more honest than
    silently dropping the imaginary part during ``asarray``.
    """
    if X.field != "real":
        raise ValueError(
            f"{fname} drives a real-valued SciPy optimizer and cannot consume a "
            f"complex domain ({type(X).__name__}, field={X.field!r}); split the "
            f"problem into real and imaginary coordinates first."
        )

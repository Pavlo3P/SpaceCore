"""External optimizer adapters (ADR-018).

Thin functions that drive a mature external optimizer from a SpaceCore
:class:`~spacecore.Functional`. Each adapter evaluates the objective through
``F.value``, converts the metric (Riesz) gradient ``F.grad`` to a coordinate
gradient with ``X.riesz`` -- the central, geometry-aware handoff of ADR-018 --
and marshals between SpaceCore elements and the optimizer's representation. The
external optimizer owns the loop, line search, and convergence.
"""

from __future__ import annotations

from ._optax import OptaxResult, minimize_optax
from ._scipy import line_search_scipy, minimize_scipy

__all__ = [
    "OptaxResult",
    "line_search_scipy",
    "minimize_optax",
    "minimize_scipy",
]

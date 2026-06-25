"""The separable Huber loss functional (ADR-019)."""
from __future__ import annotations

import math
from typing import Any

from .._base import Domain
from ...backend import Context, jax_pytree_class
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional


@jax_pytree_class
class HuberFunctional(_CoordinateFunctional[Domain]):
    r"""
    Separable Huber loss ``F(x) = sum_i h_delta(x_i)``.

    The per-coordinate loss is quadratic near the origin and linear in the tails:
    ``h_delta(r) = 1/2 r^2`` for ``|r| <= delta`` and
    ``delta (|r| - delta/2)`` otherwise. It is everywhere differentiable, with
    gradient ``r`` in the quadratic region and ``delta sign(r)`` in the tails.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    delta : float
        Transition threshold; must be finite and ``> 0``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> f = sc.HuberFunctional(X, 1.0)
    >>> float(f.value(ctx.asarray([0.5, 3.0])))  # 0.125 + (3 - 0.5)
    2.625
    """

    def __init__(self, dom: Domain, delta: Any, ctx: Context | str | None = None) -> None:
        super().__init__(dom, ctx)
        delta = float(delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"HuberFunctional requires a finite delta > 0, got {delta}.")
        self.delta = delta

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``sum_i h_delta(x_i)``."""
        o = self.ops
        d = self.delta
        a = o.abs(x)
        quadratic = 0.5 * a * a
        linear = d * (a - 0.5 * d)
        return o.sum(o.where(a <= d, quadratic, linear))

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient: ``x`` (quadratic) or ``delta sign(x)`` (tail)."""
        o = self.ops
        d = self.delta
        a = o.abs(x)
        return o.where(a <= d, x, d * o.sign(x))

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (), (self.domain, self.delta, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, delta, ctx = aux
        return cls(domain, delta, ctx)

    def _convert(self, new_ctx: Context) -> "HuberFunctional":
        """Convert this functional to ``new_ctx``."""
        return HuberFunctional(self.domain.convert(new_ctx), self.delta, new_ctx)

"""The separable Huber loss functional (ADR-019)."""
from __future__ import annotations

import math
from typing import Any

from .._base import Domain
from ...contextual import Context
from ..._check_policy import CheckLevel
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional


class HuberFunctional(_CoordinateFunctional[Domain]):
    r"""
    Separable Huber loss ``F(x) = sum_i h_delta(x_i)``.

    The per-coordinate loss is quadratic near the origin and linear in the tails:
    ``h_delta(r) = 1/2 r^2`` for ``|r| <= delta`` and
    ``delta (|r| - delta/2)`` otherwise. It is everywhere differentiable, with
    gradient ``r`` in the quadratic region and ``delta sign(r)`` in the tails
    (value and gradient agree at ``|r| = delta``, so ``h_delta`` is ``C^1``).

    **Convention.** This is Huber's original robust-statistics scaling [Huber1964]_,
    which is ``delta`` times the Moreau envelope used in the optimization
    literature: with ``M_mu`` the envelope of the absolute value from
    [Beck]_ Example 6.54, ``h_delta = delta * M_delta``. The distinction matters
    when transcribing a prox or a smoothing constant from either source — the two
    differ by exactly one factor of ``delta``.

    References
    ----------
    .. [Huber1964] P. J. Huber, "Robust estimation of a location parameter",
       *Ann. Math. Statist.* 35(1):73-101, 1964, §2 — the original
       quadratic-near-zero / linear-in-the-tails loss.
    .. [Beck] A. Beck, *First-Order Methods in Optimization*, MOS-SIAM, 2017,
       Example 6.54 (the Huber function as the Moreau envelope of the norm),
       Example 6.62 (its smoothness: gradient ``1/mu``-Lipschitz) and
       Example 6.66 (prox of the Huber function).

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

    def __init__(
        self,
        dom: Domain,
        delta: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)
        delta = float(delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"HuberFunctional requires a finite delta > 0, got {delta}.")
        self.delta = delta

    @checked_method(in_space="domain", out_scalar=True)
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

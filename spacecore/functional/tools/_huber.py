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
    Separable Huber loss ``sum_i h_delta(x_i)``.

    ``h_delta(r)`` is ``r^2 / 2`` for ``|r| <= delta`` and ``delta (|r| - delta / 2)``
    beyond; it is ``C^1``. This is Huber's scaling [Huber1964]_, which is ``delta``
    times the Moreau-envelope form of [Beck]_ Example 6.54, so prox formulas from the
    two sources differ by that factor.

    Parameters
    ----------
    dom : Space
        Domain.
    delta : float
        Positive, finite threshold.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    References
    ----------
    .. [Huber1964] P. J. Huber, Ann. Math. Statist. 35(1):73-101, 1964.
    .. [Beck] A. Beck, *First-Order Methods in Optimization*, SIAM, 2017.

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
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)
        # delta must be finite and positive.
        delta = float(delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"HuberFunctional requires a finite delta > 0, got {delta}.")
        self.delta = delta

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any) -> Any:
        """Return ``sum_i h_delta(x_i)``."""
        # Quadratic inside |x| <= delta, linear outside.
        o = self.ops
        d = self.delta
        a = o.abs(x)
        quadratic = 0.5 * a * a
        linear = d * (a - 0.5 * d)
        return o.sum(o.where(a <= d, quadratic, linear))

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient: ``x`` (quadratic) or ``delta sign(x)`` (tail)."""
        # Gradient: x inside, delta sign(x) outside.
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

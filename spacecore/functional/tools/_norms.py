"""Norm functionals: squared L2 energy and coordinate ``p``-norms (ADR-019)."""
from __future__ import annotations

import math
from typing import Any, cast

from .._base import Domain
from ...contextual import Context
from ..._checks import CheckLevel
from ._coordinate import _CoordinateFunctional, lp_coordinate_grad, lp_value


class SquaredL2NormFunctional(_CoordinateFunctional[Domain]):
    r"""
    Half the squared space norm ``<x, x>_X / 2``; its gradient is ``x``.

    Not ``LpNormFunctional(X, 2)``, which is the unsquared coordinate norm.

    Parameters
    ----------
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> f = sc.SquaredL2NormFunctional(X)
    >>> float(f.value(ctx.asarray([3.0, 4.0])))
    12.5
    >>> np.asarray(f.grad(ctx.asarray([3.0, 4.0])))
    array([3., 4.])
    """

    _fields = ("dom", "ctx")

    def _value_core(self, x: Any) -> Any:
        """``1/2 <x, x>_X`` as a real scalar."""
        return 0.5 * self.domain._real_inner_core(x, x)

    def _grad_core(self, x: Any) -> Any:
        """The Riesz gradient ``x``."""
        return x

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient ``G x`` of ``1/2 <x, x>_X``."""
        return cast(Any, self.domain).riesz(x)

class LpNormFunctional(_CoordinateFunctional[Domain]):
    r"""
    Coordinate ``p``-norm ``(sum_i |x_i|^p)^(1/p)`` for ``p >= 1``.

    The gradient is ``sign(x_i) |x_i|^(p-1) / ||x||_p^(p-1)``, Riesz-corrected; at the
    origin the zero subgradient is returned (Beck, *First-Order Methods in
    Optimization*, Example 3.3). For ``p = 1``, zero coordinates also use zero
    subgradient components (Example 3.4); elsewhere differentiate the displayed norm.

    Parameters
    ----------
    dom : Space
        Domain.
    p : float
        Finite order, at least 1.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((3,), ctx)
    >>> f = sc.LpNormFunctional(X, 1)
    >>> float(f.value(ctx.asarray([1.0, -2.0, 3.0])))
    6.0
    """

    _fields = ("dom", "p", "ctx")

    def __init__(
        self,
        dom: Domain,
        p: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)
        # p must be finite and at least 1.
        p = float(p)
        if not math.isfinite(p) or p < 1.0:
            raise ValueError(f"LpNormFunctional requires a finite p >= 1, got {p}.")
        self.p = p

    def _value_core(self, x: Any) -> Any:
        """``(sum_i |x_i|^p)^{1/p}``."""
        return lp_value(self.ops, x, self.p)

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient of the ``p``-norm (zero at the origin)."""
        return lp_coordinate_grad(self.ops, x, self.p)

def L1NormFunctional(
    dom: Domain,
    ctx: Context | str | None = None,
    check_level: CheckLevel | None = None,
) -> "LpNormFunctional[Domain]":
    r"""
    Coordinate 1-norm: ``LpNormFunctional(dom, 1)``.

    Parameters
    ----------
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Returns
    -------
    LpNormFunctional
        The ``p = 1`` norm.
    """
    return LpNormFunctional(dom, 1.0, ctx, check_level=check_level)

"""Norm functionals: squared L2 energy and coordinate ``p``-norms (ADR-019)."""
from __future__ import annotations

import math
from typing import Any, cast

from .._base import Domain
from ...backend import Context, jax_pytree_class
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional, _inner_core, lp_coordinate_grad, lp_value


@jax_pytree_class
class SquaredL2NormFunctional(_CoordinateFunctional[Domain]):
    r"""
    Half the squared space norm ``F(x) = 1/2 ||x||_X^2 = 1/2 <x, x>_X``.

    This is the smooth quadratic energy whose Riesz gradient is ``x`` and whose
    proximal operator is the clean shrinkage ``v / (1 + t)`` (see
    :func:`~spacecore.functional.prox_l2sq`). It is intentionally distinct from
    ``LpNormFunctional(X, 2)``, which is the *un-squared* coordinate 2-norm
    ``(sum_i |x_i|^2)^{1/2}``.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

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

    def __init__(self, dom: Domain, ctx: Context | str | None = None) -> None:
        super().__init__(dom, ctx)

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``1/2 <x, x>_X`` as a real scalar."""
        return 0.5 * self.ops.real(_inner_core(self.domain, x, x))

    @checked_method(in_space="domain", out_space="domain")
    def grad(self, x: Any) -> Any:
        """Return the Riesz gradient ``x``."""
        return x

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient ``G x`` of ``1/2 <x, x>_X``."""
        return cast(Any, self.domain).riesz(x)

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx: Context) -> "SquaredL2NormFunctional":
        """Convert this functional to ``new_ctx``."""
        return SquaredL2NormFunctional(self.domain.convert(new_ctx), new_ctx)


@jax_pytree_class
class LpNormFunctional(_CoordinateFunctional[Domain]):
    r"""
    Coordinate ``p``-norm ``F(x) = (sum_i |x_i|^p)^{1/p}`` for ``p >= 1``.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    p : float
        Norm order; must be finite and ``>= 1``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Notes
    -----
    The gradient at ``x != 0`` is
    ``d/dx_i ||x||_p = sign(x_i) |x_i|^{p-1} / ||x||_p^{p-1}`` (Riesz-corrected).
    At the origin the function is not differentiable; this returns the zero
    subgradient there. For ``p = 1`` the gradient is ``sign(x)``.

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

    def __init__(self, dom: Domain, p: Any, ctx: Context | str | None = None) -> None:
        super().__init__(dom, ctx)
        p = float(p)
        if not math.isfinite(p) or p < 1.0:
            raise ValueError(f"LpNormFunctional requires a finite p >= 1, got {p}.")
        self.p = p

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``(sum_i |x_i|^p)^{1/p}``."""
        return lp_value(self.ops, x, self.p)

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient of the ``p``-norm (zero at the origin)."""
        return lp_coordinate_grad(self.ops, x, self.p)

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (), (self.domain, self.p, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, p, ctx = aux
        return cls(domain, p, ctx)

    def _convert(self, new_ctx: Context) -> "LpNormFunctional":
        """Convert this functional to ``new_ctx``."""
        return LpNormFunctional(self.domain.convert(new_ctx), self.p, new_ctx)


def L1NormFunctional(
    dom: Domain, ctx: Context | str | None = None
) -> "LpNormFunctional[Domain]":
    r"""
    Coordinate 1-norm ``||x||_1`` -- a thin wrapper for ``LpNormFunctional(X, 1)``.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Returns
    -------
    LpNormFunctional
        The ``p = 1`` instance of :class:`LpNormFunctional`.
    """
    return LpNormFunctional(dom, 1.0, ctx)

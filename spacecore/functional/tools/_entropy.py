"""Entropy objectives: negative entropy and KL divergence (ADR-019)."""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain
from .._linear import _convert_space_element
from ...backend import Context, jax_pytree_class
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional


@jax_pytree_class
class NegativeEntropyFunctional(_CoordinateFunctional[Domain]):
    r"""
    Negative (Shannon) entropy ``F(x) = sum_i x_i log x_i``.

    The natural domain is the positive orthant ``x_i > 0``; the value uses the
    convention ``0 log 0 = 0`` so the origin and zero coordinates evaluate
    cleanly, but the gradient ``log x_i + 1`` is only defined for ``x_i > 0``.

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
    >>> f = sc.NegativeEntropyFunctional(X)
    >>> float(f.value(ctx.asarray([1.0, 1.0])))
    0.0
    >>> np.asarray(f.grad(ctx.asarray([1.0, 1.0])))  # log(x) + 1
    array([1., 1.])
    """

    def __init__(self, dom: Domain, ctx: Context | str | None = None) -> None:
        super().__init__(dom, ctx)

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``sum_i x_i log x_i`` with ``0 log 0 = 0``."""
        o = self.ops
        positive = x > 0
        safe = o.where(positive, x, cast(Any, 1.0))
        return o.sum(o.where(positive, x * o.log(safe), cast(Any, 0.0)))

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient ``log x_i + 1`` (defined for ``x_i > 0``)."""
        return self.ops.log(x) + 1.0

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx: Context) -> "NegativeEntropyFunctional":
        """Convert this functional to ``new_ctx``."""
        return NegativeEntropyFunctional(self.domain.convert(new_ctx), new_ctx)


@jax_pytree_class
class KLDivergenceFunctional(_CoordinateFunctional[Domain]):
    r"""
    Kullback--Leibler divergence to a fixed positive ``target``.

    ``F(x) = sum_i x_i log(x_i / t_i)`` against a strictly positive target ``t``.
    The natural domain is ``x_i >= 0`` (with ``0 log 0 = 0``) and ``t_i > 0``.
    With ``target`` equal to the all-ones element this reduces exactly to
    :class:`NegativeEntropyFunctional`, and the gradient ``log(x_i / t_i) + 1``
    reduces accordingly.

    Parameters
    ----------
    target : array-like
        Reference element ``t`` in ``dom`` with strictly positive coordinates.
    dom : Space
        Domain space ``X``. ADR-019 writes ``KLDivergenceFunctional(target)``;
        the explicit space follows the ``(data, dom, ctx)`` constructor shape
        used by :class:`~spacecore.functional.InnerProductFunctional`, because a
        bare element does not carry its space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> f = sc.KLDivergenceFunctional(ctx.asarray([1.0, 1.0]), X)
    >>> float(f.value(ctx.asarray([1.0, 1.0])))  # zero divergence at x == target
    0.0
    >>> np.asarray(f.grad(ctx.asarray([1.0, 1.0])))  # log(x / t) + 1
    array([1., 1.])
    """

    def __init__(
        self,
        target: Any,
        dom: Domain,
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, ctx)
        self._target = _convert_space_element(self.domain, target)
        if self._checks_at_least("standard"):
            self.domain._check_member(self._target)

    @property
    def target(self) -> Any:
        """Stored reference element ``t``."""
        return self._target

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``sum_i x_i log(x_i / t_i)`` with ``0 log 0 = 0``."""
        o = self.ops
        positive = x > 0
        ratio = o.where(positive, x / self._target, cast(Any, 1.0))
        return o.sum(o.where(positive, x * o.log(ratio), cast(Any, 0.0)))

    def _coordinate_grad(self, x: Any) -> Any:
        """Euclidean coordinate gradient ``log(x_i / t_i) + 1`` (for ``x_i > 0``)."""
        return self.ops.log(x / self._target) + 1.0

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (self._target,), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, ctx = aux
        return cls(children[0], domain, ctx)

    def _convert(self, new_ctx: Context) -> "KLDivergenceFunctional":
        """Convert this functional to ``new_ctx``."""
        return KLDivergenceFunctional(self._target, self.domain.convert(new_ctx), new_ctx)

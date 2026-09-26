"""
Entropy objectives: negative entropy and KL divergence (ADR-019).

The negative entropy is the standard mirror-descent kernel (Beck, *First-Order
Methods in Optimization*, 2017, Example 9.10); lifted with
:func:`~spacecore.spectralize` it becomes the von Neumann entropy.
"""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain
from .._linear import _convert_space_element
from ...contextual import Context
from ..._check_policy import CheckLevel
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional


class NegativeEntropyFunctional(_CoordinateFunctional[Domain]):
    r"""
    Negative entropy ``sum_i x_i log x_i``, with ``0 log 0 = 0``; the gradient ``log x_i + 1`` needs ``x_i > 0``.

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
    >>> f = sc.NegativeEntropyFunctional(X)
    >>> float(f.value(ctx.asarray([1.0, 1.0])))
    0.0
    >>> np.asarray(f.grad(ctx.asarray([1.0, 1.0])))  # log(x) + 1
    array([1., 1.])
    """

    def __init__(
        self,
        dom: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any) -> Any:
        """Return ``sum_i x_i log x_i`` with ``0 log 0 = 0``."""
        # sum x log x, with 0 log 0 = 0 (log evaluated on a safe input).
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


class KLDivergenceFunctional(_CoordinateFunctional[Domain]):
    r"""
    Kullback-Leibler divergence ``sum_i x_i log(x_i / t_i)`` to a positive target ``t``.

    Defined for ``x_i >= 0`` (``0 log 0 = 0``); the gradient is ``log(x_i / t_i) + 1``.
    With ``t`` all ones it is :class:`NegativeEntropyFunctional`.

    Parameters
    ----------
    target : array-like
        Target ``t`` in ``dom``, strictly positive.
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
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)
        # Store the target on the functional's backend and check it belongs.
        self._target = _convert_space_element(self.domain, target)
        if self._checks_at_least("standard"):
            self.domain._check_member(self._target)

    @property
    def target(self) -> Any:
        """Stored reference element ``t``."""
        return self._target

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any) -> Any:
        """Return ``sum_i x_i log(x_i / t_i)`` with ``0 log 0 = 0``."""
        # sum x log(x / t), with 0 log 0 = 0.
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

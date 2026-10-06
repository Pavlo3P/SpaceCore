"""
Entropy objectives: negative entropy and KL divergence (ADR-019).

The negative entropy is the standard mirror-descent kernel (Beck, *First-Order
Methods in Optimization*, Example 9.10, on the probability simplex); its spectral
lift is ``Tr(X log X)``, the negative von Neumann entropy for a density matrix.
"""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain
from .._linear import _convert_space_element
from ...contextual import Context
from ..._checks import CheckLevel
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

    _fields = ("dom", "ctx")

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

class KLDivergenceFunctional(_CoordinateFunctional[Domain]):
    r"""
    Kullback-Leibler divergence ``sum_i x_i log(x_i / t_i)`` to a positive target ``t``.

    Defined for ``x_i >= 0`` (``0 log 0 = 0``); the gradient is ``log(x_i / t_i) + 1``.
    With ``t`` all ones it is :class:`NegativeEntropyFunctional`. On probability
    vectors this is KL divergence (Beck, *First-Order Methods in Optimization*,
    Example 9.10, Eq. (9.13)); without equal total mass it can be negative.

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

    _fields = ("target:_target", "dom", "ctx")
    _dynamic = ("target",)

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

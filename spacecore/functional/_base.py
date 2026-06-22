from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .._batching import _leading_batch_size, _warn_vmap_fallback_once

# Re-exported for backward compatibility; these helpers now live in
# :mod:`spacecore._batching` and are shared with :class:`~spacecore.linop.LinOp`.
from .._batching import (  # noqa: F401
    _VMAP_FALLBACK_WARN_BATCH,
    _VMAP_FALLBACK_WARNED,
    _check_scalar_shape,
)
from .._checks import checked_method
from .._contextual import ContextBound
from ..backend import Context
from ..space import Space

if TYPE_CHECKING:
    from ..linop import LinOp


Domain = TypeVar("Domain", bound=Space)


class Functional(ContextBound, Generic[Domain]):
    r"""
    Scalar-valued map on a space.

    ``Functional`` represents a map ``F : X -> K`` without assuming any storage
    model. It mirrors the minimal ``LinOp`` contract: the domain is converted
    into the resolved context, value checks follow ``ctx.check_level``, and
    batched evaluation is implemented by a backend ``vmap`` fallback.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Attributes
    ----------
    dom : Space
        Domain space converted to ``ctx``.
    ctx : Context
        Resolved backend context.
    """

    def __init__(self, dom: Domain, ctx: Context | str | None = None) -> None:
        (self.dom,) = self._bind_context(ctx, dom)

    @property
    def domain(self) -> Domain:
        """Domain space of this scalar-valued map."""
        return self.dom

    @abstractmethod
    def value(self, x: Any) -> Any:
        """Evaluate this functional at an element of ``self.domain``."""

    def _value_core(self, x: Any) -> Any:
        """Check-free value core; the base falls back to the checked ``value``."""
        return self.value(x)

    def _grad_core(self, x: Any) -> Any:
        """Check-free gradient core; the base falls back to the checked ``grad``."""
        return self.grad(x)

    def _vvalue_core(self, xs: Any) -> Any:
        """Check-free batched-value core; the base falls back to ``vvalue``."""
        return self.vvalue(xs)

    def _vgrad_core(self, xs: Any) -> Any:
        """Check-free batched-gradient core; the base falls back to ``vgrad``."""
        return self.vgrad(xs)

    def __call__(self, x: Any) -> Any:
        """Evaluate this functional at ``x``."""
        return self.value(x)

    def compose(self, A: "LinOp") -> "Functional":
        """
        Return the pull-back ``self o A``.

        Parameters
        ----------
        A : LinOp
            Linear operator whose codomain matches this functional's domain.

        Returns
        -------
        Functional
            Functional on ``A.domain`` evaluating ``self.value(A.apply(x))``.
        """
        from ._composed import make_functional_composed

        return make_functional_composed(self, A)

    @checked_method(in_space="domain", in_batched=True)
    def vvalue(self, xs: Any) -> Any:
        """Evaluate over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        _warn_vmap_fallback_once(self, "vvalue", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.value, in_axes=0, out_axes=0)(xs)

    def assert_domain(self, x: Any) -> None:
        """Raise if ``x`` is not in the domain."""
        self.dom.check_member(x)

    @abstractmethod
    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        ...

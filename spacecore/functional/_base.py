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
from ..mapping import Mapping
from ..contextual import Context, resolve_context_priority
from .._check_policy import CheckLevel
from ..space import CoordinateSpace, Field

if TYPE_CHECKING:
    from ..linop import LinOp


Domain = TypeVar("Domain", bound=CoordinateSpace)


class Functional(Mapping, Generic[Domain]):
    r"""
    Scalar-valued map on a space.

    ``Functional`` represents a map ``F : X -> K`` without assuming any storage
    model. It mirrors the minimal ``LinOp`` contract: the domain is converted
    into the resolved context, value checks follow this object's ``check_level``,
    and batched evaluation is implemented by a backend ``vmap`` fallback.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.

    cod : Field or None, optional
        Scalar codomain, defaulting to the converted domain's ``scalars``.
        Its real or complex field is independent of the domain storage dtype.

    Attributes
    ----------
    dom : Space
        Domain space converted to ``ctx``.
    ctx : Context
        Resolved backend context.
    """

    def __init__(
        self,
        dom: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
        *,
        cod: Field | None = None,
    ) -> None:
        resolved = resolve_context_priority(ctx, dom)
        dom = dom.convert(resolved)
        if cod is None:
            cod = dom.scalars
        if not isinstance(cod, Field):
            raise TypeError("Functional codomain must be a Field.")
        self.dom, self.cod = self._bind_context(resolved, dom, cod, check_level=check_level)
        # Output checks follow the functional's policy, even when its domain
        # or explicitly supplied field uses a different validation level.
        self.cod = self.cod._with_check_level(self.check_level)

    @property
    def codomain(self) -> Field:
        """The independently bound scalar field of this functional's values."""
        return self.cod

    @property
    def domain(self) -> Domain:
        """Domain space of this scalar-valued map."""
        return self.dom

    @abstractmethod
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate this functional at an element of ``self.domain``.

        Subclasses may accept extra positional/keyword arguments — auxiliary
        parameters such as data, temperature, or a penalty weight — that are not
        part of the domain. Overrides that take only ``x`` remain valid.
        """

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Gradient at an element of ``self.domain``.

        Override in subclasses that support differentiation; the base raises
        :class:`NotImplementedError`. Any auxiliary ``*args``/``**kwargs`` mirror
        those accepted by :meth:`value`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grad().")

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(value, gradient)`` at ``x``.

        The base default evaluates :meth:`value` and :meth:`grad` separately.
        Subclasses may override with a single-pass evaluator (e.g.
        ``jax.value_and_grad``); an override must return the same pair as the
        default, with the gradient in the same geometry as :meth:`grad`.
        """
        return self.value(x, *args, **kwargs), self.grad(x, *args, **kwargs)

    def vgrad(self, xs: Any) -> Any:
        """Gradient over a leading batch axis.

        Override in subclasses that support differentiation; the base raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement vgrad().")

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free value core; the base falls back to the checked ``value``."""
        return self.value(x, *args, **kwargs)

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free gradient core; the base falls back to the checked ``grad``."""
        return self.grad(x, *args, **kwargs)

    def _vvalue_core(self, xs: Any) -> Any:
        """Check-free batched-value core; the base falls back to ``vvalue``."""
        return self.vvalue(xs)

    def _vgrad_core(self, xs: Any) -> Any:
        """Check-free batched-gradient core; the base falls back to ``vgrad``."""
        return self.vgrad(xs)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate this functional at ``x``."""
        return self.value(x, *args, **kwargs)

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

    @checked_method(in_space="domain", in_batched=True, out_batched_scalar=True)
    def vvalue(self, xs: Any) -> Any:
        """Evaluate over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        _warn_vmap_fallback_once(self, "vvalue", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.value, in_axes=0, out_axes=0)(xs)

    def assert_domain(self, x: Any) -> None:
        """Raise if ``x`` is not in the domain."""
        self.dom.check_member(x)

    # tree_flatten / tree_unflatten are inherited from PyTreeNode, which owns the
    # flatten contract for every SpaceCore container and auto-registers concrete
    # subclasses with each backend's tree protocol. Every concrete Functional is a
    # container, so the capability belongs on this base rather than being
    # re-declared (and separately registered) per functional.

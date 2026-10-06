from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .._batching import _leading_batch_size, _warn_vmap_fallback_once

from .._checks import checked_method
from ..mapping import Mapping
from ..contextual import Context, resolve_context_priority
from .._checks import CheckLevel
from ..space import Space, Field

if TYPE_CHECKING:
    from ..linop import LinOp
    from ._linear import LinearFunctional


Domain = TypeVar("Domain", bound=Space)


class Functional(Mapping, Generic[Domain]):
    r"""
    A scalar-valued map ``F : X -> K``; subclasses implement ``value`` and optionally ``grad``.

    ``grad`` is the Riesz gradient, an element of ``X``. Batched evaluation falls back
    to ``vmap``.

    Parameters
    ----------
    dom : Space
        Domain ``X``.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    cod : Field or None, optional
        Scalar codomain; the domain's ``scalars`` by default.
    """

    def __init__(
        self,
        dom: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
        *,
        cod: Field | None = None,
    ) -> None:
        # Resolve the context from the domain; the codomain defaults to its scalars.
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

    def _init_map(self, domain: Any, codomain: Any, ctx: Any, check_level: Any) -> None:
        """Bind a node of this family (the hook the shared lazy nodes in ``mapping`` call)."""
        Functional.__init__(self, domain, ctx, check_level=check_level, cod=codomain)

    @property
    def codomain(self) -> Field:
        """The independently bound scalar field of this functional's values."""
        return self.cod

    @property
    def domain(self) -> Domain:
        """Domain space of this scalar-valued map."""
        return self.dom

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate this functional at an element of ``self.domain``.

        Subclasses may accept extra positional/keyword arguments — auxiliary
        parameters such as data, temperature, or a penalty weight — that are not
        part of the domain. Overrides that take only ``x`` remain valid.
        """
        return self._value_core(x, *args, **kwargs)

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Gradient at an element of ``self.domain``.

        Override directly, or implement ``derivative_at`` with an available
        Riesz representer. Otherwise raises ``NotImplementedError``. Any auxiliary ``*args``/``**kwargs`` mirror
        those accepted by :meth:`value`.
        """
        if type(self).derivative_at is not Functional.derivative_at:
            return self.derivative_at(x, *args, **kwargs).representer
        raise NotImplementedError(f"{type(self).__name__} does not implement grad().")

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(value, gradient)`` at ``x``.

        The base default evaluates :meth:`value` and :meth:`grad` separately.
        Subclasses may override with a single-pass evaluator (e.g.
        ``jax.value_and_grad``); an override must return the same pair as the
        default, with the gradient in the same geometry as :meth:`grad`.
        """
        return self.value(x, *args, **kwargs), self.grad(x, *args, **kwargs)

    def prox(self, x: Any, step: Any) -> Any:
        r"""Minimize ``f(y) + ||y-x||_X^2 / (2 step)`` for positive ``step``.

        This optional operation uses a closed-form rule registered with
        ``register("prox", FunctionalType, GeometryType, rule)``. The rule takes
        ``(functional, geometry, x, step)`` and returns a native domain element.
        An unsupported functional/geometry pair raises ``CapabilityError``;
        sums and linear pullbacks do not acquire a prox automatically.

        Prox rules assert a proper, lower-semicontinuous convex real objective
        (also on complex spaces, using the real inner product). At check level
        ``"none"``, traced steps must be finite and positive; their values are
        the caller's responsibility. Concrete invalid Python steps still raise.
        """
        from ._prox import apply_prox

        return apply_prox(self, x, step)

    def conjugate_prox(self, x: Any, step: Any) -> Any:
        r"""Return the conjugate prox by Hilbert-space Moreau decomposition.

        The conjugate is ``f*_X(p) = sup_z Re <p,z>_X - f(z)``: dual variables
        are Riesz representers in the same space, not raw coordinate covectors.
        Thus ``prox_(step f*_X)(x) = x - step prox_(f/step)(x/step)``.
        Requires a valid convex prox and a finite, strictly positive step.
        """
        from ._prox import validate_step

        validate_step(step, self, "conjugate_prox")
        X = self.domain
        return X.axpy(-step, self.prox(X.scale(1 / step, x), 1 / step), x)

    def derivative_at(self, x: Any, *args: Any, **kwargs: Any) -> "LinearFunctional":
        """Return the differential at ``x``, acting on domain directions.

        The compatibility adapter uses the existing Riesz gradient. Real-valued
        functionals use ``Re <grad, h>``; complex-valued functionals using this
        adapter must have a complex-linear differential. Override this method
        for a native directional evaluator or a general real-linear derivative.
        At nonsmooth points a supplied subgradient is not a Frechet derivative.
        """
        return self._derivative_from_gradient(self.grad(x, *args, **kwargs))

    def _derivative_from_gradient(self, gradient: Any) -> "LinearFunctional":
        from ._linear import InnerProductFunctional

        return InnerProductFunctional(
            gradient, self.domain, self.ctx, check_level=self.check_level,
            cod=self.codomain,
        )

    def value_and_derivative(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, "LinearFunctional"]:
        """Return value and differential, preserving fused gradient evaluators.

        Derivative-first subclasses may override this method to share work.
        No representer is reconstructed from a callable differential.
        """
        if type(self).derivative_at is not Functional.derivative_at:
            return self.value(x, *args, **kwargs), self.derivative_at(x, *args, **kwargs)
        value, gradient = self.value_and_grad(x, *args, **kwargs)
        return value, self._derivative_from_gradient(gradient)

    def vgrad(self, xs: Any) -> Any:
        """Gradient over a leading batch axis.

        Override in subclasses that support differentiation; the base raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement vgrad().")

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free value core; the base falls back to an overridden ``value``."""
        if type(self).value is Functional.value:
            raise NotImplementedError(f"{type(self).__name__} defines neither value nor _value_core.")
        return self.value(x, *args, **kwargs)

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free gradient core; the base falls back to the checked ``grad``."""
        return self.grad(x, *args, **kwargs)

    def _vvalue_core(self, xs: Any) -> Any:
        """Check-free batched-value core; the base vmaps ``value`` unless ``vvalue`` is overridden."""
        if type(self).vvalue is not Functional.vvalue:
            return self.vvalue(xs)
        return self._vmap_vvalue(xs)

    def _vmap_vvalue(self, xs: Any) -> Any:
        # Generic batching: vmap the single-element value over the leading axis.
        _warn_vmap_fallback_once(self, "vvalue", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.value, in_axes=0, out_axes=0)(xs)

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
        return self._vvalue_core(xs)

    def assert_domain(self, x: Any) -> None:
        """Raise if ``x`` is not in the domain."""
        self.dom.check_member(x)

    # tree_flatten / tree_unflatten are inherited from PyTreeNode, which owns the
    # flatten contract for every SpaceCore container and auto-registers concrete
    # subclasses with each backend's tree protocol. Every concrete Functional is a
    # container, so the capability belongs on this base rather than being
    # re-declared (and separately registered) per functional.


Functional._family = Functional

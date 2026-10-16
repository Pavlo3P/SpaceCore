from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from .._batching import _leading_batch_size, _warn_vmap_fallback_once

from .._checks import checked_method
from ..mapping import Mapping
from ..contextual import Context, resolve_context_priority
from .._checks import CheckLevel
from ..space import Space, RealField

if TYPE_CHECKING:
    from ..linop import LinOp
    from ._linear import LinearFunctional


Domain = TypeVar("Domain", bound=Space)


_Method = TypeVar("_Method", bound=Callable[..., Any])


def _checked_pair(method: _Method) -> _Method:
    """Check a ``(value, second)`` evaluator: input in the domain, value in the codomain.

    ``second`` is checked against the domain only for ``value_and_grad``; a derivative is
    a functional, not an element.
    """
    pair_with_grad = method.__name__ == "value_and_grad"

    @wraps(method)
    def outputs_checked(self: Any, *args: Any, **kwargs: Any) -> Any:
        value, second = method(self, *args, **kwargs)
        if self.check_level != "none":
            self.codomain._check_member(value)
            if pair_with_grad:
                self.domain._check_member(second)
        return value, second

    return checked_method(in_space="domain")(outputs_checked)


#: How each public evaluator is checked; overrides in subclasses get the same wrapper.
_PUBLIC_CHECKS: dict[str, Callable[[Any], Any]] = {
    "value": checked_method(in_space="domain", out_space="codomain"),
    "grad": checked_method(in_space="domain", out_space="domain"),
    "vvalue": checked_method(in_space="domain", out_space="codomain", in_batched=True,
                             out_batched=True, out_batched_scalar=True),
    "vgrad": checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True),
    "value_and_grad": _checked_pair,
    "value_and_derivative": _checked_pair,
    "derivative_at": checked_method(in_space="domain"),
    "differential": checked_method(in_space="domain"),
    "prox": checked_method(in_space="domain", out_space="domain"),
}


class Functional(Mapping, Generic[Domain]):
    r"""
    A real-valued map ``F : X -> R``; subclasses implement ``value`` and optionally ``grad``.

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
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Subclasses implement cores; a public evaluator they override still gets the base checks.
        for name, check in _PUBLIC_CHECKS.items():
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_spacecore_checked", False):
                setattr(cls, name, check(method))

    def __init__(
        self,
        dom: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Domain storage may be complex; functional values are real scalars.
        resolved = resolve_context_priority(ctx, dom)
        dom = dom.convert(resolved)
        (self.dom,) = self._bind_context(resolved, dom, check_level=check_level)
        # Output checks follow the functional's policy.
        self.cod = RealField(self.ctx, check_level=self.check_level)

    def _init_map(self, domain: Any, codomain: Any, ctx: Any, check_level: Any) -> None:
        """Bind a node of this family (the hook the shared lazy nodes in ``mapping`` call)."""
        Functional.__init__(self, domain, ctx, check_level=check_level)

    @property
    def codomain(self) -> RealField:
        """The independently bound scalar field of this functional's values."""
        return self.cod

    @property
    def domain(self) -> Domain:
        """Domain space of this scalar-valued map."""
        return self.dom

    # Public evaluators: check, then call the core. Subclasses implement the cores
    # (``_value_core``, ``_grad_core``, ...); composites call their children's cores.

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate this functional at an element of ``self.domain``.

        Subclasses may accept extra positional/keyword arguments — auxiliary
        parameters such as data, temperature, or a penalty weight — that are not
        part of the domain. Overrides that take only ``x`` remain valid.
        """
        return self._value_core(x, *args, **kwargs)

    @checked_method(in_space="domain", out_space="domain")
    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Riesz gradient at an element of ``self.domain``.

        Implement ``_grad_core``, or ``_derivative_core`` with an available Riesz
        representer; otherwise raises ``NotImplementedError``. Any auxiliary
        ``*args``/``**kwargs`` mirror those accepted by :meth:`value`.
        """
        return self._grad_core(x, *args, **kwargs)

    @_checked_pair
    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(value, gradient)`` at ``x``.

        Implement ``_value_and_grad_core`` for a single-pass evaluator (e.g.
        ``jax.value_and_grad``); it must return the same pair as the separate cores,
        with the gradient in the same geometry as :meth:`grad`.
        """
        return self._value_and_grad_core(x, *args, **kwargs)

    @checked_method(in_space="domain", out_space="domain")
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

    @checked_method(in_space="domain")
    def differential(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        r"""Return the covector :math:`df(x)` with :math:`Df(x)[h] = \langle df(x), h\rangle`.

        The covector is a coordinate array paired with ``domain.pair`` and involves no
        metric. The default reads it off the Riesz gradient, ``domain.riesz(grad(x))``;
        functionals that compute coordinate partials implement ``_differential_core``.
        """
        return self._differential_core(x, *args, **kwargs)

    @checked_method(in_space="domain")
    def derivative_at(self, x: Any, *args: Any, **kwargs: Any) -> "LinearFunctional":
        """Return the differential at ``x``, acting on domain directions.

        The default wraps the Riesz gradient, ``Re <grad, h>``. Implement
        ``_derivative_core`` for a native directional evaluator. At nonsmooth points
        a supplied subgradient is not a Frechet derivative.
        """
        return self._derivative_core(x, *args, **kwargs)

    def _derivative_from_gradient(self, gradient: Any) -> "LinearFunctional":
        from ._linear import InnerProductFunctional

        return InnerProductFunctional(
            gradient, self.domain, self.ctx, check_level=self.check_level,
        )

    @_checked_pair
    def value_and_derivative(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, "LinearFunctional"]:
        """Return value and differential, preserving fused gradient evaluators.

        Derivative-first subclasses may implement ``_value_and_derivative_core`` to
        share work. No representer is reconstructed from a callable differential.
        """
        return self._value_and_derivative_core(x, *args, **kwargs)

    @checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True)
    def vgrad(self, xs: Any) -> Any:
        """Gradient over a leading batch axis; implement ``_vgrad_core`` to support it."""
        return self._vgrad_core(xs)

    # Default cores. Each falls back to a public evaluator a subclass overrode (that
    # override is itself checked), so older subclasses that write public methods work.

    def _overrides(self, name: str) -> bool:
        return getattr(type(self), name) is not getattr(Functional, name)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free value core; the base falls back to an overridden ``value``."""
        if not self._overrides("value"):
            raise NotImplementedError(f"{type(self).__name__} defines neither value nor _value_core.")
        return self.value(x, *args, **kwargs)

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free gradient core: an overridden ``grad``, else a native derivative's representer."""
        if self._overrides("grad"):
            return self.grad(x, *args, **kwargs)
        if self._overrides("_derivative_core") or self._overrides("derivative_at"):
            return self._derivative_core(x, *args, **kwargs).representer
        raise NotImplementedError(f"{type(self).__name__} does not implement grad().")

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Check-free ``(value, grad)``; the base evaluates the two cores separately."""
        if self._overrides("value_and_grad"):
            return self.value_and_grad(x, *args, **kwargs)
        return self._value_core(x, *args, **kwargs), self._grad_core(x, *args, **kwargs)

    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> "LinearFunctional":
        """Check-free derivative; the base wraps the gradient core."""
        if self._overrides("derivative_at"):
            return self.derivative_at(x, *args, **kwargs)
        return self._derivative_from_gradient(self._grad_core(x, *args, **kwargs))

    def _value_and_derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, "LinearFunctional"]:
        """Check-free ``(value, derivative)``, using a native derivative when one exists."""
        if self._overrides("value_and_derivative"):
            return self.value_and_derivative(x, *args, **kwargs)
        if self._overrides("_derivative_core") or self._overrides("derivative_at"):
            return self._value_core(x, *args, **kwargs), self._derivative_core(x, *args, **kwargs)
        value, gradient = self._value_and_grad_core(x, *args, **kwargs)
        return value, self._derivative_from_gradient(gradient)

    def _differential_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free covector; the base applies the domain's Riesz map to the gradient core."""
        if self._overrides("differential"):
            return self.differential(x, *args, **kwargs)
        return self.domain.riesz(self._grad_core(x, *args, **kwargs))

    def _vvalue_core(self, xs: Any) -> Any:
        """Check-free batched-value core; the base vmaps ``value`` unless ``vvalue`` is overridden."""
        if self._overrides("vvalue"):
            return self.vvalue(xs)
        return self._vmap_vvalue(xs)

    def _vmap_vvalue(self, xs: Any) -> Any:
        # Generic batching: vmap the single-element value over the leading axis.
        _warn_vmap_fallback_once(self, "vvalue", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.value, in_axes=0, out_axes=0)(xs)

    def _vgrad_core(self, xs: Any) -> Any:
        """Check-free batched-gradient core; the base falls back to an overridden ``vgrad``."""
        if self._overrides("vgrad"):
            return self.vgrad(xs)
        raise NotImplementedError(f"{type(self).__name__} does not implement vgrad().")

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

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True,
                    out_batched_scalar=True)
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

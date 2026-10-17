from __future__ import annotations

from typing import Any

from ._base import Domain, Functional
from .._batching import (
    _batched_real_inner,
    _check_scalar_shape,
    _leading_batch_size,
    _warn_vmap_fallback_once,
)
from ._linear import LinearFunctional, InnerProductFunctional
from .._checks import checked_method
from .._checks import CheckLevel
from ..contextual import resolve_context_priority
from ..contextual import Context
from ..linop import LinOp


class QuadraticForm(Functional[Domain]):
    """
    A scalar quadratic objective on ``dom``.

    Parameters
    ----------
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    @checked_method(in_space="domain", out_space="domain")
    def hess_apply(self, x: Any) -> Any:
        """Apply the Hessian action at ``x``; implement ``_hess_apply_core`` to support it."""
        return self._hess_apply_core(x)

    def _hess_apply_core(self, x: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not define hess_apply.")

    def _vgrad_core(self, xs: Any) -> Any:
        """Evaluate the gradient core independently over leading batch axes."""
        _warn_vmap_fallback_once(self, "vgrad", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self._grad_core, in_axes=0, out_axes=0)(xs)


class LinOpQuadraticForm(QuadraticForm[Domain]):
    r"""
    Quadratic form ``1/2 <x, Q x> + linear(x) + a`` for a Hermitian ``Q``, so ``grad = Q x + ...``.

    Hermiticity is checked where the operator's structure allows; matrix-free ``Q``
    is trusted. For a non-Hermitian ``Q``, pass its Hermitian part ``0.5 * (Q + Q.H)``.
    Expanding the quadratic gives ``D(1/2 <x,Qx>)[h] = Re<Qx,h>``; for a
    real-valued objective on a complex space the linear term must also be real.
    Beck, *First-Order Methods in Optimization*, Thm 3.29, with the real pairing.

    Parameters
    ----------
    Q : LinOp
        Hermitian operator from a space to itself.
    linear : LinearFunctional or None, optional
        Linear term on ``Q.domain``.
    a : scalar-like, optional
        Constant offset.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``Q`` and ``linear`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("Q", "linear", "a", "ctx")
    _dynamic = ("Q", "linear", "a")

    def _value_core(self, x: Any) -> Any:
        # f(x) = 1/2 <x, Q x> + l(x) + a.
        qx = self.Q._apply_core(x)
        value = 0.5 * self.domain._real_inner_core(x, qx)
        if self.linear is not None:
            value = value + self.linear._value_core(x)
        return value + self.a

    def _grad_core(self, x: Any) -> Any:
        """Riesz gradient ``Q x + grad l``: ``Q`` is Hermitian, so no extra geometry enters."""
        # grad f(x) = Q x + the representer of l.
        grad = self.Q._apply_core(x)
        if self.linear is not None:
            add = self.domain._add_core
            grad = add(grad, self.linear.representer)
        return grad

    def _vvalue_core(self, xs: Any) -> Any:
        values = 0.5 * _batched_real_inner(self.domain, xs, self.Q._vapply_core(xs))
        if self.linear is not None:
            values = values + self.linear._vvalue_core(xs)
        return values + self.a

    def _vgrad_core(self, xs: Any) -> Any:
        # Q x_i + grad l, row-wise.
        grads = self.Q._vapply_core(xs)
        if self.linear is not None:
            grads = self.domain._add_batch_core(grads, self.linear._vgrad_core(xs))
        return grads

    def __init__(
        self,
        Q: LinOp[Domain, Domain],
        linear: LinearFunctional[Domain] | None = None,
        a: Any = 0,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Q must be a LinOp; linear, if given, a LinearFunctional.
        if not isinstance(Q, LinOp):
            raise TypeError(f"Q must be a LinOp, got {type(Q).__name__}.")
        if linear is not None and not isinstance(linear, LinearFunctional):
            raise TypeError(
                f"linear must be a LinearFunctional or None, got {type(linear).__name__}."
            )

        # Put everything on one context; Q must be square and structurally Hermitian.
        resolved_ctx = resolve_context_priority(ctx, Q.domain, Q, linear)
        Q = Q.convert(resolved_ctx)
        if Q.domain != Q.codomain:
            raise ValueError("LinOpQuadraticForm requires Q.domain == Q.codomain.")
        if Q.is_hermitian() is False:
            raise ValueError("LinOpQuadraticForm requires Q to be Hermitian/self-adjoint.")
        if linear is not None:
            linear = linear.convert(resolved_ctx)
            if linear.domain != Q.domain:
                raise ValueError("linear.domain must match Q.domain.")

        super().__init__(Q.domain, resolved_ctx, check_level=check_level)
        self.Q = Q
        self.linear = linear
        # The constant offset must be a scalar.
        self.codomain._check_member(a)
        self.a = self.codomain.ctx.asarray(a)
        _check_scalar_shape(self.a, ())

    def _convert(self, new_ctx: Context) -> "LinOpQuadraticForm":
        linear = None if self.linear is None else self.linear.convert(new_ctx)
        a = self.codomain.convert(new_ctx).ctx.asarray(self.a)
        return type(self)(self.Q.convert(new_ctx), linear, a, new_ctx, self.check_level)

    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> LinearFunctional:
        """Differentiate the Hermitian quadratic with the real pairing."""
        quadratic = InnerProductFunctional(
            self.Q._apply_core(x), self.domain, self.ctx, check_level=self.check_level,
        )
        return quadratic if self.linear is None else quadratic + self.linear

    def _hess_apply_core(self, x: Any) -> Any:
        """Return the Hessian action ``Q x`` under the Hermitian assumption."""
        return self.Q._apply_core(x)

from __future__ import annotations

from typing import Any

from ._base import (
    Domain,
    Functional,
    _check_scalar_shape,
    _leading_batch_size,
    _warn_vmap_fallback_once,
)
from ._linear import LinearFunctional
from .._batching import _batched_inner
from .._checks import checked_method
from .._contextual import resolve_context_priority
from ..backend import Context, jax_pytree_class
from ..linop import LinOp


class QuadraticForm(Functional[Domain]):
    """
    Represent a scalar quadratic objective on a space.

    Parameters
    ----------
    dom : Space
        Domain space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.
    """

    def hess_apply(self, x: Any) -> Any:
        """Apply the Hessian action at ``x`` when available."""
        raise NotImplementedError(f"{type(self).__name__} does not define hess_apply.")

    def grad(self, x: Any) -> Any:
        """Return the gradient with respect to ``domain.inner`` when available."""
        raise NotImplementedError(f"{type(self).__name__} does not define grad.")

    @checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True)
    def vgrad(self, xs: Any) -> Any:
        """Evaluate ``grad`` independently over leading batch axes."""
        _warn_vmap_fallback_once(self, "vgrad", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.grad, in_axes=0, out_axes=0)(xs)


@jax_pytree_class
class LinOpQuadraticForm(QuadraticForm[Domain]):
    r"""
    Represent a quadratic form backed by a linear operator.

    Assumption:
        Q is Hermitian/self-adjoint. Under this assumption,
        grad f(x) = Q x.

    Non-Hermitian operators are not supported here. If users need the
    Hermitian part, they must construct 0.5 * (Q + Q.H) explicitly.

    The full objective is ``q(x) = 1/2 * <x, Qx> + linear(x) + a`` with
    ``Q : X -> X``. Structurally available dense and diagonal operators are
    checked at construction. Matrix-free operators are not validated; correctness
    is the caller's responsibility.

    Parameters
    ----------
    Q : LinOp
        Hermitian operator from a space to itself.
    linear : LinearFunctional or None, optional
        Optional linear term on ``Q.domain``.
    a : scalar-like, optional
        Constant scalar offset. Default is 0.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``Q`` and
        ``linear``.

    Attributes
    ----------
    Q : LinOp
        Stored Hermitian operator.
    linear : LinearFunctional or None
        Stored linear term.
    a : scalar-like
        Stored scalar offset.
    """

    def __init__(
        self,
        Q: LinOp[Domain, Domain],
        linear: LinearFunctional[Domain] | None = None,
        a: Any = 0,
        ctx: Context | str | None = None,
    ) -> None:
        if not isinstance(Q, LinOp):
            raise TypeError(f"Q must be a LinOp, got {type(Q).__name__}.")
        if linear is not None and not isinstance(linear, LinearFunctional):
            raise TypeError(
                f"linear must be a LinearFunctional or None, got {type(linear).__name__}."
            )

        resolved_ctx = resolve_context_priority(ctx, Q.domain, Q, linear)
        Q = Q.convert(resolved_ctx)
        if Q.domain != Q.codomain:
            raise ValueError("LinOpQuadraticForm requires Q.domain == Q.codomain.")
        self._check_hermitian_structure(Q)
        if linear is not None:
            linear = linear.convert(resolved_ctx)
            if linear.domain != Q.domain:
                raise ValueError("linear.domain must match Q.domain.")

        super().__init__(Q.domain, resolved_ctx)
        self.Q = Q
        self.linear = linear
        self.a = self.ctx.asarray(a)
        _check_scalar_shape(self.a, ())

    @staticmethod
    def _check_hermitian_structure(Q: LinOp[Domain, Domain]) -> None:
        """Raise when ``Q`` is structurally known to be non-Hermitian."""
        result = Q.is_hermitian()
        if result is False:
            raise ValueError("LinOpQuadraticForm requires Q to be Hermitian/self-adjoint.")

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``1/2 * <x, Qx> + linear(x) + a``."""
        qx = self.Q.apply(x)
        value = 0.5 * self.domain.inner(x, qx)
        if self.linear is not None:
            value = value + self.linear.value(x)
        return value + self.a

    @checked_method(in_space="domain", out_space="domain")
    def grad(self, x: Any) -> Any:
        """
        Return the gradient with respect to ``domain.inner``.

        This is the Riesz gradient: for Euclidean geometry it is the ordinary
        coordinate gradient, while for non-Euclidean geometry it is corrected
        by the domain inner product.

        ``LinOpQuadraticForm`` assumes ``Q`` is Hermitian/self-adjoint, so the
        quadratic contribution is exactly ``Q.apply(x)``.
        """
        grad = self.Q.apply(x)
        if self.linear is not None:
            grad = self.domain.add(grad, self.linear.representer)
        return grad

    @checked_method(in_space="domain", out_space="domain")
    def hess_apply(self, x: Any) -> Any:
        """Return the Hessian action ``Q x`` under the Hermitian assumption."""
        return self.Q.apply(x)

    @checked_method(in_space="domain", in_batched=True)
    def vvalue(self, xs: Any) -> Any:
        """Evaluate the quadratic objective over a leading batch axis."""
        qxs = self.Q.vapply(xs)
        if self.domain.is_euclidean and hasattr(xs, "shape"):
            axes = tuple(range(1, len(tuple(xs.shape))))
            values = 0.5 * self.ops.sum(self.ops.conj(xs) * qxs, axis=axes)
        else:
            values = 0.5 * _batched_inner(self.domain, xs, qxs)
        if self.linear is not None:
            values = values + self.linear.vvalue(xs)
        values = values + self.a
        if self._checks_at_least("standard"):
            _check_scalar_shape(values, (_leading_batch_size(self.domain, xs),))
        return values

    @checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True)
    def vgrad(self, xs: Any) -> Any:
        """Evaluate the Riesz gradient over a leading batch axis."""
        grads = self.Q.vapply(xs)
        if self.linear is not None:
            grads = self.domain.add_batch(grads, self.linear.vgrad(xs))
        return grads

    def __eq__(self, other: Any) -> bool:
        """Return whether another quadratic form has the same stored terms."""
        if type(other) is type(self):
            return (
                self.Q == other.Q
                and self.linear == other.linear
                and self.ops.allclose(self.a, other.a)
            )
        return False

    def tree_flatten(self):
        """Flatten this quadratic form for pytree registration."""
        children = (self.Q, self.linear, self.a)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this quadratic form from pytree data."""
        Q, linear, a = children
        return cls(Q, linear, a, Q.ctx)

    def _convert(self, new_ctx: Context) -> LinOpQuadraticForm:
        """Convert stored terms to ``new_ctx``."""
        linear = None if self.linear is None else self.linear.convert(new_ctx)
        return LinOpQuadraticForm(self.Q.convert(new_ctx), linear, self.a, new_ctx)

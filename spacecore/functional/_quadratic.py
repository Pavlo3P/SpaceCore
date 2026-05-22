from __future__ import annotations

from typing import Any

from ._base import Domain, Functional
from ._linear import LinearFunctional
from .._contextual.manager import ctx_manager
from ..backend import Context, jax_pytree_class
from ..linop import LinOp
from ..space import Space


class QuadraticForm(Functional[Domain]):
    """Scalar quadratic objective on a space."""

    def hess_apply(self, x: Any) -> Any:
        """Apply the Hessian action at ``x`` when available."""
        raise NotImplementedError(f"{type(self).__name__} does not define hess_apply.")

    def grad(self, x: Any) -> Any:
        """Gradient at ``x`` when available."""
        raise NotImplementedError(f"{type(self).__name__} does not define grad.")

    def vgrad(self, xs: Any, batch_space: Space | None = None) -> Any:
        """Evaluate ``grad`` independently over leading batch axes."""
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        batch_shape = self._require_leading_batch_axes(in_space)
        if self._enable_checks:
            in_space._check_member(xs)
        grads = self._vmap_leading(self.grad, len(batch_shape))(xs)
        if self._enable_checks:
            self._output_batch_space(self.domain, in_space)._check_member(grads)
        return grads


@jax_pytree_class
class LinOpQuadraticForm(QuadraticForm[Domain]):
    """
    Quadratic form backed by a linear operator.

    ``q(x) = 1/2 * <x, Qx> + linear(x) + a`` with ``Q : X -> X``.
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

        resolved_ctx = ctx_manager.resolve_context_priority(ctx, Q.domain, Q, linear)
        Q = Q.convert(resolved_ctx)
        if Q.domain != Q.codomain:
            raise ValueError("LinOpQuadraticForm requires Q.domain == Q.codomain.")
        if linear is not None:
            linear = linear.convert(resolved_ctx)
            if linear.domain != Q.domain:
                raise ValueError("linear.domain must match Q.domain.")

        super().__init__(Q.domain, resolved_ctx)
        self.Q = Q
        self.linear = linear
        self.a = self.ctx.asarray(a)
        if self._enable_checks:
            self._check_scalar_batch(self.a, ())

    def value(self, x: Any) -> Any:
        """Return ``1/2 * <x, Qx> + linear(x) + a``."""
        if self._enable_checks:
            self.domain._check_member(x)
        qx = self.Q.apply(x)
        value = 0.5 * self.domain.inner(x, qx)
        if self.linear is not None:
            value = value + self.linear.value(x)
        return value + self.a

    def grad(self, x: Any) -> Any:
        """
        Return the Euclidean/Riesz gradient.

        The quadratic part uses the symmetric adjoint part ``(Q + Q*) / 2``.
        For self-adjoint ``Q`` this is exactly ``Qx``.
        """
        if self._enable_checks:
            self.domain._check_member(x)
        qx = self.Q.apply(x)
        qhx = self.Q.rapply(x)
        grad = self.domain.scale(0.5, self.domain.add(qx, qhx))
        if self.linear is not None:
            grad = self.domain.add(grad, self.linear.representer)
        return grad

    def hess_apply(self, x: Any) -> Any:
        """Return the self-adjoint Hessian action ``(Q + Q*) x / 2``."""
        if self._enable_checks:
            self.domain._check_member(x)
        qx = self.Q.apply(x)
        qhx = self.Q.rapply(x)
        return self.domain.scale(0.5, self.domain.add(qx, qhx))

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return (
                self.Q == other.Q
                and self.linear == other.linear
                and self.ops.allclose(self.a, other.a)
            )
        return False

    def tree_flatten(self):
        children = (self.Q, self.linear, self.a)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        Q, linear, a = children
        return cls(Q, linear, a, Q.ctx)

    def _convert(self, new_ctx: Context) -> LinOpQuadraticForm:
        linear = None if self.linear is None else self.linear.convert(new_ctx)
        return LinOpQuadraticForm(self.Q.convert(new_ctx), linear, self.a, new_ctx)

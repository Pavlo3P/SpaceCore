from __future__ import annotations

from typing import Any, cast

from ._base import Functional
from ._linear import InnerProductFunctional, LinearFunctional
from ._quadratic import LinOpQuadraticForm
from .._checks import checked_method
from .._checks import CheckLevel, minimum_check_level
from ..contextual import Context
from ..linop import LinOp
from ..mapping import register


def _require_composable(F: Functional, A: LinOp) -> None:
    """Raise unless ``F`` can be composed with ``A``."""
    # F must be a Functional, A a LinOp, and A must land in F's domain.
    if not isinstance(F, Functional):
        raise TypeError(f"F must be a Functional, got {type(F).__name__}.")
    if not isinstance(A, LinOp):
        raise TypeError(f"A must be a LinOp, got {type(A).__name__}.")
    if A.codomain != F.domain:
        raise ValueError(
            "Functional composition requires A.codomain == F.domain; "
            f"got {A.codomain!r} and {F.domain!r}."
        )


def make_functional_composed(F: Functional, A: LinOp) -> Functional:
    """
    Return the pull-back ``F o A``, specialized where a closed form exists.

    Parameters
    ----------
    F : Functional
        Functional on ``A.codomain``.
    A : LinOp
        Operator into ``F.domain``.

    Returns
    -------
    Functional
        The pull-back.
    """
    _require_composable(F, A)
    # Closed forms: <c, A x> = <A* c, x>, and a quadratic form pulls back to A* Q A.
    if isinstance(F, InnerProductFunctional):
        return InnerProductFunctional(A.H.apply(F.representer), A.domain, A.ctx,
                                      cod=F.codomain, real_linear=F.real_linear,
                                      check_level=minimum_check_level((F.check_level, A.check_level)))
    if isinstance(F, LinOpQuadraticForm):
        Q = A.H @ F.Q @ A
        linear = None if F.linear is None else cast(Any, F.linear.compose(A))
        return LinOpQuadraticForm(Q, linear, F.a, A.ctx)
    if isinstance(F, LinearFunctional):
        return _ComposedLinearFunctional(F, A)
    # Otherwise a lazy composition.
    return ComposedFunctional(F, A)


class ComposedFunctional(Functional):
    """
    Generic pull-back of a functional through a linear operator.

    ``ComposedFunctional(F, A)`` represents ``x -> F(A x)`` on ``A.domain``.

    Parameters
    ----------
    F : Functional
        Functional defined on ``A.codomain``.
    A : LinOp
        Linear operator whose codomain is ``F.domain``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _value_core(self, x: Any) -> Any:
        return self.F._value_core(self.A._apply_core(x))

    def _grad_core(self, x: Any) -> Any:
        # Chain rule in Riesz form: grad(F o A)(x) = A^#(grad F(A x)).
        # ``rapply`` IS the metric adjoint A^# (ADR-009), so no extra Riesz map is
        # applied here; inserting one would count the geometry twice.
        return self.A._rapply_core(self.F._grad_core(self.A._apply_core(x)))

    def _vgrad_core(self, xs: Any) -> Any:
        return self.A._rvapply_core(self.F._vgrad_core(self.A._vapply_core(xs)))

    def __init__(
        self,
        F: Functional,
        A: LinOp,
        check_level: CheckLevel | None = None,
    ) -> None:
        _require_composable(F, A)
        # The pull-back lives on A's domain and keeps F's scalar codomain.
        if check_level is None:
            check_level = minimum_check_level((F.check_level, A.check_level))
        super().__init__(A.domain, A.ctx, check_level=check_level, cod=F.codomain)
        self.F = F.convert(A.ctx)
        self.A = A

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any) -> Any:
        """
        Evaluate ``F(A x)``.

        Parameters
        ----------
        x:
            Element of ``A.domain``.

        Returns
        -------
        Any
            Scalar-like value returned by the composed functional.
        """
        return self._value_core(x)

    @checked_method(in_space="domain", out_space="domain")
    def grad(self, x: Any) -> Any:
        r"""
        Return :math:`\nabla (F \circ A)(x) = A^{\#} \nabla F(A x)`.

        ``A.rapply`` is already the metric adjoint, so no Riesz map is added. Raises
        :class:`NotImplementedError` when ``F`` has no gradient. Indeed,
        ``D(F o A)[h] = DF[Ah] = <grad F, Ah> = <A^# grad F, h>``; take real parts
        for real-valued objectives on complex spaces. Beck, *First-Order Methods in
        Optimization*, Thm 3.29 (gradient pairing).

        Parameters
        ----------
        x : Any
            Element of ``A.domain``.

        Returns
        -------
        Any
            The gradient, in ``A.domain``.
        """
        return self._grad_core(x)

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(F(Ax), A^#(grad F(Ax)))`` from a single application of ``A``.

        The default base implementation would call ``value`` and ``grad``
        separately and therefore apply ``A`` twice; here the image ``A x`` is
        computed once and shared, which is the point of the fused path for a
        composition.
        """
        y = self.A.apply(x)
        value, gradient = self.F.value_and_grad(y, *args, **kwargs)
        return value, self.A.rapply(gradient)

    def derivative_at(self, x: Any, *args: Any, **kwargs: Any) -> LinearFunctional:
        return self.F.derivative_at(self.A.apply(x), *args, **kwargs).compose(self.A)

    def value_and_derivative(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, LinearFunctional]:
        value, derivative = self.F.value_and_derivative(self.A.apply(x), *args, **kwargs)
        return value, derivative.compose(self.A)

    @checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True)
    def vgrad(self, xs: Any) -> Any:
        """Evaluate the chain-rule gradient over a leading batch axis."""
        return self._vgrad_core(xs)

    def __eq__(self, other: Any) -> bool:
        """Return whether another composed functional has the same operands."""
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        return self.F == other.F and self.A == other.A

    def _repr_body(self) -> str:
        return f"{self.F._short_repr()} ∘ {self.A._short_repr()}"

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        children = (self.F, self.A)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        F, A = children
        return cls(F, A)

    def _convert(self, new_ctx: Context) -> ComposedFunctional:
        """Convert the composed functional and operator to ``new_ctx``."""
        return type(self)(self.F.convert(new_ctx), self.A.convert(new_ctx))


class _ComposedLinearFunctional(ComposedFunctional, LinearFunctional):
    """Pullback that needs only forward application for directional evaluation."""

    derivative_at = LinearFunctional.derivative_at

    @property
    def scalar_field(self) -> str:
        return "real" if self.domain.scalar_field == "real" else self.F.scalar_field

    @property
    def representer(self) -> Any:
        return self.A.rapply(self.F.representer)


register("matmul", Functional, LinOp, make_functional_composed)

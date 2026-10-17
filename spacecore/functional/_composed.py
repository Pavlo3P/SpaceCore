from __future__ import annotations

from typing import Any, cast

from ._base import Functional
from ._linear import InnerProductFunctional, LinearFunctional
from ._quadratic import LinOpQuadraticForm
from .._checks import CheckLevel, minimum_check_level
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

    _fields = ("F", "A")
    _dynamic = ("F", "A")

    def _value_core(self, x: Any) -> Any:
        return self.F._value_core(self.A._apply_core(x))

    def _grad_core(self, x: Any) -> Any:
        r"""Chain rule in Riesz form: :math:`\nabla (F \circ A)(x) = A^{\#} \nabla F(A x)`.

        ``A.rapply`` IS the metric adjoint ``A^#`` (ADR-009), so no extra Riesz map is
        applied; inserting one would count the geometry twice. Indeed
        ``D(F o A)[h] = DF[Ah] = <grad F, Ah> = <A^# grad F, h>``; take real parts for
        real-valued objectives on complex spaces. Beck, *First-Order Methods in
        Optimization*, Thm 3.29 (gradient pairing).
        """
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
        super().__init__(A.domain, A.ctx, check_level=check_level)
        self.F = F.convert(A.ctx)
        self.A = A

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """``(F(Ax), A^#(grad F(Ax)))`` from a single application of ``A``.

        Evaluating the value and gradient cores separately would apply ``A`` twice;
        here the image ``A x`` is computed once and shared.
        """
        y = self.A._apply_core(x)
        value, gradient = self.F._value_and_grad_core(y, *args, **kwargs)
        return value, self.A._rapply_core(gradient)

    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> LinearFunctional:
        return self.F._derivative_core(self.A._apply_core(x), *args, **kwargs).compose(self.A)

    def _value_and_derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, LinearFunctional]:
        value, derivative = self.F._value_and_derivative_core(self.A._apply_core(x), *args, **kwargs)
        return value, derivative.compose(self.A)

    def _repr_body(self) -> str:
        return f"{self.F._short_repr()} ∘ {self.A._short_repr()}"

class _ComposedLinearFunctional(ComposedFunctional, LinearFunctional):
    """Pullback that needs only forward application for directional evaluation."""

    _derivative_core = LinearFunctional._derivative_core
    _value_and_derivative_core = Functional._value_and_derivative_core


    @property
    def representer(self) -> Any:
        return self.A.rapply(self.F.representer)


register("matmul", Functional, LinOp, make_functional_composed)

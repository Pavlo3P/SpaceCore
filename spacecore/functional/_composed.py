from __future__ import annotations

from typing import Any

from ._base import Functional
from ._linear import InnerProductFunctional
from ._quadratic import LinOpQuadraticForm
from .._checks import checked_method
from ..backend import Context, jax_pytree_class
from ..linop import LinOp


def _require_composable(F: Functional, A: LinOp) -> None:
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
    Return the pull-back ``F o A`` with local specializations.

    Parameters
    ----------
    F:
        Functional defined on ``A.codomain``.
    A:
        Linear operator whose codomain is ``F.domain``.

    Returns
    -------
    Functional
        Specialized pull-back when available, otherwise
        :class:`ComposedFunctional`.
    """
    _require_composable(F, A)
    if isinstance(F, InnerProductFunctional):
        return InnerProductFunctional(A.H.apply(F.representer), A.domain, A.ctx)
    if isinstance(F, LinOpQuadraticForm):
        Q = A.H @ F.Q @ A
        linear = None if F.linear is None else F.linear.compose(A)
        return LinOpQuadraticForm(Q, linear, F.a, A.ctx)
    return ComposedFunctional(F, A)


@jax_pytree_class
class ComposedFunctional(Functional):
    """
    Generic pull-back of a functional through a linear operator.

    ``ComposedFunctional(F, A)`` represents ``x -> F(A x)`` on ``A.domain``.
    """

    def __init__(self, F: Functional, A: LinOp) -> None:
        _require_composable(F, A)
        super().__init__(A.domain, A.ctx)
        self.F = F.convert(A.ctx)
        self.A = A

    @checked_method(in_space="domain")
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
        return self.F.value(self.A.apply(x))

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.F == other.F and self.A == other.A
        return False

    def tree_flatten(self):
        children = (self.F, self.A)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        F, A = children
        return cls(F, A)

    def _convert(self, new_ctx: Context) -> ComposedFunctional:
        return ComposedFunctional(self.F.convert(new_ctx), self.A.convert(new_ctx))

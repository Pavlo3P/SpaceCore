from __future__ import annotations

from typing import Any, cast

from ._base import Functional
from ._linear import InnerProductFunctional
from ._quadratic import LinOpQuadraticForm
from .._checks import checked_method
from .._check_policy import CheckLevel, minimum_check_level
from ..contextual import Context
from ..kernels import core_kernels
from ..linop import LinOp


def _require_composable(F: Functional, A: LinOp) -> None:
    """Raise unless ``F`` can be composed with ``A``."""
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
    F : Functional
        Functional defined on ``A.codomain``.
    A : LinOp
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
        linear = None if F.linear is None else cast(Any, F.linear.compose(A))
        return LinOpQuadraticForm(Q, linear, F.a, A.ctx)
    return ComposedFunctional(F, A)


@core_kernels("composed-functional")
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
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.
    """

    def __init__(
        self,
        F: Functional,
        A: LinOp,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        _require_composable(F, A)
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
        Return the Riesz gradient of ``F o A`` by the chain rule.

        For :math:`G = F \circ A`, differentiating gives
        :math:`DG(x)[h] = DF(Ax)[Ah] = \langle \nabla F(Ax), Ah\rangle_Y`.
        Moving ``A`` across the pairing with the adjoint's defining identity
        :math:`\langle Au, v\rangle_Y = \langle u, A^{\#}v\rangle_X` (and
        conjugate symmetry) turns that into
        :math:`\langle A^{\#}\nabla F(Ax), h\rangle_X`, so

        .. math::

            \nabla (F \circ A)(x) = A^{\#}\, \nabla F(A x).

        ``LinOp.rapply`` **is** :math:`A^{\#}`, the metric adjoint (ADR-009), so
        the geometry of both spaces is already accounted for; applying a Riesz
        map on top would count it twice. Correspondingly there is no explicit
        conjugation here — the adjoint identity absorbs it.

        Raises :class:`NotImplementedError` when the inner ``F`` has no gradient.

        Parameters
        ----------
        x:
            Element of ``A.domain``.

        Returns
        -------
        Any
            Riesz gradient in ``A.domain``.
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
        return ComposedFunctional(self.F.convert(new_ctx), self.A.convert(new_ctx))

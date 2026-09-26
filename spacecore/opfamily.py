r"""
The functional-weighted map ``m(x) = F(x) A x``: a :class:`~spacecore.Mapping` that is not linear.

``at(x)`` is the frozen member ``F(x) A``, with ``at(x).apply(x) == m(x)``;
``linearize_at(x)`` is the derivative ``Dm(x)``. The two agree only for constant ``F``.
"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from ._checks import checked_method
from ._check_policy import CheckLevel, minimum_check_level
from .contextual import Context
from .functional import Functional
from .linop import LinOp, MatrixFreeLinOp
from .mapping import Mapping, register
from .linop._algebra import make_scaled
from .space import Space

Domain = TypeVar("Domain", bound=Space)
Codomain = TypeVar("Codomain", bound=Space)


class FunctionalScaledOperator(Mapping, Generic[Domain, Codomain]):
    r"""
    The functional-weighted map :math:`m(x) = F(x)\, A x`; ``F`` and ``A`` share a domain.

    Parameters
    ----------
    functional : Functional
        Weight ``F`` on ``op.domain``.
    op : LinOp
        Operator ``A``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the less strict of the two operands' when omitted.
    """

    def __init__(
        self,
        functional: Functional,
        op: LinOp,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        if not isinstance(functional, Functional):
            raise TypeError(
                f"functional must be a Functional, got {type(functional).__name__}."
            )
        if not isinstance(op, LinOp):
            raise TypeError(f"op must be a LinOp, got {type(op).__name__}.")
        if functional.domain != op.domain:
            raise ValueError(
                "FunctionalScaledOperator requires functional.domain == op.domain; "
                f"got {functional.domain!r} and {op.domain!r}."
            )
        if check_level is None:
            check_level = minimum_check_level((functional.check_level, op.check_level))
        self.dom, self.cod = self._bind_context(
            op.ctx, op.domain, op.codomain, check_level=check_level
        )
        self.functional = functional.convert(self.ctx)
        self.op = op.convert(self.ctx)

    @property
    def domain(self) -> Domain:
        """Domain space ``X``: the index set of the family and each member's domain."""
        return self.dom

    @property
    def codomain(self) -> Codomain:
        """Codomain space ``Y`` of each member."""
        return self.cod

    def __call__(self, x: Any) -> Any:
        """Evaluate the map at ``x``."""
        return self.apply(x)

    def _arrow(self) -> str:
        """Return the ``domain ⇝ codomain`` descriptor (``⇝``: not linear)."""
        from ._repr import describe_space

        return f"{describe_space(self.dom)} ⇝ {describe_space(self.cod)}"

    @checked_method(in_space="domain")
    def at(self, x: Any) -> LinOp:
        """Return the frozen member ``F(x) · A``."""
        return make_scaled(self.functional.value(x), self.op)

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``F(x) · (A x)`` without building the intermediate node."""
        return self.codomain.scale(self.functional.value(x), self.op.apply(x))

    @checked_method(in_space="domain")
    def linearize_at(self, x: Any) -> LinOp:
        r"""
        Return the derivative :math:`Dm(x)[h] = \langle \nabla F(x), h\rangle A x + F(x) A h` as a ``LinOp``.

        It differs from :meth:`at` by the rank-one first term. Its adjoint is
        :math:`\langle Ax, w\rangle_Y \nabla F(x) + \overline{F(x)} A^{\#} w`, in the
        codomain geometry.
        """
        X, Y = self.domain, self.codomain
        value, gradient = self.functional.value_and_grad(x)
        ax = self.op.apply(x)
        conj_value = self.ops.conj(value)

        def _apply(h: Any) -> Any:
            return Y.add(
                Y.scale(X.inner(gradient, h), ax),
                Y.scale(value, self.op.apply(h)),
            )

        def _rapply(w: Any) -> Any:
            return X.add(
                X.scale(Y.inner(ax, w), gradient),
                X.scale(conj_value, self.op.rapply(w)),
            )

        # ``_rapply`` is already the *metric* adjoint: it pairs with ``Y.inner``
        # and defers to ``self.op.rapply`` (itself metric-aware). Stating that
        # explicitly documents the choice and silences the construction advisory.
        return MatrixFreeLinOp(
            _apply, _rapply, X, Y, self.ctx, euclidean_adjoint=False
        )

    def __eq__(self, other: Any) -> bool:
        """Return whether another family has the same weight and operator."""
        if not self.same_math(other):
            return NotImplemented
        return self.functional == other.functional and self.op == other.op

    def tree_flatten(self):
        """Flatten this family for pytree registration."""
        return (self.functional, self.op), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this family from pytree data."""
        functional, op = children
        return cls(functional, op)

    def _convert(self, new_ctx: Context) -> "FunctionalScaledOperator":
        """Convert both operands to ``new_ctx``."""
        return FunctionalScaledOperator(
            self.functional.convert(new_ctx), self.op.convert(new_ctx)
        )

    def _repr_body(self) -> str:
        return f"{self.functional._short_repr()} · {self.op._short_repr()}, {self._arrow()}"


def make_functional_scaled_operator(functional: Functional, op: LinOp) -> Any:
    """
    Return ``F * A``: a ``ScaledLinOp`` when ``F`` is a constant (or zero) functional, else a :class:`FunctionalScaledOperator`.

    Parameters
    ----------
    functional : Functional
        Weight ``F``.
    op : LinOp
        Operator ``A``.

    Returns
    -------
    LinOp or FunctionalScaledOperator
        The product.
    """
    from .functional import ConstantFunctional, ZeroFunctional

    if not isinstance(functional, Functional):
        raise TypeError(
            f"functional must be a Functional, got {type(functional).__name__}."
        )
    if not isinstance(op, LinOp):
        raise TypeError(f"op must be a LinOp, got {type(op).__name__}.")
    if functional.domain != op.domain:
        raise ValueError(
            "F · A requires functional.domain == op.domain; "
            f"got {functional.domain!r} and {op.domain!r}."
        )
    if isinstance(functional, ZeroFunctional):
        return make_scaled(0, op)
    if isinstance(functional, ConstantFunctional):
        return make_scaled(functional.constant, op)
    return FunctionalScaledOperator(functional, op)


register("mul", Functional, LinOp, make_functional_scaled_operator)
register("mul", LinOp, Functional, lambda op, f: make_functional_scaled_operator(f, op))

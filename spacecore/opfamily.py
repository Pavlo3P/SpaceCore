r"""Point-indexed families of linear operators, and the functional-weighted map.

A :class:`~spacecore.linop.LinOp` is a *single* linear map. Several constructions
that look like operators are really a **family** of them, one per point of the
domain: the object is linear once a point is fixed, but the point-to-operator
assignment is not.

The motivating case is the functional-weighted map

.. math::

    m(x) = F(x) \, A x,

with ``F`` a scalar-valued :class:`~spacecore.functional.Functional` and ``A`` a
linear operator. This is **not** linear in ``x`` — both the scale ``F(x)`` and the
direction ``A x`` move with ``x`` — so it cannot be a ``LinOp`` without breaking
the contract that ``rapply``, ``compose_chain``, and the kernel fusion rules rely
on. It *is* linear the moment ``x`` is frozen, and that is exactly what
:meth:`OperatorFamily.at` returns.

Two different linear operators are attached to each point, and conflating them is
the easy mistake:

* :meth:`OperatorFamily.at` — the **frozen member** :math:`A_x`. For the
  functional-weighted family that is ``F(x) · A``, so ``at(x).apply(x) == m(x)``.
* :meth:`OperatorFamily.linearize_at` — the **derivative** :math:`Dm(x)`, the
  linear map ``h -> Dm(x)[h]`` used by Newton/Gauss-Newton. These coincide only
  when the family is constant (i.e. ``F`` is constant).

This module deliberately sits outside both ``linop`` and ``functional``: it
depends on both, and neither depends on it.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from ._checks import checked_method
from ._check_policy import CheckLevel, minimum_check_level
from .backend import PyTreeNode
from .contextual import Context, ContextBound
from .functional import Functional
from .linop import LinOp, MatrixFreeLinOp
from .linop._algebra import make_scaled
from .space import Space

Domain = TypeVar("Domain", bound=Space)
Codomain = TypeVar("Codomain", bound=Space)


class OperatorFamily(PyTreeNode, ContextBound, Generic[Domain, Codomain]):
    r"""
    A point-indexed family of linear operators :math:`x \mapsto A_x \in L(X, Y)`.

    Not a :class:`~spacecore.linop.LinOp`, and deliberately not a subclass of one:
    the map :meth:`apply` is non-linear in general. A ``LinOp`` is the special
    case of a *constant* family, and every member of a family is a genuine
    ``LinOp`` obtained with :meth:`at`.

    Subclasses implement :meth:`at`. Everything else is derived from it, though
    subclasses may override :meth:`apply` with a fused evaluation that avoids
    building the intermediate node.

    Parameters
    ----------
    dom : Space
        Domain space ``X`` — both the index set of the family and the domain of
        each member.
    cod : Space
        Codomain space ``Y`` of each member.
    ctx : Context, str, or None, optional
        Backend context specification.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this object. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used.
    """

    def __init__(
        self,
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        self.dom, self.cod = self._bind_context(ctx, dom, cod, check_level=check_level)

    @property
    def domain(self) -> Domain:
        """Domain space of every member of this family."""
        return self.dom

    @property
    def codomain(self) -> Codomain:
        """Codomain space of every member of this family."""
        return self.cod

    @abstractmethod
    def at(self, x: Any) -> LinOp:
        """Return the family member at ``x`` — a genuine :class:`LinOp`.

        Freezing the index point is what recovers linearity, so the result
        supports the whole operator algebra: ``@``, ``+``, ``.H``, ``fuse()``.
        """

    def apply(self, x: Any) -> Any:
        """Evaluate the map ``x -> A_x x`` at ``x``.

        Note the double role of ``x``: it selects the member *and* is the vector
        the member is applied to. That coupling is precisely what makes this
        non-linear. To apply one member to a *different* vector, use
        ``family.at(x).apply(h)``.
        """
        return self.at(x).apply(x)

    def __call__(self, x: Any) -> Any:
        """Evaluate the map at ``x``."""
        return self.apply(x)

    def linearize_at(self, x: Any) -> LinOp:
        r"""Return the derivative :math:`Dm(x)` as a :class:`LinOp`.

        The Fréchet derivative of ``m(x) = A_x x`` at a fixed ``x``, i.e. the
        linear map ``h -> Dm(x)[h]``. This is **not** :meth:`at`: ``at(x)``
        ignores how the family varies, while ``linearize_at(x)`` accounts for it.
        They agree exactly when the family is constant.

        Override in subclasses that can differentiate; the base raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement linearize_at()."
        )

    def _arrow(self) -> str:
        """Return the ``domain ⇝ codomain`` descriptor (``⇝``: not linear)."""
        from ._repr import describe_space

        return f"{describe_space(self.dom)} ⇝ {describe_space(self.cod)}"

    def _repr_body(self) -> str:
        return self._arrow()

    def _short_repr(self) -> str:
        """Return a bounded ``ClassName(domain ⇝ codomain)`` form for nesting."""
        return f"{type(self).__name__}({self._arrow()})"


class FunctionalScaledOperator(OperatorFamily[Domain, Codomain]):
    r"""
    The functional-weighted map :math:`m(x) = F(x)\, A x`.

    ``F`` and ``A`` must share a domain. The family member at ``x`` is the
    ordinary scalar multiple ``F(x) · A``, built through
    :func:`~spacecore.linop.make_scaled`, so it folds and canonicalizes like any
    other scaled operator.

    Parameters
    ----------
    functional : Functional
        Scalar weight ``F``, defined on ``op.domain``.
    op : LinOp
        Linear operator ``A``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy. When omitted, the least-strict of the two
        operands is used, matching the functional algebra's combinator rule.

    Attributes
    ----------
    functional : Functional
        The scalar weight.
    op : LinOp
        The scaled operator.
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
        super().__init__(op.domain, op.codomain, op.ctx, check_level=check_level)
        self.functional = functional.convert(self.ctx)
        self.op = op.convert(self.ctx)

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
        r"""Return the derivative of ``m(x) = F(x) A x`` at ``x``.

        By the product rule, for a direction ``h``

        .. math::

            Dm(x)[h] = DF(x)[h] \; A x \;+\; F(x)\, A h
                     = \langle \nabla F(x), h\rangle \, Ax + F(x)\, A h,

        using the library's Riesz convention ``<grad F(x), h> = DF(x)[h]``. The
        first term is a **rank-one** operator (its output is always a multiple of
        the fixed vector ``Ax``), the second the frozen member — so the
        derivative and :meth:`at` differ by exactly that rank-one correction, and
        coincide when ``F`` is constant.

        The adjoint follows from
        :math:`\langle Dm(x)h, w\rangle_Y = \langle h, Dm(x)^{\#}w\rangle_X`:

        .. math::

            Dm(x)^{\#}[w] = \langle Ax, w\rangle_Y \, \nabla F(x)
                          + \overline{F(x)}\, A^{\#} w .

        Both coefficients are inner products in the *codomain* geometry, not
        coordinate dot products, so this is the metric adjoint (ADR-009).
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
    Return ``F · A`` as a :class:`LinOp` when possible, else as a family.

    A :class:`~spacecore.functional.ConstantFunctional` weight is a constant
    family, so the result collapses to an ordinary
    :class:`~spacecore.linop.ScaledLinOp` — the linear case is not forced through
    the non-linear type. Everything else builds a
    :class:`FunctionalScaledOperator`.

    As in the functional algebra, the collapse is **structural**: a functional
    that merely happens to be constant is not recognized, because that is a fact
    about values, not about the expression.

    Parameters
    ----------
    functional : Functional
        Scalar weight ``F``.
    op : LinOp
        Linear operator ``A``.

    Returns
    -------
    LinOp or FunctionalScaledOperator
        ``ScaledLinOp`` for a constant weight, otherwise the family.
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

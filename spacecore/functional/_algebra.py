"""Lazy functional algebra: scalar multiples, sums, and pointwise products.

Mirrors ``linop/_algebra.py`` for the parts they share. `Functional` gains the
additive/scalar algebra `LinOp` already has, so objectives compose as ``a * F``,
``F + G``, ``F - G``, ``-F``. The operator overloads live on
:class:`~spacecore.functional.Functional` and delegate to the ``make_*`` factories
here, which do local canonicalization (fold nested scalars, flatten nested sums).

Unlike the operator algebra this one is also **multiplicative**: functionals are
scalar-valued, so ``F * G`` is the pointwise product
(:class:`ProductFunctional`), with :class:`ConstantFunctional` as the embedding
of a plain scalar. Multiplying by a constant functional folds back to
:class:`ScaledFunctional` — scaling is the constant-factor case of a product.
Canonicalization stays *structural*: it reads node types, never values, so a
functional that merely happens to be constant is not recognized as one.

A functional's ``grad`` is a *metric (Riesz) gradient* -- an element of the domain
``X`` -- so the algebra combines child gradients through the domain's own vector
ops (``X.add`` / ``X.scale``), never raw ``+`` / ``*`` (which would be wrong on a
tree/stacked domain, where an element is a pytree).
"""
from __future__ import annotations

from typing import Any

from ._base import Functional
from .._checks import checked_method
from .._check_policy import CheckLevel, minimum_check_level
from ..contextual import Context
from .._lazy_algebra import (
    finalize_sum,
    flatten_sum,
    fold_scaled,
    is_scalar_like as is_scalar_like,  # re-exported for functional/_base.py
    scalar_eq,
)


def _require_same_domain(terms: Any, node: str = "SumFunctional") -> None:
    """Raise unless every functional in ``terms`` shares the first term's domain.

    Domain equality folds in the backend/dtype context, so this also rejects a
    same-shape space on a different backend or dtype.

    Parameters
    ----------
    terms : sequence of Functional
        Operands to compare.
    node : str, optional
        Node name used in the error message; sums and products share this check.
    """
    domain = terms[0].domain
    for i, term in enumerate(terms[1:], start=1):
        if term.domain != domain:
            raise ValueError(
                f"All {node} operands must have the same domain; operand 0 has "
                f"domain {domain!r}, operand {i} has domain {term.domain!r}."
            )


class ScaledFunctional(Functional):
    """
    Lazy scalar multiple ``scalar * functional``.

    Parameters
    ----------
    scalar : scalar-like
        Scalar coefficient.
    functional : Functional
        Functional to scale.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.
    """

    def __init__(
        self,
        scalar: Any,
        functional: Functional,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        if not isinstance(functional, Functional):
            raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
        if not is_scalar_like(scalar):
            raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
        # Default the policy from the OPERAND, not from its domain space, so the
        # result inherits the least-strict operand independently of order.
        if check_level is None:
            check_level = functional.check_level
        super().__init__(functional.domain, functional.ctx, check_level=check_level)
        self.scalar = scalar
        self.functional = functional.convert(self.ctx)

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return ``scalar * functional.value(x)``."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free scaled value."""
        return self.scalar * self.functional._value_core(x, *args, **kwargs)

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the Riesz gradient ``conj(scalar) * functional.grad(x)``.

        The value scales by ``scalar`` but the metric gradient scales by its
        conjugate: the domain inner product conjugates its first argument, so
        ``<conj(a) g, h> = a <g, h>`` recovers ``D(a F)(x)[h]`` (mirrors
        ``ScaledLinOp.rapply``). For a real scalar this is just ``scalar``.
        """
        conj_scalar = self.ops.conj(self.scalar)
        return self.domain.scale(conj_scalar, self.functional.grad(x, *args, **kwargs))

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(scalar * value, conj(scalar) * grad)`` from one fused child eval."""
        value, grad = self.functional.value_and_grad(x, *args, **kwargs)
        conj_scalar = self.ops.conj(self.scalar)
        return self.scalar * value, self.domain.scale(conj_scalar, grad)

    def __eq__(self, other: Any) -> bool:
        """Return whether another scaled functional has the same scalar and operand."""
        if not self.same_math(other):
            return NotImplemented
        return scalar_eq(self.scalar, other.scalar) and self.functional == other.functional

    def tree_flatten(self):
        """Flatten this functional for pytree registration (scalar is a traced child)."""
        return (self.scalar, self.functional), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        scalar, functional = children
        return cls(scalar, functional)

    def _convert(self, new_ctx: Context) -> "ScaledFunctional":
        """Convert the scaled functional to ``new_ctx``."""
        return ScaledFunctional(self.scalar, self.functional.convert(new_ctx))


def make_scaled_functional(scalar: Any, functional: Functional) -> Functional:
    """
    Return a locally simplified scalar multiple of a functional.

    Unit scalars pass ``functional`` through unchanged and nested
    :class:`ScaledFunctional` nodes fold into a single scalar; no other
    simplification is attempted.

    Parameters
    ----------
    scalar : scalar-like
        Scalar coefficient.
    functional : Functional
        Functional to scale.

    Returns
    -------
    Functional
        Simplified scalar multiple.
    """
    if not isinstance(functional, Functional):
        raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
    if not is_scalar_like(scalar):
        raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")

    return fold_scaled(
        scalar,
        functional,
        is_zero=lambda f: isinstance(f, ZeroFunctional),
        unwrap_scaled=lambda f: (
            (f.scalar, f.functional) if isinstance(f, ScaledFunctional) else None
        ),
        make_zero=lambda: ZeroFunctional(functional.domain, functional.ctx),
        make_scaled_node=ScaledFunctional,
    )


class SumFunctional(Functional):
    """
    Lazy sum of finitely many functionals on a common domain.

    Parameters
    ----------
    terms : sequence of Functional
        Nonempty sequence of functionals sharing one domain.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.
    """

    def __init__(self, terms: Any, check_level: CheckLevel | bool | None = None) -> None:
        parts = tuple(terms)
        if not parts:
            raise ValueError(
                "SumFunctional requires a nonempty sequence of Functional operands."
            )
        for i, term in enumerate(parts):
            if not isinstance(term, Functional):
                raise TypeError(f"operand {i} must be a Functional, got {type(term).__name__}.")
        _require_same_domain(parts)
        if check_level is None:
            check_level = minimum_check_level(tuple(term.check_level for term in parts))
        super().__init__(parts[0].domain, parts[0].ctx, check_level=check_level)
        self.terms = tuple(term.convert(self.ctx) for term in parts)

    @property
    def parts(self) -> tuple[Functional, ...]:
        """Return the summed terms in order."""
        return self.terms

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the sum of the term values at ``x``."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free sum of term values."""
        total = None
        for term in self.terms:
            value = term._value_core(x, *args, **kwargs)
            total = value if total is None else total + value
        return total

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the domain-sum of the term gradients (combined via ``X.add``)."""
        domain = self.domain
        terms = iter(self.terms)
        total = next(terms).grad(x, *args, **kwargs)
        for term in terms:
            total = domain.add(total, term.grad(x, *args, **kwargs))
        return total

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(sum values, domain-sum grads)`` from one fused pass per term."""
        domain = self.domain
        value_total = None
        grad_total = None
        for term in self.terms:
            value, grad = term.value_and_grad(x, *args, **kwargs)
            value_total = value if value_total is None else value_total + value
            grad_total = grad if grad_total is None else domain.add(grad_total, grad)
        return value_total, grad_total

    def __eq__(self, other: Any) -> bool:
        """Return whether another sum has the same ordered terms."""
        if not self.same_math(other):
            return NotImplemented
        if len(self.terms) != len(other.terms):
            return False
        return all(a == b for a, b in zip(self.terms, other.terms))

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return self.terms, ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        return cls(tuple(children))

    def _convert(self, new_ctx: Context) -> "SumFunctional":
        """Convert every term to ``new_ctx``."""
        return SumFunctional(tuple(term.convert(new_ctx) for term in self.terms))


def make_functional_sum(terms: Any) -> Functional:
    """
    Return a locally simplified lazy sum of functionals.

    Nested :class:`SumFunctional` nodes are flattened; a single surviving term is
    returned unwrapped. Domain and context compatibility is validated by
    :class:`SumFunctional`.

    Parameters
    ----------
    terms : sequence of Functional
        Nonempty sequence of functionals sharing one domain.

    Returns
    -------
    Functional
        Simplified lazy sum, or the single operand when only one remains.
    """
    terms = tuple(terms)
    if not terms:
        raise ValueError(
            "make_functional_sum requires a nonempty sequence of Functional operands."
        )
    for i, term in enumerate(terms):
        if not isinstance(term, Functional):
            raise TypeError(f"operand {i} must be a Functional, got {type(term).__name__}.")
    flat = flatten_sum(
        terms,
        is_sum=lambda t: isinstance(t, SumFunctional),
        parts=lambda t: t.terms,
    )
    # Validate all terms' domains BEFORE dropping zeros, so a domain mismatch is
    # never swallowed by the single-survivor unwrap or the all-zero collapse.
    _require_same_domain(flat)
    return finalize_sum(
        flat,
        is_zero=lambda f: isinstance(f, ZeroFunctional),
        make_zero=lambda: ZeroFunctional(flat[0].domain, flat[0].ctx),
        make_sum_node=SumFunctional,
    )


class ZeroFunctional(Functional):
    """
    The zero functional: value ``0``, gradient the domain's zero element.

    The additive identity of the functional algebra (``make_functional_sum``
    drops it and ``make_scaled_functional`` returns it for a zero scalar).

    Parameters
    ----------
    dom : Space
        Domain space.
    ctx : Context, str, or None, optional
        Backend context specification.
    check_level : {{"none", "cheap", "standard", "strict"}}, optional
        Runtime validation policy for this object. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used.
    """

    def __init__(
        self,
        dom: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the scalar zero."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free scalar zero in the domain dtype."""
        return self.ctx.asarray(0.0)

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the domain's zero element."""
        return self.domain.zeros()

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(0, X.zeros())``."""
        return self._value_core(x, *args, **kwargs), self.domain.zeros()

    def __eq__(self, other: Any) -> bool:
        """Return whether another zero functional has the same domain."""
        if not self.same_math(other):
            return NotImplemented
        return self.domain == other.domain

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        dom, ctx = aux
        return cls(dom, ctx)

    def _convert(self, new_ctx: Context) -> "ZeroFunctional":
        """Convert the zero functional to ``new_ctx``."""
        return ZeroFunctional(self.domain.convert(new_ctx), new_ctx)


class ConstantFunctional(Functional):
    """
    The constant functional ``x -> constant``: fixed value, zero gradient.

    The embedding of a scalar into the functional algebra. It generalizes
    :class:`ZeroFunctional` (which is the ``constant = 0`` case, kept separate
    because it is the additive identity the canonicalizers recognize), and it is
    what :func:`make_functional_product` folds against: multiplying by a constant
    functional is exactly scaling by its value, so ``C * F`` collapses to a
    :class:`ScaledFunctional` rather than building a product node.

    Parameters
    ----------
    dom : Space
        Domain space.
    constant : scalar-like
        Value returned at every point.
    ctx : Context, str, or None, optional
        Backend context specification.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.

    Attributes
    ----------
    constant : scalar-like
        The stored value.
    """

    def __init__(
        self,
        dom: Any,
        constant: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        if not is_scalar_like(constant):
            raise TypeError(f"constant must be scalar-like, got {type(constant).__name__}.")
        super().__init__(dom, ctx, check_level=check_level)
        self.constant = constant

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the stored constant."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free constant in the domain dtype (mirrors ``ZeroFunctional``)."""
        return self.ctx.asarray(self.constant)

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the domain's zero element (a constant has zero derivative)."""
        return self.domain.zeros()

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(constant, X.zeros())``."""
        return self._value_core(x, *args, **kwargs), self.domain.zeros()

    def __eq__(self, other: Any) -> bool:
        """Return whether another constant functional has the same domain and value."""
        if not self.same_math(other):
            return NotImplemented
        return self.domain == other.domain and scalar_eq(self.constant, other.constant)

    def tree_flatten(self):
        """Flatten this functional for pytree registration (constant is a traced child)."""
        return (self.constant,), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        dom, ctx = aux
        (constant,) = children
        return cls(dom, constant, ctx)

    def _convert(self, new_ctx: Context) -> "ConstantFunctional":
        """Convert the constant functional to ``new_ctx``."""
        return ConstantFunctional(self.domain.convert(new_ctx), self.constant, new_ctx)


def make_constant_functional(
    dom: Any,
    constant: Any,
    ctx: Context | str | None = None,
) -> Functional:
    """
    Return a locally simplified constant functional on ``dom``.

    A zero constant collapses to :class:`ZeroFunctional`, so the additive
    identity keeps exactly one representation and the sum/scale canonicalizers
    continue to recognize it.

    Parameters
    ----------
    dom : Space
        Domain space.
    constant : scalar-like
        Value returned at every point.
    ctx : Context, str, or None, optional
        Backend context specification.

    Returns
    -------
    Functional
        :class:`ZeroFunctional` for a zero constant, else
        :class:`ConstantFunctional`.
    """
    if not is_scalar_like(constant):
        raise TypeError(f"constant must be scalar-like, got {type(constant).__name__}.")
    if scalar_eq(constant, 0):
        return ZeroFunctional(dom, ctx)
    return ConstantFunctional(dom, constant, ctx)


class ShiftedFunctional(Functional):
    """
    Affine shift ``functional + offset``: value shifted, gradient unchanged.

    Parameters
    ----------
    functional : Functional
        Functional to shift.
    offset : scalar-like
        Constant added to the value.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.
    """

    def __init__(
        self,
        functional: Functional,
        offset: Any,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        if not isinstance(functional, Functional):
            raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
        if not is_scalar_like(offset):
            raise TypeError(f"offset must be scalar-like, got {type(offset).__name__}.")
        if check_level is None:
            check_level = functional.check_level
        super().__init__(functional.domain, functional.ctx, check_level=check_level)
        self.functional = functional.convert(self.ctx)
        self.offset = offset

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return ``functional.value(x) + offset``."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free shifted value."""
        return self.functional._value_core(x, *args, **kwargs) + self.offset

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return ``functional.grad(x)`` (a constant shift has zero gradient)."""
        return self.functional.grad(x, *args, **kwargs)

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(value + offset, grad)`` from one fused child evaluation."""
        value, grad = self.functional.value_and_grad(x, *args, **kwargs)
        return value + self.offset, grad

    def __eq__(self, other: Any) -> bool:
        """Return whether another shifted functional has the same offset and operand."""
        if not self.same_math(other):
            return NotImplemented
        return scalar_eq(self.offset, other.offset) and self.functional == other.functional

    def tree_flatten(self):
        """Flatten this functional for pytree registration (offset is a traced child)."""
        return (self.functional, self.offset), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        functional, offset = children
        return cls(functional, offset)

    def _convert(self, new_ctx: Context) -> "ShiftedFunctional":
        """Convert the shifted functional to ``new_ctx``."""
        return ShiftedFunctional(self.functional.convert(new_ctx), self.offset)


def make_shifted_functional(functional: Functional, offset: Any) -> Functional:
    """
    Return a locally simplified affine shift of a functional.

    A zero offset passes ``functional`` through unchanged and nested
    :class:`ShiftedFunctional` nodes fold into one offset.

    Parameters
    ----------
    functional : Functional
        Functional to shift.
    offset : scalar-like
        Constant added to the value.

    Returns
    -------
    Functional
        Simplified affine shift.
    """
    if not isinstance(functional, Functional):
        raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
    if not is_scalar_like(offset):
        raise TypeError(f"offset must be scalar-like, got {type(offset).__name__}.")
    if scalar_eq(offset, 0):
        return functional
    if isinstance(functional, ShiftedFunctional):
        return make_shifted_functional(functional.functional, functional.offset + offset)
    return ShiftedFunctional(functional, offset)


class ProductFunctional(Functional):
    r"""
    Lazy pointwise product ``(F * G)(x) = F(x) * G(x)`` on a shared domain.

    Deliberately **binary**: the gradient is the two-factor product rule, which
    does not generalize to an n-ary node without the full Leibniz expansion, and
    nesting ``(F*G)*H`` expresses the same thing with the same cost.

    The gradient conjugates each cofactor, mirroring
    :meth:`ScaledFunctional.grad`. Writing :math:`D` for the derivative,

    .. math::

        D(FG)(x)[h] = G(x)\, DF(x)[h] + F(x)\, DG(x)[h],

    and the Riesz gradient is the element pairing to that under the domain inner
    product. Since the inner product conjugates its *first* argument,
    :math:`\langle \overline{a} g, h\rangle = a \langle g, h\rangle`, so the
    coefficients enter conjugated:

    .. math::

        \nabla(FG)(x) = \overline{G(x)}\, \nabla F(x)
                      + \overline{F(x)}\, \nabla G(x).

    For real-valued factors — the usual case — the conjugations are identities.
    The two terms are combined through the domain's own ``scale``/``add``, never
    raw ``*``/``+``, because a domain element may be a pytree.

    Parameters
    ----------
    left, right : Functional
        Factors sharing one domain.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this functional. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Validation
        policy is a property of the functional, not of the ``Context``.

    Attributes
    ----------
    left, right : Functional
        The two factors, converted into this node's context.
    """

    def __init__(
        self,
        left: Functional,
        right: Functional,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        for name, factor in (("left", left), ("right", right)):
            if not isinstance(factor, Functional):
                raise TypeError(
                    f"{name} must be a Functional, got {type(factor).__name__}."
                )
        _require_same_domain((left, right), node="ProductFunctional")
        # Default the policy from the OPERANDS, not from their domain space, so
        # the result inherits the least-strict operand independently of order
        # (mirrors SumFunctional).
        if check_level is None:
            check_level = minimum_check_level((left.check_level, right.check_level))
        super().__init__(left.domain, left.ctx, check_level=check_level)
        self.left = left.convert(self.ctx)
        self.right = right.convert(self.ctx)

    @property
    def factors(self) -> tuple[Functional, Functional]:
        """Return the two factors in order."""
        return (self.left, self.right)

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return ``left.value(x) * right.value(x)``."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free pointwise product of the factor values."""
        return (
            self.left._value_core(x, *args, **kwargs)
            * self.right._value_core(x, *args, **kwargs)
        )

    def _product_grad(self, lv: Any, lg: Any, rv: Any, rg: Any) -> Any:
        """Combine factor values/gradients by the product rule, in domain ops."""
        domain = self.domain
        conj = self.ops.conj
        return domain.add(domain.scale(conj(rv), lg), domain.scale(conj(lv), rg))

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the product-rule Riesz gradient at ``x``.

        Both factors are evaluated through ``value_and_grad`` because the rule
        needs each factor's *value* as well as its gradient; asking for the
        gradients alone would evaluate the values a second time.
        """
        lv, lg = self.left.value_and_grad(x, *args, **kwargs)
        rv, rg = self.right.value_and_grad(x, *args, **kwargs)
        return self._product_grad(lv, lg, rv, rg)

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(value, grad)`` from one fused evaluation per factor."""
        lv, lg = self.left.value_and_grad(x, *args, **kwargs)
        rv, rg = self.right.value_and_grad(x, *args, **kwargs)
        return lv * rv, self._product_grad(lv, lg, rv, rg)

    def __eq__(self, other: Any) -> bool:
        """Return whether another product has the same ordered factors.

        Ordered, like :class:`SumFunctional`: the operation commutes, but this is
        structural equality of expression trees, not semantic equivalence.
        """
        if not self.same_math(other):
            return NotImplemented
        return self.left == other.left and self.right == other.right

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (self.left, self.right), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        left, right = children
        return cls(left, right)

    def _convert(self, new_ctx: Context) -> "ProductFunctional":
        """Convert both factors to ``new_ctx``."""
        return ProductFunctional(
            self.left.convert(new_ctx), self.right.convert(new_ctx)
        )


def make_functional_product(left: Functional, right: Functional) -> Functional:
    """
    Return a locally simplified pointwise product of two functionals.

    Only *structural* simplifications are attempted — the ones visible from the
    node types, with no evaluation:

    * a :class:`ZeroFunctional` factor collapses the product to zero (matching
      how :func:`make_scaled_functional` treats a zero scalar);
    * a :class:`ConstantFunctional` factor becomes a
      :class:`ScaledFunctional` on the other factor, since multiplying by a
      constant *is* scaling.

    There is deliberately no attempt to recognize a functional that merely
    *happens* to be constant or zero: that is a fact about values, not about the
    expression, and is not decidable here.

    Parameters
    ----------
    left, right : Functional
        Factors sharing one domain.

    Returns
    -------
    Functional
        Simplified product.
    """
    for name, factor in (("left", left), ("right", right)):
        if not isinstance(factor, Functional):
            raise TypeError(f"{name} must be a Functional, got {type(factor).__name__}.")
    # Validate domains BEFORE any collapse, so a mismatch is never swallowed by
    # the zero/constant shortcuts.
    _require_same_domain((left, right), node="ProductFunctional")

    if isinstance(left, ZeroFunctional) or isinstance(right, ZeroFunctional):
        return ZeroFunctional(left.domain, left.ctx)
    if isinstance(left, ConstantFunctional):
        return make_scaled_functional(left.constant, right)
    if isinstance(right, ConstantFunctional):
        return make_scaled_functional(right.constant, left)
    return ProductFunctional(left, right)

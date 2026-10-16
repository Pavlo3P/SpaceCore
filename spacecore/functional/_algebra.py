"""
Lazy functional algebra: scaled, sum, product and constant nodes.

Functionals are scalar-valued, so beyond the operator algebra ``F * G`` is a
pointwise product and a scalar embeds as :class:`ConstantFunctional`.
Canonicalization reads node types, never values. Gradients combine through the
domain's ``add``/``scale``, because elements may be pytrees.
"""
from __future__ import annotations

import operator
from functools import reduce
from typing import Any

from ._base import Functional
from ._linear import LinearFunctional
from ..mapping import Scalar, ScaledNode, SumNode, ZeroNode, make_scaled_of, make_sum_of, register
from .._checks import CheckLevel, minimum_check_level
from ..contextual import Context
from ..mapping._lazy_algebra import is_scalar_like, scalar_eq


def _require_factors(left: Any, right: Any) -> None:
    """Raise unless both product factors are functionals on one domain.

    Domain equality folds in the backend/dtype context, so this also rejects a
    same-shape space on a different backend or dtype.
    """
    for name, factor in (("left", left), ("right", right)):
        if not isinstance(factor, Functional):
            raise TypeError(f"{name} must be a Functional, got {type(factor).__name__}.")
    if right.domain != left.domain:
        raise ValueError(
            "All ProductFunctional operands must have the same domain; operand 0 has "
            f"domain {left.domain!r}, operand 1 has domain {right.domain!r}."
        )


class ScaledFunctional(ScaledNode, Functional):
    """
    Lazy scalar multiple ``scalar * functional``.

    Parameters
    ----------
    scalar : scalar-like
        Scalar coefficient.
    functional : Functional
        Functional to scale.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("scalar", "functional:operand")
    _dynamic = ("scalar", "functional")

    def __init__(self, scalar: Any, functional: Functional, check_level: CheckLevel | None = None) -> None:
        super().__init__(scalar, functional, check_level)
        self.codomain._check_member(scalar)

    def _convert(self, new_ctx: Context) -> "ScaledFunctional":
        scalar = self.codomain.convert(new_ctx).ctx.asarray(self.scalar)
        return type(self)(scalar, self.functional.convert(new_ctx), self.check_level)

    @property
    def functional(self) -> Functional:
        """The scaled functional."""
        return self.operand

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free scaled value."""
        return self.scalar * self.functional._value_core(x, *args, **kwargs)

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """The real scalar times the child's Riesz gradient."""
        return self.domain._scale_core(self.scalar, self.functional._grad_core(x, *args, **kwargs))

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """``(scalar * value, scalar * grad)`` from one fused child evaluation."""
        value, grad = self.functional._value_and_grad_core(x, *args, **kwargs)
        return self.scalar * value, self.domain._scale_core(self.scalar, grad)

    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> LinearFunctional:
        return self.scalar * self.functional._derivative_core(x, *args, **kwargs)

    def _value_and_derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, LinearFunctional]:
        value, derivative = self.functional._value_and_derivative_core(x, *args, **kwargs)
        return self.scalar * value, self.scalar * derivative


def make_scaled_functional(scalar: Any, functional: Functional) -> Functional:
    """
    Return ``scalar * functional``: ``1`` passes it through, nested scalars fold.

    Parameters
    ----------
    scalar : scalar-like
        Coefficient.
    functional : Functional
        Functional to scale.

    Returns
    -------
    Functional
        The scaled functional.
    """
    family = LinearFunctional if isinstance(functional, LinearFunctional) else Functional
    return make_scaled_of(family, scalar, functional)


class SumFunctional(SumNode, Functional):
    """
    Lazy sum of finitely many functionals on a common domain.

    Parameters
    ----------
    terms : sequence of Functional
        Nonempty sequence of functionals sharing one domain.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("terms:parts",)
    _dynamic = ("terms",)

    def __init__(self, terms: Any, check_level: CheckLevel | None = None) -> None:
        super().__init__(terms, check_level)

    @property
    def terms(self) -> tuple[Functional, ...]:
        """The summed functionals (alias of ``parts``)."""
        return self.parts

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free sum of term values."""
        # Sum of the terms' values (plain scalars); _checked_parts guarantees a term.
        return reduce(operator.add, (term._value_core(x, *args, **kwargs) for term in self.terms))

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """The domain-sum of the term gradients (combined via the domain's add)."""
        # Constant terms have zero gradient and are skipped, so ``F + c`` costs no extra add.
        grads = [term._grad_core(x, *args, **kwargs) for term in self.terms if not isinstance(term, ConstantFunctional)]
        return reduce(self.domain._add_core, grads) if grads else self.domain.zeros()

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """``(sum values, domain-sum grads)`` from one fused pass per term."""
        # One fused evaluation per term; values add as scalars, gradients in the domain.
        domain = self.domain
        value_total = None
        grad_total = None
        for term in self.terms:
            if isinstance(term, ConstantFunctional):
                value_total = term._value_core(x) if value_total is None else value_total + term._value_core(x)
                continue
            value, grad = term._value_and_grad_core(x, *args, **kwargs)
            value_total = value if value_total is None else value_total + value
            grad_total = grad if grad_total is None else domain._add_core(grad_total, grad)
        return value_total, domain.zeros() if grad_total is None else grad_total


    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> LinearFunctional:
        return make_functional_sum(tuple(t._derivative_core(x, *args, **kwargs) for t in self.terms))

    def _value_and_derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, LinearFunctional]:
        pairs = [t._value_and_derivative_core(x, *args, **kwargs) for t in self.terms]
        return sum(v for v, _ in pairs), make_functional_sum(tuple(d for _, d in pairs))


def make_functional_sum(terms: Any) -> Functional:
    """
    Return the lazy sum of functionals, flattened, with zero terms dropped.

    Parameters
    ----------
    terms : sequence of Functional
        Functionals on one domain.

    Returns
    -------
    Functional
        The sum, or the single remaining term.
    """
    terms = tuple(terms)
    family = LinearFunctional if terms and all(isinstance(t, LinearFunctional) for t in terms) else Functional
    return make_sum_of(family, terms)


class ZeroFunctional(ZeroNode, Functional):
    """
    The zero functional: value ``0``, gradient the domain's zero; the sum's identity.

    Parameters
    ----------
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    @classmethod
    def _of(cls, domain: Any, codomain: Any, ctx: Any, check_level: Any = None) -> "ZeroFunctional":
        return cls(domain, ctx, check_level)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free scalar zero in the codomain dtype."""
        return self.codomain.ctx.asarray(0.0)

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """The domain's zero element."""
        return self.domain.zeros()

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """``(0, X.zeros())``."""
        return self._value_core(x, *args, **kwargs), self.domain.zeros()


class ConstantFunctional(Functional):
    """
    The constant functional ``x -> constant``, with zero gradient.

    A product with it folds to scaling by ``constant``.

    Parameters
    ----------
    dom : Space
        Domain.
    constant : scalar-like
        The value.
    ctx : Context, str, or None, optional
        Backend context.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("dom", "constant", "ctx", "check_level")
    _dynamic = ("constant",)

    def __init__(
        self,
        dom: Any,
        constant: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        if not is_scalar_like(constant):
            raise TypeError(f"constant must be scalar-like, got {type(constant).__name__}.")
        super().__init__(dom, ctx, check_level=check_level)
        self.codomain._check_member(constant)
        self.constant = constant

    def _convert(self, new_ctx: Context) -> "ConstantFunctional":
        cod = self.codomain.convert(new_ctx)
        return type(self)(self.domain.convert(new_ctx), cod.ctx.asarray(self.constant),
                          new_ctx, self.check_level)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free constant in the codomain dtype (mirrors ``ZeroFunctional``)."""
        return self.codomain.ctx.asarray(self.constant)

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """The domain's zero element (a constant has zero derivative)."""
        return self.domain.zeros()

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """``(constant, X.zeros())``."""
        return self._value_core(x, *args, **kwargs), self.domain.zeros()

def make_constant_functional(
    dom: Any,
    constant: Any,
    ctx: Context | str | None = None,
) -> Functional:
    """
    Return the constant functional on ``dom``, or :class:`ZeroFunctional` for ``0``.

    Parameters
    ----------
    dom : Space
        Domain.
    constant : scalar-like
        The value.
    ctx : Context, str, or None, optional
        Backend context.

    Returns
    -------
    Functional
        The constant (or zero) functional.
    """
    # A zero constant is the additive identity, kept as ZeroFunctional.
    if not is_scalar_like(constant):
        raise TypeError(f"constant must be scalar-like, got {type(constant).__name__}.")
    if scalar_eq(constant, 0):
        return ZeroFunctional(dom, ctx)
    return ConstantFunctional(dom, constant, ctx)


def make_shifted_functional(functional: Functional, offset: Any) -> Functional:
    """
    Return ``functional + offset`` as a sum with a :class:`ConstantFunctional`; ``0`` passes it through.

    Parameters
    ----------
    functional : Functional
        Functional to shift.
    offset : scalar-like
        Constant added to the value.

    Returns
    -------
    Functional
        The shifted functional.
    """
    if not isinstance(functional, Functional):
        raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
    if not is_scalar_like(offset):
        raise TypeError(f"offset must be scalar-like, got {type(offset).__name__}.")
    if scalar_eq(offset, 0):
        return functional
    constant = ConstantFunctional(functional.domain, offset, functional.ctx,
                                  check_level=functional.check_level)
    return make_functional_sum((functional, constant))


class ProductFunctional(Functional):
    r"""
    Lazy pointwise product ``(F * G)(x) = F(x) G(x)`` on a shared domain.

    The gradient is :math:`G(x) \nabla F(x) + F(x) \nabla G(x)`.
    Expanding ``D(FG)[h] = G DF[h] + F DG[h]`` gives this rule with the real pairing.

    Parameters
    ----------
    left, right : Functional
        Factors on one domain.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("left", "right")
    _dynamic = ("left", "right")

    def __init__(
        self,
        left: Functional,
        right: Functional,
        check_level: CheckLevel | None = None,
    ) -> None:
        _require_factors(left, right)
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

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free pointwise product ``left.value(x) * right.value(x)``."""
        # Pointwise product of the two values.
        return (
            self.left._value_core(x, *args, **kwargs)
            * self.right._value_core(x, *args, **kwargs)
        )

    def _product_grad(self, lv: Any, lg: Any, rv: Any, rg: Any) -> Any:
        """Combine factor values/gradients by the product rule, in domain ops."""
        domain = self.domain
        return domain._add_core(domain._scale_core(rv, lg), domain._scale_core(lv, rg))

    def _grad_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """The product-rule Riesz gradient at ``x``.

        Both factors are evaluated through ``value_and_grad`` because the rule
        needs each factor's *value* as well as its gradient; asking for the
        gradients alone would evaluate the values a second time.
        """
        return self._value_and_grad_core(x, *args, **kwargs)[1]

    def _value_and_grad_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """``(value, grad)`` from one fused evaluation per factor."""
        lv, lg = self.left._value_and_grad_core(x, *args, **kwargs)
        rv, rg = self.right._value_and_grad_core(x, *args, **kwargs)
        return lv * rv, self._product_grad(lv, lg, rv, rg)

    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> LinearFunctional:
        return self._value_and_derivative_core(x, *args, **kwargs)[1]

    def _value_and_derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, LinearFunctional]:
        lv, ld = self.left._value_and_derivative_core(x, *args, **kwargs)
        rv, rd = self.right._value_and_derivative_core(x, *args, **kwargs)
        return lv * rv, rv * ld + lv * rd

def make_functional_product(left: Functional, right: Functional) -> Functional:
    """
    Return ``left * right``; a zero factor gives zero and a constant factor gives scaling.

    Only node types are inspected, so a functional that merely evaluates to a constant
    is not folded.

    Parameters
    ----------
    left, right : Functional
        Factors on one domain.

    Returns
    -------
    Functional
        The product.
    """
    # Validate BEFORE any collapse, so a mismatch is never swallowed by the shortcuts.
    _require_factors(left, right)

    # Structural shortcuts: a zero factor gives zero, a constant factor gives scaling.
    if isinstance(left, ZeroFunctional) or isinstance(right, ZeroFunctional):
        return ZeroFunctional(left.domain, left.ctx)
    if isinstance(left, ConstantFunctional):
        return make_scaled_functional(left.constant, right)
    if isinstance(right, ConstantFunctional):
        return make_scaled_functional(right.constant, left)
    return ProductFunctional(left, right)


class _ScaledLinearFunctional(ScaledFunctional, LinearFunctional):
    """Linear algebra preserves the linear-functional contract."""

    _derivative_core = LinearFunctional._derivative_core
    _grad_core = LinearFunctional._grad_core
    _vgrad_core = LinearFunctional._vgrad_core
    _value_and_grad_core = Functional._value_and_grad_core
    _value_and_derivative_core = Functional._value_and_derivative_core

    @property
    def representer(self) -> Any:
        return self.domain.scale(self.scalar, self.functional.representer)


class _SumLinearFunctional(SumFunctional, LinearFunctional):
    """Sum evaluated lazily, with a representer only when all terms have one."""

    _derivative_core = LinearFunctional._derivative_core
    _grad_core = LinearFunctional._grad_core
    _vgrad_core = LinearFunctional._vgrad_core
    _value_and_grad_core = Functional._value_and_grad_core
    _value_and_derivative_core = Functional._value_and_derivative_core

    @property
    def representer(self) -> Any:
        return reduce(self.domain.add, (term.representer for term in self.terms))


class _ZeroLinearFunctional(ZeroFunctional, LinearFunctional):
    """Zero in the linear-functional algebra."""

    @property
    def representer(self) -> Any:
        return self.domain.zeros()


LinearFunctional._Sum = _SumLinearFunctional
LinearFunctional._Scaled = _ScaledLinearFunctional
LinearFunctional._Zero = _ZeroLinearFunctional


Functional._Sum, Functional._Scaled, Functional._Zero = SumFunctional, ScaledFunctional, ZeroFunctional

register("add", Functional, Functional, lambda f, g: make_functional_sum((f, g)))
register("add", Functional, Scalar, make_shifted_functional)
register("add", Scalar, Functional, lambda s, f: make_shifted_functional(f, s))
register("mul", Scalar, Functional, make_scaled_functional)
register("mul", Functional, Scalar, lambda f, s: make_scaled_functional(s, f))
register("mul", Functional, Functional, make_functional_product)

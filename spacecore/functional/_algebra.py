"""
Lazy functional algebra: scaled, sum, product, constant and shifted nodes.

Functionals are scalar-valued, so beyond the operator algebra ``F * G`` is a
pointwise product and a scalar embeds as :class:`ConstantFunctional`.
Canonicalization reads node types, never values. Gradients combine through the
domain's ``add``/``scale``, because elements may be pytrees.
"""
from __future__ import annotations

from typing import Any

from ._base import Functional
from ..mapping import Scalar, ScaledNode, SumNode, ZeroNode, make_scaled_of, make_sum_of, register
from ..space import Field
from .._checks import checked_method
from .._check_policy import CheckLevel, minimum_check_level
from ..contextual import Context
from .._lazy_algebra import (
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

    def __init__(self, scalar: Any, functional: Functional, check_level: CheckLevel | bool | None = None) -> None:
        super().__init__(scalar, functional, check_level)

    @property
    def functional(self) -> Functional:
        """The scaled functional."""
        return self.operand

    @checked_method(in_space="domain", out_space="codomain")
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
    return make_scaled_of(Functional, scalar, functional)


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

    def __init__(self, terms: Any, check_level: CheckLevel | bool | None = None) -> None:
        super().__init__(terms, check_level)

    @property
    def terms(self) -> tuple[Functional, ...]:
        """The summed functionals (alias of ``parts``)."""
        return self.parts

    @checked_method(in_space="domain", out_space="codomain")
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
    return make_sum_of(Functional, terms)


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
    cod : Field or None, optional
        Scalar codomain; the domain's ``scalars`` by default.
    """

    @classmethod
    def _of(cls, domain: Any, codomain: Any, ctx: Any, check_level: Any = None) -> "ZeroFunctional":
        return cls(domain, ctx, check_level, cod=codomain)

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the scalar zero."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free scalar zero in the codomain dtype."""
        return self.codomain.ctx.asarray(0.0)

    def grad(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the domain's zero element."""
        return self.domain.zeros()

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(0, X.zeros())``."""
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
    cod : Field or None, optional
        Scalar codomain; the domain's ``scalars`` by default.
    """

    def __init__(
        self,
        dom: Any,
        constant: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
        *,
        cod: Field | None = None,
    ) -> None:
        if not is_scalar_like(constant):
            raise TypeError(f"constant must be scalar-like, got {type(constant).__name__}.")
        super().__init__(dom, ctx, check_level=check_level, cod=cod)
        self.constant = constant

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the stored constant."""
        return self._value_core(x, *args, **kwargs)

    def _value_core(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Check-free constant in the codomain dtype (mirrors ``ZeroFunctional``)."""
        return self.codomain.ctx.asarray(self.constant)

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
        return self.domain == other.domain and self.codomain == other.codomain and scalar_eq(self.constant, other.constant)

    def tree_flatten(self):
        """Flatten this functional for pytree registration (constant is a traced child)."""
        return (self.constant,), (self.domain, self.ctx, self.codomain, self.check_level)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        dom, ctx, cod, check_level = aux
        (constant,) = children
        return cls(dom, constant, ctx, check_level=check_level, cod=cod)

    def _convert(self, new_ctx: Context) -> "ConstantFunctional":
        """Convert the constant functional to ``new_ctx``."""
        return ConstantFunctional(self.domain.convert(new_ctx), self.constant, new_ctx,
                                  check_level=self.check_level, cod=self.codomain)


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
        Validation level; the ambient default when omitted.
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
        super().__init__(functional.domain, functional.ctx, check_level=check_level,
                         cod=functional.codomain)
        self.functional = functional.convert(self.ctx)
        self.offset = offset

    @checked_method(in_space="domain", out_space="codomain")
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
    Return ``functional + offset``: ``0`` passes it through, nested shifts fold.

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
    if isinstance(functional, ShiftedFunctional):
        return make_shifted_functional(functional.functional, functional.offset + offset)
    return ShiftedFunctional(functional, offset)


class ProductFunctional(Functional):
    r"""
    Lazy pointwise product ``(F * G)(x) = F(x) G(x)`` on a shared domain.

    The gradient is :math:`\overline{G(x)} \nabla F(x) + \overline{F(x)} \nabla G(x)`:
    the coefficients are conjugated because the inner product conjugates its first
    argument.

    Parameters
    ----------
    left, right : Functional
        Factors on one domain.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
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
        super().__init__(left.domain, left.ctx, check_level=check_level, cod=left.codomain)
        self.left = left.convert(self.ctx)
        self.right = right.convert(self.ctx)

    @property
    def factors(self) -> tuple[Functional, Functional]:
        """Return the two factors in order."""
        return (self.left, self.right)

    @checked_method(in_space="domain", out_space="codomain")
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
    for name, factor in (("left", left), ("right", right)):
        if not isinstance(factor, Functional):
            raise TypeError(f"{name} must be a Functional, got {type(factor).__name__}.")
    # Validate domains BEFORE any collapse, so a mismatch is never swallowed by
    # the zero/constant shortcuts.
    _require_same_domain((left, right), node="ProductFunctional")

    if isinstance(left, ZeroFunctional) or isinstance(right, ZeroFunctional):
        return ZeroFunctional(left.domain, left.ctx, cod=left.codomain)
    if isinstance(left, ConstantFunctional):
        return make_scaled_functional(left.constant, right)
    if isinstance(right, ConstantFunctional):
        return make_scaled_functional(right.constant, left)
    return ProductFunctional(left, right)


Functional._Sum, Functional._Scaled, Functional._Zero = SumFunctional, ScaledFunctional, ZeroFunctional

register("add", Functional, Functional, lambda f, g: make_functional_sum((f, g)))
register("add", Functional, Scalar, make_shifted_functional)
register("add", Scalar, Functional, lambda s, f: make_shifted_functional(f, s))
register("mul", Scalar, Functional, make_scaled_functional)
register("mul", Functional, Scalar, lambda f, s: make_scaled_functional(s, f))
register("mul", Functional, Functional, make_functional_product)

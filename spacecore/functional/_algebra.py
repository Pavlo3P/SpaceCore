"""Lazy functional algebra: scalar multiples and sums (mirrors ``linop/_algebra.py``).

`Functional` gains the additive/scalar algebra `LinOp` already has, so objectives
compose as ``a * F``, ``F + G``, ``F - G``, ``-F``. The operator overloads live on
:class:`~spacecore.functional.Functional` and delegate to the ``make_*`` factories
here, which do local canonicalization (fold nested scalars, flatten nested sums).

A functional's ``grad`` is a *metric (Riesz) gradient* -- an element of the domain
``X`` -- so the algebra combines child gradients through the domain's own vector
ops (``X.add`` / ``X.scale``), never raw ``+`` / ``*`` (which would be wrong on a
tree/stacked domain, where an element is a pytree).
"""
from __future__ import annotations

from numbers import Number
from typing import Any

from ._base import Functional
from .._checks import checked_method
from ..backend import Context, jax_pytree_class


def is_scalar_like(value: Any) -> bool:
    """Return whether ``value`` can be used as a scalar multiplier for a functional."""
    if isinstance(value, Number):
        return True
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape) == ()
    ndim = getattr(value, "ndim", None)
    return ndim == 0


def _scalar_eq(a: Any, b: Any) -> bool:
    """Return whether two scalar-likes are equal, NaN-reflexive, as a real ``bool``."""
    if bool(a == b):
        return True
    try:
        return bool(a != a) and bool(b != b)
    except Exception:
        return False


def _require_same_domain(terms: Any) -> None:
    """Raise unless every functional in ``terms`` shares the first term's domain.

    Domain equality folds in the backend/dtype context, so this also rejects a
    same-shape space on a different backend or dtype.
    """
    domain = terms[0].domain
    for i, term in enumerate(terms[1:], start=1):
        if term.domain != domain:
            raise ValueError(
                "All SumFunctional operands must have the same domain; operand 0 has "
                f"domain {domain!r}, operand {i} has domain {term.domain!r}."
            )


@jax_pytree_class
class ScaledFunctional(Functional):
    """
    Lazy scalar multiple ``scalar * functional``.

    Parameters
    ----------
    scalar : scalar-like
        Scalar coefficient.
    functional : Functional
        Functional to scale.
    """

    def __init__(self, scalar: Any, functional: Functional) -> None:
        if not isinstance(functional, Functional):
            raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
        if not is_scalar_like(scalar):
            raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
        super().__init__(functional.domain, functional.ctx)
        self.scalar = scalar
        self.functional = functional.convert(self.ctx)

    @checked_method(in_space="domain")
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
        if not self._eq_backend_compatible(other):
            return NotImplemented
        return _scalar_eq(self.scalar, other.scalar) and self.functional == other.functional

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
    if _scalar_eq(scalar, 0):
        return ZeroFunctional(functional.domain, functional.ctx)
    if _scalar_eq(scalar, 1):
        return functional
    if isinstance(functional, ZeroFunctional):
        return functional
    if isinstance(functional, ScaledFunctional):
        return make_scaled_functional(scalar * functional.scalar, functional.functional)
    return ScaledFunctional(scalar, functional)


@jax_pytree_class
class SumFunctional(Functional):
    """
    Lazy sum ``F_1 + ... + F_n`` of functionals on a common domain.

    Parameters
    ----------
    terms : sequence of Functional
        Nonempty sequence of functionals sharing one domain.
    """

    def __init__(self, terms: Any) -> None:
        parts = tuple(terms)
        if not parts:
            raise ValueError(
                "SumFunctional requires a nonempty sequence of Functional operands."
            )
        for i, term in enumerate(parts):
            if not isinstance(term, Functional):
                raise TypeError(f"operand {i} must be a Functional, got {type(term).__name__}.")
        _require_same_domain(parts)
        super().__init__(parts[0].domain, parts[0].ctx)
        self.terms = tuple(term.convert(self.ctx) for term in parts)

    @property
    def parts(self) -> tuple[Functional, ...]:
        """Return the summed terms in order."""
        return self.terms

    @checked_method(in_space="domain")
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
        if not self._eq_backend_compatible(other):
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


def _flatten_functional_sum_terms(terms: Any) -> tuple[Functional, ...]:
    """Flatten nested :class:`SumFunctional` nodes into a flat term tuple."""
    flat: list[Functional] = []
    for i, term in enumerate(terms):
        if not isinstance(term, Functional):
            raise TypeError(f"operand {i} must be a Functional, got {type(term).__name__}.")
        if isinstance(term, SumFunctional):
            flat.extend(term.terms)
        else:
            flat.append(term)
    return tuple(flat)


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
    flat = _flatten_functional_sum_terms(terms)
    # Validate all terms' domains BEFORE dropping zeros, so a domain mismatch is
    # never swallowed by the single-survivor unwrap or the all-zero collapse.
    _require_same_domain(flat)
    nonzero = tuple(term for term in flat if not isinstance(term, ZeroFunctional))
    if not nonzero:
        return ZeroFunctional(flat[0].domain, flat[0].ctx)
    if len(nonzero) == 1:
        return nonzero[0]
    return SumFunctional(nonzero)


@jax_pytree_class
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
    """

    def __init__(self, dom: Any, ctx: Context | str | None = None) -> None:
        super().__init__(dom, ctx)

    @checked_method(in_space="domain")
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
        if not self._eq_backend_compatible(other):
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


@jax_pytree_class
class ShiftedFunctional(Functional):
    """
    Affine shift ``functional + offset``: value shifted, gradient unchanged.

    Parameters
    ----------
    functional : Functional
        Functional to shift.
    offset : scalar-like
        Constant added to the value.
    """

    def __init__(self, functional: Functional, offset: Any) -> None:
        if not isinstance(functional, Functional):
            raise TypeError(f"functional must be a Functional, got {type(functional).__name__}.")
        if not is_scalar_like(offset):
            raise TypeError(f"offset must be scalar-like, got {type(offset).__name__}.")
        super().__init__(functional.domain, functional.ctx)
        self.functional = functional.convert(self.ctx)
        self.offset = offset

    @checked_method(in_space="domain")
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
        if not self._eq_backend_compatible(other):
            return NotImplemented
        return _scalar_eq(self.offset, other.offset) and self.functional == other.functional

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
    if _scalar_eq(offset, 0):
        return functional
    if isinstance(functional, ShiftedFunctional):
        return make_shifted_functional(functional.functional, functional.offset + offset)
    return ShiftedFunctional(functional, offset)

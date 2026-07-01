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
        """Return ``scalar * functional.grad(x)`` scaled in the domain geometry."""
        return self.domain.scale(self.scalar, self.functional.grad(x, *args, **kwargs))

    def value_and_grad(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(scalar * value, scalar * grad)`` from one fused child evaluation."""
        value, grad = self.functional.value_and_grad(x, *args, **kwargs)
        return self.scalar * value, self.domain.scale(self.scalar, grad)

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
    if _scalar_eq(scalar, 1):
        return functional
    if isinstance(functional, ScaledFunctional):
        return make_scaled_functional(scalar * functional.scalar, functional.functional)
    return ScaledFunctional(scalar, functional)

"""``Mapping``: the common parent of ``LinOp`` and ``Functional``, owning the lazy algebra.

The operator overloads live here once. What ``a + b``, ``a * b`` and ``a @ b`` *build*
is looked up in a rule table keyed by the two operand kinds, so a family plugs into
the algebra by registering rules — the way a backend plugs in by supplying ops.

Families register three primitive operations: ``"add"``, ``"mul"`` and ``"matmul"``.
Subtraction, negation, the reflected forms and ``0 + m == m`` (so builtin ``sum``
works) are derived here. Same-family and cross-family rules are the same mechanism:
``(LinOp, LinOp)`` and ``(Functional, LinOp)`` are just different keys. A pair with no
rule returns ``NotImplemented``, so Python raises ``TypeError``.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ._lazy_algebra import is_scalar_like, scalar_eq
from ._repr import describe_space
from .backend import PyTreeNode
from .contextual import ContextBound


class Scalar:
    """Rule-table key standing for any scalar-like operand."""


_RULES: dict[tuple[str, type, type], Callable[[Any, Any], Any]] = {}


def register(op: str, left: type, right: type, rule: Callable[[Any, Any], Any]) -> None:
    """Register ``rule(a, b)`` as the result of ``a <op> b`` for these operand kinds.

    ``op`` is ``"add"``, ``"mul"`` or ``"matmul"``; ``left`` and ``right`` are
    classes (matched along the MRO) or :class:`Scalar`.
    """
    _RULES[op, left, right] = rule


def _kinds(x: Any) -> tuple[type, ...]:
    if not isinstance(x, Mapping) and is_scalar_like(x):
        return (Scalar,)
    return type(x).__mro__


def _is_zero(x: Any) -> bool:
    return not isinstance(x, Mapping) and is_scalar_like(x) and scalar_eq(x, 0)


def _combine(op: str, a: Any, b: Any) -> Any:
    """Build ``a <op> b`` from the most specific registered rule, or ``NotImplemented``."""
    if op == "add":
        if _is_zero(b):
            return a
        if _is_zero(a):
            return b
    for left in _kinds(a):
        for right in _kinds(b):
            rule = _RULES.get((op, left, right))
            if rule is not None:
                return rule(a, b)
    return NotImplemented


def _negate(x: Any) -> Any:
    if isinstance(x, Mapping):
        return _combine("mul", -1, x)
    return -x if is_scalar_like(x) else NotImplemented


class Mapping(PyTreeNode, ContextBound):
    """
    A map from ``domain`` to ``codomain``, with the lazy algebra of the family.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Backend context specification.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this object. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used.
    """

    @property
    @abstractmethod
    def domain(self) -> Any:
        """Domain space."""

    @property
    @abstractmethod
    def codomain(self) -> Any:
        """Codomain: a space for operators, a :class:`~spacecore.space.Field` for functionals."""

    def __add__(self, other: Any) -> Any:
        """Return ``self + other`` per the registered ``add`` rule."""
        return _combine("add", self, other)

    def __radd__(self, other: Any) -> Any:
        """Return ``other + self``; ``0 + self`` is ``self``, so builtin ``sum`` works."""
        return _combine("add", other, self)

    def __sub__(self, other: Any) -> Any:
        """Return ``self + (-1) * other``."""
        neg = _negate(other)
        return NotImplemented if neg is NotImplemented else _combine("add", self, neg)

    def __rsub__(self, other: Any) -> Any:
        """Return ``other + (-1) * self``."""
        neg = _negate(self)
        return NotImplemented if neg is NotImplemented else _combine("add", other, neg)

    def __neg__(self) -> Any:
        """Return ``(-1) * self``."""
        neg = _negate(self)
        if neg is NotImplemented:
            raise TypeError(f"{type(self).__name__} has no scalar multiplication rule.")
        return neg

    def __mul__(self, other: Any) -> Any:
        """Return ``self * other`` per the registered ``mul`` rule."""
        return _combine("mul", self, other)

    def __rmul__(self, other: Any) -> Any:
        """Return ``other * self`` per the registered ``mul`` rule."""
        return _combine("mul", other, self)

    def __matmul__(self, other: Any) -> Any:
        """Return the composition ``self @ other`` per the registered ``matmul`` rule."""
        return _combine("matmul", self, other)

    def __rmatmul__(self, other: Any) -> Any:
        """Return the composition ``other @ self`` per the registered ``matmul`` rule."""
        return _combine("matmul", other, self)

    def __eq__(self, other: Any) -> bool:
        """No algebraic equality here; ``NotImplemented`` falls back to identity symmetrically."""
        return NotImplemented

    def _arrow(self) -> str:
        """Return the ``domain → codomain`` descriptor."""
        return f"{describe_space(self.domain)} → {describe_space(self.codomain)}"

    def _repr_body(self) -> str:
        return self._arrow()

    def _short_repr(self) -> str:
        """Return a bounded ``ClassName(domain → codomain)`` form, so nested trees stay short."""
        return f"{type(self).__name__}({self._arrow()})"

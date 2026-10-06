"""
``Mapping``, the parent of ``LinOp`` and ``Functional``, and the shared lazy algebra.

:class:`Algebraic` holds the operator overloads for ``Mapping`` and ``Element``.
What ``a + b``, ``a * b`` and ``a @ b`` build is looked up in a rule table keyed by
the operand kinds. A family registers ``add``, ``mul`` and ``matmul`` rules, and
subtraction, negation and ``sum()`` are derived; a pair with no rule raises
``TypeError``. :class:`ScaledNode`, :class:`SumNode` and :class:`ZeroNode` are the
shared lazy nodes. A family provides ``_family``, ``_init_map`` and its
``_Sum``/``_Scaled``/``_Zero`` classes.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from .._checks import minimum_check_level
from ._lazy_algebra import is_scalar_like, scalar_eq
from .._repr import describe_space
from ..backend import PyTreeNode
from ..backend._container import field_values
from ..contextual import ContextBound


class Scalar:
    """Rule-table key standing for any scalar-like operand."""


_RULES: dict[tuple[str, type, type], Callable[..., Any]] = {}


def register(op: str, left: type, right: type, rule: Callable[..., Any]) -> None:
    """Register ``rule(a, b)`` as the result of ``a <op> b`` for these operand kinds.

    ``op`` is ``"add"``, ``"mul"`` or ``"matmul"``; ``left`` and ``right`` are
    classes (matched along the MRO) or :class:`Scalar`. For ``"prox"``, keys
    are functional and geometry classes, and the rule takes ``(f, geometry, x,
    step)``. ``object`` matches any geometry, including a native space without
    a separate geometry object; register it only for geometry-independent rules.
    """
    _RULES[op, left, right] = rule


def _kinds(x: Any) -> tuple[type, ...]:
    if not isinstance(x, Algebraic) and is_scalar_like(x):
        return (Scalar,)
    return type(x).__mro__


def _is_zero(x: Any) -> bool:
    return not isinstance(x, Algebraic) and is_scalar_like(x) and scalar_eq(x, 0)


def _find_rule(op: str, a: Any, b: Any) -> Callable[..., Any] | None:
    # Most specific rule wins: walk both MROs, left operand first.
    for left in _kinds(a):
        for right in _kinds(b):
            rule = _RULES.get((op, left, right))
            if rule is not None:
                return rule
    return None


def _combine(op: str, a: Any, b: Any) -> Any:
    """Build ``a <op> b`` from the most specific registered rule, or ``NotImplemented``."""
    # m + 0 and 0 + m are m, so builtin sum() can start from 0.
    if op == "add":
        if _is_zero(b):
            return a
        if _is_zero(a):
            return b
    rule = _find_rule(op, a, b)
    if rule is not None:
        return rule(a, b)
    # No rule: Python tries the reflected operator, then raises TypeError.
    return NotImplemented


def _negate(x: Any) -> Any:
    if isinstance(x, Algebraic):
        return _combine("mul", -1, x)
    return -x if is_scalar_like(x) else NotImplemented


class Algebraic:
    """Operator overloads driven by the rule table; shared by ``Mapping`` and ``Element``."""

    __slots__ = ()

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
        # Negation is (-1) * self; a family without scalar mul cannot negate.
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


class Mapping(Algebraic, PyTreeNode, ContextBound):
    """
    A map from ``domain`` to ``codomain``, with the lazy algebra of the family.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Backend context specification.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    @property
    @abstractmethod
    def domain(self) -> Any:
        """Domain space."""

    @property
    @abstractmethod
    def codomain(self) -> Any:
        """Codomain: a space for operators, a :class:`~spacecore.space.Field` for functionals."""

    def __eq__(self, other: Any) -> bool:
        """Compare declared ``_fields``; without them, fall back to identity symmetrically."""
        if not getattr(type(self), "_fields", ()) or not self.same_math(other):
            return NotImplemented
        ops = self.ops
        mine, theirs = field_values(self), field_values(other)
        for name, a in mine.items():
            b = theirs[name]
            if name in ("ctx", "check_level"):
                continue
            if ops.is_dense(a) or ops.is_dense(b):
                # Arrays: same shape and numerically close, NaN matching NaN.
                if not (ops.is_dense(a) and ops.is_dense(b) and tuple(a.shape) == tuple(b.shape)
                        and bool(ops.allclose(a, b, equal_nan=True))):
                    return False
            elif callable(a) and not isinstance(a, Mapping):
                # User callables: equal only when they are the same object.
                if a is not b:
                    return False
            elif not scalar_eq(a, b):
                return False
        return True

    def _arrow(self) -> str:
        """Return the ``domain → codomain`` descriptor."""
        return f"{describe_space(self.domain)} → {describe_space(self.codomain)}"

    def _repr_body(self) -> str:
        return self._arrow()

    def _short_repr(self) -> str:
        """Return a bounded ``ClassName(domain → codomain)`` form, so nested trees stay short."""
        return f"{type(self).__name__}({self._arrow()})"


def same_space_for_algebra(left: Any, right: Any) -> bool:
    """Return whether two spaces are compatible for algebraic combination."""
    # Equal spaces, or the same type and math context after conversion.
    if left == right:
        return True
    if type(left) is not type(right) or not left.ctx.same_math(right.ctx):
        return False
    try:
        return left.convert(right.ctx) == right
    except Exception:
        return False


def _checked_parts(family: type, parts: Any, node: str) -> tuple[Any, ...]:
    """Return ``parts`` as a tuple after checking type, count, and shared endpoints."""
    # Reject empty input and operands of another family first.
    parts = tuple(parts)
    if not parts:
        raise ValueError(f"{node} requires a nonempty sequence of {family.__name__} operands.")
    for i, part in enumerate(parts):
        if not isinstance(part, family):
            raise TypeError(f"operand {i} must be a {family.__name__}, got {type(part).__name__}.")
    # Every operand must share operand 0's context, domain and codomain.
    ctx, dom, cod = parts[0].ctx, parts[0].domain, parts[0].codomain
    for i, part in enumerate(parts[1:], start=1):
        if not ctx.same_math(part.ctx):
            raise ValueError(
                f"All {family.__name__} operands in an algebraic expression must have the same "
                f"ctx; operand 0 has ctx {ctx!r}, operand {i} has ctx {part.ctx!r}."
            )
        if not (same_space_for_algebra(part.domain, dom) and same_space_for_algebra(part.codomain, cod)):
            raise ValueError(
                f"All {node} operands must have the same domain and codomain; operand 0 maps "
                f"{dom!r} -> {cod!r}, operand {i} maps {part.domain!r} -> {part.codomain!r}."
            )
    return parts


class ScaledNode:
    """Lazy ``scalar * operand``, shared by every family; the family class adds evaluation."""

    def __init__(self, scalar: Any, operand: Any, check_level: Any = None) -> None:
        # Only operands of this node's own family can be scaled here.
        if not isinstance(operand, self._family):
            raise TypeError(f"operand must be a {self._family.__name__}, got {type(operand).__name__}.")
        if not is_scalar_like(scalar):
            raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
        # The policy defaults from the operand, so the result does not depend on operand order.
        level = operand.check_level if check_level is None else check_level
        # Bind with the operand's endpoints and context.
        self._init_map(operand.domain, operand.codomain, operand.ctx, level)
        self.scalar, self.operand = scalar, operand.convert(self.ctx)

class SumNode:
    """Lazy finite sum of operands sharing domain and codomain; the family class adds evaluation."""

    def __init__(self, parts: Any, check_level: Any = None) -> None:
        # Same family, same endpoints; the least strict operand's level by default.
        parts = _checked_parts(self._family, parts, type(self).__name__)
        if check_level is None:
            check_level = minimum_check_level(tuple(part.check_level for part in parts))
        self._init_map(parts[0].domain, parts[0].codomain, parts[0].ctx, check_level)
        self.parts = tuple(part.convert(self.ctx) for part in parts)

class ZeroNode:
    """The zero map of a family; the family class adds evaluation."""

    @classmethod
    def _of(cls, domain: Any, codomain: Any, ctx: Any, check_level: Any = None) -> Any:
        """Build the zero map; a family whose constructor orders arguments differently overrides this."""
        return cls(domain, codomain, ctx, check_level)

    def __eq__(self, other: Any) -> bool:
        """Return whether ``other`` is the zero map between the same spaces."""
        if not self.same_math(other):
            return NotImplemented
        return self.domain == other.domain and self.codomain == other.codomain

    def tree_flatten(self):
        """No children: the spaces, context and policy are static."""
        return (), (self.domain, self.codomain, self.ctx, self.check_level)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the static data."""
        return cls._of(*aux)

    def _convert(self, new_ctx: Any) -> Any:
        return self._of(
            self.domain.convert(new_ctx), self.codomain.convert(new_ctx), new_ctx, self.check_level
        )


def make_sum_of(family: type, parts: Any) -> Any:
    """Return the canonical lazy sum: nested sums flattened, zeros dropped, singletons unwrapped.

    Every operand's type and endpoints are checked before any zero is dropped, so a
    mismatched zero term cannot vanish silently.
    """
    # Type-check before flattening, so a foreign operand fails clearly.
    parts = tuple(parts)
    for i, part in enumerate(parts):
        if not isinstance(part, family):
            raise TypeError(f"operand {i} must be a {family.__name__}, got {type(part).__name__}.")
    # Flatten nested sums, then validate every term, zeros included.
    def flatten(terms):
        for term in terms:
            if isinstance(term, SumNode):
                yield from flatten(term.parts)
            else:
                yield term

    flat = _checked_parts(family, flatten(parts), family._Sum.__name__)
    # Drop zeros; collapse to a zero map, a single term, or a sum node.
    nonzero = tuple(term for term in flat if not isinstance(term, ZeroNode))
    if not nonzero:
        return family._Zero._of(flat[0].domain, flat[0].codomain, flat[0].ctx)
    return nonzero[0] if len(nonzero) == 1 else family._Sum(nonzero)


def make_scaled_of(family: type, scalar: Any, operand: Any) -> Any:
    """Return the canonical ``scalar * operand``: ``0`` gives zero, ``1`` the operand, nested scalars fold."""
    if not isinstance(operand, family):
        raise TypeError(f"operand must be a {family.__name__}, got {type(operand).__name__}.")
    if not is_scalar_like(scalar):
        raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
    # 0 gives zero, 1 gives the operand, nested scalars multiply out.
    current = operand
    while True:
        if scalar_eq(scalar, 0):
            return family._Zero._of(operand.domain, operand.codomain, operand.ctx)
        if scalar_eq(scalar, 1) or isinstance(current, ZeroNode):
            return current
        if not isinstance(current, ScaledNode):
            return family._Scaled(scalar, current)
        scalar, current = scalar * current.scalar, current.operand

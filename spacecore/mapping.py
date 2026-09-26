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

from ._check_policy import minimum_check_level
from ._lazy_algebra import finalize_sum, flatten_sum, fold_scaled, is_scalar_like, scalar_eq
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
    if not isinstance(x, Algebraic) and is_scalar_like(x):
        return (Scalar,)
    return type(x).__mro__


def _is_zero(x: Any) -> bool:
    return not isinstance(x, Algebraic) and is_scalar_like(x) and scalar_eq(x, 0)


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


def same_space_for_algebra(left: Any, right: Any) -> bool:
    """Return whether two spaces are compatible for algebraic combination."""
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
    parts = tuple(parts)
    if not parts:
        raise ValueError(f"{node} requires a nonempty sequence of {family.__name__} operands.")
    for i, part in enumerate(parts):
        if not isinstance(part, family):
            raise TypeError(f"operand {i} must be a {family.__name__}, got {type(part).__name__}.")
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
        if not isinstance(operand, self._family):
            raise TypeError(f"operand must be a {self._family.__name__}, got {type(operand).__name__}.")
        if not is_scalar_like(scalar):
            raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
        # The policy defaults from the operand, so the result does not depend on operand order.
        level = operand.check_level if check_level is None else check_level
        self._init_map(operand.domain, operand.codomain, operand.ctx, level)
        self.scalar, self.operand = scalar, operand.convert(self.ctx)

    def __eq__(self, other: Any) -> bool:
        """Return whether ``other`` has the same scalar and operand."""
        if not self.same_math(other):
            return NotImplemented
        return scalar_eq(self.scalar, other.scalar) and self.operand == other.operand

    def tree_flatten(self):
        """Expose the scalar (traced) and the operand as children."""
        return (self.scalar, self.operand), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the scalar and the operand."""
        return cls(*children)

    def _convert(self, new_ctx: Any) -> Any:
        return type(self)(self.scalar, self.operand.convert(new_ctx))


class SumNode:
    """Lazy finite sum of operands sharing domain and codomain; the family class adds evaluation."""

    def __init__(self, parts: Any, check_level: Any = None) -> None:
        parts = _checked_parts(self._family, parts, type(self).__name__)
        if check_level is None:
            check_level = minimum_check_level(tuple(part.check_level for part in parts))
        self._init_map(parts[0].domain, parts[0].codomain, parts[0].ctx, check_level)
        self.parts = tuple(part.convert(self.ctx) for part in parts)

    def __eq__(self, other: Any) -> bool:
        """Return whether ``other`` has the same operands, in order (``A + B != B + A``)."""
        if not self.same_math(other):
            return NotImplemented
        return len(self.parts) == len(other.parts) and all(
            a == b for a, b in zip(self.parts, other.parts)
        )

    def tree_flatten(self):
        """Expose the operands as children."""
        return self.parts, ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the operands."""
        return cls(tuple(children))

    def _convert(self, new_ctx: Any) -> Any:
        return type(self)(tuple(part.convert(new_ctx) for part in self.parts))


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
    parts = tuple(parts)
    for i, part in enumerate(parts):
        if not isinstance(part, family):
            raise TypeError(f"operand {i} must be a {family.__name__}, got {type(part).__name__}.")
    flat = flatten_sum(parts, is_sum=lambda t: isinstance(t, SumNode), parts=lambda t: t.parts)
    flat = _checked_parts(family, flat, family._Sum.__name__)
    return finalize_sum(
        flat,
        is_zero=lambda t: isinstance(t, ZeroNode),
        make_zero=lambda: family._Zero._of(flat[0].domain, flat[0].codomain, flat[0].ctx),
        make_sum_node=family._Sum,
    )


def make_scaled_of(family: type, scalar: Any, operand: Any) -> Any:
    """Return the canonical ``scalar * operand``: ``0`` gives zero, ``1`` the operand, nested scalars fold."""
    if not isinstance(operand, family):
        raise TypeError(f"operand must be a {family.__name__}, got {type(operand).__name__}.")
    if not is_scalar_like(scalar):
        raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
    return fold_scaled(
        scalar,
        operand,
        is_zero=lambda o: isinstance(o, ZeroNode),
        unwrap_scaled=lambda o: (o.scalar, o.operand) if isinstance(o, ScaledNode) else None,
        make_zero=lambda: family._Zero._of(operand.domain, operand.codomain, operand.ctx),
        make_scaled_node=family._Scaled,
    )

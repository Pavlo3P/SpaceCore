"""Canonicalization primitives shared by the lazy-algebra factories.

The LinOp and Functional algebras build the same kind of expression tree — lazy
*sums*, *scalar multiples*, and *zero* elements with only local canonicalization
(flatten nested sums, drop zeros, unwrap singletons, fold nested scalings). The
node *semantics* differ (``apply``/``rapply`` for operators, ``value``/``grad``
for functionals) and stay in each domain; only this canonicalization skeleton is
shared here, so ``make_sum`` / ``make_scaled`` are not duplicated between
:mod:`spacecore.linop` and :mod:`spacecore.functional`.

Each generic is parameterized by a tiny "node vocabulary" (predicates and node
factories) so a caller stays fully in control of the concrete node types.
"""
from __future__ import annotations

from numbers import Number
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")


def is_scalar_like(value: Any) -> bool:
    """Return whether ``value`` can be used as a scalar multiplier."""
    if isinstance(value, Number):
        return True
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape) == ()
    return getattr(value, "ndim", None) == 0


def scalar_eq(a: Any, b: Any) -> bool:
    """Return whether two scalar-likes are equal, NaN-reflexive, as a real ``bool``.

    Two matching NaN scalars compare equal (mirrors ``equal_nan=True``), so a
    NaN-scaled node equals itself. Always returns a genuine Python ``bool`` — a
    0-d backend-array ``==`` would yield ``np.bool_``, which leaks through the
    ``and`` combinator of any container's ``__eq__``.
    """
    try:
        if bool(a == b):
            return True
        # ``x != x`` is True only for NaN (including a complex value with a NaN
        # component), so this matches NaN against NaN.
        return bool(a != a) and bool(b != b)
    except Exception:
        return False


def flatten_sum(
    terms: Sequence[T],
    *,
    is_sum: Callable[[T], bool],
    parts: Callable[[T], Sequence[T]],
) -> tuple[T, ...]:
    """Flatten nested sum nodes into a flat tuple of terms.

    Parameters
    ----------
    terms:
        Operands to flatten (already validated by the caller).
    is_sum:
        Predicate that is ``True`` for a sum node.
    parts:
        Accessor returning the child terms of a sum node.
    """
    flat: list[T] = []
    for term in terms:
        if is_sum(term):
            flat.extend(flatten_sum(parts(term), is_sum=is_sum, parts=parts))
        else:
            flat.append(term)
    return tuple(flat)


def finalize_sum(
    terms: Sequence[T],
    *,
    is_zero: Callable[[T], bool],
    make_zero: Callable[[], T],
    make_sum_node: Callable[[tuple[T, ...]], T],
) -> T:
    """Drop zero terms, then collapse to zero, unwrap a singleton, or build a node.

    The caller is expected to have already flattened and validated ``terms``.
    """
    nonzero = tuple(term for term in terms if not is_zero(term))
    if not nonzero:
        return make_zero()
    if len(nonzero) == 1:
        return nonzero[0]
    return make_sum_node(nonzero)


def fold_scaled(
    scalar: Any,
    operand: T,
    *,
    is_zero: Callable[[T], bool],
    unwrap_scaled: Callable[[T], "tuple[Any, T] | None"],
    make_zero: Callable[[], T],
    make_scaled_node: Callable[[Any, T], T],
) -> T:
    """Canonicalize ``scalar * operand``.

    ``0 -> zero``, ``1 -> operand``, a zero operand passes through, nested
    scalings fold into one coefficient, otherwise a scaled node is built.
    ``unwrap_scaled(operand)`` returns ``(inner_scalar, inner_operand)`` for a
    scaled node, or ``None`` otherwise.
    """
    if scalar_eq(scalar, 0):
        return make_zero()
    if scalar_eq(scalar, 1):
        return operand
    if is_zero(operand):
        return operand
    nested = unwrap_scaled(operand)
    if nested is not None:
        inner_scalar, inner = nested
        return fold_scaled(
            scalar * inner_scalar,
            inner,
            is_zero=is_zero,
            unwrap_scaled=unwrap_scaled,
            make_zero=make_zero,
            make_scaled_node=make_scaled_node,
        )
    return make_scaled_node(scalar, operand)

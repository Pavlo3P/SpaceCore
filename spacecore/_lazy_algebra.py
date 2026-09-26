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


def conjugate_scalar(value: Any) -> Any:
    """Return the scalar conjugate when the value supports conjugation.

    Duck-typed rather than backend-dispatched: Python numbers, NumPy/CuPy
    scalars, and 0-d Torch/JAX arrays all expose ``conjugate`` or ``conj``, and
    a value that exposes neither is its own conjugate as far as this module is
    concerned.
    """
    if hasattr(value, "conjugate"):
        return value.conjugate()
    if hasattr(value, "conj"):
        return value.conj()
    return value


def scalar_eq(a: Any, b: Any) -> bool:
    """
    Return whether two scalars are recognizably equal, as a plain ``bool``.

    A traced scalar cannot be compared, so the answer is ``False``, and a canonicalizer
    then skips the simplification, which is always sound. Only ``TypeError`` is caught.
    Two NaNs compare equal.
    """
    try:
        if bool(a == b):
            return True
        # ``x != x`` is True only for NaN (including a complex value with a NaN
        # component), so this matches NaN against NaN.
        return bool(a != a) and bool(b != b)
    except TypeError:
        return False


def is_recognizably_nonreal(value: Any) -> bool:
    """Return whether ``value`` is *provably* a non-real scalar.

    The polarity matters: this answers "can we see an imaginary part?", not "is
    this real?". A traced scalar has no value to inspect, so it is reported as
    ``False`` (not provably non-real) and callers let it through. Rejecting on
    ``True`` therefore never fires spuriously under ``jax.jit``; it only rejects
    a concrete value that genuinely carries an imaginary part.

    Used by spaces whose scalar field is narrower than their entry dtype (see
    :attr:`spacecore.space.Space.scalar_field`), where an imaginary multiplier
    would silently produce an element outside the space.
    """
    try:
        # A NaN in any component makes ``!=`` uninformative (``nan != nan`` is
        # True for a *real* NaN), so nothing is provable and we let it through.
        if bool(value != value):
            return False
        return bool(value != conjugate_scalar(value))
    except TypeError:
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
    # Recursively splice the parts of nested sums.
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
    # 0 gives zero, 1 gives the operand, a zero operand stays zero.
    if scalar_eq(scalar, 0):
        return make_zero()
    if scalar_eq(scalar, 1):
        return operand
    if is_zero(operand):
        return operand
    # Fold s * (t * A) into (s t) * A.
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

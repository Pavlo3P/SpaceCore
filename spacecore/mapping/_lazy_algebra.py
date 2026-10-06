"""Scalar helpers shared by the operator and functional algebras."""
from __future__ import annotations

from numbers import Number
from typing import Any


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

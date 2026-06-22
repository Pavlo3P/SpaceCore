"""Shared helpers for compact, Jupyter-friendly ``__repr__`` output.

The goal is a single, consistent representation style across spaces, linear
operators, functionals, and other context-bound values:

* **algebra first** — what the object *is* mathematically (shape, field,
  domain/codomain) leads the line;
* **backend second** — a terse ``backend='numpy', dtype=float64`` tag follows;
* **never dump arrays** — array fields are summarized as
  ``<array shape=(m, n), dtype=float64>`` rather than printed in full.

This module deliberately imports nothing from the rest of ``spacecore`` so it
can be used by the lowest layers (``ContextBound``) without import cycles.
"""

from __future__ import annotations

from typing import Any, Iterable, cast

#: Mathematical-field glyphs used in space descriptors.
_FIELD_SYMBOLS = {"real": "ℝ", "complex": "ℂ"}  # ℝ, ℂ

#: 0-d dtype strings recognized as boolean.
_BOOL_DTYPES = {"bool", "bool_", "torch.bool"}


def field_symbol(field: str | None) -> str:
    """Return the glyph (``ℝ``/``ℂ``) for a scalar field name."""
    if field is None:
        return "?"
    return _FIELD_SYMBOLS.get(field, "?")


def format_dtype(dtype: Any) -> str:
    """Return a short dtype name (``float64``) rather than ``dtype('float64')``.

    Backend-qualified names such as torch's ``torch.float32`` are stripped to the
    bare scalar name, since the ``backend=`` tag already records the family. Bare
    scalar *types* (rather than dtype instances) fall back to their class name.
    """
    if dtype is None:
        return "None"
    if isinstance(dtype, type):
        return getattr(dtype, "__name__", str(dtype))
    name = getattr(dtype, "name", None)
    text = name if isinstance(name, str) else str(dtype)
    return text.rsplit(".", 1)[-1]


def describe_space(space: Any) -> str:
    """Return a space's compact math descriptor (e.g. ``ℝ^(2, 3)``).

    Falls back to ``repr`` for objects that do not expose
    :meth:`_space_descriptor`, so operator/functional arrows stay robust during
    partial rollouts.
    """
    descriptor = getattr(space, "_space_descriptor", None)
    if callable(descriptor):
        return cast(str, descriptor())
    return repr(space)


def _format_scalar(value: Any, dtype: Any = None) -> str:
    """Format a real or complex scalar compactly (``.6g``), complex-aware."""
    dstr = str(dtype) if dtype is not None else ""
    kind = getattr(dtype, "kind", None)
    try:
        if kind == "c" or "complex" in dstr or isinstance(value, complex):
            return f"{complex(value):.6g}"
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return repr(value)


def summarize_value(value: Any) -> str:
    """Return a compact one-line summary for arrays, scalars, and pytrees.

    Context-bound objects describe themselves via :meth:`_short_repr`; dense or
    sparse arrays collapse to ``<array shape=..., dtype=...>``; 0-d arrays and
    Python scalars render numerically (complex-aware); tuples/lists recurse.

    The function is written to never raise on pathological inputs, so it is safe
    to use from any ``__repr__``.
    """
    short = getattr(value, "_short_repr", None)
    if callable(short):
        return cast(str, short())

    try:
        shape = getattr(value, "shape", None)
    except Exception:
        shape = None
    if shape is not None:
        try:
            shape_text = tuple(shape)
        except TypeError:
            shape_text = None
        if shape_text is not None:
            dtype = getattr(value, "dtype", None)
            if shape_text == ():
                dstr = str(dtype)
                if dstr in _BOOL_DTYPES or getattr(dtype, "kind", None) == "b":
                    try:
                        return repr(bool(value))
                    except Exception:
                        return repr(value)
                return _format_scalar(value, dtype)
            dtype_text = "" if dtype is None else f", dtype={format_dtype(dtype)}"
            return f"<array shape={shape_text}{dtype_text}>"

    # Python scalars: keep ints/bools exact, collapse floats/complex to .6g so a
    # Python ``2.0`` and a 0-d backend ``2.0`` array render identically.
    if isinstance(value, bool) or isinstance(value, int):
        return repr(value)
    if isinstance(value, (float, complex)):
        return _format_scalar(value)
    if isinstance(value, tuple):
        inner = ", ".join(summarize_value(part) for part in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(summarize_value(part) for part in value) + "]"
    return repr(value)


def shape_descriptor(field: str | None, shape: Any) -> str:
    """Render a coordinate descriptor like ``ℝ^5`` or ``ℝ^(2, 3)``.

    A scalar (empty shape) renders as the bare field glyph; a 1-D shape drops
    the tuple parentheses for readability.
    """
    sym = field_symbol(field)
    shape = tuple(shape)
    if not shape:
        return sym
    if len(shape) == 1:
        return f"{sym}^{shape[0]}"
    return f"{sym}^{shape!r}"


def truncated_join(parts: Iterable[str], sep: str, limit: int = 6) -> str:
    """Join ``parts`` with ``sep``, abbreviating once more than ``limit`` items.

    Keeps wide operand lists (e.g. a 20-term sum) from producing an unreadable
    one-liner.
    """
    parts = list(parts)
    if len(parts) <= limit:
        return sep.join(parts)
    head = sep.join(parts[:limit])
    return f"{head}{sep}…(+{len(parts) - limit} more)"

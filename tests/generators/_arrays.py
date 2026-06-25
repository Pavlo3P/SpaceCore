from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from spacecore.backend import Context
from spacecore.types import DenseArray

from ._protocol import GeneratedCase
from ._seed import resolve_rng


Field = Literal["real", "complex"]
DEFAULT_DENSE_SHAPES: tuple[tuple[int, ...], ...] = ((), (3,), (2, 3), (2, 2, 2))


def _shape(value: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(dimension) for dimension in value)
    if any(dimension <= 0 for dimension in result):
        raise ValueError(f"{name} must contain only positive dimensions, got {result!r}.")
    return result


def _numpy_dtype(ctx: Context) -> np.dtype[Any]:
    try:
        return np.dtype(ctx.dtype)
    except TypeError:
        text = str(ctx.dtype)
        for name in ("complex128", "complex64", "float64", "float32"):
            if name in text:
                return np.dtype(name)
    raise TypeError(f"Cannot map context dtype {ctx.dtype!r} to a NumPy dtype.")


def _field(ctx: Context, requested: Field | None) -> Field:
    context_field: Field = "complex" if ctx.ops.is_complex_dtype(ctx.dtype) else "real"
    if requested not in (None, "real", "complex"):
        raise ValueError(f"field must be 'real', 'complex', or None, got {requested!r}.")
    result = context_field if requested is None else requested
    if result == "complex" and context_field == "real":
        raise TypeError("Cannot generate complex-valued data for a real context.")
    return result


def _shape_id(shape: tuple[int, ...]) -> str:
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def dense_array_case(
    ctx: Context,
    shape: Sequence[int],
    *,
    batch_shape: Sequence[int] = (),
    field: Field | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GeneratedCase[DenseArray]:
    """Generate one backend dense array and its NumPy reference."""
    generator = resolve_rng(seed=seed, rng=rng)
    element_shape = _shape(shape, "shape")
    prefix = _shape(batch_shape, "batch_shape")
    value_field = _field(ctx, field)
    full_shape = prefix + element_shape
    reference = generator.standard_normal(full_shape)
    if value_field == "complex":
        reference = reference + 1j * generator.standard_normal(full_shape)
    reference = np.asarray(reference, dtype=_numpy_dtype(ctx))
    capabilities = {"dense", value_field}
    if prefix:
        capabilities.add("batched")
    return GeneratedCase(
        obj=ctx.asarray(reference),
        reference={
            "array": reference.copy(),
            "shape": element_shape,
            "batch_shape": prefix,
            "field": value_field,
            "dtype": reference.dtype,
        },
        capabilities=frozenset(capabilities),
        id=(f"dense-{value_field}-{_shape_id(element_shape)}" + (f"-batch-{'x'.join(map(str, prefix))}" if prefix else "")),
    )


def dense_array_cases(
    ctx: Context,
    *,
    shapes: Sequence[Sequence[int]] = DEFAULT_DENSE_SHAPES,
    batch_shape: Sequence[int] = (),
    field: Field | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GeneratedCase[DenseArray], ...]:
    """Generate scalar, vector, matrix, and small-tensor dense cases."""
    generator = resolve_rng(seed=seed, rng=rng)
    return tuple(
        dense_array_case(
            ctx,
            shape,
            batch_shape=batch_shape,
            field=field,
            seed=None,
            rng=generator,
        )
        for shape in shapes
    )

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spacecore.backend import Context
from spacecore.types import DenseArray

from ._arrays import _field, _numpy_dtype, _shape_id
from ._protocol import GeneratedCase
from ._seed import resolve_rng


def hermitian_case(
    ctx: Context,
    size: int,
    *,
    batch_shape: Sequence[int] = (),
    complex: bool | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GeneratedCase[DenseArray]:
    """Generate a real symmetric or complex Hermitian square matrix."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}.")
    prefix = tuple(int(dimension) for dimension in batch_shape)
    if any(dimension <= 0 for dimension in prefix):
        raise ValueError(f"batch_shape must contain positive dimensions, got {prefix!r}.")
    inferred_complex = ctx.ops.is_complex_dtype(ctx.dtype)
    use_complex = inferred_complex if complex is None else complex
    _field(ctx, "complex" if use_complex else "real")
    generator = resolve_rng(seed=seed, rng=rng)
    shape = prefix + (size, size)
    raw = generator.standard_normal(shape)
    if use_complex:
        raw = raw + 1j * generator.standard_normal(shape)
    matrix = (raw + np.swapaxes(raw.conj(), -1, -2)) / 2
    matrix = np.asarray(matrix, dtype=_numpy_dtype(ctx))
    capabilities = {"hermitian", "complex" if use_complex else "real"}
    if prefix:
        capabilities.add("batched")
    return GeneratedCase(
        obj=ctx.asarray(matrix),
        reference={
            "matrix": matrix.copy(),
            "size": size,
            "batch_shape": prefix,
            "field": "complex" if use_complex else "real",
        },
        capabilities=frozenset(capabilities),
        id=(f"hermitian-{'complex' if use_complex else 'real'}-{size}" + (f"-batch-{_shape_id(prefix)}" if prefix else "")),
    )


def hermitian_cases(
    ctx: Context,
    *,
    sizes: Sequence[int] = (2, 3),
    batch_shape: Sequence[int] = (),
    complex: bool | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GeneratedCase[DenseArray], ...]:
    """Generate a deterministic collection of Hermitian matrices."""
    generator = resolve_rng(seed=seed, rng=rng)
    return tuple(
        hermitian_case(
            ctx,
            size,
            batch_shape=batch_shape,
            complex=complex,
            seed=None,
            rng=generator,
        )
        for size in sizes
    )

from __future__ import annotations

import numpy as np

from spacecore.backend import Context
from spacecore.types import DenseArray

from ._arrays import _field, _numpy_dtype
from ._protocol import GeneratedCase
from ._seed import resolve_rng


def spd_metric_case(
    ctx: Context,
    size: int,
    *,
    condition_number: float = 10.0,
    complex: bool | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GeneratedCase[DenseArray]:
    """Generate a small SPD or Hermitian-positive-definite metric matrix."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}.")
    if condition_number < 1 or not np.isfinite(condition_number):
        raise ValueError("condition_number must be finite and at least 1.")
    inferred_complex = ctx.ops.is_complex_dtype(ctx.dtype)
    use_complex = inferred_complex if complex is None else complex
    _field(ctx, "complex" if use_complex else "real")
    generator = resolve_rng(seed=seed, rng=rng)
    raw = generator.standard_normal((size, size))
    if use_complex:
        raw = raw + 1j * generator.standard_normal((size, size))
    basis, _ = np.linalg.qr(raw)
    eigenvalues = np.geomspace(1.0, float(condition_number), size)
    matrix = (basis * eigenvalues) @ basis.conj().T
    matrix = (matrix + matrix.conj().T) / 2
    matrix = np.asarray(matrix, dtype=_numpy_dtype(ctx))
    inverse = np.asarray(np.linalg.inv(matrix), dtype=matrix.dtype)
    condition_estimate = float(np.linalg.cond(matrix))
    return GeneratedCase(
        obj=ctx.asarray(matrix),
        reference={
            "matrix": matrix.copy(),
            "inverse": inverse,
            "eigenvalues": eigenvalues,
            "condition_estimate": condition_estimate,
            "condition_number": float(condition_number),
            "field": "complex" if use_complex else "real",
        },
        capabilities=frozenset(
            {"metric", "spd", "hermitian", "complex" if use_complex else "real"}
        ),
        id=f"spd-{'complex' if use_complex else 'real'}-{size}-cond-{condition_number:g}",
    )


def spd_metric_cases(
    ctx: Context,
    *,
    sizes: tuple[int, ...] = (2, 3),
    condition_number: float = 10.0,
    complex: bool | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GeneratedCase[DenseArray], ...]:
    """Generate deterministic metric cases for several small dimensions."""
    generator = resolve_rng(seed=seed, rng=rng)
    return tuple(
        spd_metric_case(
            ctx,
            size,
            condition_number=condition_number,
            complex=complex,
            seed=None,
            rng=generator,
        )
        for size in sizes
    )

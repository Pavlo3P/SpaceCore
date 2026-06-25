from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pytest

from spacecore.backend import CHECK_LEVELS, CheckLevel

from ._contexts import context_cases


@dataclass(frozen=True, slots=True)
class BatchCase:
    """Describe whether generated values carry a leading batch shape."""

    batched: bool
    batch_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.batched != bool(self.batch_shape):
            raise ValueError("batched must be true exactly when batch_shape is nonempty.")
        if any(dimension <= 0 for dimension in self.batch_shape):
            raise ValueError("batch_shape dimensions must be positive.")

    @property
    def id(self) -> str:
        return "unbatched" if not self.batched else f"batch-{'x'.join(map(str, self.batch_shape))}"


def check_level_params(
    levels: Iterable[CheckLevel | str] = CHECK_LEVELS,
) -> tuple[Any, ...]:
    """Return stable pytest parameters for every accepted runtime check level."""
    params = []
    for level in levels:
        if level not in CHECK_LEVELS:
            allowed = ", ".join(repr(item) for item in CHECK_LEVELS)
            raise ValueError(f"Unknown check level {level!r}. Expected one of: {allowed}.")
        params.append(pytest.param(level, id=f"checks-{level}"))
    return tuple(params)


def batch_cases(batch_size: int = 2) -> tuple[BatchCase, BatchCase]:
    """Return the standard unbatched and leading-axis batch descriptors."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
    return BatchCase(False, ()), BatchCase(True, (int(batch_size),))


def batch_params(batch_size: int = 2) -> tuple[Any, ...]:
    """Return stable pytest parameters for unbatched and batched cases."""
    return tuple(pytest.param(case, id=case.id) for case in batch_cases(batch_size))


def context_params(**kwargs: Any) -> tuple[Any, ...]:
    """Return contexts as pytest parameters, preserving skip marks and readable ids."""
    return tuple(
        pytest.param(case.obj, marks=case.marks, id=case.id) for case in context_cases(**kwargs)
    )

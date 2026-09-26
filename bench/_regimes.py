"""Configuration regimes of ADR-023, reduced to the baseline.

The regime axis existed to time each operation with the ADR-016 dispatch layer
off, on, and in verify mode. That layer was removed in 0.5.0a (its specs applied
only to eager NumPy, where they measured mostly slower, and never under
``jax.jit``), so every probe runs the ``baseline`` regime. The names stay so
result rows keep their ``regime`` coordinate.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

Regime = Literal["baseline"]

REGIMES: tuple[Regime, ...] = ("baseline",)
"""Every regime the result model and dashboard understand."""

BASELINE: Regime = "baseline"


def regimes_for(family: str, requested: tuple[Regime, ...] | None) -> tuple[Regime, ...]:
    """Return the regimes to sweep for one probe family: always the baseline."""
    return (BASELINE,)


@contextmanager
def benchmark_regime(regime: Regime) -> Iterator[None]:
    """Run the timed section under ``regime``; the baseline needs no configuration."""
    yield

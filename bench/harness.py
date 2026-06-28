from __future__ import annotations

import gc
import tracemalloc
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter_ns
from typing import Any, Callable


def sync_result(value: Any) -> Any:
    """Synchronize asynchronous backend results before timing stops."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, (tuple, list)):
        for item in value:
            sync_result(item)
    elif isinstance(value, dict):
        for item in value.values():
            sync_result(item)
    else:
        try:
            import torch

            if isinstance(value, torch.Tensor) and value.is_cuda:
                torch.cuda.synchronize(value.device)
        except Exception:
            pass
    return value


def time_op(
    fn: Callable[[], Any],
    *,
    repeat: int = 20,
    number: int = 100,
    warmup: int = 2,
) -> dict[str, float]:
    """Return best and median per-call runtime in nanoseconds."""
    if repeat <= 0 or number <= 0 or warmup < 0:
        raise ValueError("repeat and number must be positive; warmup must be non-negative.")
    for _ in range(warmup):
        sync_result(fn())
    samples: list[float] = []
    for _ in range(repeat):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = perf_counter_ns()
            result = None
            for _ in range(number):
                result = fn()
            sync_result(result)
            elapsed = perf_counter_ns() - start
        finally:
            if was_enabled:
                gc.enable()
        samples.append(elapsed / number)
    return {"best_ns": min(samples), "median_ns": float(median(samples))}


def time_op_first_call(fn: Callable[[], Any]) -> float:
    """Time one call, including any one-time compilation or dispatch setup."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = perf_counter_ns()
        sync_result(fn())
        return float(perf_counter_ns() - start)
    finally:
        if was_enabled:
            gc.enable()


def measure_peak_memory(fn: Callable[[], Any]) -> dict[str, int]:
    """Return ``{"current_bytes": ..., "peak_bytes": ...}`` for one call.

    Uses :mod:`tracemalloc`, which observes Python-level allocations.
    Backend libraries that allocate via C extensions (NumPy, JAX, Torch)
    show up as the C objects' Python wrappers — the absolute number is
    not the resident set size, but it is reproducible and useful for
    *relative* comparison of an optimized kernel against its generic
    reference on the same backend.

    The function tears down any pre-existing :mod:`tracemalloc` state
    that the caller may have started, so it is safe to call inside
    benchmarks that compose with other instrumentation as long as the
    caller does not depend on a separate tracemalloc trace running.
    """
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    tracemalloc.start()
    try:
        sync_result(fn())
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if was_tracing:
        tracemalloc.start()
    return {"current_bytes": int(current), "peak_bytes": int(peak)}


@dataclass(slots=True)
class BenchResult:
    """Serializable result for one bare-vs-SpaceCore timing cell."""

    case_id: str
    label: str
    backend: str
    operator_type: str
    operation: str
    geometry: str
    shape_kind: str
    size_name: str
    size: int
    checks: bool
    batch: int | None
    bare_label: str
    sc_label: str
    bare_best_ns: float
    bare_median_ns: float
    sc_best_ns: float
    sc_median_ns: float
    overhead_ns: float
    ratio: float
    predicted_overhead_ns: float
    gap: str
    components: list[dict[str, float]]
    verdict: str
    compile_bare_ns: float | None = None
    compile_sc_ns: float | None = None
    breakdown: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

"""Group-by summaries over a list of :class:`MacroResult` rows.

The dashboard and the CLI ``summary`` command consume aggregates keyed
by ``(benchmark_name, backend, size_label, mode, check_level)``. Each
aggregate carries median, mean, std, min, max for every numeric metric
that varies across seeds.

Aggregates are computed in pure Python so they are JSON-serializable
without external dependencies.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from ._schema import MacroResult


_NUMERIC_FIELDS = (
    "run_time_ns",
    "setup_time_ns",
    "compile_time_ns",
    "time_per_iteration_ns",
    "throughput",
    "memory_peak_bytes",
    "error_vs_bare",
    "residual",
    "objective",
)


def _stats(values: list[float]) -> dict[str, float] | None:
    """Return min/max/mean/median/std for a non-empty list, else None."""
    nums = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not nums:
        return None
    return {
        "median": float(statistics.median(nums)),
        "mean": float(statistics.fmean(nums)),
        "std": float(statistics.stdev(nums)) if len(nums) >= 2 else 0.0,
        "min": float(min(nums)),
        "max": float(max(nums)),
        "n": len(nums),
    }


def group_summaries(rows: Iterable[MacroResult]) -> list[dict[str, Any]]:
    """Aggregate by ``(benchmark, backend, size_label, mode, check_level)``.

    Returns
    -------
    list of dict
        One row per group with ``key`` plus a ``metrics`` dict whose
        entries are the per-metric stats dicts.
    """
    rows = list(rows)
    buckets: dict[tuple[str, str, str, str, str | None], list[MacroResult]] = {}
    for r in rows:
        key = (r.benchmark_name, r.backend, r.size_label, r.mode, r.check_level)
        buckets.setdefault(key, []).append(r)
    out: list[dict[str, Any]] = []
    for (name, backend, size_label, mode, check_level), group in buckets.items():
        metric_stats: dict[str, dict[str, float]] = {}
        for field in _NUMERIC_FIELDS:
            stats = _stats([getattr(r, field) for r in group])
            if stats is not None:
                metric_stats[field] = stats
        first = group[0]
        out.append(
            {
                "benchmark_name": name,
                "workload": first.workload,
                "backend": backend,
                "device": first.device,
                "size_label": size_label,
                "size_params": dict(first.size_params),
                "mode": mode,
                "check_level": check_level,
                "iterations": first.iterations,
                "metrics": metric_stats,
                "n_seeds": len(group),
            }
        )
    out.sort(
        key=lambda row: (
            row["benchmark_name"],
            row["backend"],
            row["size_label"],
            row["mode"],
        )
    )
    return out


def runtime_ratio_table(
    rows: Iterable[MacroResult],
    *,
    reference_mode: str = "bare",
) -> dict[tuple[str, str, str, str], float]:
    """Compute ``runtime_ratio = mode_runtime / reference_runtime``.

    Returns a dict ``{(benchmark, backend, size, mode): ratio}``. The
    reference mode rows themselves yield ratio ``1.0``.
    """
    summaries = group_summaries(rows)
    # Index reference rows by (benchmark, backend, size).
    ref_runtime: dict[tuple[str, str, str], float] = {}
    for row in summaries:
        if row["mode"] != reference_mode:
            continue
        runtime = row["metrics"].get("run_time_ns", {}).get("median")
        if runtime is not None and runtime > 0:
            ref_runtime[
                (row["benchmark_name"], row["backend"], row["size_label"])
            ] = runtime
    out: dict[tuple[str, str, str, str], float] = {}
    for row in summaries:
        runtime = row["metrics"].get("run_time_ns", {}).get("median")
        if runtime is None or runtime <= 0:
            continue
        key3 = (row["benchmark_name"], row["backend"], row["size_label"])
        ref = ref_runtime.get(key3)
        if ref is None or ref <= 0:
            continue
        out[(row["benchmark_name"], row["backend"], row["size_label"], row["mode"])] = (
            runtime / ref
        )
    return out

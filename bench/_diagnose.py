"""Overhead diagnosis for benchmark results.

The verdict module (:mod:`bench._verdict`) tells you *whether* a
SpaceCore call is slower than the bare reference. This module explains
*why*. Each :class:`ProbeResult` is examined against a battery of
heuristics — validation-cost dominance, JAX compile latency, seed
jitter, memory overhead — and tagged with one or more :class:`Reason`
codes plus a human-readable summary.

The :func:`overall_diagnosis` rollup produces the structured payload
the dashboard surfaces in its verdict section: dominant-reason
histogram, top overhead cases, top wins, JAX compile summary, and a
short prose narrative.

The heuristics here are intentionally numeric and order-independent —
they fire on raw thresholds against the :class:`ProbeResult` fields, so
the diagnosis is deterministic and easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any

from ._probes import ProbeResult
from ._verdict import _FAMILY_TOL


class Reason(str, Enum):
    """Why a benchmark case looks the way it does.

    Multiple reasons can apply to one case. The first reason in the
    tuple returned by :func:`diagnose` is the dominant one and drives
    the summary string.
    """

    CONSTANT_VALIDATION_COST = "CONSTANT_VALIDATION_COST"
    BARE_SATURATES_OP = "BARE_SATURATES_OP"
    BARE_TOO_SMALL_TO_COMPARE = "BARE_TOO_SMALL_TO_COMPARE"
    JAX_COMPILE_DOMINANT = "JAX_COMPILE_DOMINANT"
    JAX_TRACE_OVERHEAD = "JAX_TRACE_OVERHEAD"
    TORCH_EAGER_OVERHEAD = "TORCH_EAGER_OVERHEAD"
    HIGH_SEED_JITTER = "HIGH_SEED_JITTER"
    MEMORY_OVERHEAD = "MEMORY_OVERHEAD"
    CORRECTNESS_FAILURE = "CORRECTNESS_FAILURE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """A diagnosis for one :class:`ProbeResult`.

    ``reasons`` is ordered from dominant to least dominant. ``summary``
    is a single human-readable sentence keyed on the dominant reason.
    """

    reasons: tuple[Reason, ...]
    summary: str


# ---------------------------------------------------------------------------
# Per-case diagnosis


def _fires_constant_validation(r: ProbeResult) -> bool:
    return (
        r.bare_median_ns < 1500.0
        and r.speedup < 0.10
        and r.sc_median_ns > 5000.0
    )


def _fires_bare_saturates(r: ProbeResult) -> bool:
    return r.speedup >= 0.95


def _fires_bare_too_small(r: ProbeResult) -> bool:
    return r.bare_median_ns < 100.0


def _fires_jax_compile_dominant(r: ProbeResult) -> bool:
    return (
        r.backend == "jax"
        and r.compile_ns_median is not None
        and r.sc_median_ns > 0.0
        and r.compile_ns_median > 10.0 * r.sc_median_ns
    )


def _fires_jax_trace_overhead(r: ProbeResult) -> bool:
    if r.backend != "jax" or r.speedup >= 0.05:
        return False
    return not _fires_constant_validation(r)


def _fires_torch_eager_overhead(r: ProbeResult) -> bool:
    return r.backend == "torch" and r.speedup < 0.10


def _fires_high_seed_jitter(r: ProbeResult) -> bool:
    denom = max(r.speedup, 1e-9)
    return (r.speedup_std / denom) > 0.20 and r.speedup_std > 0.05


def _fires_memory_overhead(r: ProbeResult) -> bool:
    bare = max(r.bare_peak_bytes_median, 64)
    return r.sc_peak_bytes_median > 4 * bare


def _fires_correctness_failure(r: ProbeResult) -> bool:
    tol = _FAMILY_TOL.get(r.family, 1e-9)
    return r.error_max > tol


def _summary_for(reason: Reason, r: ProbeResult) -> str:
    bare_us = r.bare_median_ns / 1_000.0
    sc_us = r.sc_median_ns / 1_000.0
    if reason is Reason.CONSTANT_VALIDATION_COST:
        return (
            f"Validation overhead dominates: bare {bare_us:.2f}us "
            f"vs SC {sc_us:.2f}us."
        )
    if reason is Reason.BARE_SATURATES_OP:
        return "SpaceCore is within 5% of bare (problem size large enough)."
    if reason is Reason.BARE_TOO_SMALL_TO_COMPARE:
        return (
            f"Bare reference is {r.bare_median_ns:.0f} ns "
            "— too small for meaningful comparison."
        )
    if reason is Reason.JAX_COMPILE_DOMINANT:
        compile_ms = (r.compile_ns_median or 0.0) / 1_000_000.0
        return (
            f"JAX compile ({compile_ms:.2f} ms) > 10x steady-state "
            f"({sc_us:.2f} us)."
        )
    if reason is Reason.JAX_TRACE_OVERHEAD:
        return (
            f"JAX trace overhead: SC {sc_us:.2f} us vs bare {bare_us:.2f} us "
            f"(speedup {r.speedup:.3f}x)."
        )
    if reason is Reason.TORCH_EAGER_OVERHEAD:
        return (
            f"Torch eager dispatch overhead: SC {sc_us:.2f} us vs "
            f"bare {bare_us:.2f} us (speedup {r.speedup:.2f}x)."
        )
    if reason is Reason.HIGH_SEED_JITTER:
        return (
            f"High seed jitter: speedup {r.speedup:.2f}x +/- "
            f"{r.speedup_std:.2f}x."
        )
    if reason is Reason.MEMORY_OVERHEAD:
        ratio = r.sc_peak_bytes_median / max(r.bare_peak_bytes_median, 64)
        return (
            f"Memory overhead: SC peak {r.sc_peak_bytes_median:,} B "
            f"is {ratio:.1f}x bare peak."
        )
    if reason is Reason.CORRECTNESS_FAILURE:
        tol = _FAMILY_TOL.get(r.family, 1e-9)
        return (
            f"Correctness failure: error {r.error_max:.2e} "
            f"exceeds family tolerance {tol:.0e}."
        )
    return f"Speedup {r.speedup:.2f}x — no specific overhead source identified."


# Order matters: earliest entry that fires becomes the dominant reason.
_HEURISTICS: tuple[tuple[Reason, Any], ...] = (
    (Reason.CORRECTNESS_FAILURE, _fires_correctness_failure),
    (Reason.JAX_COMPILE_DOMINANT, _fires_jax_compile_dominant),
    (Reason.JAX_TRACE_OVERHEAD, _fires_jax_trace_overhead),
    (Reason.TORCH_EAGER_OVERHEAD, _fires_torch_eager_overhead),
    (Reason.CONSTANT_VALIDATION_COST, _fires_constant_validation),
    (Reason.BARE_SATURATES_OP, _fires_bare_saturates),
    (Reason.BARE_TOO_SMALL_TO_COMPARE, _fires_bare_too_small),
    (Reason.HIGH_SEED_JITTER, _fires_high_seed_jitter),
    (Reason.MEMORY_OVERHEAD, _fires_memory_overhead),
)


def diagnose(result: ProbeResult) -> Diagnosis:
    """Return a :class:`Diagnosis` for one :class:`ProbeResult`."""
    fired: list[Reason] = []
    for reason, predicate in _HEURISTICS:
        if predicate(result):
            fired.append(reason)
    if not fired:
        return Diagnosis(reasons=(Reason.NEUTRAL,), summary=_summary_for(Reason.NEUTRAL, result))
    summary = _summary_for(fired[0], result)
    return Diagnosis(reasons=tuple(fired), summary=summary)


# ---------------------------------------------------------------------------
# Cross-case rollup


def _dominant_reason_label(d: Diagnosis) -> str:
    return d.reasons[0].value if d.reasons else Reason.NEUTRAL.value


def overall_diagnosis(results: list[ProbeResult]) -> dict[str, Any]:
    """Aggregate diagnoses across every probe result.

    Returns the structured payload the dashboard renders in its verdict
    section. See the module docstring of :mod:`bench._diagnose` for the
    field-by-field contract.
    """
    if not results:
        return {
            "dominant_reason_counts": {},
            "top_overhead_cases": [],
            "top_wins": [],
            "jax_compile_summary": None,
            "family_overhead_ranking": [],
            "narrative": "No benchmark results to diagnose.",
        }

    diagnoses = [diagnose(r) for r in results]

    reason_counts: dict[str, int] = {}
    for d in diagnoses:
        for reason in d.reasons:
            reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1

    def overhead_ns(r: ProbeResult) -> float:
        return r.sc_median_ns - r.bare_median_ns

    def overhead_factor(r: ProbeResult) -> float:
        return r.sc_median_ns / max(r.bare_median_ns, 1.0)

    top_overhead = sorted(
        results,
        key=lambda r: (overhead_factor(r), overhead_ns(r)),
        reverse=True,
    )[:5]
    top_overhead_cases = [
        (
            r.operation_name,
            r.size,
            r.backend,
            overhead_factor(r),
            overhead_ns(r),
        )
        for r in top_overhead
    ]

    top_wins_sorted = sorted(results, key=lambda r: r.speedup, reverse=True)[:5]
    top_wins = [
        (r.operation_name, r.size, r.backend, r.speedup) for r in top_wins_sorted
    ]

    jax_compiles = [
        r.compile_ns_median
        for r in results
        if r.backend == "jax" and r.compile_ns_median is not None
    ]
    if jax_compiles:
        jax_compile_summary: dict[str, float] | None = {
            "cases": len(jax_compiles),
            "median_compile_ms": float(median(jax_compiles)) / 1_000_000.0,
        }
    else:
        jax_compile_summary = None

    family_buckets: dict[str, list[float]] = {}
    for r in results:
        family_buckets.setdefault(r.family, []).append(overhead_factor(r))
    family_overhead_ranking = sorted(
        ((family, float(median(factors))) for family, factors in family_buckets.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    n_backends = len({r.backend for r in results})
    overall_median = float(median(r.speedup for r in results))
    median_overhead_factor = 1.0 / max(overall_median, 1e-12)
    overhead_reason_counts = {
        reason: count
        for reason, count in reason_counts.items()
        if reason != Reason.NEUTRAL.value
    }
    dominant_reason = max(
        (overhead_reason_counts or reason_counts).items(),
        key=lambda kv: kv[1],
    )[0]

    if jax_compile_summary is not None:
        jax_note = (
            f"JAX compile latency seen on {jax_compile_summary['cases']} case(s); "
            f"median {jax_compile_summary['median_compile_ms']:.2f} ms"
        )
    else:
        jax_note = "No JAX compile latency recorded"

    narrative = (
        f"{len(results)} cases across {n_backends} backend(s). "
        f"Median speedup {overall_median:.2f}x "
        f"({median_overhead_factor:.2f}x runtime vs bare). "
        f"Biggest overhead source is {dominant_reason}. "
        f"{jax_note}."
    )

    return {
        "dominant_reason_counts": reason_counts,
        "top_overhead_cases": top_overhead_cases,
        "top_wins": top_wins,
        "jax_compile_summary": jax_compile_summary,
        "family_overhead_ranking": family_overhead_ranking,
        "narrative": narrative,
    }

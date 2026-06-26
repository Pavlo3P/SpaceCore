"""Verdict categorization and text reporting.

The verdict module turns a flat list of :class:`ProbeResult` records into:

* per-case categorization (``WIN`` / ``NEUTRAL`` / ``LOSS`` /
  ``CORRECTNESS_FAILURE`` / ``REGRESSION``);
* per-family rollups (median speedup, count by status);
* overall summary statistics (median + percentiles);
* a printable text report for the ``python -m bench summary`` command.

A ``LOSS`` means the SpaceCore call is slower than the bare reference but
within the expected overhead envelope. A ``REGRESSION`` only fires when a
*baseline* run is supplied for comparison — the verdict module does not
guess at thresholds without a reference run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Iterable

from ._probes import ProbeResult


class Status(str, Enum):
    """Per-case status label.

    ``WIN`` — SpaceCore matches or beats the bare reference (speedup ≥ 0.95).
    ``NEUTRAL`` — speedup between 0.50 and 0.95: small overhead, often
    constant-factor.
    ``LOSS`` — speedup between 0.10 and 0.50: meaningful overhead.
    ``HEAVY_LOSS`` — speedup below 0.10: SpaceCore is more than 10× the
    bare reference; usually a small-problem-size phenomenon.
    ``CORRECTNESS_FAILURE`` — error_max exceeds the per-family tolerance.
    ``REGRESSION`` — only set during baseline comparison, see
    :func:`compare_to_baseline`.
    """

    WIN = "WIN"
    NEUTRAL = "NEUTRAL"
    LOSS = "LOSS"
    HEAVY_LOSS = "HEAVY_LOSS"
    CORRECTNESS_FAILURE = "CORRECTNESS_FAILURE"
    REGRESSION = "REGRESSION"


# Per-family correctness tolerance. ``inf`` for linalg results because
# iterative solvers approximate rather than reproduce the bare reference.
_FAMILY_TOL = {
    "space": 1e-9,
    "linop": 1e-9,
    "functional": 1e-9,
    "linalg": float("inf"),
    "kernel": 1e-12,
}


def _categorize_speedup(speedup: float) -> Status:
    if speedup >= 0.95:
        return Status.WIN
    if speedup >= 0.50:
        return Status.NEUTRAL
    if speedup >= 0.10:
        return Status.LOSS
    return Status.HEAVY_LOSS


def categorize(result: ProbeResult) -> Status:
    """Return the categorical status for one ``ProbeResult``.

    Tolerance is widened when the result was measured on a device or
    backend that defaults to ``float32`` (Torch MPS, Torch float32
    default, JAX without x64), because the bare NumPy reference is in
    float64.
    """
    tol = _FAMILY_TOL.get(result.family, 1e-9)
    if _is_low_precision_device(result):
        tol = max(tol, 1e-4)
    if result.error_max > tol:
        return Status.CORRECTNESS_FAILURE
    return _categorize_speedup(result.speedup)


def _is_low_precision_device(result: ProbeResult) -> bool:
    """Return whether this case ran in float32 by default.

    Torch MPS executes float32, Torch CPU's default is float32 unless
    overridden, and JAX without x64 also reduces to float32.
    """
    if result.backend == "torch" and result.device == "mps":
        return True
    if result.backend == "torch" and result.device == "cpu":
        return True
    return False


@dataclass(frozen=True, slots=True)
class FamilyRollup:
    """Aggregate stats for one operation family."""

    family: str
    case_count: int
    median_speedup: float
    wins: int
    neutral: int
    losses: int
    heavy_losses: int
    correctness_failures: int


@dataclass(frozen=True, slots=True)
class Verdict:
    """Top-level verdict: per-case status, rollups, summary stats."""

    statuses: dict[str, Status] = field(default_factory=dict)
    """``operation_name@size`` → :class:`Status`."""

    families: tuple[FamilyRollup, ...] = ()
    overall_median_speedup: float = 0.0
    overall_min_speedup: float = 0.0
    overall_max_speedup: float = 0.0
    top_wins: tuple[tuple[str, int, float], ...] = ()
    top_losses: tuple[tuple[str, int, float], ...] = ()
    regressions: tuple[tuple[str, int, float, float], ...] = ()
    """``(name, size, old_speedup, new_speedup)`` per regression."""

    @property
    def n_cases(self) -> int:
        return sum(f.case_count for f in self.families)


def _key(result: ProbeResult) -> str:
    return (
        f"{result.operation_name}@{result.size}/{result.backend}/"
        f"{result.device}/{result.check_level}"
    )


def make_verdict(results: Iterable[ProbeResult]) -> Verdict:
    """Build a :class:`Verdict` from raw results.

    A baseline comparison is optional and lives in
    :func:`compare_to_baseline`; the bare ``make_verdict`` only
    categorizes against the per-family tolerance and the speedup bands.
    """
    results = list(results)
    statuses = {_key(r): categorize(r) for r in results}
    speedups = [r.speedup for r in results]
    family_buckets: dict[str, list[ProbeResult]] = {}
    for r in results:
        family_buckets.setdefault(r.family, []).append(r)

    families: list[FamilyRollup] = []
    for family, group in family_buckets.items():
        counts = {s: 0 for s in Status}
        for r in group:
            counts[statuses[_key(r)]] += 1
        families.append(
            FamilyRollup(
                family=family,
                case_count=len(group),
                median_speedup=float(median(r.speedup for r in group)),
                wins=counts[Status.WIN],
                neutral=counts[Status.NEUTRAL],
                losses=counts[Status.LOSS],
                heavy_losses=counts[Status.HEAVY_LOSS],
                correctness_failures=counts[Status.CORRECTNESS_FAILURE],
            )
        )

    sorted_by_speed = sorted(results, key=lambda r: r.speedup, reverse=True)
    top_wins = tuple(
        (r.operation_name, r.size, r.speedup) for r in sorted_by_speed[:5]
    )
    top_losses = tuple(
        (r.operation_name, r.size, r.speedup)
        for r in sorted(results, key=lambda r: r.speedup)[:5]
    )

    return Verdict(
        statuses=statuses,
        families=tuple(sorted(families, key=lambda f: f.family)),
        overall_median_speedup=float(median(speedups)) if speedups else 0.0,
        overall_min_speedup=min(speedups) if speedups else 0.0,
        overall_max_speedup=max(speedups) if speedups else 0.0,
        top_wins=top_wins,
        top_losses=top_losses,
    )


def compare_to_baseline(
    current: Iterable[ProbeResult],
    baseline: Iterable[ProbeResult],
    *,
    threshold: float = 0.20,
    noise_floor_ns: float = 1_000.0,
) -> tuple[Verdict, list[str]]:
    """Compare a current run against a baseline run.

    A case regresses when its median SpaceCore time grew by more than
    ``threshold`` (default 20%) plus ``noise_floor_ns`` over the
    baseline. The verdict ``statuses`` map is updated to mark each
    regression :class:`Status.REGRESSION`, overriding the un-baselined
    categorization.

    Returns the verdict and a list of human-readable regression
    summaries suitable for ``print``.
    """
    verdict = make_verdict(current)
    current_by_key = {_key(r): r for r in current}
    baseline_by_key = {_key(r): r for r in baseline}
    regressions: list[tuple[str, int, float, float]] = []
    lines: list[str] = []
    for key, cur in current_by_key.items():
        base = baseline_by_key.get(key)
        if base is None:
            continue
        old_ns = max(base.sc_median_ns, 1.0)
        new_ns = cur.sc_median_ns
        if new_ns > old_ns * (1.0 + threshold) + noise_floor_ns:
            verdict.statuses[key] = Status.REGRESSION
            regressions.append((cur.operation_name, cur.size, base.speedup, cur.speedup))
            lines.append(
                f"{cur.operation_name}@n={cur.size}: median {new_ns:,.0f}ns > "
                f"baseline {old_ns:,.0f}ns × {1.0 + threshold} + {noise_floor_ns:,.0f}ns"
            )
    verdict_with_reg = Verdict(
        statuses=verdict.statuses,
        families=verdict.families,
        overall_median_speedup=verdict.overall_median_speedup,
        overall_min_speedup=verdict.overall_min_speedup,
        overall_max_speedup=verdict.overall_max_speedup,
        top_wins=verdict.top_wins,
        top_losses=verdict.top_losses,
        regressions=tuple(regressions),
    )
    return verdict_with_reg, lines


# ---------------------------------------------------------------------------
# Text rendering


def _fmt_ns(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:6.2f} ms"
    if value >= 1e3:
        return f"{value / 1e3:6.2f} us"
    return f"{value:6.0f} ns"


def render_text(results: list[ProbeResult], verdict: Verdict) -> str:
    """Render a human-readable verdict to text.

    Used by ``python -m bench summary``. Output is wide; pipe through
    ``less -S`` if the terminal is narrow.
    """
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("SpaceCore benchmark summary")
    lines.append("=" * 96)
    lines.append(
        f"  cases:    {verdict.n_cases:>4d}   "
        f"median speedup: {verdict.overall_median_speedup:6.2f}x   "
        f"range: [{verdict.overall_min_speedup:.2f}x, {verdict.overall_max_speedup:.2f}x]"
    )
    lines.append("")
    lines.append("Family rollup")
    lines.append("-" * 96)
    lines.append(
        f"  {'family':<14s} {'n':>4s} {'median':>10s} "
        f"{'WIN':>5s} {'NEUT':>5s} {'LOSS':>5s} {'HEAVY':>5s} {'CORR':>5s}"
    )
    for f in verdict.families:
        lines.append(
            f"  {f.family:<14s} {f.case_count:>4d} {f.median_speedup:>9.2f}x "
            f"{f.wins:>5d} {f.neutral:>5d} {f.losses:>5d} {f.heavy_losses:>5d} "
            f"{f.correctness_failures:>5d}"
        )
    lines.append("")
    lines.append("Top 5 wins (speedup highest)")
    lines.append("-" * 96)
    for name, size, speedup in verdict.top_wins:
        lines.append(f"  {name:<40s} n={size:<6d} {speedup:6.2f}x")
    lines.append("")
    lines.append("Top 5 losses (speedup lowest)")
    lines.append("-" * 96)
    for name, size, speedup in verdict.top_losses:
        lines.append(f"  {name:<40s} n={size:<6d} {speedup:6.2f}x")
    if verdict.regressions:
        lines.append("")
        lines.append("Regressions vs baseline")
        lines.append("-" * 96)
        for name, size, old, new in verdict.regressions:
            lines.append(
                f"  {name:<40s} n={size:<6d} old={old:6.2f}x new={new:6.2f}x"
            )
    lines.append("")
    lines.append("Per-case detail")
    lines.append("-" * 96)
    lines.append(
        f"  {'name':<30s} {'backend':>8s} {'checks':>7s} {'n':>6s} "
        f"{'bare':>10s} {'spacecore':>10s} {'jit':>10s}"
    )
    for r in sorted(
        results,
        key=lambda r: (r.family, r.operation_name, r.backend, r.size, r.check_level),
    ):
        lines.append(
            f"  {r.operation_name:<30s} {r.backend:>8s} {r.check_level:>7s} {r.size:>6d} "
            f"{_fmt_ns(r.bare_median_ns)} "
            f"{_fmt_ns(r.sc_median_ns)} "
            f"{_fmt_ns(r.jit_median_ns) if r.jit_median_ns is not None else '       n/a'}"
        )
    lines.append("=" * 96)
    return "\n".join(lines)

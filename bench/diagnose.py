from __future__ import annotations

from collections import defaultdict
from typing import Any

from .cost_model import classify_gap


def dominant_component(components: list[dict[str, float]]) -> str:
    visible = [c for c in components if c["name"] != "amortized_per_element"]
    if not visible:
        return "fixed wrapper"
    return max(visible, key=lambda c: c["ns"])["name"]


def make_verdict(
    *,
    case: Any,
    overhead_ns: float,
    bare_ns: float,
    predicted_ns: float,
    components: list[dict[str, float]],
    gap: str,
) -> str:
    dominant = dominant_component(components)
    pct = 0.0 if bare_ns <= 0 else 100.0 * max(overhead_ns, 0.0) / bare_ns
    if gap == "ok":
        if case.batch:
            per = max(overhead_ns, 0.0) / case.batch
            return (
                f"✅ vapply-style overhead {overhead_ns:.0f} ns over batch {case.batch} "
                f"= {per:.0f} ns/element; dominant predicted component: {dominant}."
            )
        return (
            f"✅ overhead {overhead_ns:.0f} ns ({pct:.2f}% of bare), "
            f"matches prediction {predicted_ns:.0f} ns; dominant: {dominant}."
        )
    if gap == "slightly_high":
        return (
            f"⚠️ overhead {overhead_ns:.0f} ns is moderately above prediction "
            f"{predicted_ns:.0f} ns. Dominant predicted component: {dominant}."
        )
    hint = "Check for unexpected Python loops, non-broadcast Riesz maps, or retracing."
    if "vmap_fallback" in {c["name"] for c in components}:
        hint = "Likely batched Riesz/vmap fallback; make geometry broadcast over leading axes."
    return (
        f"⚠️ overhead {overhead_ns:.0f} ns, >2x predicted {predicted_ns:.0f} ns. "
        f"Dominant: {dominant}. {hint}"
    )


def add_trend_flags(results: list[dict[str, Any]]) -> None:
    """Classify size-sweep trends in-place for matching operation groups."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        key = (
            row["backend"],
            row["operator_type"],
            row["operation"],
            row["geometry"],
            row["shape_kind"],
            row["checks"],
        )
        groups[key].append(row)

    rank = {"tiny": 0, "small": 1, "large": 2}
    for rows in groups.values():
        rows.sort(key=lambda r: rank.get(r["size_name"], 99))
        if len(rows) < 3:
            for row in rows:
                row["trend"] = "insufficient"
            continue
        tiny, _, large = rows[0], rows[1], rows[-1]
        if large["ratio"] <= 1.02 or large["ratio"] < tiny["ratio"]:
            trend = "decays-to-1.0"
        elif large["ratio"] > tiny["ratio"] * 1.2:
            trend = "grows-with-size"
        else:
            trend = "constant-above-1.0"
        for row in rows:
            row["trend"] = trend
            if row is large and trend != "decays-to-1.0" and row["gap"] == "ok" and row["ratio"] > 1.15:
                row["gap"] = "slightly_high"
                row["verdict"] = row["verdict"].replace("✅", "⚠️", 1) + f" Trend: {trend}."

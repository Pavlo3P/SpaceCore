"""Self-contained interactive HTML dashboard generator for SpaceCore bench runs.

This module produces a single static ``.html`` file that opens in any
browser without a network connection (Plotly.js is loaded once from a
CDN at view time; everything else — data, styles, JS — is inlined). The
generated page provides a maintainer-grade view of a bench run for
exploratory analysis, much richer than the static PNG produced by
``bench/_plots.py``.

Layout of the generated HTML
============================

1. **Title bar** with run metadata: Python version, SpaceCore version,
   platform string, and total case count.
2. **Summary cards**: total cases, median speedup, and one card per
   verdict status (WIN / NEUTRAL / LOSS / HEAVY_LOSS /
   CORRECTNESS_FAILURE / REGRESSION).
3. **Diagnosis panels**: dominant reasons, narrative, top overhead
   cases, top wins, JAX compile summary.
4. **Filter controls**:
   - Backend checkbox group (numpy / jax / torch / cupy, as present).
   - Check-mode selector (all / none / cheap).
   - Family checkbox group (space, linop, functional, linalg, kernel).
   - Status checkbox group (WIN ... REGRESSION).
   - Substring search on ``operation_name``.
   - Min/max speedup numeric inputs.
5. **Interactive Plotly charts** (rendered client-side from
   ``window.BENCH_DATA``):
   - Speedup distribution (bare / SpaceCore, log-x, per backend).
   - Worst-case overhead decomposition (extra ns over bare, split into
     validation and non-validation overhead).
   - Overhead persistence by problem size (SpaceCore / bare runtime ratio).
   - Per-seed jitter scatter (log-x, colored by backend).
   - Per-family median memory overhead ratio (SpaceCore peak / bare peak).
   - Optional baseline scatter (current vs baseline speedup, log-log)
     when a ``baseline`` argument is supplied.
6. **Sortable, filterable table** with one row per
   ``(probe, size, backend, device, check_level)`` case. Vanilla JS; no framework
   dependencies. Includes a Compile (ms) column for JAX rows and a
   Diagnosis chip per row.

The public surface is intentionally small:

* :func:`render_dashboard` — writes the HTML file and returns its path.
* :func:`open_in_browser`  — convenience helper that opens the file in
  the default browser.

The data shape consumed is :class:`bench._probes.ProbeResult`;
categorization reuses :class:`bench._verdict.Status` and the per-family
tolerance map. Diagnosis annotations come from
:mod:`bench._diagnose` when that module is available; otherwise the
dashboard renders with empty diagnosis panels.
"""
from __future__ import annotations

import html
import json
import webbrowser
from pathlib import Path
from statistics import median
from typing import Iterable

from ._io import _metadata
from ._probes import ProbeResult
from ._verdict import Status, categorize


# --- Optional diagnosis module ---------------------------------------------
# ``bench._diagnose`` may not yet be present (the diagnosis agent lands
# alongside this dashboard work). We degrade gracefully when it is missing
# so dashboards keep rendering — the diagnosis panels simply show empty
# state and the per-row chip falls back to ``NEUTRAL``.
try:  # pragma: no cover - import guard
    from ._diagnose import (  # type: ignore
        Reason,
        diagnose as _diagnose_one,
        overall_diagnosis as _overall_diagnosis,
    )
    _HAS_DIAGNOSE = True
except Exception:  # pragma: no cover - dependency not yet available
    Reason = None  # type: ignore[assignment]
    _diagnose_one = None  # type: ignore[assignment]
    _overall_diagnosis = None  # type: ignore[assignment]
    _HAS_DIAGNOSE = False


# Plotly CDN — single external dependency for the rendered page.
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# Verdict-band colors — match the conventions used elsewhere in bench/.
_STATUS_COLORS: dict[str, str] = {
    Status.WIN.value: "#2ca02c",
    Status.NEUTRAL.value: "#7f7f7f",
    Status.LOSS.value: "#ff7f0e",
    Status.HEAVY_LOSS.value: "#d62728",
    Status.CORRECTNESS_FAILURE.value: "#8c564b",
    Status.REGRESSION.value: "#e377c2",
}


# Family colors mirror ``bench/_plots.py`` so static and interactive
# views stay visually consistent.
_FAMILY_COLORS: dict[str, str] = {
    "space": "#4C78A8",
    "linop": "#F58518",
    "functional": "#54A24B",
    "linalg": "#B279A2",
    "kernel": "#E45756",
}


# Backend colors — used as the *primary* color axis throughout the
# multi-backend dashboard so the same backend always reads the same
# across panels.
_BACKEND_COLORS: dict[str, str] = {
    "numpy": "#4C78A8",
    "jax":   "#54A24B",
    "torch": "#F58518",
    "cupy":  "#E45756",
}


# Reason chip colors. Kept here (and not in ``_diagnose``) so the
# dashboard renders even when the diagnosis module is absent.
_REASON_COLORS: dict[str, str] = {
    "CONSTANT_VALIDATION_COST":   "#ff7f0e",
    "BARE_SATURATES_OP":          "#2ca02c",
    "BARE_TOO_SMALL_TO_COMPARE":  "#7f7f7f",
    "JAX_COMPILE_DOMINANT":       "#9467bd",
    "JAX_TRACE_OVERHEAD":         "#1f77b4",
    "TORCH_EAGER_OVERHEAD":       "#e377c2",
    "KERNEL_WIN":                 "#2ca02c",
    "KERNEL_NEUTRAL":             "#7f7f7f",
    "HIGH_SEED_JITTER":           "#bcbd22",
    "SOLVER_FIXED_ITERATIONS":    "#17becf",
    "MEMORY_OVERHEAD":            "#d62728",
    "CORRECTNESS_FAILURE":        "#8c564b",
    "NEUTRAL":                    "#d3d3d3",
}


_REGRESSION_THRESHOLD = 0.20
_REGRESSION_NOISE_FLOOR_NS = 1_000.0


def render_dashboard(
    results: Iterable[ProbeResult],
    out_path: str | Path,
    baseline: Iterable[ProbeResult] | None = None,
) -> Path:
    """Render an interactive HTML dashboard for a bench run.

    Parameters
    ----------
    results
        Iterable of :class:`ProbeResult` for the current run. Each
        result carries a ``backend`` field; the dashboard groups
        traces and offers a filter row by backend.
    out_path
        Where to write the HTML file. Parent directories are created.
    baseline
        Optional iterable of :class:`ProbeResult` from a prior run.
        When supplied, regressions (current SC ns > 1.20 × baseline SC
        ns + 1000 ns) are marked :class:`Status.REGRESSION` and a sixth
        plot — a current-vs-baseline speedup scatter — is included.

    Returns
    -------
    Path
        The resolved path of the saved file.
    """
    results = list(results)
    if not results:
        raise ValueError("render_dashboard: no results")
    baseline_list = list(baseline) if baseline is not None else None

    statuses = _build_statuses(results, baseline_list)
    diagnoses = _build_diagnoses(results)
    rows = [
        _row_payload(r, statuses[_key(r)], diagnoses.get(_key(r)))
        for r in results
    ]
    baseline_rows = (
        _baseline_payload(results, baseline_list, statuses)
        if baseline_list is not None
        else []
    )

    summary = _summary(rows)
    meta = _metadata()
    overall = _build_overall(results, diagnoses)

    backends = sorted({r["backend"] for r in rows})

    payload = {
        "rows": rows,
        "baseline": baseline_rows,
        "summary": summary,
        "meta": meta,
        "status_colors": _STATUS_COLORS,
        "family_colors": _FAMILY_COLORS,
        "backend_colors": _BACKEND_COLORS,
        "reason_colors": _REASON_COLORS,
        "families": sorted({r["family"] for r in rows}),
        "backends": backends,
        "check_levels": sorted({r["check_level"] for r in rows}),
        "statuses": [s.value for s in Status],
        "has_baseline": bool(baseline_rows),
        "overall_diagnosis": overall,
    }

    html_text = _render_html(payload)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def open_in_browser(path: str | Path) -> None:
    """Open ``path`` in the default browser via ``file://`` URL."""
    webbrowser.open(f"file://{Path(path).resolve()}")


# ---------------------------------------------------------------------------
# Categorization + payload assembly


def _key(r: ProbeResult) -> str:
    return (
        f"{r.operation_name}@{r.size}@{r.backend}@{r.device}@{r.check_level}"
    )


def _build_statuses(
    results: list[ProbeResult],
    baseline: list[ProbeResult] | None,
) -> dict[str, Status]:
    """Categorize every result, applying the regression override if asked."""
    statuses = {_key(r): categorize(r) for r in results}
    if baseline is None:
        return statuses
    baseline_by_key = {_key(b): b for b in baseline}
    for r in results:
        base = baseline_by_key.get(_key(r))
        if base is None:
            continue
        old_ns = max(base.sc_median_ns, 1.0)
        if r.sc_median_ns > old_ns * (1.0 + _REGRESSION_THRESHOLD) + _REGRESSION_NOISE_FLOOR_NS:
            statuses[_key(r)] = Status.REGRESSION
    return statuses


def _build_diagnoses(results: list[ProbeResult]) -> dict[str, dict]:
    """Run ``diagnose`` on every result; ``{}`` when diagnose is absent."""
    if not _HAS_DIAGNOSE or _diagnose_one is None:
        return {}
    out: dict[str, dict] = {}
    for r in results:
        try:
            verdict = _diagnose_one(r)
        except Exception:  # pragma: no cover - defensive
            continue
        out[_key(r)] = _diagnosis_to_dict(verdict)
    return out


def _diagnosis_to_dict(verdict) -> dict:
    """Coerce a diagnose() return value into the JSON-safe row payload.

    ``diagnose`` is expected to return either:
    * a dataclass / object with ``reason`` and ``summary`` attributes, or
    * a mapping with at least ``reason`` (and optionally ``summary``).

    Both shapes are accepted so the dashboard stays robust to small
    changes in the diagnosis module's API.
    """
    if verdict is None:
        return {"reason": "NEUTRAL", "summary": ""}
    if isinstance(verdict, dict):
        reason = verdict.get("reason", "NEUTRAL")
        summary = verdict.get("summary", "")
    else:
        reason = getattr(verdict, "reason", None)
        if reason is None:
            reasons = getattr(verdict, "reasons", ())
            reason = reasons[0] if reasons else "NEUTRAL"
        summary = getattr(verdict, "summary", "")
    # ``reason`` may itself be an Enum.
    if hasattr(reason, "value"):
        reason = reason.value
    elif hasattr(reason, "name"):
        reason = reason.name
    return {"reason": str(reason), "summary": str(summary)}


def _overhead_factor(r: ProbeResult) -> float:
    return r.sc_median_ns / max(r.bare_median_ns, 1.0)


def _overhead_ns(r: ProbeResult) -> float:
    return r.sc_median_ns - r.bare_median_ns


def _build_overall(
    results: list[ProbeResult],
    diagnoses: dict[str, dict],
) -> dict:
    """Build the data structure used by the Diagnosis section.

    Always returns a dict so the JS side can render fixed panels
    without branching. When the diagnosis module isn't available we
    still populate the "top overhead" and "top wins" panels from raw
    timings — they remain useful even without the per-row reason.
    """
    # ----- panel 3: top overhead cases (worst SC-vs-bare ratio) --------
    by_overhead = sorted(
        results,
        key=lambda r: (_overhead_factor(r), _overhead_ns(r)),
        reverse=True,
    )
    top_overhead = [
        {
            "operation_name": r.operation_name,
            "size": r.size,
            "backend": r.backend,
            "check_level": r.check_level,
            "sc_median_ns": r.sc_median_ns,
            "bare_median_ns": r.bare_median_ns,
            "overhead_ns": _overhead_ns(r),
            "overhead_factor": _overhead_factor(r),
            "validation_overhead_ns": r.validation_overhead_ns,
            "speedup": r.speedup,
            "reason": (diagnoses.get(_key(r)) or {}).get("reason", "NEUTRAL"),
            "summary": (diagnoses.get(_key(r)) or {}).get("summary", ""),
        }
        for r in by_overhead[:5]
    ]

    # ----- panel 4: top wins (highest effective speedup) ---------------
    def _eff_speedup(r: ProbeResult) -> float:
        if r.family == "kernel" and r.optimized_speedup is not None:
            return r.optimized_speedup
        return r.speedup

    by_speed = sorted(results, key=_eff_speedup, reverse=True)
    top_wins = [
        {
            "operation_name": r.operation_name,
            "size": r.size,
            "backend": r.backend,
            "check_level": r.check_level,
            "speedup": _eff_speedup(r),
            "is_optimized": (r.family == "kernel" and r.optimized_speedup is not None),
            "reason": (diagnoses.get(_key(r)) or {}).get("reason", "NEUTRAL"),
        }
        for r in by_speed[:5]
    ]

    # ----- panel 5: JAX compile summary --------------------------------
    jax_compiles = [
        r.compile_ns_median for r in results
        if r.backend == "jax" and r.compile_ns_median is not None
    ]
    if jax_compiles:
        jax_summary = {
            "cases": len(jax_compiles),
            "median_compile_ns": float(median(jax_compiles)),
        }
    else:
        jax_summary = None

    # ----- panels 1 + 2: ask the diagnose module ----------------------
    dominant_reason_counts: list[tuple[str, int]] = []
    narrative = ""
    if _HAS_DIAGNOSE and _overall_diagnosis is not None:
        try:
            res = _overall_diagnosis(results)
        except Exception:  # pragma: no cover - defensive
            res = None
        if isinstance(res, dict):
            raw_counts = res.get("dominant_reason_counts") or {}
            if isinstance(raw_counts, dict):
                items = list(raw_counts.items())
            else:
                items = list(raw_counts)
            normalized: list[tuple[str, int]] = []
            for entry in items:
                if isinstance(entry, tuple) and len(entry) == 2:
                    key, count = entry
                else:
                    # Fall back for ``{reason: count}`` flattened tuples.
                    key, count = entry, raw_counts[entry] if isinstance(raw_counts, dict) else 0
                if hasattr(key, "value"):
                    key = key.value
                elif hasattr(key, "name"):
                    key = key.name
                try:
                    normalized.append((str(key), int(count)))
                except (TypeError, ValueError):
                    continue
            normalized.sort(key=lambda kv: kv[1], reverse=True)
            dominant_reason_counts = normalized[:5]
            narrative = str(res.get("narrative", ""))

    persistent_overhead = _persistent_overhead(results)
    return {
        "dominant_reason_counts": dominant_reason_counts,
        "narrative": narrative,
        "top_overhead": top_overhead,
        "top_wins": top_wins,
        "jax_compile_summary": jax_summary,
        "persistent_overhead": persistent_overhead,
    }


def _persistent_overhead(results: list[ProbeResult]) -> list[dict]:
    """Find operations whose overhead remains visible at the largest size."""
    by_key: dict[tuple[str, str, str], list[ProbeResult]] = {}
    for r in results:
        by_key.setdefault((r.operation_name, r.backend, r.check_level), []).append(r)
    out: list[dict] = []
    for (name, backend, check_level), rows in by_key.items():
        ordered = sorted(rows, key=lambda r: r.size)
        if len(ordered) < 2:
            continue
        first, last = ordered[0], ordered[-1]
        first_factor = _overhead_factor(first)
        last_factor = _overhead_factor(last)
        if last_factor < 1.10:
            continue
        trend = "persists" if last_factor >= first_factor * 0.80 else "shrinks"
        out.append(
            {
                "operation_name": name,
                "backend": backend,
                "check_level": check_level,
                "small_size": first.size,
                "large_size": last.size,
                "small_overhead_factor": first_factor,
                "large_overhead_factor": last_factor,
                "trend": trend,
            }
        )
    out.sort(key=lambda row: row["large_overhead_factor"], reverse=True)
    return out[:5]


def _row_payload(r: ProbeResult, status: Status, diagnosis: dict | None) -> dict:
    """Flatten a :class:`ProbeResult` into a JSON-safe row dict."""
    seeds = [
        {
            "seed": s.seed,
            "sc_median_ns": s.sc_median_ns,
            "bare_median_ns": s.bare_median_ns,
            "error": s.error_vs_reference,
        }
        for s in r.seeds
    ]
    diag = diagnosis or {"reason": "NEUTRAL", "summary": ""}
    return {
        "operation_name": r.operation_name,
        "family": r.family,
        "backend": r.backend,
        "device": r.device,
        "check_level": r.check_level,
        "size": r.size,
        "bare_median_ns": r.bare_median_ns,
        "sc_median_ns": r.sc_median_ns,
        "overhead_ns": _overhead_ns(r),
        "overhead_factor": _overhead_factor(r),
        "abstraction_overhead_ns": r.abstraction_overhead_ns,
        "validation_overhead_ns": r.validation_overhead_ns,
        "speedup": r.speedup,
        "speedup_std": r.speedup_std,
        "error_max": r.error_max,
        "sc_peak_bytes": r.sc_peak_bytes_median,
        "bare_peak_bytes": r.bare_peak_bytes_median,
        "optimized_speedup": r.optimized_speedup,
        "compile_ns_median": r.compile_ns_median,
        "notes": r.notes,
        "status": status.value,
        "diagnosis_reason": diag["reason"],
        "diagnosis_summary": diag["summary"],
        "seeds": seeds,
    }


def _baseline_payload(
    results: list[ProbeResult],
    baseline: list[ProbeResult],
    statuses: dict[str, Status],
) -> list[dict]:
    """One row per case that exists in both runs — for the comparison plot."""
    baseline_by_key = {_key(b): b for b in baseline}
    out: list[dict] = []
    for r in results:
        base = baseline_by_key.get(_key(r))
        if base is None:
            continue
        out.append(
            {
                "operation_name": r.operation_name,
                "family": r.family,
                "backend": r.backend,
                "device": r.device,
                "check_level": r.check_level,
                "size": r.size,
                "baseline_speedup": base.speedup,
                "current_speedup": r.speedup,
                "baseline_sc_ns": base.sc_median_ns,
                "current_sc_ns": r.sc_median_ns,
                "status": statuses[_key(r)].value,
            }
        )
    return out


def _summary(rows: list[dict]) -> dict:
    """Top-line counts shown in the summary cards."""
    counts = {s.value: 0 for s in Status}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    speedups = [row["speedup"] for row in rows]
    return {
        "total": len(rows),
        "median_speedup": float(median(speedups)) if speedups else 0.0,
        "min_speedup": min(speedups) if speedups else 0.0,
        "max_speedup": max(speedups) if speedups else 0.0,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# HTML rendering


def _render_html(payload: dict) -> str:
    """Inject ``payload`` into the HTML template and return the full document."""
    # Embed payload as a JSON literal. We dump twice — once for the data
    # itself (allow_nan=False to keep things JSON-strict where possible)
    # and once to escape it as a string literal that JS can ``JSON.parse``.
    # The double-encoding lets us drop the result straight into a
    # ``<script>`` body with no further escaping concerns.
    raw_json = json.dumps(payload, default=_json_default, allow_nan=True)
    embedded = json.dumps(raw_json)
    # Defense in depth against an inline ``</script>`` sequence in any
    # operation name or note — escape HTML in the string form as well.
    embedded_html_safe = embedded.replace("</", "<\\/")

    meta = payload["meta"]
    versions = meta.get("versions", {})
    sc_version = html.escape(versions.get("spacecore", "unknown"))
    py_version = html.escape(meta.get("python", "unknown"))
    plat = html.escape(meta.get("platform", "unknown"))
    n_cases = payload["summary"]["total"]
    median_speedup = payload["summary"]["median_speedup"]
    counts = payload["summary"]["counts"]

    summary_cards = _summary_cards_html(n_cases, median_speedup, counts)
    filter_controls = _filter_controls_html(payload["families"], payload["backends"])
    diagnosis_section = _diagnosis_section_html(payload["overall_diagnosis"])

    return _TEMPLATE.format(
        plotly_cdn=_PLOTLY_CDN,
        css=_CSS,
        sc_version=sc_version,
        py_version=py_version,
        platform=plat,
        n_cases=n_cases,
        summary_cards=summary_cards,
        filter_controls=filter_controls,
        diagnosis_section=diagnosis_section,
        embedded_data=embedded_html_safe,
        plot_baseline_block=_PLOT_BASELINE_BLOCK if payload["has_baseline"] else "",
        js=_JS,
    )


def _json_default(obj):
    # Numpy / non-standard fallthrough — avoid crashing on stray scalars.
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _summary_cards_html(n_cases: int, median_speedup: float, counts: dict) -> str:
    """Build the summary card row HTML."""
    parts: list[str] = []
    parts.append(_card("Total cases", f"{n_cases}", "#1f77b4"))
    parts.append(_card("Median speedup", f"{median_speedup:.2f}x", "#1f77b4"))
    for status in Status:
        color = _STATUS_COLORS[status.value]
        parts.append(_card(status.value, str(counts.get(status.value, 0)), color))
    return "\n".join(parts)


def _card(label: str, value: str, color: str) -> str:
    return (
        f'<div class="card" style="border-top:4px solid {color};">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value" style="color:{color};">{html.escape(value)}</div>'
        f"</div>"
    )


def _filter_controls_html(families: list[str], backends: list[str]) -> str:
    backend_boxes = "\n".join(
        f'<label class="chip" data-backend-chip="{html.escape(b)}">'
        f'<input type="checkbox" class="f-backend" value="{html.escape(b)}" checked>'
        f'<span style="color:{_BACKEND_COLORS.get(b, "#333")};">{html.escape(b)}</span>'
        f"</label>"
        for b in backends
    )
    family_boxes = "\n".join(
        f'<label class="chip"><input type="checkbox" class="f-family" value="{html.escape(f)}" checked>'
        f'<span style="color:{_FAMILY_COLORS.get(f, "#333")};">{html.escape(f)}</span></label>'
        for f in families
    )
    status_boxes = "\n".join(
        f'<label class="chip"><input type="checkbox" class="f-status" value="{s.value}" checked>'
        f'<span style="color:{_STATUS_COLORS[s.value]};">{s.value}</span></label>'
        for s in Status
    )
    return f"""
<div class="filter-row" id="backendFilter">
  <div class="filter-group">
    <div class="filter-label">Backend</div>
    <div class="chips">{backend_boxes}</div>
  </div>
  <div class="filter-group">
    <div class="filter-label">Check mode</div>
    <select id="f-check-level">
      <option value="all" selected>all</option>
      <option value="none">none</option>
      <option value="cheap">cheap</option>
    </select>
  </div>
</div>
<div class="filter-row">
  <div class="filter-group">
    <div class="filter-label">Family</div>
    <div class="chips">{family_boxes}</div>
  </div>
  <div class="filter-group">
    <div class="filter-label">Status</div>
    <div class="chips">{status_boxes}</div>
  </div>
  <div class="filter-group">
    <div class="filter-label">Search</div>
    <input type="text" id="f-search" placeholder="substring of operation_name" />
  </div>
  <div class="filter-group">
    <div class="filter-label">Speedup range</div>
    <div class="range-row">
      <input type="number" id="f-min" placeholder="min" step="0.01" />
      <span>–</span>
      <input type="number" id="f-max" placeholder="max" step="0.01" />
    </div>
  </div>
  <div class="filter-group">
    <div class="filter-label">&nbsp;</div>
    <button id="f-reset" type="button">Reset filters</button>
  </div>
</div>
"""


def _diagnosis_section_html(overall: dict) -> str:
    """Build the static parts of the Diagnosis section.

    The five panels are emitted as empty containers; ``_JS`` fills them
    from ``window.BENCH_DATA.overall_diagnosis`` so it can re-render
    when filters change in future work. Panel 5 stays hidden until
    JS reveals it.
    """
    has_jax = overall.get("jax_compile_summary") is not None
    jax_hidden = "" if has_jax else 'style="display:none;"'
    return f"""
<h2>What's causing overhead and wins</h2>
<div class="diagnosis-grid" id="diagnosis-section">
  <div class="diag-panel" id="diag-dominant">
    <div class="diag-title">Dominant reasons</div>
    <div class="diag-body" id="diag-dominant-body"></div>
  </div>
  <div class="diag-panel" id="diag-narrative">
    <div class="diag-title">Narrative</div>
    <div class="diag-body narrative" id="diag-narrative-body"></div>
  </div>
  <div class="diag-panel" id="diag-overhead">
    <div class="diag-title">Worst overhead ratios</div>
    <ol class="diag-list" id="diag-overhead-body"></ol>
  </div>
  <div class="diag-panel" id="diag-persistent">
    <div class="diag-title">Overhead persistence</div>
    <ol class="diag-list" id="diag-persistent-body"></ol>
  </div>
  <div class="diag-panel" id="diag-wins">
    <div class="diag-title">Top wins</div>
    <ol class="diag-list" id="diag-wins-body"></ol>
  </div>
  <div class="diag-panel" id="diag-jax" {jax_hidden}>
    <div class="diag-title">JAX compile summary</div>
    <div class="diag-body" id="diag-jax-body"></div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# CSS — kept inline so the file is truly self-contained.


_CSS = """
  :root {
    --bg: #f7f8fa;
    --fg: #1d2330;
    --muted: #5a6473;
    --card: #ffffff;
    --border: #e4e7ec;
    --accent: #1f77b4;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
    margin: 0;
    padding: 24px 32px 64px;
    line-height: 1.4;
  }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 24px 0 8px; color: var(--muted);
       text-transform: uppercase; letter-spacing: 0.06em; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .meta code { background: #eef0f3; padding: 1px 6px; border-radius: 3px;
               font-size: 12px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    min-width: 120px;
    flex: 0 0 auto;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .card-label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                letter-spacing: 0.06em; }
  .card-value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .filter-row {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 18px;
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: flex-start;
  }
  .filter-group { display: flex; flex-direction: column; gap: 6px; }
  .filter-label { font-size: 11px; color: var(--muted);
                  text-transform: uppercase; letter-spacing: 0.06em; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; max-width: 480px; }
  .chip {
    display: inline-flex; align-items: center; gap: 4px;
    border: 1px solid var(--border); border-radius: 999px;
    padding: 2px 10px; font-size: 12px; cursor: pointer;
    background: #fff;
  }
  .chip input { margin: 0; }
  input[type="text"], input[type="number"] {
    padding: 4px 8px; font-size: 13px;
    border: 1px solid var(--border); border-radius: 4px;
    font-family: inherit;
  }
  #f-search { width: 220px; }
  .range-row { display: flex; align-items: center; gap: 6px; }
  .range-row input { width: 90px; }
  button {
    padding: 5px 12px; font-size: 13px; cursor: pointer;
    border: 1px solid var(--border); border-radius: 4px;
    background: #fff;
  }
  button:hover { background: #f0f1f4; }
  .diagnosis-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }
  .diag-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
  }
  .diag-title {
    font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 8px;
  }
  .diag-body { font-size: 13px; }
  .diag-body.narrative {
    background: #eef0f3;
    border-radius: 4px;
    padding: 8px 10px;
    color: var(--fg);
    white-space: pre-wrap;
    min-height: 32px;
  }
  .diag-list {
    margin: 0; padding-left: 18px; font-size: 12px;
    line-height: 1.5;
  }
  .diag-list li { margin-bottom: 4px; }
  .reason-chip {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    color: #fff; font-size: 11px; font-weight: 600;
    margin: 2px 4px 2px 0;
  }
  .backend-tag {
    display: inline-block; padding: 0 6px; border-radius: 4px;
    color: #fff; font-size: 10px; font-weight: 600;
    margin-left: 4px; vertical-align: middle;
  }
  .reason-count { color: var(--muted); font-size: 11px; margin-left: 4px; }
  .plot-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .plot-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .plot-card.wide { grid-column: 1 / -1; }
  .plot {
    padding: 8px;
    min-height: 360px;
  }
  .plot-caption {
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
    padding: 8px 12px 10px;
    background: #fbfcfd;
  }
  table {
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
    font-size: 12px;
  }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: #eef0f3; cursor: pointer; user-select: none;
       font-weight: 600; position: sticky; top: 0; z-index: 1; }
  th:hover { background: #e3e7ec; }
  th.sort-asc::after  { content: " \\25B2"; color: var(--muted); }
  th.sort-desc::after { content: " \\25BC"; color: var(--muted); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    color: #fff; font-size: 11px; font-weight: 600;
  }
  .table-wrap {
    max-height: 70vh; overflow: auto; border: 1px solid var(--border);
    border-radius: 6px; margin-top: 8px;
  }
  .table-wrap table { border: 0; border-radius: 0; }
  .footer { color: var(--muted); font-size: 12px; margin-top: 16px; }
"""


# ---------------------------------------------------------------------------
# JS — builds plots from window.BENCH_DATA and wires filters.


_PLOT_BASELINE_BLOCK = """
  <div class="plot-card wide">
    <div id="plot-baseline" class="plot"></div>
    <div class="plot-caption">
      Compares current and baseline speedup for matching cases. Points below the
      diagonal mean the current run is slower than the baseline.
    </div>
  </div>
"""


_JS = r"""
(function () {
  const DATA = window.BENCH_DATA;
  const FAMILY_COLORS = DATA.family_colors;
  const STATUS_COLORS = DATA.status_colors;
  const BACKEND_COLORS = DATA.backend_colors || {};
  const REASON_COLORS = DATA.reason_colors || {};
  const ROWS = DATA.rows;
  const BASELINE = DATA.baseline || [];
  const STATUSES = DATA.statuses;
  const FAMILIES = DATA.families;
  const BACKENDS = DATA.backends || [];
  const CHECK_LEVELS = DATA.check_levels || ["none", "cheap"];
  const OVERALL = DATA.overall_diagnosis || {
    dominant_reason_counts: [],
    narrative: "",
    top_overhead: [],
    top_wins: [],
    jax_compile_summary: null,
    persistent_overhead: [],
  };

  // ---- Filter state -----------------------------------------------------
  const state = {
    backends: new Set(BACKENDS),
    families: new Set(FAMILIES),
    statuses: new Set(STATUSES),
    checkLevel: "all",
    search: "",
    minSpeedup: null,
    maxSpeedup: null,
  };

  function filtered() {
    return ROWS.filter(function (r) {
      if (!state.backends.has(r.backend)) return false;
      if (state.checkLevel !== "all" && r.check_level !== state.checkLevel) return false;
      if (!state.families.has(r.family)) return false;
      if (!state.statuses.has(r.status)) return false;
      if (state.search && r.operation_name.indexOf(state.search) === -1) return false;
      if (state.minSpeedup !== null && r.speedup < state.minSpeedup) return false;
      if (state.maxSpeedup !== null && r.speedup > state.maxSpeedup) return false;
      return true;
    });
  }

  function backendColor(b) {
    return BACKEND_COLORS[b] || "#888";
  }
  function checkDash(level) {
    return level === "none" ? "solid" : "dash";
  }
  function seriesLabel(backend, level) {
    return backend + " / SpaceCore " + level;
  }
  function reasonColor(r) {
    return REASON_COLORS[r] || REASON_COLORS.NEUTRAL || "#d3d3d3";
  }

  // ---- Plot builders ----------------------------------------------------
  const LAYOUT_BASE = {
    margin: { l: 50, r: 20, t: 40, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    font: { family: "-apple-system, sans-serif", size: 12, color: "#1d2330" },
  };

  function backendsPresent(rows) {
    const seen = {};
    rows.forEach(function (r) { seen[r.backend] = true; });
    return BACKENDS.filter(function (b) { return seen[b]; });
  }

  function seriesPresent(rows) {
    const seen = {};
    rows.forEach(function (r) { seen[r.backend + "||" + r.check_level] = true; });
    const out = [];
    BACKENDS.forEach(function (b) {
      CHECK_LEVELS.forEach(function (level) {
        if (seen[b + "||" + level]) out.push({ backend: b, check_level: level });
      });
    });
    return out;
  }

  function speedupDistribution(rows) {
    const present = seriesPresent(rows);
    const allSpeedups = rows.map(function (r) { return r.speedup; });
    const minS = allSpeedups.length ? Math.min.apply(null, allSpeedups) : 0.01;
    const maxS = allSpeedups.length ? Math.max.apply(null, allSpeedups) : 10;
    const lo = Math.log10(Math.max(minS, 1e-3));
    const hi = Math.log10(Math.max(maxS, 1.0));
    const traces = present.map(function (series) {
      const xs = rows
        .filter(function (r) {
          return r.backend === series.backend && r.check_level === series.check_level;
        })
        .map(function (r) { return Math.log10(Math.max(r.speedup, 1e-6)); });
      return {
        x: xs,
        type: "histogram",
        name: seriesLabel(series.backend, series.check_level),
        autobinx: false,
        xbins: { start: lo, end: hi, size: (hi - lo) / 24 },
        marker: { color: backendColor(series.backend), line: { width: 1, color: "#1F3552" } },
        opacity: 0.7,
        hovertemplate: seriesLabel(series.backend, series.check_level) +
          "<br>check_level: " + series.check_level +
          "<br>speedup: %{x:.3f} (log10)<br>cases: %{y}<extra></extra>",
      };
    });
    const shapes = [];
    [
      { x: 0.1, color: STATUS_COLORS.HEAVY_LOSS },
      { x: 0.5, color: STATUS_COLORS.LOSS },
      { x: 1.0, color: STATUS_COLORS.NEUTRAL },
      { x: 5.0, color: STATUS_COLORS.WIN },
    ].forEach(function (g) {
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: Math.log10(g.x), x1: Math.log10(g.x), y0: 0, y1: 1,
        line: { color: g.color, dash: "dash", width: 2 },
      });
    });
    const layout = Object.assign({}, LAYOUT_BASE, {
      title: "1. Speedup distribution per backend (log-x; dashed lines at 0.1x / 0.5x / 1x / 5x)",
      xaxis: { title: "speedup (log10)", tickformat: ".2f" },
      yaxis: { title: "case count" },
      barmode: "stack",
      shapes: shapes,
      bargap: 0.05,
      showlegend: true,
      legend: { orientation: "h", y: -0.2 },
    });
    Plotly.react("plot-distribution", traces, layout, { displaylogo: false, responsive: true });
  }

  function overheadDecomposition(rows) {
    const worst = rows
      .filter(function (r) { return r.overhead_ns > 0; })
      .slice()
      .sort(function (a, b) {
        return (b.overhead_factor - a.overhead_factor) ||
               (b.overhead_ns - a.overhead_ns);
      })
      .slice(0, 18)
      .reverse();
    const labels = worst.map(function (r) {
      return r.operation_name + "<br>n=" + r.size + " " + r.backend + "/" + r.check_level;
    });
    const validation = worst.map(function (r) {
      if (r.validation_overhead_ns == null) return 0;
      return Math.max(0, Math.min(r.validation_overhead_ns, r.overhead_ns));
    });
    const other = worst.map(function (r, i) {
      return Math.max(0, r.overhead_ns - validation[i]);
    });
    const text = worst.map(function (r) {
      return r.operation_name +
             "<br>backend: " + r.backend +
             "<br>check_level: " + r.check_level +
             "<br>size: " + r.size +
             "<br>bare: " + r.bare_median_ns.toFixed(0) + " ns" +
             "<br>SpaceCore: " + r.sc_median_ns.toFixed(0) + " ns" +
             "<br>extra: " + r.overhead_ns.toFixed(0) + " ns" +
             "<br>runtime ratio: " + r.overhead_factor.toFixed(2) + "x";
    });
    const traces = [
      {
        x: other, y: labels, text: text,
        name: "non-validation overhead",
        type: "bar", orientation: "h",
        marker: { color: "#4C78A8" },
        hovertemplate: "%{text}<br>non-validation: %{x:.0f} ns<extra></extra>",
      },
      {
        x: validation, y: labels, text: text,
        name: "cheap validation overhead",
        type: "bar", orientation: "h",
        marker: { color: "#F58518" },
        hovertemplate: "%{text}<br>validation: %{x:.0f} ns<extra></extra>",
      },
    ];
    const layout = Object.assign({}, LAYOUT_BASE, {
      title: "2. Worst overhead decomposition (SpaceCore time minus bare time)",
      xaxis: { title: "extra runtime over bare (ns)" },
      yaxis: { automargin: true },
      barmode: "stack",
      showlegend: true,
      legend: { orientation: "h", y: -0.2 },
    });
    Plotly.react("plot-overhead", traces, layout, { displaylogo: false, responsive: true });
  }

  function scalingCurves(rows) {
    // One line per (operation, backend, check level).
    const byKey = {};
    rows.forEach(function (r) {
      const k = r.operation_name + "||" + r.backend + "||" + r.check_level;
      (byKey[k] = byKey[k] || []).push(r);
    });
    const traces = [];
    Object.keys(byKey).sort().forEach(function (k) {
      const pts = byKey[k].slice().sort(function (a, b) { return a.size - b.size; });
      if (pts.length < 2) return;
      const backend = pts[0].backend;
      const checkLevel = pts[0].check_level;
      const opname = pts[0].operation_name;
      traces.push({
        x: pts.map(function (p) { return p.size; }),
        y: pts.map(function (p) { return p.overhead_factor; }),
        name: opname + " [" + backend + ", SpaceCore " + checkLevel + "]",
        mode: "lines+markers",
        type: "scatter",
        line: { color: backendColor(backend), width: 1.5, dash: checkDash(checkLevel) },
        marker: { color: backendColor(backend), size: 6 },
        text: pts.map(function (p) {
          return p.operation_name +
                 "<br>backend: " + p.backend +
                 "<br>check_level: " + p.check_level +
                 "<br>size: " + p.size +
                 "<br>sc median: " + p.sc_median_ns.toFixed(0) + " ns" +
                 "<br>bare median: " + p.bare_median_ns.toFixed(0) + " ns" +
                 "<br>runtime ratio: " + p.overhead_factor.toFixed(2) + "x" +
                 "<br>speedup: " + p.speedup.toFixed(2) + "x";
        }),
        hovertemplate: "%{text}<extra></extra>",
      });
    });
    const layout = Object.assign({}, LAYOUT_BASE, {
      title: "3. Overhead persistence by size (1.0x means SpaceCore matches bare)",
      xaxis: { type: "log", title: "problem size" },
      yaxis: { type: "log", title: "runtime ratio: SpaceCore / bare" },
      showlegend: true,
      legend: { orientation: "h", y: -0.2 },
      shapes: [
        {
          type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1,
          line: { color: "#2ca02c", dash: "dash", width: 1 },
        },
        {
          type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 2, y1: 2,
          line: { color: "#ff7f0e", dash: "dot", width: 1 },
        },
      ],
    });
    Plotly.react("plot-scaling", traces, layout, { displaylogo: false, responsive: true });
  }

  function perSeedJitter(rows) {
    const present = seriesPresent(rows);
    const traces = present.map(function (series) {
      const xs = [], ys = [], texts = [];
      rows.filter(function (r) {
        return r.backend === series.backend && r.check_level === series.check_level;
      }).forEach(function (r) {
        if (r.sc_median_ns <= 0) return;
        (r.seeds || []).forEach(function (s) {
          xs.push(r.size);
          ys.push(s.sc_median_ns / r.sc_median_ns);
          texts.push(r.operation_name + " [" + r.backend + "]" +
            "<br>check_level: " + r.check_level + "<br>seed " + s.seed);
        });
      });
      return {
        x: xs, y: ys, text: texts,
        name: seriesLabel(series.backend, series.check_level),
        mode: "markers", type: "scatter",
        marker: { color: backendColor(series.backend), size: 7, opacity: 0.7,
                  symbol: series.check_level === "none" ? "circle" : "diamond" },
        hovertemplate: "%{text}<br>size: %{x}<br>ratio: %{y:.3f}<extra></extra>",
      };
    }).filter(function (t) { return t.x.length > 0; });
    const layout = Object.assign({}, LAYOUT_BASE, {
      title: "4. Per-seed jitter, colored by backend (closer to 1.0 = lower noise)",
      xaxis: { type: "log", title: "problem size" },
      yaxis: { title: "per-seed median / aggregate" },
      showlegend: true,
      legend: { orientation: "h", y: -0.2 },
      shapes: [{
        type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1,
        line: { color: "#999", dash: "dash", width: 1 },
      }],
    });
    Plotly.react("plot-jitter", traces, layout, { displaylogo: false, responsive: true });
  }

  function memoryBars(rows) {
    const byFamily = {};
    rows.forEach(function (r) {
      (byFamily[r.family] = byFamily[r.family] || []).push(r);
    });
    const fams = Object.keys(byFamily).sort();
    if (!fams.length) {
      Plotly.react("plot-memory", [], LAYOUT_BASE, { responsive: true });
      return;
    }
    function med(arr) {
      if (!arr.length) return 0;
      const s = arr.slice().sort(function (a, b) { return a - b; });
      const mid = Math.floor(s.length / 2);
      return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }
    const present = seriesPresent(rows);
    const traces = present.map(function (series) {
      return {
        x: fams, type: "bar", name: seriesLabel(series.backend, series.check_level),
        offsetgroup: series.backend + "-" + series.check_level,
        y: fams.map(function (f) {
          const vals = byFamily[f]
            .filter(function (r) {
              return r.backend === series.backend && r.check_level === series.check_level;
            })
            .map(function (r) {
              return r.sc_peak_bytes / Math.max(r.bare_peak_bytes || 0, 64);
            });
          return med(vals);
        }),
        marker: { color: backendColor(series.backend),
                  pattern: { shape: series.check_level === "none" ? "" : "/" } },
        hovertemplate: "check_level: " + series.check_level +
          "<br>%{x}<br>%{y:.2f}x peak memory<extra></extra>",
      };
    });
    const layout = Object.assign({}, LAYOUT_BASE, {
      title: "5. Median memory overhead ratio per family",
      barmode: "group",
      yaxis: { type: "log", title: "SpaceCore peak / bare peak" },
      xaxis: { title: "family" },
      showlegend: true,
      legend: { orientation: "h", y: -0.2 },
      shapes: [{
        type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1,
        line: { color: "#999", dash: "dash", width: 1 },
      }],
    });
    Plotly.react("plot-memory", traces, layout, { displaylogo: false, responsive: true });
  }

  function baselineScatter(rows) {
    if (!BASELINE.length) return;
    // Build a set of visible keys from the current filtered rows.
    const visible = new Set(rows.map(function (r) {
      return r.operation_name + "@" + r.size + "@" + r.backend + "@" +
             r.device + "@" + r.check_level;
    }));
    const pts = BASELINE.filter(function (b) {
      return visible.has(b.operation_name + "@" + b.size + "@" + b.backend + "@" +
        b.device + "@" + b.check_level);
    });
    const xs = pts.map(function (p) { return p.baseline_speedup; });
    const ys = pts.map(function (p) { return p.current_speedup; });
    const colors = pts.map(function (p) {
      return p.status === "REGRESSION" ? STATUS_COLORS.REGRESSION : backendColor(p.backend);
    });
    const text = pts.map(function (p) {
      return p.operation_name + " [" + p.backend + "]<br>size " + p.size +
             "<br>check_level: " + p.check_level +
             "<br>baseline: " + p.baseline_speedup.toFixed(2) + "x" +
             "<br>current: " + p.current_speedup.toFixed(2) + "x" +
             "<br>" + p.status;
    });
    const trace = {
      x: xs, y: ys, text: text,
      mode: "markers", type: "scatter",
      marker: { color: colors, size: 8, opacity: 0.8,
                line: { color: "#222", width: 0.5 } },
      hovertemplate: "%{text}<extra></extra>",
    };
    const allVals = xs.concat(ys);
    const minV = allVals.length ? Math.max(Math.min.apply(null, allVals), 1e-3) : 0.01;
    const maxV = allVals.length ? Math.max(Math.max.apply(null, allVals), 1) : 10;
    const diag = {
      x: [minV, maxV], y: [minV, maxV],
      mode: "lines", type: "scatter", showlegend: false,
      line: { color: "#999", dash: "dash", width: 1 },
      hoverinfo: "skip",
    };
    const layout = Object.assign({}, LAYOUT_BASE, {
      title: "6. Current vs baseline speedup (log-log; below diagonal = regression)",
      xaxis: { type: "log", title: "baseline speedup" },
      yaxis: { type: "log", title: "current speedup" },
      showlegend: false,
    });
    Plotly.react("plot-baseline", [diag, trace], layout, { displaylogo: false, responsive: true });
  }

  // ---- Diagnosis section ------------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }
  function backendTag(b) {
    return '<span class="backend-tag" style="background:' + backendColor(b) + ';">' +
           escapeHtml(b) + "</span>";
  }
  function reasonChip(reason, summary) {
    const title = summary ? ' title="' + escapeHtml(summary) + '"' : "";
    return '<span class="reason-chip" style="background:' + reasonColor(reason) + ';"' +
           title + ">" + escapeHtml(reason) + "</span>";
  }
  function fmtSpeedup(v) { return (v == null ? "-" : Number(v).toFixed(2)) + "x"; }
  function fmtFactor(v) { return (v == null ? "-" : Number(v).toFixed(2)) + "x"; }
  function fmtNsShort(v) {
    if (v == null) return "-";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + " ms";
    if (v >= 1e3) return (v / 1e3).toFixed(2) + " us";
    return v.toFixed(0) + " ns";
  }

  function renderDiagnosisSection() {
    // Panel 1: dominant reasons.
    const dom = document.getElementById("diag-dominant-body");
    if (dom) {
      const counts = OVERALL.dominant_reason_counts || [];
      if (!counts.length) {
        dom.innerHTML = '<span class="reason-count">No dominant reasons reported.</span>';
      } else {
        dom.innerHTML = counts.map(function (kv) {
          const reason = kv[0], count = kv[1];
          return reasonChip(reason, "") +
                 '<span class="reason-count">x' + count + "</span>";
        }).join(" ");
      }
    }
    // Panel 2: narrative.
    const narr = document.getElementById("diag-narrative-body");
    if (narr) {
      const text = OVERALL.narrative || "";
      narr.textContent = text || "No narrative available.";
    }
    // Panel 3: top overhead cases.
    const oh = document.getElementById("diag-overhead-body");
    if (oh) {
      const cases = OVERALL.top_overhead || [];
      if (!cases.length) {
        oh.innerHTML = "<li>No data.</li>";
      } else {
        oh.innerHTML = cases.map(function (c) {
          return "<li>" + escapeHtml(c.operation_name) +
                 " (n=" + c.size + ") " + backendTag(c.backend) +
                 " <span class=\"reason-count\">checks=" + escapeHtml(c.check_level) + "</span>" +
                 " &mdash; " + fmtFactor(c.overhead_factor) + " runtime" +
                 " &middot; +" + fmtNsShort(c.overhead_ns) +
                 " " + reasonChip(c.reason, c.summary || "") + "</li>";
        }).join("");
      }
    }
    // Panel 4: persistent overhead.
    const persist = document.getElementById("diag-persistent-body");
    if (persist) {
      const cases = OVERALL.persistent_overhead || [];
      if (!cases.length) {
        persist.innerHTML = "<li>No size-persistent overhead above 1.10x.</li>";
      } else {
        persist.innerHTML = cases.map(function (c) {
          return "<li>" + escapeHtml(c.operation_name) +
                 " " + backendTag(c.backend) +
                 " <span class=\"reason-count\">checks=" + escapeHtml(c.check_level) + "</span>" +
                 " &mdash; n=" + c.small_size + " " + fmtFactor(c.small_overhead_factor) +
                 " to n=" + c.large_size + " " + fmtFactor(c.large_overhead_factor) +
                 " <span class=\"reason-count\">" + escapeHtml(c.trend) + "</span></li>";
        }).join("");
      }
    }
    // Panel 5: top wins.
    const win = document.getElementById("diag-wins-body");
    if (win) {
      const cases = OVERALL.top_wins || [];
      if (!cases.length) {
        win.innerHTML = "<li>No data.</li>";
      } else {
        win.innerHTML = cases.map(function (c) {
          const tag = c.is_optimized ? ' <span class="reason-count">(optimized)</span>' : "";
          return "<li>" + escapeHtml(c.operation_name) +
                 " (n=" + c.size + ") " + backendTag(c.backend) +
                 " <span class=\"reason-count\">checks=" + escapeHtml(c.check_level) + "</span>" +
                 " &mdash; " + fmtSpeedup(c.speedup) + tag +
                 " " + reasonChip(c.reason, "") + "</li>";
        }).join("");
      }
    }
    // Panel 6: JAX compile summary.
    const jaxPanel = document.getElementById("diag-jax");
    const jaxBody = document.getElementById("diag-jax-body");
    if (jaxPanel && jaxBody) {
      const j = OVERALL.jax_compile_summary;
      if (!j) {
        jaxPanel.style.display = "none";
      } else {
        jaxPanel.style.display = "";
        const ms = (j.median_compile_ns / 1e6).toFixed(2);
        jaxBody.textContent = j.cases + " JAX cases; median compile time " + ms + "ms";
      }
    }
  }

  // ---- Table ------------------------------------------------------------
  let sortKey = "operation_name";
  let sortDir = 1; // 1 asc, -1 desc

  const TABLE_COLUMNS = [
    { key: "family",            label: "family",       num: false },
    { key: "backend",           label: "backend",      num: false },
    { key: "check_level",       label: "checks",       num: false },
    { key: "operation_name",    label: "name",         num: false },
    { key: "size",              label: "size",         num: true  },
    { key: "status",            label: "status",       num: false },
    { key: "diagnosis_reason",  label: "diagnosis",    num: false },
    { key: "bare_median_ns",    label: "bare ns",      num: true  },
    { key: "sc_median_ns",      label: "sc ns",        num: true  },
    { key: "overhead_factor",   label: "runtime ratio", num: true },
    { key: "abstraction_overhead_ns", label: "overhead ns", num: true },
    { key: "validation_overhead_ns", label: "validation ns", num: true },
    { key: "speedup",           label: "speedup",      num: true  },
    { key: "speedup_std",       label: "std",          num: true  },
    { key: "compile_ns_median", label: "Compile (ms)", num: true  },
    { key: "error_max",         label: "err max",      num: true  },
    { key: "sc_peak_bytes",     label: "sc peak",      num: true  },
  ];

  function renderTableHead() {
    const thead = document.getElementById("table-head");
    thead.innerHTML = "";
    const tr = document.createElement("tr");
    TABLE_COLUMNS.forEach(function (c) {
      const th = document.createElement("th");
      th.textContent = c.label;
      th.dataset.key = c.key;
      if (c.num) th.classList.add("num");
      if (c.key === sortKey) th.classList.add(sortDir > 0 ? "sort-asc" : "sort-desc");
      th.addEventListener("click", function () {
        if (sortKey === c.key) sortDir = -sortDir;
        else { sortKey = c.key; sortDir = 1; }
        renderTableHead();
        renderTableBody();
      });
      tr.appendChild(th);
    });
    thead.appendChild(tr);
  }

  function fmtNs(v) {
    if (v == null) return "-";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + " ms";
    if (v >= 1e3) return (v / 1e3).toFixed(2) + " us";
    return v.toFixed(0) + " ns";
  }
  function fmtNum(v) { return v == null ? "-" : Number(v).toFixed(2); }
  function fmtBytes(v) {
    if (v == null) return "-";
    if (v >= 1024 * 1024) return (v / (1024 * 1024)).toFixed(2) + " MB";
    if (v >= 1024) return (v / 1024).toFixed(1) + " KB";
    return v + " B";
  }
  function fmtErr(v) { return v == null ? "-" : Number(v).toExponential(2); }
  function fmtCompileMs(v) {
    if (v == null) return "&mdash;";
    return (v / 1e6).toFixed(2);
  }

  function renderTableBody() {
    const rows = filtered().slice().sort(function (a, b) {
      const av = a[sortKey], bv = b[sortKey];
      if (av === bv) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return av < bv ? -sortDir : sortDir;
    });
    const body = document.getElementById("table-body");
    body.innerHTML = "";
    rows.forEach(function (r) {
      const tr = document.createElement("tr");
      const statusColor = STATUS_COLORS[r.status] || "#888";
      const reasonChipHtml = reasonChip(
        r.diagnosis_reason || "NEUTRAL",
        r.diagnosis_summary || ""
      );
      const cells = [
        '<td style="color:' + (FAMILY_COLORS[r.family] || "#333") + ';">' + escapeHtml(r.family) + "</td>",
        '<td><span class="backend-tag" style="background:' + backendColor(r.backend) + ';">' +
          escapeHtml(r.backend) + "</span></td>",
        "<td>" + escapeHtml(r.check_level) + "</td>",
        "<td>" + escapeHtml(r.operation_name) + "</td>",
        '<td class="num">' + r.size + "</td>",
        '<td><span class="badge" style="background:' + statusColor + ';">' + r.status + "</span></td>",
        "<td>" + reasonChipHtml + "</td>",
        '<td class="num">' + fmtNs(r.bare_median_ns) + "</td>",
        '<td class="num">' + fmtNs(r.sc_median_ns) + "</td>",
        '<td class="num">' + fmtNum(r.overhead_factor) + "x</td>",
        '<td class="num">' + fmtNs(r.abstraction_overhead_ns) + "</td>",
        '<td class="num">' + fmtNs(r.validation_overhead_ns) + "</td>",
        '<td class="num">' + fmtNum(r.speedup) + "x</td>",
        '<td class="num">' + fmtNum(r.speedup_std) + "</td>",
        '<td class="num">' + fmtCompileMs(r.compile_ns_median) + "</td>",
        '<td class="num">' + fmtErr(r.error_max) + "</td>",
        '<td class="num">' + fmtBytes(r.sc_peak_bytes) + "</td>",
      ];
      tr.innerHTML = cells.join("");
      body.appendChild(tr);
    });
    document.getElementById("table-count").textContent = rows.length;
  }

  // ---- Orchestration ----------------------------------------------------
  function refresh() {
    const rows = filtered();
    speedupDistribution(rows);
    overheadDecomposition(rows);
    scalingCurves(rows);
    perSeedJitter(rows);
    memoryBars(rows);
    baselineScatter(rows);
    renderTableBody();
  }

  function wireFilters() {
    document.querySelectorAll(".f-backend").forEach(function (el) {
      el.addEventListener("change", function () {
        if (el.checked) state.backends.add(el.value);
        else state.backends.delete(el.value);
        refresh();
      });
    });
    document.getElementById("f-check-level").addEventListener("change", function (e) {
      state.checkLevel = e.target.value;
      refresh();
    });
    document.querySelectorAll(".f-family").forEach(function (el) {
      el.addEventListener("change", function () {
        if (el.checked) state.families.add(el.value);
        else state.families.delete(el.value);
        refresh();
      });
    });
    document.querySelectorAll(".f-status").forEach(function (el) {
      el.addEventListener("change", function () {
        if (el.checked) state.statuses.add(el.value);
        else state.statuses.delete(el.value);
        refresh();
      });
    });
    document.getElementById("f-search").addEventListener("input", function (e) {
      state.search = e.target.value || "";
      refresh();
    });
    document.getElementById("f-min").addEventListener("input", function (e) {
      const v = parseFloat(e.target.value);
      state.minSpeedup = isFinite(v) ? v : null;
      refresh();
    });
    document.getElementById("f-max").addEventListener("input", function (e) {
      const v = parseFloat(e.target.value);
      state.maxSpeedup = isFinite(v) ? v : null;
      refresh();
    });
    document.getElementById("f-reset").addEventListener("click", function () {
      state.backends = new Set(BACKENDS);
      state.families = new Set(FAMILIES);
      state.statuses = new Set(STATUSES);
      state.checkLevel = "all";
      state.search = ""; state.minSpeedup = null; state.maxSpeedup = null;
      document.querySelectorAll(".f-backend, .f-family, .f-status").forEach(function (el) {
        el.checked = true;
      });
      document.getElementById("f-search").value = "";
      document.getElementById("f-check-level").value = "all";
      document.getElementById("f-min").value = "";
      document.getElementById("f-max").value = "";
      refresh();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderTableHead();
    wireFilters();
    renderDiagnosisSection();
    refresh();
  });
})();
"""


# ---------------------------------------------------------------------------
# The HTML skeleton. Placeholders are filled by ``str.format``.


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SpaceCore bench dashboard</title>
<script src="{plotly_cdn}" charset="utf-8"></script>
<style>
{css}
</style>
</head>
<body>
<h1>SpaceCore bench dashboard</h1>
<div class="meta">
  <code>spacecore {sc_version}</code>
  &nbsp;|&nbsp; Python <code>{py_version}</code>
  &nbsp;|&nbsp; <code>{platform}</code>
  &nbsp;|&nbsp; <code>{n_cases} cases</code>
</div>

<div class="cards">
{summary_cards}
</div>

{diagnosis_section}

{filter_controls}

<h2>Plots</h2>
<div class="plot-grid">
  <div class="plot-card">
    <div id="plot-distribution" class="plot"></div>
    <div class="plot-caption">
      Speedup is bare median time divided by SpaceCore median time. Values below
      1.0x are overhead; values above 1.0x are wins.
    </div>
  </div>
  <div class="plot-card">
    <div id="plot-overhead" class="plot"></div>
    <div class="plot-caption">
      Shows the worst cases by runtime ratio and splits extra time into cheap
      validation cost and other abstraction/runtime cost where paired data exists.
    </div>
  </div>
  <div class="plot-card wide">
    <div id="plot-scaling" class="plot"></div>
    <div class="plot-caption">
      Shows whether overhead amortizes with larger inputs. Lines trending toward
      1.0x shrink with size; flat high lines indicate persistent overhead.
    </div>
  </div>
  <div class="plot-card">
    <div id="plot-jitter" class="plot"></div>
    <div class="plot-caption">
      Compares per-seed timing to the aggregate median. Wide vertical spread means
      the measurement is noisy and should be treated cautiously.
    </div>
  </div>
  <div class="plot-card">
    <div id="plot-memory" class="plot"></div>
    <div class="plot-caption">
      Memory overhead is SpaceCore peak Python allocation divided by bare peak
      allocation. The dashed 1.0x line means no extra peak allocation.
    </div>
  </div>
  {plot_baseline_block}
</div>

<h2>Cases (<span id="table-count">0</span>)</h2>
<div class="table-wrap">
  <table>
    <thead id="table-head"></thead>
    <tbody id="table-body"></tbody>
  </table>
</div>

<div class="footer">
  Generated by <code>bench/_dashboard.py</code>. Charts are rendered client-side
  from inlined data; no network requests are made for data — only for Plotly.js.
</div>

<script>
window.BENCH_DATA = JSON.parse({embedded_data});
</script>
<script>
{js}
</script>
</body>
</html>
"""

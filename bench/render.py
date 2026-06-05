from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_json(data: dict[str, Any]) -> str:
    """Render an interactive, self-contained dashboard for an overhead artifact."""
    embedded = html.escape(json.dumps(data, allow_nan=False), quote=False)
    return _HTML_TEMPLATE.replace("__EMBEDDED_JSON__", embedded)


def render_file(json_path: str | Path, html_path: str | Path) -> None:
    data = json.loads(Path(json_path).read_text())
    Path(html_path).write_text(render_json(data))


_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpaceCore Overhead Diagnostics</title>
<style>
:root {
  --bg: #f6f8fb;
  --panel: #ffffff;
  --border: #d8dee9;
  --muted: #5b6776;
  --text: #1f2933;
  --blue: #2563eb;
  --green: #2f9e44;
  --amber: #f08c00;
  --red: #d9480f;
  --purple: #7c3aed;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}
header {
  padding: 18px 22px 14px;
  background: #111827;
  color: white;
}
header h1 { margin: 0 0 6px; font-size: 22px; }
header .sub { color: #cbd5e1; font-size: 13px; }
main {
  display: grid;
  grid-template-columns: minmax(620px, 1fr) 440px;
  gap: 14px;
  padding: 14px;
}
@media (max-width: 1180px) { main { grid-template-columns: 1fr; } }
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 14px;
}
.panel h2 { margin: 0 0 10px; font-size: 16px; }
.panel h3 { margin: 8px 0 6px; font-size: 13px; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
}
.card {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px;
  background: #fbfdff;
}
.card .label { font-size: 12px; color: var(--muted); }
.card .value { font-size: 18px; font-weight: 700; margin-top: 2px; }
.controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px 14px;
}
.facet { min-width: 0; }
.facet-title { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip {
  border: 1px solid var(--border);
  background: #f8fafc;
  color: var(--text);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
  user-select: none;
}
.chip.active { background: #dbeafe; border-color: #93c5fd; color: #1d4ed8; }
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}
input[type="search"], select {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 8px;
  background: white;
  color: var(--text);
}
button {
  border: 1px solid var(--border);
  background: white;
  border-radius: 6px;
  padding: 6px 9px;
  cursor: pointer;
}
button.primary { background: var(--blue); color: white; border-color: var(--blue); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.tab.active { background: #111827; color: white; border-color: #111827; }
.active-filters { color: var(--muted); font-size: 12px; margin-top: 8px; }
.table-wrap { overflow: auto; max-height: 72vh; border: 1px solid var(--border); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; white-space: nowrap; }
th { position: sticky; top: 0; background: #f4f6f8; z-index: 1; cursor: pointer; }
tr:hover td { background: #f8fbff; }
tr.selected td { background: #eef6ff; }
.muted { color: var(--muted); }
.verdict { white-space: normal; min-width: 260px; }
.group-row td { background: #f9fafb; font-weight: 700; }
.child-row td:first-child { padding-left: 26px; }
.pill { border-radius: 999px; padding: 2px 7px; font-weight: 700; }
.ok { color: var(--green); }
.slightly_high { color: var(--amber); }
.anomalous { color: var(--red); }
.chart { width: 100%; min-height: 220px; border: 1px solid var(--border); border-radius: 6px; background: white; margin-bottom: 10px; }
.chart svg { width: 100%; height: 240px; display: block; }
.chart-title { padding: 8px 9px 0; font-size: 13px; font-weight: 700; }
.chart-note { padding: 0 9px 7px; font-size: 11px; color: var(--muted); }
svg .axis { stroke: #9aa5b1; stroke-width: 1; }
svg .grid { stroke: #e5e7eb; stroke-width: 1; }
svg text { fill: #5b6776; font-size: 11px; }
.heatmap { display: grid; gap: 1px; overflow: auto; border: 1px solid var(--border); }
.heat-cell { padding: 8px; min-width: 92px; min-height: 42px; font-size: 12px; }
.heat-head { background: #f4f6f8; font-weight: 700; }
.mini-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }
.mini { border: 1px solid var(--border); border-radius: 6px; padding: 6px; }
.mini svg { width: 100%; height: 120px; }
.load-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
.empty { padding: 18px; color: var(--muted); text-align: center; }
</style>
</head>
<body>
<header>
  <h1>SpaceCore Overhead Diagnostics</h1>
  <div class="sub">Interactive view over <code>overhead.json</code>. Default sort and color use absolute overhead, not ratio.</div>
  <div class="load-row">
    <button id="fetch-json">Reload ./overhead.json</button>
    <label>Load JSON: <input id="file-json" type="file" accept="application/json,.json"></label>
    <span id="load-status" class="sub"></span>
  </div>
</header>
<main>
  <section>
    <div class="panel">
      <h2>Summary</h2>
      <div id="summary" class="summary-grid"></div>
    </div>
    <div class="panel">
      <h2>Filters</h2>
      <div id="facets" class="controls"></div>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Search case_id or label">
        <label>Group by <select id="group-by" multiple size="4"></select></label>
        <label>Aggregate <select id="aggregate">
          <option value="median_ratio">Median ratio</option>
          <option value="max_ratio">Max ratio</option>
          <option value="median_overhead_ns">Median overhead</option>
          <option value="gap_factor">Measured/predicted gap</option>
        </select></label>
        <label>Color by <select id="color-by">
          <option value="overhead_ns">Absolute overhead</option>
          <option value="ratio">Ratio</option>
          <option value="gap_factor">Measured/predicted gap</option>
        </select></label>
        <label>Timing <select id="timing">
          <option value="median">Median</option>
          <option value="best">Best</option>
        </select></label>
        <label>Layer <select id="layer-filter"><option value="">Check overhead</option></select></label>
        <label>Breakdown <select id="breakdown-mode">
          <option value="absolute">Absolute ns</option>
          <option value="share">Share of total</option>
        </select></label>
        <button id="clear">Clear filters</button>
        <button id="csv" class="primary">Export CSV</button>
      </div>
      <div id="active-filters" class="active-filters"></div>
    </div>
    <div class="panel">
      <div class="tabs">
        <button class="tab active" data-view="table">Table</button>
        <button class="tab" data-view="heatmap">Heatmap</button>
        <button class="tab" data-view="backend">Backend comparison</button>
        <button class="tab" data-view="smallmultiples">Size sweeps</button>
        <button class="tab" data-view="flagged">Flagged only</button>
      </div>
      <div id="view"></div>
    </div>
  </section>
  <aside>
    <div class="panel">
      <h2>Plots</h2>
      <div id="ratio-chart" class="chart"></div>
      <div id="decomp-chart" class="chart"></div>
      <div id="scatter-chart" class="chart"></div>
    </div>
  </aside>
</main>
<script id="embedded-json" type="application/json">__EMBEDDED_JSON__</script>
<script>
(() => {
  const FACETS = ["backend", "operator_type", "operation", "geometry", "shape_kind", "size_name", "checks", "gap", "trend"];
  const GROUP_FIELDS = ["backend", "operator_type", "operation", "geometry", "size_name", "checks", "gap", "trend"];
  const GAP_RANK = {ok: 0, slightly_high: 1, anomalous: 2};
  const SIZE_RANK = {tiny: 0, small: 1, large: 2};
  const TREND_COLOR = {"decays-to-1.0": "#2f9e44", "constant-above-1.0": "#f08c00", "grows-with-size": "#d9480f", insufficient: "#7c3aed"};
  const GAP_COLOR = {ok: "#2f9e44", slightly_high: "#f08c00", anomalous: "#d9480f"};
  let artifact = null;
  let rows = [];
  let selectedId = null;
  let searchTimer = null;
  const state = {
    filters: Object.fromEntries(FACETS.map(f => [f, new Set()])),
    search: "",
    groupBy: [],
    aggregate: "median_ratio",
    colorBy: "overhead_ns",
    timing: "median",
    layerFilter: "",
    breakdownMode: "absolute",
    view: "table",
    sortKey: "overhead_ns",
    sortDir: "desc"
  };

  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const fmtNs = ns => {
    ns = Number(ns || 0);
    const sign = ns < 0 ? "-" : "";
    ns = Math.abs(ns);
    if (ns >= 1e6) return `${sign}${(ns / 1e6).toFixed(3)} ms`;
    if (ns >= 1e3) return `${sign}${(ns / 1e3).toFixed(3)} us`;
    return `${sign}${ns.toFixed(0)} ns`;
  };
  const fmtRatio = r => r === null || r === undefined || !Number.isFinite(Number(r)) ? "n/a" : `${Number(r).toFixed(3)}x`;
  const median = vals => {
    const v = vals.map(Number).filter(Number.isFinite).sort((a,b) => a-b);
    if (!v.length) return null;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m-1] + v[m]) / 2;
  };
  const sum = vals => vals.reduce((a,b) => a + Number(b || 0), 0);
  const ratioFor = r => state.timing === "best"
    ? safeDiv(r.sc_best_ns, r.bare_best_ns)
    : r.ratio;
  const overheadFor = r => state.timing === "best"
    ? Number(r.sc_best_ns || 0) - Number(r.bare_best_ns || 0)
    : Number(r.overhead_ns || 0);
  const safeDiv = (a,b) => b ? Number(a || 0) / Number(b) : Infinity;
  const gapFactor = r => safeDiv(Math.max(overheadFor(r), 0), Math.max(Number(r.predicted_overhead_ns || 0), 250));
  const flagged = r => r.gap !== "ok" || (["constant-above-1.0", "grows-with-size"].includes(r.trend) && ratioFor(r) > 1.15);
  const fieldValue = (r, f) => String(r[f] ?? "");

  function rowWithDerived(r) {
    return {
      ...r,
      _ratio: ratioFor(r),
      _overhead: overheadFor(r),
      _gap_factor: gapFactor(r),
      _flagged: flagged(r)
    };
  }

  function initData(data, source) {
    artifact = data;
    rows = (data.results || []).map(rowWithDerived);
    selectedId = rows[0]?.case_id || null;
    $("load-status").textContent = `Loaded ${rows.length} rows from ${source}.`;
    buildControls();
    restoreHash();
    render();
  }

  async function tryFetch() {
    try {
      const response = await fetch("./overhead.json", {cache: "no-store"});
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      initData(await response.json(), "./overhead.json");
    } catch (err) {
      const embedded = JSON.parse($("embedded-json").textContent);
      initData(embedded, "embedded JSON (fetch blocked or unavailable; use file picker to load another)");
    }
  }

  function buildControls() {
    $("facets").innerHTML = FACETS.map(f => {
      const values = [...new Set(rows.map(r => fieldValue(r, f)).filter(Boolean))].sort((a,b) => {
        if (f === "size_name") return (SIZE_RANK[a] ?? 99) - (SIZE_RANK[b] ?? 99);
        if (f === "gap") return (GAP_RANK[a] ?? 99) - (GAP_RANK[b] ?? 99);
        return a.localeCompare(b);
      });
      const chips = values.map(v => `<button class="chip" data-facet="${esc(f)}" data-value="${esc(v)}">${esc(v)}</button>`).join("");
      return `<div class="facet"><div class="facet-title">${esc(f)}</div><div class="chips">${chips}</div></div>`;
    }).join("");
    $("group-by").innerHTML = GROUP_FIELDS.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join("");
    const layers = new Set();
    for (const r of rows) {
      for (const layer of (r.breakdown?.ladder || [])) layers.add(layer.layer);
      for (const key of Object.keys(r.breakdown?.measured_primitives || {})) layers.add(key);
    }
    $("layer-filter").innerHTML = `<option value="">Check overhead</option>` + [...layers].sort().map(x => `<option value="${esc(x)}">${esc(x)}</option>`).join("");
  }

  function filteredRows() {
    const text = state.search.trim().toLowerCase();
    return rows.filter(r => {
      for (const f of FACETS) {
        const selected = state.filters[f];
        if (selected.size && !selected.has(fieldValue(r, f))) return false;
      }
      if (text && !(`${r.case_id} ${r.label || ""}`.toLowerCase().includes(text))) return false;
      return true;
    }).map(rowWithDerived);
  }

  function aggregateRows(items) {
    const ratios = items.map(r => r._ratio);
    const overheads = items.map(r => r._overhead);
    const gaps = items.map(r => r._gap_factor);
    const worst = items.reduce((a,r) => (GAP_RANK[r.gap] ?? 0) > (GAP_RANK[a] ?? 0) ? r.gap : a, "ok");
    return {
      count: items.length,
      median_ratio: median(ratios),
      best_ratio: Math.min(...ratios),
      max_ratio: Math.max(...ratios),
      median_overhead_ns: median(overheads),
      total_overhead_ns: sum(overheads),
      gap_factor: median(gaps),
      median_predicted_ns: median(items.map(r => r.predicted_overhead_ns)),
      worst_gap: worst,
      verdict: items.find(r => r.gap === worst)?.verdict || ""
    };
  }

  function metricValue(rowOrAgg) {
    if (state.aggregate === "median_ratio") return rowOrAgg.median_ratio ?? rowOrAgg._ratio;
    if (state.aggregate === "max_ratio") return rowOrAgg.max_ratio ?? rowOrAgg._ratio;
    if (state.aggregate === "median_overhead_ns") return rowOrAgg.median_overhead_ns ?? rowOrAgg._overhead;
    return rowOrAgg.gap_factor ?? rowOrAgg._gap_factor;
  }

  function colorForValue(value, metric = state.colorBy) {
    value = Number(value || 0);
    if (metric === "ratio") {
      if (value <= 1.05) return "#e6f4ea";
      if (value <= 1.25) return "#fff3bf";
      return "#ffe3e3";
    }
    if (metric === "gap_factor") {
      if (value <= 1.5) return "#e6f4ea";
      if (value <= 2.0) return "#fff3bf";
      return "#ffe3e3";
    }
    const abs = Math.abs(value);
    if (abs <= 2_000) return "#e6f4ea";
    if (abs <= 20_000) return "#fff3bf";
    return "#ffe3e3";
  }

  function sortRows(items) {
    const key = state.sortKey;
    const dir = state.sortDir === "asc" ? 1 : -1;
    return [...items].sort((a,b) => {
      const av = key === "ratio" ? a._ratio : key === "overhead_ns" ? a._overhead : key === "gap_factor" ? a._gap_factor : key === "layer" ? selectedLayerCost(a) : a[key];
      const bv = key === "ratio" ? b._ratio : key === "overhead_ns" ? b._overhead : key === "gap_factor" ? b._gap_factor : key === "layer" ? selectedLayerCost(b) : b[key];
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
    });
  }

  function renderSummary(items) {
    const ratios = items.map(r => r._ratio);
    const jit = items.filter(r => r.backend === "jax-jit").map(r => r._ratio);
    const large = items.filter(r => Number(r.size) > 1_000).map(r => r._ratio);
    const cards = [
      ["Rows", items.length],
      ["Median ratio", fmtRatio(median(ratios))],
      ["Median >10^3", fmtRatio(median(large))],
      ["Median JIT", jit.length ? fmtRatio(median(jit)) : "n/a"],
      ["Flagged", items.filter(flagged).length],
      ["Median overhead", fmtNs(median(items.map(r => r._overhead)) || 0)]
    ];
    $("summary").innerHTML = cards.map(([label, value]) => `<div class="card"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`).join("");
  }

  function selectedLayerCost(r) {
    const breakdown = r.breakdown;
    if (!breakdown || breakdown.error) return null;
    if (state.layerFilter) {
      const layer = (breakdown.ladder || []).find(x => x.layer === state.layerFilter);
      if (layer) return Math.max(Number(layer.delta_ns || 0), 0);
      const primitive = breakdown.measured_primitives?.[state.layerFilter];
      return primitive == null ? null : Number(primitive);
    }
    const check = (breakdown.ladder || []).find(x => x.layer === "checks");
    return check ? Math.max(Number(check.delta_ns || 0), 0) : null;
  }

  function rowHtml(r, cls = "") {
    const colorMetric = state.colorBy === "ratio" ? r._ratio : state.colorBy === "gap_factor" ? r._gap_factor : r._overhead;
    const style = `background:${colorForValue(colorMetric)}`;
    const layerValue = selectedLayerCost(r);
    return `<tr class="${cls} ${selectedId === r.case_id ? "selected" : ""}" data-id="${esc(r.case_id)}">
      <td class="${esc(r.gap)}">${esc(r.gap)}</td>
      <td>${esc(r.label || r.case_id)}<div class="muted">${esc(r.case_id)}</div></td>
      <td>${esc(r.backend)}</td><td>${esc(r.operator_type)}</td><td>${esc(r.operation)}</td><td>${esc(r.geometry)}</td>
      <td>${esc(r.size_name)}<div class="muted">size ${esc(r.size)}</div></td>
      <td style="${style}">${fmtRatio(r._ratio)}<div class="muted">${fmtNs(r._overhead)}</div></td>
      <td>${fmtNs(state.timing === "best" ? r.bare_best_ns : r.bare_median_ns)}</td>
      <td>${fmtNs(state.timing === "best" ? r.sc_best_ns : r.sc_median_ns)}</td>
      <td>${fmtNs(r.predicted_overhead_ns)}<div class="muted">${r._gap_factor.toFixed(2)}x gap</div></td>
      <td>${layerValue == null ? "n/a" : fmtNs(layerValue)}<div class="muted">${esc(state.layerFilter || "checks")}</div></td>
      <td class="verdict">${esc(r.verdict || "")}</td>
    </tr>`;
  }

  function tableHeader() {
    const cols = [
      ["gap", "Gap"], ["label", "Case"], ["backend", "Backend"], ["operator_type", "Operator"], ["operation", "Operation"],
      ["geometry", "Geometry"], ["size", "Size"], ["ratio", "Ratio + overhead"], ["bare_median_ns", "Bare"], ["sc_median_ns", "SpaceCore"],
      ["predicted_overhead_ns", "Predicted"], ["layer", "Layer"], ["verdict", "Verdict"]
    ];
    return `<thead><tr>${cols.map(([k, t]) => `<th data-sort="${k}">${esc(t)}${state.sortKey === k ? (state.sortDir === "asc" ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead>`;
  }

  function renderTable(items) {
    if (!items.length) return `<div class="empty">No rows match the current filters.</div>`;
    if (state.groupBy.length) return renderGrouped(items);
    const body = sortRows(items).map(r => rowHtml(r)).join("");
    return `<div class="table-wrap"><table>${tableHeader()}<tbody>${body}</tbody></table></div>`;
  }

  function groupTree(items, fields, depth = 0) {
    if (!fields.length) {
      return `<div class="table-wrap" style="max-height:none;margin:8px 0 12px;"><table>${tableHeader()}<tbody>${sortRows(items).map(r => rowHtml(r, "child-row")).join("")}</tbody></table></div>`;
    }
    const [field, ...rest] = fields;
    const groups = new Map();
    for (const r of items) {
      const key = fieldValue(r, field) || "(blank)";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    const entries = [...groups.entries()].map(([key, rows]) => [key, rows, aggregateRows(rows)]);
    entries.sort((a,b) => metricValue(b[2]) - metricValue(a[2]));
    return entries.map(([key, rows, agg]) => {
      const style = `background:${colorForValue(metricValue(agg), state.aggregate.includes("ratio") ? "ratio" : state.aggregate === "gap_factor" ? "gap_factor" : "overhead_ns")}`;
      return `<details open style="margin-left:${depth * 12}px"><summary class="group-row" style="${style}; padding:8px; border:1px solid var(--border); cursor:pointer;">
        ${esc(field)} = ${esc(key)} · count ${agg.count} · median ${fmtRatio(agg.median_ratio)} · max ${fmtRatio(agg.max_ratio)} · median overhead ${fmtNs(agg.median_overhead_ns || 0)} · total overhead ${fmtNs(agg.total_overhead_ns || 0)} · worst ${esc(agg.worst_gap)}
      </summary>${groupTree(rows, rest, depth + 1)}</details>`;
    }).join("");
  }

  function renderGrouped(items) {
    return groupTree(items, state.groupBy);
  }

  function renderHeatmap(items) {
    const rowsV = [...new Set(items.map(r => r.operator_type))].sort();
    const colsV = [...new Set(items.map(r => r.operation))].sort();
    const cell = (op, operation) => {
      const group = items.filter(r => r.operator_type === op && r.operation === operation);
      if (!group.length) return `<div class="heat-cell"></div>`;
      const agg = aggregateRows(group);
      const v = metricValue(agg);
      const display = state.aggregate.includes("ratio") ? fmtRatio(v) : state.aggregate === "median_overhead_ns" ? fmtNs(v) : `${v.toFixed(2)}x`;
      return `<div class="heat-cell" style="background:${colorForValue(v, state.aggregate.includes("ratio") ? "ratio" : state.aggregate === "gap_factor" ? "gap_factor" : "overhead_ns")}">${display}<div class="muted">${group.length} rows</div></div>`;
    };
    return `<div class="heatmap" style="grid-template-columns: 160px repeat(${colsV.length}, minmax(92px, 1fr));">
      <div class="heat-cell heat-head">operator</div>${colsV.map(c => `<div class="heat-cell heat-head">${esc(c)}</div>`).join("")}
      ${rowsV.map(op => `<div class="heat-cell heat-head">${esc(op)}</div>${colsV.map(c => cell(op,c)).join("")}`).join("")}
    </div>`;
  }

  function renderBackend(items) {
    const keys = [...new Set(items.map(r => `${r.operator_type}.${r.operation}.${r.geometry}.${r.size_name}`))].sort();
    const backs = [...new Set(items.map(r => r.backend))].sort();
    const body = keys.map(k => {
      const [op, operation, geometry, sizeName] = k.split(".");
      const cells = backs.map(b => {
        const r = items.find(x => `${x.operator_type}.${x.operation}.${x.geometry}.${x.size_name}` === k && x.backend === b);
        return `<td>${r ? `${fmtRatio(r._ratio)}<div class="muted">${fmtNs(r._overhead)}</div>` : "n/a"}</td>`;
      }).join("");
      const eager = items.find(x => `${x.operator_type}.${x.operation}.${x.geometry}.${x.size_name}` === k && x.backend === "numpy-eager");
      const jit = items.find(x => `${x.operator_type}.${x.operation}.${x.geometry}.${x.size_name}` === k && x.backend === "jax-jit");
      return `<tr><td>${esc(op)}</td><td>${esc(operation)}</td><td>${esc(geometry)}</td><td>${esc(sizeName)}</td>${cells}<td>${eager && jit ? `${fmtRatio(eager._ratio)} → ${fmtRatio(jit._ratio)}` : "n/a"}</td></tr>`;
    }).join("");
    return `<div class="table-wrap"><table><thead><tr><th>Operator</th><th>Operation</th><th>Geometry</th><th>Size</th>${backs.map(b => `<th>${esc(b)}</th>`).join("")}<th>eager → jit</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderSmallMultiples(items) {
    const groups = groupBy(items, r => `${r.operator_type}.${r.operation}.${r.geometry}`);
    const cards = [...groups.entries()].map(([key, group]) => `<div class="mini"><strong>${esc(key)}</strong>${ratioSvg(group, 220, 110, [key])}</div>`).join("");
    return `<div class="mini-grid">${cards || '<div class="empty">No size sweeps in current filters.</div>'}</div>`;
  }

  function renderFlagged(items) {
    return renderTable(items.filter(flagged));
  }

  function renderView(items) {
    if (state.view === "heatmap") return renderHeatmap(items);
    if (state.view === "backend") return renderBackend(items);
    if (state.view === "smallmultiples") return renderSmallMultiples(items);
    if (state.view === "flagged") return renderFlagged(items);
    return renderTable(items);
  }

  function groupBy(items, fn) {
    const m = new Map();
    for (const item of items) {
      const k = fn(item);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(item);
    }
    return m;
  }

  function ratioSvg(items, width = 400, height = 220, explicitKeys = null) {
    const groupField = state.groupBy[0] || "operator_type";
    const groups = explicitKeys
      ? new Map(explicitKeys.map(k => [k, items]))
      : groupBy(items, r => fieldValue(r, groupField) || "all");
    const valid = items.filter(r => Number(r.size) > 0 && Number.isFinite(r._ratio));
    if (!valid.length) return `<div class="empty">No ratio data.</div>`;
    const pad = {l: 44, r: 14, t: 14, b: 34};
    const xMin = Math.min(...valid.map(r => r.size));
    const xMax = Math.max(...valid.map(r => r.size));
    const yMin = Math.min(1, ...valid.map(r => r._ratio));
    const yMax = Math.max(1.05, ...valid.map(r => r._ratio));
    const lx = x => {
      const lo = Math.log10(xMin), hi = Math.log10(xMax);
      return hi === lo ? pad.l : pad.l + (Math.log10(x) - lo) * (width - pad.l - pad.r) / (hi - lo);
    };
    const yy = y => pad.t + (yMax - y) * (height - pad.t - pad.b) / (yMax - yMin || 1);
    let content = `<line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${height-pad.b}"></line><line class="axis" x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}"></line><line class="grid" x1="${pad.l}" y1="${yy(1)}" x2="${width-pad.r}" y2="${yy(1)}"></line><text x="4" y="${yy(1)+4}">1.00x</text>`;
    let i = 0;
    for (const [key, group] of groups) {
      const pts = [...group].filter(r => r.size > 0).sort((a,b) => a.size - b.size);
      if (!pts.length) continue;
      const color = TREND_COLOR[pts.find(r => r.trend)?.trend] || ["#2563eb", "#7c3aed", "#0891b2", "#db2777"][i++ % 4];
      const path = pts.map((p, j) => `${j ? "L" : "M"}${lx(p.size).toFixed(1)},${yy(p._ratio).toFixed(1)}`).join(" ");
      content += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"><title>${esc(key)}</title></path>`;
      content += pts.map(p => `<circle data-id="${esc(p.case_id)}" cx="${lx(p.size).toFixed(1)}" cy="${yy(p._ratio).toFixed(1)}" r="${selectedId === p.case_id ? 5 : 3}" fill="${color}"><title>${esc(key)} ${p.size_name}: ${fmtRatio(p._ratio)}, ${fmtNs(p._overhead)}</title></circle>`).join("");
    }
    content += `<text x="${pad.l}" y="${height-10}">size log scale</text><text x="${width-92}" y="14">ratio</text>`;
    return `<svg viewBox="0 0 ${width} ${height}">${content}</svg>`;
  }

  function renderRatioChart(items) {
    $("ratio-chart").innerHTML = `<div class="chart-title">Ratio vs size</div><div class="chart-note">One line per ${esc(state.groupBy[0] || "operator_type")}; log-x. Color follows trend.</div>${ratioSvg(items)}`;
  }

  function renderDecompChart(items) {
    const selected = items.find(r => r.case_id === selectedId);
    if (selected?.breakdown && !selected.breakdown.error) {
      $("decomp-chart").innerHTML = `<div class="chart-title">Measured timing breakdown</div><div class="chart-note">${esc(selected.case_id)} · waterfall + predicted/measured comparison</div>${waterfallHtml(selected)}${threeWayHtml(selected)}`;
      return;
    }
    $("decomp-chart").innerHTML = `<div class="chart-title">Measured timing breakdown</div><div class="empty">Select a profiled row to see the waterfall. Rows without a breakdown still show predicted components in the table.</div>`;
  }

  function waterfallHtml(row) {
    const b = row.breakdown;
    const layers = b.ladder || [];
    const base = layers.find(x => x.layer === b.baseline_layer) || layers[0];
    const full = layers.find(x => x.layer === b.full_layer) || layers[layers.length - 1];
    const deltas = layers.slice(Math.max(1, layers.indexOf(base) + 1), layers.indexOf(full) + 1).filter(x => x.delta_ns != null);
    const check = layers.find(x => x.layer === "checks");
    if (check && !deltas.includes(check)) deltas.push(check);
    const total = Math.max(...deltas.map(x => Math.max(Number(x.delta_ns || 0), 0)), 1);
    const rows = [`<div style="display:grid;grid-template-columns:110px 1fr 80px;gap:6px;margin:5px 8px;align-items:center;"><strong>${esc(base.layer)}</strong><div style="height:16px;background:#dbeafe;border-radius:4px;"></div><span>${fmtNs(base.time_ns)}</span></div>`]
      .concat(deltas.map(layer => {
        const raw = Number(layer.delta_ns || 0);
        const ns = Math.max(raw, 0);
        const label = raw < 0 ? "noise" : layer.delta_status === "below_noise" ? "≈0" : fmtNs(ns);
        const color = layer.layer === "checks" ? "#f08c00" : layer.layer.includes("riesz") ? "#7c3aed" : "#2563eb";
        const value = state.breakdownMode === "share" && row._overhead ? `${Math.max(0, 100 * ns / Math.abs(row._overhead)).toFixed(1)}%` : label;
        return `<div style="display:grid;grid-template-columns:110px 1fr 80px;gap:6px;margin:5px 8px;align-items:center;"><span>${esc(layer.layer)}</span><div style="height:16px;background:#e5e7eb;border-radius:4px;"><div style="height:16px;width:${Math.max(2, ns / total * 100).toFixed(1)}%;background:${color};border-radius:4px;"></div></div><span>${value}</span></div>`;
      }));
    return `<div style="margin-top:6px">${rows.join("")}<div class="chart-note">ladder sum ${fmtNs(b.ladder_sum_ns || 0)}; ladder vs total gap ${fmtNs(b.ladder_vs_total_gap_ns || 0)}</div></div>`;
  }

  function threeWayHtml(row) {
    const predicted = new Map();
    for (const c of row.components || []) if (c.name !== "amortized_per_element") predicted.set(c.name, (predicted.get(c.name) || 0) + Number(c.ns || 0));
    const ladder = new Map();
    for (const layer of row.breakdown?.ladder || []) if (layer.delta_ns != null) ladder.set(layer.layer, Math.max(Number(layer.delta_ns || 0), 0));
    const primitives = new Map(Object.entries(row.breakdown?.measured_primitives || {}).filter(([,v]) => v != null).map(([k,v]) => [k, Number(v)]));
    const names = [...new Set([...predicted.keys(), ...ladder.keys(), ...primitives.keys()])];
    const max = Math.max(1, ...names.flatMap(n => [predicted.get(n) || 0, ladder.get(n) || 0, primitives.get(n) || 0]));
    const line = (name, value, color) => `<div style="height:8px;background:#e5e7eb;border-radius:4px;"><div style="height:8px;width:${Math.max(1, value / max * 100).toFixed(1)}%;background:${color};border-radius:4px;"></div></div>`;
    const rows = names.map(n => `<div style="display:grid;grid-template-columns:120px 1fr 1fr 1fr;gap:5px;margin:5px 8px;align-items:center;"><span>${esc(n)}</span>${line(n, predicted.get(n)||0, "#2f9e44")}${line(n, ladder.get(n)||0, "#2563eb")}${line(n, primitives.get(n)||0, "#7c3aed")}</div>`).join("");
    return `<div class="chart-note">Three-way comparison: predicted green · ladder blue · measured primitives purple</div>${rows}`;
  }

  function renderScatter(items) {
    const width = 400, height = 240, pad = {l: 48, r: 18, t: 16, b: 34};
    const vals = items.map(r => [Math.max(r.predicted_overhead_ns || 0, 1), Math.max(r._overhead, 1)]);
    if (!vals.length) { $("scatter-chart").innerHTML = `<div class="chart-title">Measured vs predicted</div><div class="empty">No rows.</div>`; return; }
    const max = Math.max(10, ...vals.flat());
    const pos = v => pad.l + (Math.log10(v) / Math.log10(max)) * (width - pad.l - pad.r);
    const ypos = v => height - pad.b - (Math.log10(v) / Math.log10(max)) * (height - pad.t - pad.b);
    const pts = items.map(r => `<circle data-id="${esc(r.case_id)}" cx="${pos(Math.max(r.predicted_overhead_ns || 0, 1)).toFixed(1)}" cy="${ypos(Math.max(r._overhead, 1)).toFixed(1)}" r="${selectedId === r.case_id ? 5 : 3}" fill="${GAP_COLOR[r.gap] || "#7c3aed"}"><title>${esc(r.case_id)}: measured ${fmtNs(r._overhead)}, predicted ${fmtNs(r.predicted_overhead_ns)}</title></circle>`).join("");
    const diag = `<line class="grid" x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${pad.t}"></line>`;
    $("scatter-chart").innerHTML = `<div class="chart-title">Measured vs predicted overhead</div><div class="chart-note">Log scale. Diagonal is y=x; points above are surprises.</div><svg viewBox="0 0 ${width} ${height}"><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${height-pad.b}"></line><line class="axis" x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}"></line>${diag}${pts}<text x="${pad.l}" y="${height-10}">predicted</text><text x="${width-75}" y="14">measured</text></svg>`;
  }

  function renderPlots(items) {
    renderRatioChart(items);
    renderDecompChart(items);
    renderScatter(items);
  }

  function updateActiveFilterText() {
    const parts = [];
    for (const f of FACETS) if (state.filters[f].size) parts.push(`${f}: ${[...state.filters[f]].join(", ")}`);
    if (state.search) parts.push(`search: "${state.search}"`);
    if (state.groupBy.length) parts.push(`group: ${state.groupBy.join(" → ")}`);
    $("active-filters").textContent = parts.length ? `Active filters: ${parts.join(" · ")}` : "No active filters. Selecting no chips in a facet means all values.";
  }

  function render() {
    const items = filteredRows();
    renderSummary(items);
    updateActiveFilterText();
    $("view").innerHTML = renderView(items);
    renderPlots(items);
    syncControlState();
    saveHash();
  }

  function syncControlState() {
    document.querySelectorAll(".chip").forEach(chip => chip.classList.toggle("active", state.filters[chip.dataset.facet].has(chip.dataset.value)));
    $("search").value = state.search;
    $("aggregate").value = state.aggregate;
    $("color-by").value = state.colorBy;
    $("timing").value = state.timing;
    $("layer-filter").value = state.layerFilter;
    $("breakdown-mode").value = state.breakdownMode;
    [...$("group-by").options].forEach(opt => opt.selected = state.groupBy.includes(opt.value));
    document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.view === state.view));
  }

  function saveHash() {
    const plain = {...state, filters: Object.fromEntries(FACETS.map(f => [f, [...state.filters[f]]]))};
    history.replaceState(null, "", "#" + encodeURIComponent(JSON.stringify(plain)));
  }
  function restoreHash() {
    if (!location.hash) return;
    try {
      const saved = JSON.parse(decodeURIComponent(location.hash.slice(1)));
      for (const f of FACETS) state.filters[f] = new Set(saved.filters?.[f] || []);
      state.search = saved.search || "";
      state.groupBy = saved.groupBy || [];
      state.aggregate = saved.aggregate || state.aggregate;
      state.colorBy = saved.colorBy || state.colorBy;
      state.timing = saved.timing || state.timing;
      state.view = saved.view || state.view;
      state.sortKey = saved.sortKey || state.sortKey;
      state.sortDir = saved.sortDir || state.sortDir;
    } catch {}
  }

  function exportCsv(items) {
    const cols = ["case_id","label","backend","operator_type","operation","geometry","shape_kind","size_name","size","checks","batch","bare_median_ns","sc_median_ns","overhead_ns","ratio","predicted_overhead_ns","gap","trend","verdict"];
    const csv = [cols.join(",")].concat(items.map(r => cols.map(c => `"${String(r[c] ?? "").replaceAll('"', '""')}"`).join(","))).join("\n");
    const blob = new Blob([csv], {type: "text/csv"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "spacecore-overhead-filtered.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  document.addEventListener("click", ev => {
    const chip = ev.target.closest(".chip");
    if (chip) {
      const set = state.filters[chip.dataset.facet];
      set.has(chip.dataset.value) ? set.delete(chip.dataset.value) : set.add(chip.dataset.value);
      render();
      return;
    }
    const tab = ev.target.closest(".tab");
    if (tab) { state.view = tab.dataset.view; render(); return; }
    const th = ev.target.closest("th[data-sort]");
    if (th) {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      else { state.sortKey = key; state.sortDir = "desc"; }
      render();
      return;
    }
    const row = ev.target.closest("tr[data-id]");
    if (row) { selectedId = row.dataset.id; render(); return; }
    const point = ev.target.closest("circle[data-id]");
    if (point) { selectedId = point.dataset.id; render(); return; }
  });
  $("search").addEventListener("input", ev => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.search = ev.target.value; render(); }, 120);
  });
  $("group-by").addEventListener("change", ev => { state.groupBy = [...ev.target.selectedOptions].map(o => o.value); render(); });
  $("aggregate").addEventListener("change", ev => { state.aggregate = ev.target.value; render(); });
  $("color-by").addEventListener("change", ev => { state.colorBy = ev.target.value; render(); });
  $("timing").addEventListener("change", ev => { state.timing = ev.target.value; rows = rows.map(rowWithDerived); render(); });
  $("layer-filter").addEventListener("change", ev => { state.layerFilter = ev.target.value; render(); });
  $("breakdown-mode").addEventListener("change", ev => { state.breakdownMode = ev.target.value; render(); });
  $("clear").addEventListener("click", () => { for (const f of FACETS) state.filters[f].clear(); state.search = ""; state.groupBy = []; render(); });
  $("csv").addEventListener("click", () => exportCsv(filteredRows()));
  $("fetch-json").addEventListener("click", tryFetch);
  $("file-json").addEventListener("change", ev => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => initData(JSON.parse(reader.result), file.name);
    reader.readAsText(file);
  });

  tryFetch();
})();
</script>
</body>
</html>"""

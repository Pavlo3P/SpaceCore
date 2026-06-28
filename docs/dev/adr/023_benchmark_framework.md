# ADR-023: Benchmark framework and result aggregation

## Status

Proposed.

This ADR specifies the benchmark framework SpaceCore *should* provide: a
top-level `bench` package runnable as `python -m bench` that measures SpaceCore
overhead against bare backend calls, and — the focus of this ADR — a first-class
**result-aggregation facility** that rolls measured cells up along any subset of
dimensions (size, module/family, operation, backend, device, check level) so a
run can be read at any altitude from "one number for the whole suite" down to a
single `(operation, size, backend)` cell.

## Context

Every optimization claim in the project — the kernel-layer speedups
([ADR-016](016_kernel_layers.md)), lazy-algebra fusion
([ADR-021](021_lazy_operator_algebra_and_simplification.md)), the materialized
cache ([ADR-022](022_caching.md)) — must be *measured*, reproducibly, before and
after the change. A raw benchmark run produces hundreds of
`(operation, size, backend, device, check_level, seed)` cells. That granularity
is necessary for measurement but useless for judgment: a reviewer asking "did
`linop` get faster at large sizes on NumPy?" or "what is the suite-wide median
overhead at `cheap`?" cannot answer it from a flat list of cells.

The missing piece is **aggregation**: a defined way to collapse the cell grid
along chosen axes and summarize the rest. Without it, every consumer (the CLI
summary, the dashboard, a CI gate, a release report) re-implements ad-hoc
group-by logic with inconsistent statistics and inconsistent handling of the
speedup ratio. This ADR makes aggregation a single, tested, reusable layer with
one contract, and pins the framework conventions the aggregation layer depends
on (fixed seeds, factory probes, capped check levels).

## Decision

Build `bench` as a top-level package (not under `spacecore`; the library must
never import it) with the layering below. The load-bearing additions over a
naive timer harness are the **probe data model**, the **fixed-seed
reproducibility contract**, and the **aggregation layer**.

### Framework conventions to establish

1. **Factory probes.** A probe is a *factory* `(backend, seed, size) -> ProbeCase`
   (or `(backend, device, seed, size)` when device-aware), not a closed callable,
   so input construction happens outside the timed region. Each `ProbeCase`
   carries the closed zero-arg callables to time: the **pure-array-library
   baseline** `bare`, the SpaceCore call `sc`, and optional `reference` /
   `optimized` / `unchecked`. The `bare` baseline is normative — see below.
2. **Fixed seed quartet.** Every probe runs on `SEEDS = (0, 1, 2, 3)` and nothing
   else — the sole entropy source, so a cell is reproducible and within-cell
   seed spread is the jitter estimate.
3. **Capped check levels.** Benchmark runs use `check_level` `none` and `cheap`
   only; the suite measures the overhead floor, not strict-mode validation.
4. **Configuration regimes.** A run sweeps the optimization toggles the
   performance ADRs introduce so the *same* probe is timed under each — see the
   regime sub-section below. A regime is part of every cell's coordinate.
5. **Tagged result rows.** Each measured cell is a `ProbeResult` carrying its full
   coordinate — `operation_name`, `family`, `size`, `backend`, `device`,
   `check_level`, `regime` — plus per-seed timings and the metrics below. The
   coordinate is what the aggregation layer groups on; it must be complete and
   stable.

### The pure-array-library baseline

The number that matters is **how much the SpaceCore abstraction costs over the
best code a user could have written by hand in the raw array library** — not over
some convenient or naive loop. So every probe's `bare` callable is held to a
contract: it is the **most efficient implementation of the same mathematical
operation expressible in the target array library alone**, written idiomatically
*for that library*, importing nothing from `spacecore`.

- **Per-backend, idiomatic.** "Efficient" is library-specific, so `bare` is
  resolved per backend: the NumPy baseline is a single vectorized `np` expression
  (e.g. `A @ x`, `np.einsum`, a fused BLAS call); the JAX baseline is the
  jit-friendly `jnp` equivalent (and, for `jit_compatible` probes, the
  `jax.jit`-wrapped form so SpaceCore is compared against *compiled* JAX, not
  traced); the Torch baseline is the idiomatic `torch` op on the right device. A
  probe declares one `bare` per backend it supports; the factory returns the one
  matching its `backend` argument.
- **Hand-optimal, not strawman.** `bare` must not allocate needlessly, must reuse
  the same algorithm SpaceCore uses (no comparing an O(n²) hand loop against an
  O(n) library kernel, in either direction), and must exploit the obvious library
  fast path (batched matmul over a Python loop, in-place where idiomatic). A
  deliberately slow baseline would inflate SpaceCore's apparent standing and is a
  correctness bug in the probe.
- **Same result as the reference.** `bare` computes the identical mathematical
  value as `sc` and the `reference`; the runner records `bare`'s error against the
  reference too, so a baseline that "wins" only by computing something cheaper and
  wrong is caught, not rewarded.
- **The fixed floor.** `bare` is **regime-independent** — it knows nothing of
  dispatch or caching — so it is the single stationary reference line every
  SpaceCore regime is measured against. `speedup = bare / sc` and
  `abstraction_overhead_ns = sc - bare` are therefore "overhead over the
  hand-optimal array-library floor," the headline the suite exists to report. The
  goal state is `sc ≈ bare` (the abstraction is free); a regime reaching
  `speedup ≥ 1` means SpaceCore matched or beat hand-written library code.

This makes the comparison adversarial in SpaceCore's disfavor by construction: the
library is always measured against the strongest pure-array-library opponent the
probe author can write, on every backend.

### Configuration regimes

The point of the benchmark suite is to *prove* the optimization ADRs, and those
optimizations are toggles, not always-on behavior:
[ADR-016](016_kernel_layers.md) dispatch ships **off by default**, and the
[ADR-022](022_caching.md) materialized-form cache only builds while dispatch is
on. A number measured in one configuration says nothing about the other. So a run
must time the *same* probe under each **regime** — a named bundle of the
optimization toggles — and tag the cell with it:

- `baseline` — dispatch **off** (today's default path); the cache is inert.
- `dispatch` — `dispatch_mode("on")`, cache **off**.
- `dispatch+cache` — `dispatch_mode("on")` with the materialized-form cache
  **on**.
- `verify` — `dispatch_mode("verify")` (routes *and* checks the routed result
  against the generic), used to confirm correctness, not for the headline speed
  number.

The runner applies a regime by entering the corresponding context
(`set_dispatch_mode` / `dispatch_mode(...)`, plus the cache toggle) around case
construction and execution, exactly as it already brackets `check_level`. The
toggles are **not fully orthogonal**: `cache=on` is meaningful only when dispatch
routes, so the runner enumerates the *valid* combinations above and skips
incoherent ones (cache-on with dispatch-off) rather than emitting empty cells.
The regime set is a small, named, extensible enumeration — a future optimization
(e.g. ADR-021 `fuse()`) adds a regime without touching the probe catalog.

Because every regime runs the identical probe on the identical `(seed, size)`
inputs, the cross-regime comparison is apples-to-apples: the speed and memory
effect of dispatch, or of the cache on top of dispatch, is the *within-run* delta
between two regime cells — the same paired-measurement idea as `cheap` vs `none`,
generalized to the optimization axis.

### Metrics each cell must expose

So aggregates are meaningful, every `ProbeResult` records, per cell:

- `bare_median_ns`, `sc_median_ns`, and the derived `speedup = bare / sc`;
- `abstraction_overhead_ns = sc_median - bare_median`;
- `validation_overhead_ns = sc(cheap) - sc(none)` (paired across check levels);
- `regime_speedup` — the within-run ratio of this cell against the `baseline`
  regime cell at the same `(operation, size, backend, device, check_level)`
  (e.g. `dispatch+cache` vs `baseline`), so "what did the optimization buy" is a
  first-class metric, not something the reader reconstructs;
- `error_max` against the reference (for the correctness verdict) — the `verify`
  regime additionally records the routed-vs-generic agreement;
- `sc_peak_bytes_median` / `bare_peak_bytes_median` (so a *materializing* fast
  path's memory cost is visible against its speed win);
- per-seed records retained so an aggregate can report spread.

### Aggregation — computed and rendered in the generated HTML

Aggregation is a feature of the **generated dashboard HTML**, not a Python
reporting helper. The dashboard `bench/_dashboard.py` produces is a single,
self-contained HTML file that **embeds the raw `ProbeResult` cells as JSON** and
ships a small embedded aggregation engine (vanilla JS, no CDN, no server) that
rolls those cells up *in the browser* and re-renders on demand. The user opens
one file and pivots the data interactively; the roll-ups are not pre-baked into
static tables.

- **Interactive group-by.** The page exposes controls to choose the grouping
  dimensions from a fixed vocabulary — `family` (the module roll-up: `space` /
  `linop` / `functional`), `operation`, `size`, `backend`,
  `device`, `check_level`, `regime`. Selecting `family` answers "per module";
  `size` answers "per size"; `regime` answers "what did dispatch / caching buy";
  `family + size + backend` gives a cube face; selecting nothing collapses to one
  row for the whole run. Changing the selection recomputes and redraws
  client-side with no re-run.
- **Regime is also a *pivot* axis, not just a group key.** The page can hold
  `regime` as columns and put the rest of the coordinate on rows, so each row
  shows `baseline` / `dispatch` / `dispatch+cache` side by side with the
  `regime_speedup` between them — the direct "on vs off" read for ADR-016/022.
- **Each rendered group** shows its key, the `count` of cells folded in, and per
  metric a `median` / `mean` / `std` / `min` / `max` / `p25` / `p75` summary, as
  both a sortable pivot table and the corresponding chart (e.g. speedup-vs-size
  curves grouped by module).
- **Speedup is summarized with the geometric mean** (a ratio, not an additive
  quantity), and the worst-case `min` is always shown alongside it so a bad cell
  cannot hide inside a healthy median.
- **Correctness is a gate, not an average**: a group is flagged
  `CORRECTNESS_FAILURE` and visually marked if *any* member cell exceeds its
  family tolerance, rather than averaging `error_max`.
- **Self-contained and offline.** All aggregation logic lives in the emitted HTML;
  the artifact carries the raw cells, so the same file re-pivots anywhere with no
  Python, no network, and no recompute step. The JSON artifact from `bench/_io.py`
  remains the raw, machine-readable source the HTML is generated from.

The **grouping vocabulary and the summary rules** (geometric mean for speedup,
worst-case `min`, correctness-as-gate) are the single normative definition. The
HTML implements them for interactive viewing; the headless `compare`/`track`
paths below reuse the *same* definition so a CI gate and the dashboard never
disagree on what "per-module speedup" means.

### Comparison mode — tracking performance changes

Aggregation answers "how does *this* run look." Comparison answers "what
*changed* since last time," which is the question that actually gates a merge.
This is a defined layer — `bench/_compare.py` — not ad-hoc subtraction.

```text
compare(current, baseline, *, by, rel_threshold=0.10, noise_band="seed") -> CompareReport
```

- **Join, then diff.** Both runs are aggregated under the same `by`, then the two
  sets of `AggregateRow`s are joined on their group key. For each matched group,
  report the **speedup ratio change** (`geomean_current / geomean_baseline`) and
  the **overhead delta** (`abstraction_overhead_ns` and `validation_overhead_ns`
  current − baseline), as both absolute and percentage change.
- **Per-group classification.** Each group is labelled `IMPROVED`, `UNCHANGED`, or
  `REGRESSED` by the relative change against `rel_threshold`, plus `NEW` /
  `DROPPED` for group keys present in only one of the two runs (so adding or
  removing a probe is visible, not silently absorbed).
- **Noise band.** A change smaller than the run's own seed jitter is *not* a
  regression. `noise_band="seed"` derives the floor from the per-seed spread the
  cells already carry; a change inside the band is `UNCHANGED` regardless of
  `rel_threshold`. This keeps a CI gate from flapping on measurement noise.
- **Environment guard.** Absolute nanoseconds are only comparable on the same
  machine. `compare` checks the two artifacts' `meta` (platform, processor,
  backend library versions) and **refuses** to emit a regression verdict across
  incompatible environments — it downgrades to an advisory diff and says why,
  rather than reporting a fake regression caused by different hardware.
- **Regime is part of the join key.** `compare` matches like regime to like
  (`dispatch+cache` vs `dispatch+cache`), so a change in the dispatch path is never
  confused with a change in the baseline path. The *within-run* "on vs off" effect
  is read from `regime_speedup` and is **not** a regression signal — `compare`
  diffs that effect *across* runs (did the dispatch win shrink since last release),
  which is the regression that matters for an optimization.
- **Direction is explicit.** A `REGRESSION` only ever fires in `compare` against a
  supplied baseline; a single run never self-declares one.

For tracking a metric across *many* runs (release-over-release drift, not just
current-vs-previous), provide a `track` reduction:

```text
track(artifacts, *, by, metric="speedup") -> list[TrendRow]
```

It ingests an ordered list of artifacts — each stamped in `meta` with an external
`label` (commit SHA, version tag, or timestamp passed in at run time, never read
from the wall clock inside a probe) — joins their aggregates on the group key, and
emits one time series per group. That feeds a trend panel in the dashboard and a
"performance changelog" a release can quote ("`linop` apply at n=8192 on NumPy:
1.9x → 2.4x over 0.4.0→0.4.1"). `track` is the same join logic as `compare`
applied N-wise; it does not introduce a second comparison contract.

### Surfaces built on the layer

- **Dashboard (the aggregation surface)** (`bench/_dashboard.py`): the
  self-contained HTML described above — interactive group-by, pivot tables, and
  speedup-vs-size charts grouped by module, all computed client-side from the
  embedded raw cells. This is where a human reads aggregated results.
- **CLI** (`python -m bench`): `run`, `compare`, `plot`, `summary`, `list`. `run`
  emits the JSON artifact and generates the dashboard HTML; `plot` regenerates the
  HTML from a stored artifact. `summary` prints a quick text roll-up for terminal
  use (a convenience over the same shared grouping rules, not the primary surface).
  `compare current baseline [--by ...] [--rel-threshold ...]` joins two runs at the
  group level, prints the per-group change table (improved / unchanged / regressed
  / new / dropped), and exits non-zero if any group regressed — only against an
  explicit baseline on a compatible environment. `track artifact... --by ...
  --metric speedup` prints the release-over-release trend.
- **Persistence** (`bench/_io.py`): JSON artifact with `meta` (Python / platform /
  library versions) plus the raw `ProbeResult` cells. The artifact is the
  machine-readable source of truth; the HTML aggregation is generated from it and
  the headless `compare`/`track` paths read it directly.
- **Macro layer** (`bench/macro/`): a separate runner/registry/schema for
  algorithm-level workloads (CG, Lanczos, PDHG, etc.) whose per-mode rows feed the
  *same* grouping vocabulary and render in the same HTML, keyed by
  `(benchmark, backend, size_label, mode)`.

## Rationale

- **Aggregation in the generated HTML.** The grid is too large to read raw, and a
  static pre-baked table forces the author to guess which roll-up the reader
  wants. Embedding the raw cells plus a client-side group-by engine lets the
  reader pivot by size, module, backend, etc. on demand from a single offline
  file — the natural home for interactive exploration. A fixed grouping vocabulary
  and shared summary rules keep that view, the CI gate, and any release report
  computing the same numbers the same way.
- **Compare against the hand-optimal array-library floor.** The honest question a
  user asks is "what does this abstraction cost me versus writing the raw `np` /
  `jnp` / `torch` myself, well?" Measuring `sc` against a *naive* baseline would
  flatter SpaceCore; measuring against the strongest per-backend, idiomatic
  implementation makes `abstraction_overhead_ns` a number worth trusting and turns
  `speedup ≥ 1` into a real claim ("matched or beat hand-written library code").
  Requiring `bare` to match the reference value keeps a fast-but-wrong baseline
  from gaming the comparison.
- **Regimes as a first-class axis.** The optimizations being validated are
  off-by-default toggles, so a single-configuration benchmark cannot speak to them.
  Timing the same probe under `baseline` / `dispatch` / `dispatch+cache` on
  identical inputs turns "is dispatch worth it" into a measured within-run delta,
  and turns "did caching regress on top of dispatch" into a cross-run diff. Naming
  regimes (rather than free-form kwargs) keeps the set small, enumerable, and
  groupable; enumerating only the coherent combinations avoids meaningless cells.
- **Geometric mean + worst-case for speedup.** A ratio averaged arithmetically is
  biased and lets a 0.1x cell be cancelled by a 10x cell. Geometric mean is the
  correct central tendency for ratios; reporting `min` alongside it keeps a single
  bad cell visible at every altitude.
- **Correctness as a gate, not an average.** A benchmark that is fast but wrong is
  worthless; folding `error_max` by averaging would dilute a real failure. Any
  failing member must fail the group.
- **Factory probes + fixed seeds.** Aggregation is only trustworthy if each cell
  is reproducible and construction cost is excluded; keying probes on
  `(seed, size)` with a fixed quartet is what makes a re-run — and therefore a
  before/after comparison — meaningful.
- **Self-contained HTML, embedded JS.** Computing the roll-ups in the emitted page
  (vanilla JS, no CDN, no server) means one file re-pivots anywhere with no
  recompute and no Python; the raw cells travel with it. The headless
  `compare`/`track` paths reuse the same grouping vocabulary so they never diverge
  from what the dashboard shows.
- **Comparison joins aggregates, not raw cells.** Diffing at the group level is
  what makes a change reviewable ("`linop` regressed at large sizes") and lets the
  same `by` vocabulary drive both summary and regression detection. A noise band
  tied to the run's own seed jitter is what separates a real regression from
  measurement flutter, and the environment guard is what keeps a cross-machine
  comparison from manufacturing a fake one.

## Alternatives considered

- **`pandas` groupby for aggregation.** Rejected: adds a heavy dependency for a
  fixed, small set of group-by shapes, and pins aggregation to a Python session
  rather than the self-contained HTML the reader actually opens.
- **Pre-baked static aggregate tables in the HTML.** Rejected: forces the author to
  choose the roll-up at generation time. Embedding the raw cells and aggregating
  client-side lets the reader pick `by` dimensions interactively from one file.
- **`pytest-benchmark` / `asv`.** Rejected: neither models the *paired*
  bare-vs-SpaceCore comparison nor the module/size roll-up that is the point here.
- **Let each consumer group results itself.** Rejected: this is the status quo the
  ADR removes — divergent statistics (arithmetic vs. geometric speedup),
  inconsistent correctness handling, and duplicated code across summary, dashboard,
  and CI.
- **One unified micro+macro runner and schema.** Rejected: a microprobe is a
  single closed call while a macro has amortized setup and per-mode compile cost;
  one schema would carry mostly-null fields. They diverge at the runner and
  converge on the shared grouping vocabulary in the HTML instead.
- **Arithmetic mean of speedups.** Rejected as statistically wrong for ratios (see
  Rationale).

## Consequences

- Every optimization ADR gets a probe whose effect is readable at the module and
  size altitude, not just per cell; a release report can quote one geometric-mean
  overhead per module — and that overhead is stated honestly against hand-optimal
  raw-array-library code, the floor a user could realistically reach themselves.
- The dispatch (ADR-016) and caching (ADR-022) speed claims become *measured*
  within-run deltas (`baseline` vs `dispatch` vs `dispatch+cache`) on identical
  inputs, and their erosion over releases is a cross-run regression the gate
  catches — closing the "implemented but never benchmarked" gap those ADRs flag.
- Aggregated results are read by opening one HTML file and pivoting by size /
  module / backend interactively — no Python session, no re-run.
- A CI performance gate becomes a thin wrapper: run, `compare` against a committed
  baseline, fail on a per-group regression. (To be wired once the baseline-artifact
  convention is settled.)
- Comparison and `track` give a release a quotable performance changelog and make
  drift across versions visible per module/size, not just at the moment of merge.
- Raw cells in the artifact make stored runs self-describing: the same JSON
  regenerates the interactive HTML and feeds the headless diff without re-running.
- The framework is dependency-heavier than the core (Plotly for dashboards), which
  is acceptable only because `bench` is never on the library import path.

## Contributor invariants

- Add a microprobe as a **factory** `(backend, seed, size) -> ProbeCase`; never
  build inputs inside a timed closure; register it under a **unique** dotted name
  in the correct family.
- The `bare` callable must be the **most efficient pure-array-library**
  implementation of the operation, written idiomatically **per backend**
  (vectorized `np`, jit-wrapped `jnp` for `jit_compatible` probes, idiomatic
  `torch`), importing **nothing from `spacecore`**, and must compute the **same
  value as the `reference`** (the runner checks `bare`'s error too). A naive or
  deliberately slow baseline is a probe bug — it must be the strongest opponent you
  can write.
- A probe's problem must be a deterministic function of `(seed, size)` only — no
  wall-clock, no unseeded RNG, no ambient global state. `(0, 1, 2, 3)` is the only
  entropy source.
- A `ProbeResult` must carry its **complete coordinate** (`family`, `operation`,
  `size`, `backend`, `device`, `check_level`, `regime`); aggregation groups on it,
  so a missing or unstable field silently corrupts every roll-up.
- A run sweeps the named **regimes** by entering the dispatch/cache contexts around
  the *same* probe on the *same* `(seed, size)` inputs; enumerate only the coherent
  combinations (no `cache=on` under dispatch-off). Add a new optimization as a new
  named regime, never as a special-case branch inside a probe.
- `compare`/`track` join on `regime` too — like regime to like; the within-run
  `regime_speedup` is the optimization's effect, not a regression.
- Summarize speedup with the **geometric mean** and always report the worst-case
  `min`; never arithmetic-mean a ratio.
- Fold correctness as a **gate**: an aggregate is a `CORRECTNESS_FAILURE` if any
  member cell fails its family tolerance.
- Aggregation logic lives in the **generated HTML** (embedded vanilla JS, no CDN,
  no server); the emitted dashboard must stay self-contained and offline, with the
  raw cells embedded so it re-pivots without a recompute step.
- The grouping vocabulary and summary rules (geometric-mean speedup, worst-case
  `min`, correctness-as-gate) are **one normative definition** shared by the HTML
  and the headless `compare`/`track`; do not let the two drift.
- Benchmark runs use `check_level` `none` or `cheap` only.
- No `spacecore -> bench` import edge; importing `bench` mutates no global state
  except the explicit, runner-invoked `enable_jax_x64()` / `enable_torch_x64()`
  (every backend runs float64 for a fair comparison; Apple MPS is the float32-only
  exception, handled by the device probe and a widened correctness gate).
- A `REGRESSION` requires an explicit baseline artifact passed to `compare`; never
  hard-code an absolute timing threshold.
- `compare` and `track` join on the **group key** and must refuse a regression
  verdict across incompatible `meta` (different platform / processor / backend
  versions); absolute nanoseconds compare only within one machine.
- A change inside the run's **seed-jitter noise band** is `UNCHANGED`, never a
  regression — do not gate CI on flutter.
- Stamp every run's `meta` with an external `label` (commit / version / timestamp)
  passed in at run time; `track` orders on it. Never read the wall clock inside a
  probe.

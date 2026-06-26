# SpaceCore bench framework

Reproducible, generator-driven benchmarks for the SpaceCore public API. Every
probe runs on the same four seeds (`0`, `1`, `2`, `3`), reports per-seed
timings, peak memory, and correctness against a NumPy reference, and
plugs into a single CLI with an interactive HTML dashboard.

Every operation is emitted as separate `check_level="none"` and
`check_level="cheap"` result rows. Each row compares SpaceCore against the
matching backend-native bare timing. The cheap row also reports validation
overhead relative to its paired none row. JAX-compatible probes additionally
report eager and steady-state `jax.jit` time; compilation latency is recorded
separately and excluded from steady-state timing. The dashboard can display
both modes together or isolate either mode.

## CLI

```
python -m bench run        --json out.json --html dashboard.html [--open]
python -m bench summary    out.json
python -m bench plot       out.json --out dashboard.html [--open]
python -m bench compare    current.json baseline.json --html cmp.html
python -m bench list       [--family space|linop|functional]
```

`run` filters:
- `--family <name>` (repeatable) restricts to one operation family.
- `--regime <name>` (repeatable) selects dispatch/cache regimes for `linop`
  probes (`baseline`, `dispatch`, `dispatch_cache`, `verify`); default is
  `baseline` + `dispatch_cache`.
- `--match <substring>` filters by probe name.
- `--quiet` suppresses per-probe progress.
- `--html <path>` writes a self-contained interactive dashboard.
- `--open` opens the dashboard in the default browser after writing.

A non-zero exit from `compare` indicates a regression vs the baseline.

## Coverage

The micro surface follows the 0.4.1 benchmark-surface spec
(`docs/dev/0.4.1-bench-surface.md`): per-call SpaceCore overhead measured
against a hand-optimal pure-array-library bare, across three families. `linalg`,
the synthetic `kernel`-comparison probes, `check_member`, and the tree-space
probes are intentionally out of scope (the optimized folds are still measured on
the real `linop` operators under the dispatch regimes).

| Family | Probes | What is measured |
|---|---|---|
| `space` | `add`, `scale`, `inner`, `norm`, `zeros`, `flatten`/`unflatten`, `convert`, `stacked.*`, `hermitian.*`, `elementwise_jordan.*` | space arithmetic / geometry / spectra vs idiomatic `numpy` |
| `linop` | `dense.{apply,rapply,vapply,rvapply}`, `diagonal.*`, `sparse.*`, `identity`, `zero`, `scaled`, `composed.*`, `summed`, `block_diagonal`/`stacked`/`sum_to_single` (uniform **and** ragged), `matrix_free.*` | LinOp apply / metric-adjoint / batched / structured paths vs raw `matmul` / `*`, swept across dispatch regimes |
| `functional` | `inner_product.value`/`grad`, `quadratic.value`, `matrix_free.value`, `generated_linear.value` | Functional value + gradient vs analytic NumPy |

Sizes are picked per probe to span small / medium / large regimes. Block
operators are measured both uniform (where the ADR-016 batched fold applies) and
ragged (where it does not).

## Verdict bands

Each case is categorized from the SpaceCore-vs-bare median speedup:

- `WIN` — speedup ≥ 0.95 (matches or beats the reference)
- `NEUTRAL` — 0.50 ≤ speedup < 0.95
- `LOSS` — 0.10 ≤ speedup < 0.50
- `HEAVY_LOSS` — speedup < 0.10
- `CORRECTNESS_FAILURE` — error_max exceeds family tolerance (overrides the band)
- `REGRESSION` — only set during `compare`; current median > 1.20× baseline + 1 µs

The verdict report shows family rollups, the five best and five worst cases by
speedup, and a per-case detail table.

## Interactive dashboard

`--html` (or `python -m bench plot`) renders a single self-contained HTML file
with embedded Plotly.js charts and a filterable / sortable case table:

1. **Speedup distribution** — histogram of bare median ÷ SpaceCore median;
   values below `1.0x` are overhead and values above `1.0x` are wins.
2. **Worst overhead decomposition** — extra SpaceCore time over bare for the
   worst runtime-ratio cases, split into cheap validation cost and remaining
   abstraction/runtime cost when paired `none` / `cheap` data exists.
3. **Overhead persistence by size** — SpaceCore median ÷ bare median on a
   log-log size curve; lines trending toward `1.0x` amortize with size, while
   flat high lines indicate persistent overhead.
4. **Per-seed jitter** — scatter of per-seed median ÷ aggregate median by size.
5. **Median memory overhead per family** — grouped bars of SpaceCore peak ÷
   bare peak.
6. **Current vs baseline** (only when `--html` is used with `compare`) —
   log-log scatter with regressing points highlighted.

The table at the bottom shows every case, sortable by any column, with status
badges colored to match the verdict bands. Filter controls (family checkboxes,
status checkboxes, name substring, min/max speedup) update both the plots and
the table.

## Adding a probe

Implement a factory `(seed, size) -> ProbeCase` in `bench/_operations.py`,
then register a `Probe(name, family, factory, sizes)`:

```python
def _make_my_op(seed: int, size: int) -> ProbeCase:
    ctx = _numpy_ctx()
    space, x, x_np = _dense_vector(ctx, size, seed)
    return ProbeCase(
        bare_label="numpy",
        sc_label="SpaceCore",
        bare=lambda: x_np * 2,
        sc=lambda: space.scale(2.0, x),
        reference=lambda: x_np * 2,
    )

registry.register(Probe(
    name="space.my_op",
    family="space",
    factory=_make_my_op,
    sizes=(256, 4096),
))
```

The smoke test `tests/bench/test_bench_smoke.py::test_probe_factory_builds_a_runnable_case`
will pick up the new probe automatically. The `bare` callable must be the
hand-optimal pure-array-library implementation and must reproduce the
`reference` value (enforced by `tests/bench/test_bare_baseline.py`).

## Module layout

```
bench/
  __main__.py        # CLI: python -m bench {run, compare, plot, summary, list}
  _seeds.py          # SEEDS = (0, 1, 2, 3)
  _regimes.py        # dispatch/cache regimes (baseline/dispatch/dispatch_cache/verify)
  _probes.py         # Probe / ProbeCase / SeedTiming / ProbeResult / ProbeRegistry
  _operations.py     # Probe definitions registered at import time
  _run.py            # Multi-seed runner with per-size warmup/repeat
  _verdict.py        # Status enum, family rollup, render_text, compare_to_baseline
  _dashboard.py      # Self-contained interactive Plotly HTML dashboard
  _io.py             # JSON save/load with run metadata
  harness.py         # time_op, time_op_first_call, measure_peak_memory
```

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
python -m bench list       [--family space|linop|functional|linalg|kernel]
```

`run` filters:
- `--family <name>` (repeatable) restricts to one operation family.
- `--match <substring>` filters by probe name.
- `--quiet` suppresses per-probe progress.
- `--html <path>` writes a self-contained interactive dashboard.
- `--open` opens the dashboard in the default browser after writing.

A non-zero exit from `compare` indicates a regression vs the baseline.

## Coverage

25 probes across five families, each run on every size:

| Family | Probes | What is measured |
|---|---|---|
| `space` | `add`, `scale`, `inner`, `norm`, `check_member`, `zeros` | `DenseCoordinateSpace` arithmetic and validation vs `numpy` |
| `linop` | `dense.apply / rapply / vapply`, `diagonal.apply`, `sparse.apply`, `identity.apply`, `composed.apply`, `summed.apply`, `scaled.apply` | LinOp apply / metric-adjoint / batched paths vs raw `matmul` / `*` |
| `functional` | `inner_product.value / grad`, `quadratic.value` | Functional value + gradient vs analytic NumPy |
| `linalg` | `cg.diagonal`, `power_iteration.diagonal` | Iterative solvers vs closed-form references |
| `kernel` | `composed_chain.k{2,4,8}`, `block_diagonal_dense.b{4,16}` | Optimized kernel paths vs generic SpaceCore path |

Sizes are picked per probe to span small / medium / large regimes.

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
will pick up the new probe automatically.

### Kernel probes specifically

A kernel probe additionally sets `ProbeCase.optimized` to the optimized kernel
callable. The bench compares the generic and optimized variants and surfaces
the optimized speedup separately. Update the `_KERNEL_PROBE_TO_BENCHMARK_ID`
mapping in `bench/_operations.py` so `kernel_benchmark_ids()` reports the
matching `KernelSpec.benchmark_id`; the kernel-policy test refuses to pass
until every registered kernel has a corresponding bench probe.

## Module layout

```
bench/
  __main__.py        # CLI: python -m bench {run, compare, plot, summary, list}
  _seeds.py          # SEEDS = (0, 1, 2, 3)
  _probes.py         # Probe / ProbeCase / SeedTiming / ProbeResult / ProbeRegistry
  _operations.py     # Probe definitions registered at import time
                     # + kernel_benchmark_ids / kernel_probe_cases helpers
  _run.py            # Multi-seed runner with per-size warmup/repeat
  _verdict.py        # Status enum, family rollup, render_text, compare_to_baseline
  _dashboard.py      # Self-contained interactive Plotly HTML dashboard
  _io.py             # JSON save/load with run metadata
  harness.py         # time_op, time_op_first_call, measure_peak_memory
```

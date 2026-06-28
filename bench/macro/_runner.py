"""Macrobenchmark runner.

Walks the macro registry, executes each benchmark across the requested
backend / device / size / seed / mode matrix, and emits one
:class:`MacroResult` per ``(benchmark, backend, device, size_label,
seed, mode)`` cell.

The runner is intentionally separate from :mod:`bench._run` because the
shape of a macrobenchmark differs from a microprobe: a macro performs
non-trivial work, has setup that must be amortized out of the timed
section, and may have a JAX-jitted path whose first-call compile time
must be reported separately. Sharing types would force the micro path
to carry every macro-specific field.

The runner does not own randomness. Each ``factory(...)`` call receives
the seed; the benchmark module decides what to randomize.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from typing import Any, Iterable

from .._devices import devices_for
from ._registry import MacroBenchmark, registry
from ._schema import MODE_CHECK_LEVEL, RUN_MODES, MacroResult, ModeName


_PER_MODE_REPEAT = 3
"""How many times the inner loop is repeated for steady-state timing.

A small number is enough because each macro iteration is much larger
than a microprobe call. The runner reports the median across repeats.
"""


def _backend_available(backend: str) -> bool:
    if backend == "numpy":
        return True
    if backend == "jax":
        try:
            import jax  # noqa: F401

            return True
        except ImportError:
            return False
    if backend == "torch":
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def _sync(value: Any) -> Any:
    """Block on a JAX or Torch async device-side result."""
    if value is None:
        return value
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, (tuple, list)):
        for v in value:
            _sync(v)
        return value
    if isinstance(value, dict):
        for v in value.values():
            _sync(v)
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            torch.cuda.synchronize() if value.is_cuda else None
    except Exception:
        pass
    return value


def _time_call(fn) -> tuple[float, Any]:
    """Return ``(wall_ns, result)`` for one call to ``fn``."""
    was_gc = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        result = fn()
        _sync(result)
        elapsed = time.perf_counter_ns() - start
        return float(elapsed), result
    finally:
        if was_gc:
            gc.enable()


def _measure_peak_memory(fn) -> tuple[int, Any]:
    """Run ``fn`` once and return ``(peak_bytes, result)``."""
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    tracemalloc.start()
    try:
        result = fn()
        _sync(result)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if was_tracing:
        tracemalloc.start()
    return int(peak), result


def _median(values: list[float]) -> float:
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return float(sorted_vals[mid])
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _safe_error_vs(ref_metrics: dict[str, float], cur_metrics: dict[str, float]) -> float | None:
    """L_inf error of cur against ref over keys that are present in both."""
    common = set(ref_metrics) & set(cur_metrics)
    if not common:
        return None
    diffs = []
    import math
    for key in common:
        try:
            ref = float(ref_metrics[key])
            cur = float(cur_metrics[key])
        except (TypeError, ValueError):
            continue
        if math.isnan(ref) or math.isnan(cur):
            continue
        diffs.append(abs(ref - cur))
    if not diffs:
        return None
    return max(diffs)


def _build_result(
    *,
    benchmark: MacroBenchmark,
    backend: str,
    device: str,
    size_label: str,
    size_params: dict[str, Any],
    seed: int,
    mode: ModeName,
    iterations: int,
    setup_time_ns: float,
    run_time_ns: float,
    compile_time_ns: float | None,
    memory_peak_bytes: int | None,
    error_vs_bare: float | None,
    metrics: dict[str, Any],
    extra: dict[str, Any],
    throughput_per_iteration: float | None,
    notes: str,
) -> MacroResult:
    time_per_iteration = run_time_ns / iterations if iterations else None
    throughput = None
    if throughput_per_iteration is not None and run_time_ns:
        # work units / second
        throughput = throughput_per_iteration * iterations * 1e9 / run_time_ns
    return MacroResult(
        benchmark_name=benchmark.name,
        workload=benchmark.workload,
        backend=backend,
        device=device,
        mode=mode,
        check_level=MODE_CHECK_LEVEL[mode],
        size_label=size_label,
        size_params=dict(size_params),
        seed=int(seed),
        iterations=int(iterations),
        setup_time_ns=float(setup_time_ns),
        compile_time_ns=compile_time_ns,
        run_time_ns=float(run_time_ns),
        time_per_iteration_ns=time_per_iteration,
        throughput=throughput,
        memory_peak_bytes=memory_peak_bytes,
        error_vs_bare=error_vs_bare,
        residual=metrics.get("residual"),
        objective=metrics.get("objective"),
        notes=notes,
        extra=dict(extra),
    )


def _run_one_seed(
    benchmark: MacroBenchmark,
    backend: str,
    device: str,
    size_label: str,
    size_params: dict[str, Any],
    seed: int,
    *,
    modes: tuple[ModeName, ...],
    progress: bool,
) -> list[MacroResult]:
    """Run every mode of one benchmark at one ``(backend, size, seed)``."""
    # Setup (timed once per seed, attributed to every emitted row).
    setup_start = time.perf_counter_ns()
    payload = benchmark.factory(
        backend=backend,
        device=device,
        seed=seed,
        size_params=size_params,
    )
    setup_time_ns = float(time.perf_counter_ns() - setup_start)
    if payload.setup is not None:
        s_start = time.perf_counter_ns()
        payload.setup()
        setup_time_ns += float(time.perf_counter_ns() - s_start)

    iterations = payload.iterations
    bare_ref_metrics: dict[str, float] = {}
    rows: list[MacroResult] = []
    for mode in modes:
        if mode not in payload.mode_callables:
            continue
        callable_ = payload.mode_callables[mode]
        compile_ns: float | None = None

        # JAX modes need compile/steady separation. ``spacecore_lowered``
        # on JAX is by convention the JIT-compiled path.
        if backend == "jax" and mode in {"bare", "spacecore_lowered"}:
            try:
                import jax  # noqa: F401

                first_ns, _ = _time_call(callable_)
                compile_ns = first_ns
            except ImportError:
                compile_ns = None

        # Steady-state runs: median of repeats.
        runtimes: list[float] = []
        last_result: Any = None
        for _ in range(_PER_MODE_REPEAT):
            run_ns, result = _time_call(callable_)
            runtimes.append(run_ns)
            last_result = result
        run_ns = _median(runtimes)
        peak_bytes, _ = _measure_peak_memory(callable_)

        # Numerical metrics. The bare mode is the canonical reference.
        cur_metrics: dict[str, float] = {}
        cur_extra: dict[str, Any] = {}
        if payload.reference_metric_extractor is not None:
            cur_metrics = payload.reference_metric_extractor(last_result) or {}
        if payload.extra_metric_extractor is not None:
            cur_extra = payload.extra_metric_extractor(last_result) or {}
        if mode == "bare":
            bare_ref_metrics = dict(cur_metrics)
            error_vs_bare = 0.0
        else:
            error_vs_bare = _safe_error_vs(bare_ref_metrics, cur_metrics)

        rows.append(
            _build_result(
                benchmark=benchmark,
                backend=backend,
                device=device,
                size_label=size_label,
                size_params=size_params,
                seed=seed,
                mode=mode,
                iterations=iterations,
                setup_time_ns=setup_time_ns,
                run_time_ns=run_ns,
                compile_time_ns=compile_ns,
                memory_peak_bytes=peak_bytes,
                error_vs_bare=error_vs_bare,
                metrics=cur_metrics,
                extra=cur_extra,
                throughput_per_iteration=payload.throughput_per_iteration,
                notes=benchmark.notes,
            )
        )
        if progress:
            label = f"{benchmark.name}/{backend}/{device}/{size_label}/{mode}"
            tpi = (run_ns / iterations) if iterations else 0.0
            print(
                f"    {label:60s} run={run_ns/1e6:8.2f}ms  iters={iterations:>5d}  "
                f"per_iter={tpi/1e3:8.2f}us",
                flush=True,
            )
    return rows


def run_benchmarks(
    benchmarks: Iterable[MacroBenchmark] | None = None,
    *,
    backends: tuple[str, ...] | None = None,
    devices: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    sizes: tuple[str, ...] | None = None,
    modes: tuple[ModeName, ...] = RUN_MODES,
    quick: bool = False,
    progress: bool = True,
) -> list[MacroResult]:
    """Run every macrobenchmark and return the flat list of result rows.

    Parameters
    ----------
    benchmarks
        Optional iterable of :class:`MacroBenchmark` to run. ``None``
        runs every registered benchmark.
    backends
        Optional backend filter.
    devices
        Optional device filter.
    seeds
        Seeds to run each benchmark on.
    sizes
        Optional size-label filter.
    modes
        Subset of :data:`bench.macro._schema.RUN_MODES`.
    quick
        Run only the smallest configured size per benchmark, with a
        single seed. Intended for CI / smoke testing.
    """
    from .. import enable_jax_x64

    enable_jax_x64()

    if benchmarks is None:
        benchmarks = registry.all()
    benchmarks = tuple(benchmarks)

    if quick:
        seeds = (seeds[0],)

    rows: list[MacroResult] = []
    for benchmark in benchmarks:
        eligible_backends = benchmark.backends if backends is None else tuple(
            b for b in benchmark.backends if b in backends
        )
        size_filter = sizes
        if quick:
            size_filter = benchmark.quick_size_labels()
        size_items = [
            (label, params) for label, params in benchmark.sizes.items()
            if size_filter is None or label in size_filter
        ]
        for backend in eligible_backends:
            if not _backend_available(backend):
                continue
            avail_devs = devices_for(backend)
            if not avail_devs:
                continue
            eligible_devs = avail_devs if devices is None else tuple(
                d for d in avail_devs if d in devices
            )
            if not eligible_devs:
                continue
            # Use the first eligible device per backend (CPU usually).
            # Benchmark-defined factories that want device-aware
            # behavior should consult device themselves.
            device = eligible_devs[0]
            for size_label, size_params in size_items:
                if progress:
                    print(
                        f"  {benchmark.name} [{backend}/{device}/{size_label}]",
                        flush=True,
                    )
                for seed in seeds:
                    try:
                        rows.extend(
                            _run_one_seed(
                                benchmark,
                                backend,
                                device,
                                size_label,
                                size_params,
                                seed,
                                modes=modes,
                                progress=progress,
                            )
                        )
                    except Exception as err:
                        if progress:
                            print(
                                f"    seed={seed}: {type(err).__name__}: {err}",
                                flush=True,
                            )
    return rows

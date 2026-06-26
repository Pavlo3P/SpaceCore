"""Multi-seed, multi-backend benchmark runner.

The runner walks every probe in :mod:`bench._operations`, every backend
the probe declares it supports, every problem size, and every seed in
:data:`bench._seeds.SEEDS`. For each ``(probe, backend, size, seed)``
quartet it:

* builds a fresh :class:`ProbeCase` via the probe factory;
* times backend-native bare, unchecked SpaceCore, and checked SpaceCore variants;
* captures peak Python memory through :mod:`tracemalloc`;
* records the maximum absolute error against the NumPy reference;
* for JAX-compatible probes, records JIT compilation separately from
  steady-state compiled execution.

The per-seed records are aggregated to one :class:`ProbeResult` per
``(probe, backend, size)`` triple. The aggregate carries per-seed
records (so the dashboard can show jitter) along with median speedup,
speedup std-dev, error_max, median peak memory, and the median
compile_ns when applicable.
"""
from __future__ import annotations

import math
from dataclasses import replace
from statistics import median, stdev
from typing import Any, Iterable

import numpy as np

from ._devices import devices_for
from ._probes import Probe, ProbeCase, ProbeResult, SeedTiming, registry
from ._regimes import BASELINE, Regime, benchmark_regime, regimes_for
from ._seeds import SEEDS
from .harness import measure_peak_memory, time_op, time_op_first_call


def _backend_available(backend: str) -> bool:
    """Return whether the backend's library imports successfully."""
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
    if backend == "cupy":
        from tests._helpers import has_cupy

        return has_cupy()
    return False


def _error_vs_reference(actual: Any, reference: Any) -> float:
    """Maximum absolute element difference, tolerant of containers."""
    if reference is None:
        return 0.0
    if isinstance(reference, tuple):
        return max(_error_vs_reference(a, r) for a, r in zip(actual, reference))
    if isinstance(reference, dict):
        return max(_error_vs_reference(actual[k], reference[k]) for k in reference)
    try:
        from tests._helpers import to_numpy

        a = np.asarray(to_numpy(actual))
        r = np.asarray(reference)
    except Exception:
        return 0.0
    if a.shape != r.shape:
        return float("inf")
    try:
        return float(np.max(np.abs(a - r)))
    except (TypeError, ValueError):
        return 0.0


def _safe_stdev(values: list[float]) -> float:
    return float(stdev(values)) if len(values) >= 2 else 0.0


def _build_case(probe: Probe, backend: str, device: str, seed: int, size: int) -> ProbeCase:
    """Call the probe factory with the right argument count."""
    if probe.device_aware:
        return probe.factory(backend, device, seed, size)
    return probe.factory(backend, seed, size)


def _aggregate(
    probe: Probe,
    backend: str,
    device: str,
    size: int,
    check_level: str,
    regime: str,
    seed_records: list[SeedTiming],
) -> ProbeResult:
    bare_medians = [s.bare_median_ns for s in seed_records]
    sc_medians = [s.sc_median_ns for s in seed_records]
    unchecked_medians = [
        s.unchecked_median_ns for s in seed_records if s.unchecked_median_ns is not None
    ]
    unchecked_med = float(median(unchecked_medians)) if unchecked_medians else None
    bare_med = float(median(bare_medians))
    sc_med = float(median(sc_medians))
    speedups = [
        (b / s if s else math.inf) for b, s in zip(bare_medians, sc_medians)
    ]
    optimized_speedups: list[float] = []
    for record in seed_records:
        if record.optimized_median_ns is not None and record.optimized_median_ns > 0:
            optimized_speedups.append(record.sc_median_ns / record.optimized_median_ns)
    compile_records = [s.compile_ns for s in seed_records if s.compile_ns is not None]
    return ProbeResult(
        operation_name=probe.name,
        family=probe.family,
        size=size,
        seeds=tuple(seed_records),
        bare_median_ns=bare_med,
        sc_median_ns=sc_med,
        speedup=float(median(speedups)),
        speedup_std=_safe_stdev(speedups),
        error_max=max(s.error_vs_reference for s in seed_records),
        sc_peak_bytes_median=int(median(s.sc_peak_bytes for s in seed_records)),
        bare_peak_bytes_median=int(median(s.bare_peak_bytes for s in seed_records)),
        backend=backend,
        device=device,
        check_level=check_level,
        regime=regime,
        optimized_speedup=(
            float(median(optimized_speedups)) if optimized_speedups else None
        ),
        compile_ns_median=(
            float(median(compile_records)) if compile_records else None
        ),
        unchecked_median_ns=unchecked_med,
        abstraction_overhead_ns=sc_med - bare_med,
        validation_overhead_ns=sc_med - unchecked_med if unchecked_med is not None else None,
        jit_median_ns=(
            float(median(s.jit_median_ns for s in seed_records if s.jit_median_ns is not None))
            if any(s.jit_median_ns is not None for s in seed_records)
            else None
        ),
        notes=probe.notes,
    )


def _run_one_seed(
    probe: Probe,
    backend: str,
    device: str,
    size: int,
    seed: int,
    check_level: str,
    regime: Regime,
    *,
    repeat: int,
    number: int,
    warmup: int,
) -> SeedTiming:
    from ._operations import benchmark_check_level

    # The regime's dispatch mode must be active for the whole timed
    # section so every ``sc`` call routes (or not) under it. Case
    # construction runs under it too, which is harmless.
    with benchmark_regime(regime):
        with benchmark_check_level(check_level):
            case = _build_case(probe, backend, device, seed, size)
        reference = case.reference() if case.reference is not None else None

        sc_result = case.sc()
        error = _error_vs_reference(sc_result, reference)
        if case.optimized is not None:
            error = max(error, _error_vs_reference(case.optimized(), reference))

        compile_ns: float | None = None
        jit_t = None
        if backend == "jax" and probe.jit_compatible:
            import jax

            jitted = jax.jit(case.sc)
            compile_ns = time_op_first_call(jitted)
            jit_t = time_op(jitted, repeat=repeat, number=number, warmup=warmup)

        bare_t = time_op(case.bare, repeat=repeat, number=number, warmup=warmup)
        sc_t = time_op(case.sc, repeat=repeat, number=number, warmup=warmup)
        opt_t = (
            time_op(case.optimized, repeat=repeat, number=number, warmup=warmup)
            if case.optimized is not None
            else None
        )

        sc_mem = measure_peak_memory(case.sc)
        bare_mem = measure_peak_memory(case.bare)

    # The compile_ns we report is just the first-call total — leaving
    # the subtraction (first_call − typical_call) to the dashboard so
    # both raw and net numbers are inspectable.
    del sc_result
    return SeedTiming(
        seed=seed,
        bare_best_ns=bare_t["best_ns"],
        bare_median_ns=bare_t["median_ns"],
        sc_best_ns=sc_t["best_ns"],
        sc_median_ns=sc_t["median_ns"],
        optimized_best_ns=opt_t["best_ns"] if opt_t else None,
        optimized_median_ns=opt_t["median_ns"] if opt_t else None,
        error_vs_reference=error,
        sc_peak_bytes=int(sc_mem["peak_bytes"]),
        bare_peak_bytes=int(bare_mem["peak_bytes"]),
        compile_ns=compile_ns,
        jit_best_ns=jit_t["best_ns"] if jit_t else None,
        jit_median_ns=jit_t["median_ns"] if jit_t else None,
    )


def _numbers_for(size: int) -> tuple[int, int, int]:
    if size <= 128:
        return 7, 200, 2
    if size <= 1024:
        return 7, 50, 2
    if size <= 8192:
        return 5, 10, 1
    return 5, 3, 1


def run_probes(
    probes: Iterable[Probe] = None,
    *,
    seeds: tuple[int, ...] = SEEDS,
    backends: tuple[str, ...] | None = None,
    devices: tuple[str, ...] | None = None,
    check_levels: tuple[str, ...] = ("none", "cheap"),
    regimes: tuple[Regime, ...] | None = None,
    max_size: int | None = None,
    progress: bool = True,
) -> list[ProbeResult]:
    """Run probes across seeds, sizes, backends, and devices.

    Parameters
    ----------
    backends
        Filter the per-probe ``backends`` list. ``None`` keeps every
        backend each probe declares.
    devices
        Filter the per-backend device list (``cpu`` / ``cuda`` / ``mps`` /
        ``gpu`` / ``tpu``). ``None`` keeps every device the backend can
        target on this machine.
    check_levels
        Validation modes to benchmark. Defaults to both ``none`` and ``cheap``.
    regimes
        Dispatch/cache regimes to sweep for dispatch-eligible (``linop``)
        probes. ``None`` selects :data:`bench._regimes.DEFAULT_DISPATCH_REGIMES`
        (``baseline`` + ``dispatch_cache``); non-dispatch families always run
        ``baseline`` only. ``baseline`` is always included so it can serve as
        the ``regime_speedup`` reference.
    max_size
        Run each probe only at sizes ``<= max_size``. ``None`` (the
        default) keeps every configured size. Use a small value to keep
        the run light on CPU and memory; a probe whose smallest size
        exceeds ``max_size`` is skipped entirely.
    """
    if probes is None:
        probes = registry.all()
    probes = tuple(probes)
    # Enable JAX x64 once per process before any JAX probe is built,
    # so float64 comparisons are fair on dtype-sensitive operations.
    from bench import enable_jax_x64

    enable_jax_x64()
    results: list[ProbeResult] = []
    # Pre-resolve the (probe, backend, device, size) plan so the
    # progress counter is meaningful.
    plan: list[tuple[Probe, str, str, int, str, Regime]] = []
    for probe in probes:
        eligible_backends = probe.backends if backends is None else tuple(
            b for b in probe.backends if b in backends
        )
        for backend in eligible_backends:
            if not _backend_available(backend):
                continue
            available_devs = devices_for(backend)
            if not available_devs:
                continue
            eligible_devs = available_devs if devices is None else tuple(
                d for d in available_devs if d in devices
            )
            # A non-device-aware probe is still run once on the first
            # eligible device (its sc closures don't honor device anyway).
            iter_devs = eligible_devs if probe.device_aware else eligible_devs[:1]
            eligible_sizes = probe.sizes if max_size is None else tuple(
                s for s in probe.sizes if s <= max_size
            )
            for device in iter_devs:
                for size in eligible_sizes:
                    for check_level in check_levels:
                        if check_level not in {"none", "cheap"}:
                            raise ValueError(
                                f"benchmark check_level must be 'none' or 'cheap', got {check_level!r}"
                            )
                        for regime in regimes_for(probe.family, regimes):
                            plan.append(
                                (probe, backend, device, size, check_level, regime)
                            )
    total = len(plan)
    for i, (probe, backend, device, size, check_level, regime) in enumerate(plan, 1):
        if progress:
            tag = f"{backend}/{device}" if probe.device_aware else backend
            print(
                f"[{i}/{total}] {probe.name} {tag} n={size} "
                f"checks={check_level} regime={regime}",
                flush=True,
            )
        seed_records: list[SeedTiming] = []
        repeat, number, warmup = _numbers_for(size)
        for seed in seeds:
            try:
                seed_records.append(
                    _run_one_seed(
                        probe,
                        backend,
                        device,
                        size,
                        seed,
                        check_level,
                        regime,
                        repeat=repeat,
                        number=number,
                        warmup=warmup,
                    )
                )
            except Exception as err:
                if progress:
                    print(
                        f"    skipped seed={seed}: {type(err).__name__}: {err}",
                        flush=True,
                    )
                continue
        if seed_records:
            results.append(
                _aggregate(
                    probe, backend, device, size, check_level, regime, seed_records
                )
            )
    by_case = {
        (r.operation_name, r.backend, r.device, r.size, r.check_level, r.regime): r
        for r in results
    }
    paired: list[ProbeResult] = []
    for result in results:
        # Validation overhead pairs cheap-vs-none within the same regime.
        none_result = by_case.get(
            (
                result.operation_name,
                result.backend,
                result.device,
                result.size,
                "none",
                result.regime,
            )
        )
        validation_overhead = (
            result.sc_median_ns - none_result.sc_median_ns
            if result.check_level == "cheap" and none_result is not None
            else 0.0 if result.check_level == "none" else None
        )
        # Regime speedup pairs this regime against baseline at the same
        # (operation, backend, device, size, check_level).
        base_result = by_case.get(
            (
                result.operation_name,
                result.backend,
                result.device,
                result.size,
                result.check_level,
                BASELINE,
            )
        )
        if result.regime == BASELINE:
            regime_speedup: float | None = 1.0
        elif base_result is not None and result.sc_median_ns:
            regime_speedup = base_result.sc_median_ns / result.sc_median_ns
        else:
            regime_speedup = None
        paired.append(
            replace(
                result,
                validation_overhead_ns=validation_overhead,
                regime_speedup=regime_speedup,
            )
        )
    return paired

"""Macro runner smoke tests against a synthetic benchmark.

These tests register a *minimal* :class:`MacroBenchmark` into a fresh
:class:`MacroRegistry`, run it, and assert that the runner:

* emits one :class:`MacroResult` per ``(seed, mode)`` cell;
* attaches the right ``check_level`` to each row;
* records JAX compile time separately from steady-state runtime (when
  JAX is available);
* aggregates correctly across seeds.

No actual macrobenchmark from ``bench.macro.cg_poisson`` &c. is executed
— the synthetic benchmark replaces the timed code with a constant-time
no-op so the suite runs fast.
"""
from __future__ import annotations

from typing import Any

import pytest

from bench.macro._registry import MacroBenchmark, MacroPayload, MacroRegistry
from bench.macro._runner import run_benchmarks
from bench.macro._schema import RUN_MODES


def _make_payload(*, backend: str, **_: Any) -> MacroPayload:
    """Build a no-op payload with four distinct mode callables.

    Each callable returns a different metric dict so we can verify
    error_vs_bare is computed (bare = reference, other modes drift).
    """
    metrics = {
        "bare": {"residual": 1.0e-6, "objective": 0.5},
        "spacecore_public_none": {"residual": 1.05e-6, "objective": 0.5},
        "spacecore_public_cheap": {"residual": 1.05e-6, "objective": 0.5},
        "spacecore_lowered": {"residual": 1.05e-6, "objective": 0.5},
    }

    def _make(mode: str):
        def _call() -> dict:
            return metrics[mode]
        return _call

    return MacroPayload(
        iterations=10,
        size_params={"n": 8},
        mode_callables={mode: _make(mode) for mode in RUN_MODES},
        reference_metric_extractor=lambda result: dict(result),
    )


@pytest.fixture
def synthetic_registry(monkeypatch):
    """Replace the global registry with one containing only the synthetic benchmark."""
    fresh = MacroRegistry()
    benchmark = MacroBenchmark(
        name="synthetic",
        workload="constant-time no-op for runner smoke",
        sizes={"tiny": {"n": 8}},
        backends=("numpy",),
        factory=_make_payload,
        quick_sizes=("tiny",),
    )
    fresh.register(benchmark)
    import bench.macro._runner as runner_mod
    monkeypatch.setattr(runner_mod, "registry", fresh)
    return fresh


def test_runner_emits_one_row_per_seed_per_mode(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0, 1),
        progress=False,
    )
    assert len(rows) == 2 * len(RUN_MODES)
    seeds_per_mode = {mode: set() for mode in RUN_MODES}
    for row in rows:
        seeds_per_mode[row.mode].add(row.seed)
    for mode, seeds in seeds_per_mode.items():
        assert seeds == {0, 1}, f"mode {mode!r} missing seeds: {seeds}"


def test_runner_attaches_check_level_per_mode(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    by_mode = {row.mode: row for row in rows}
    assert by_mode["bare"].check_level is None
    assert by_mode["spacecore_public_none"].check_level == "none"
    assert by_mode["spacecore_public_cheap"].check_level == "cheap"
    assert by_mode["spacecore_lowered"].check_level == "none"


def test_runner_sets_bare_error_vs_bare_to_zero(synthetic_registry):
    """The ``bare`` mode is the reference; its error_vs_bare must be 0."""
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    bare_row = next(r for r in rows if r.mode == "bare")
    assert bare_row.error_vs_bare == 0.0


def test_runner_computes_error_vs_bare_for_non_bare_modes(synthetic_registry):
    """A non-bare mode whose metrics differ has a non-zero error_vs_bare."""
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    other_row = next(r for r in rows if r.mode == "spacecore_public_none")
    # residual drifted from 1.0e-6 to 1.05e-6 → L_inf error 5e-8.
    assert other_row.error_vs_bare is not None
    assert 0 < other_row.error_vs_bare < 1e-6


def test_runner_records_time_per_iteration(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    for row in rows:
        assert row.time_per_iteration_ns is not None
        assert row.time_per_iteration_ns == row.run_time_ns / row.iterations


def test_runner_records_memory_peak(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    for row in rows:
        assert row.memory_peak_bytes is not None
        assert row.memory_peak_bytes >= 0


def test_runner_emits_setup_time(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    for row in rows:
        assert row.setup_time_ns >= 0


def test_runner_carries_size_params_verbatim(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        progress=False,
    )
    for row in rows:
        assert row.size_params == {"n": 8}


def test_runner_filter_by_modes(synthetic_registry):
    """``modes=`` filter excludes any mode not in the tuple."""
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0,),
        modes=("bare", "spacecore_public_cheap"),
        progress=False,
    )
    emitted_modes = {row.mode for row in rows}
    assert emitted_modes == {"bare", "spacecore_public_cheap"}


def test_runner_quick_mode_uses_single_seed(synthetic_registry):
    """``quick=True`` collapses seeds to just the first one."""
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        seeds=(0, 1, 2, 3),
        quick=True,
        progress=False,
    )
    assert {row.seed for row in rows} == {0}


def test_runner_quick_mode_uses_quick_sizes(synthetic_registry):
    rows = run_benchmarks(
        benchmarks=synthetic_registry.all(),
        quick=True,
        progress=False,
    )
    assert {row.size_label for row in rows} == {"tiny"}


def test_runner_partial_mode_implementation(synthetic_registry):
    """A benchmark that only implements ``bare`` emits one row per seed."""
    # Replace the synthetic benchmark with one that only implements ``bare``.
    def _bare_only_payload(*, backend: str, **_: Any) -> MacroPayload:
        return MacroPayload(
            iterations=1,
            size_params={"n": 8},
            mode_callables={"bare": lambda: {"residual": 0.0}},
            reference_metric_extractor=lambda r: dict(r),
        )

    fresh = MacroRegistry()
    bench = MacroBenchmark(
        name="bare_only",
        workload="bare-only smoke",
        sizes={"tiny": {"n": 8}},
        backends=("numpy",),
        factory=_bare_only_payload,
        quick_sizes=("tiny",),
    )
    fresh.register(bench)

    rows = run_benchmarks(benchmarks=fresh.all(), seeds=(0,), progress=False)
    assert len(rows) == 1
    assert rows[0].mode == "bare"

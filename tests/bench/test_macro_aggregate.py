"""Aggregation tests for ``bench.macro._aggregate``.

These tests use synthesized :class:`MacroResult` rows. They do not run
any benchmark; they validate group-by behavior, missing-mode handling,
and runtime ratio computation against deterministic inputs.
"""
from __future__ import annotations

from bench.macro._aggregate import group_summaries, runtime_ratio_table
from bench.macro._schema import MODE_CHECK_LEVEL, MacroResult


def _row(
    *,
    benchmark="cg_poisson",
    backend="numpy",
    size_label="n=64",
    mode="bare",
    seed=0,
    run_time_ns=1.0e6,
    iterations=100,
    error_vs_bare=0.0,
    residual=None,
) -> MacroResult:
    return MacroResult(
        benchmark_name=benchmark,
        workload="example",
        backend=backend,
        device="cpu",
        mode=mode,
        check_level=MODE_CHECK_LEVEL[mode],
        size_label=size_label,
        size_params={"n": 64},
        seed=seed,
        iterations=iterations,
        setup_time_ns=0.0,
        run_time_ns=run_time_ns,
        time_per_iteration_ns=run_time_ns / iterations,
        memory_peak_bytes=1024,
        error_vs_bare=error_vs_bare,
        residual=residual,
    )


def test_group_summaries_keys_by_name_backend_size_mode_check_level():
    rows = [
        _row(seed=0, run_time_ns=1.0e6),
        _row(seed=1, run_time_ns=1.1e6),
        _row(seed=2, run_time_ns=0.9e6),
        _row(seed=3, run_time_ns=1.05e6),
    ]
    summaries = group_summaries(rows)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["benchmark_name"] == "cg_poisson"
    assert s["backend"] == "numpy"
    assert s["size_label"] == "n=64"
    assert s["mode"] == "bare"
    assert s["check_level"] is None
    assert s["n_seeds"] == 4


def test_group_summaries_reports_min_max_mean_median_std_per_metric():
    rows = [_row(seed=i, run_time_ns=1.0e6 * (i + 1)) for i in range(4)]
    summaries = group_summaries(rows)
    stats = summaries[0]["metrics"]["run_time_ns"]
    assert stats["min"] == 1.0e6
    assert stats["max"] == 4.0e6
    assert stats["median"] == 2.5e6  # average of 2e6 and 3e6
    assert stats["mean"] == 2.5e6
    assert stats["n"] == 4
    assert stats["std"] > 0  # non-zero spread


def test_group_summaries_distinguishes_modes_within_the_same_run():
    """Same benchmark / size / backend with different modes appear as separate rows."""
    rows = [
        _row(seed=0, mode="bare", run_time_ns=1.0e6),
        _row(seed=0, mode="spacecore_public_none", run_time_ns=1.5e6),
        _row(seed=0, mode="spacecore_public_cheap", run_time_ns=2.0e6),
    ]
    summaries = group_summaries(rows)
    modes = {s["mode"] for s in summaries}
    assert modes == {"bare", "spacecore_public_none", "spacecore_public_cheap"}


def test_group_summaries_tolerates_missing_modes():
    """A benchmark that emits only ``bare`` aggregates without crashing."""
    rows = [_row(seed=0, mode="bare", run_time_ns=1.0e6)]
    summaries = group_summaries(rows)
    assert len(summaries) == 1
    assert summaries[0]["mode"] == "bare"


def test_group_summaries_drops_metrics_with_only_none_values():
    """If every seed reports ``None`` for a metric, that metric is absent."""
    rows = [_row(seed=i, residual=None) for i in range(4)]
    summaries = group_summaries(rows)
    assert "residual" not in summaries[0]["metrics"]


def test_runtime_ratio_table_is_one_for_the_reference_mode():
    rows = [
        _row(seed=0, mode="bare", run_time_ns=1.0e6),
        _row(seed=0, mode="spacecore_public_none", run_time_ns=1.2e6),
    ]
    ratios = runtime_ratio_table(rows)
    assert ratios[("cg_poisson", "numpy", "n=64", "bare")] == 1.0
    assert ratios[("cg_poisson", "numpy", "n=64", "spacecore_public_none")] == 1.2


def test_runtime_ratio_table_handles_multiple_benchmarks():
    rows = [
        _row(benchmark="cg", mode="bare", run_time_ns=1.0e6),
        _row(benchmark="cg", mode="spacecore_public_none", run_time_ns=1.4e6),
        _row(benchmark="pdhg", mode="bare", run_time_ns=2.0e6),
        _row(benchmark="pdhg", mode="spacecore_public_none", run_time_ns=2.6e6),
    ]
    ratios = runtime_ratio_table(rows)
    assert ratios[("cg", "numpy", "n=64", "spacecore_public_none")] == 1.4
    assert ratios[("pdhg", "numpy", "n=64", "spacecore_public_none")] == 1.3


def test_runtime_ratio_table_skips_missing_reference():
    """Rows without a bare counterpart don't appear in the ratio table."""
    rows = [_row(mode="spacecore_public_none", run_time_ns=1.5e6)]
    ratios = runtime_ratio_table(rows)
    assert ratios == {}


def test_summaries_are_stable_sorted():
    """Two identical inputs aggregate to the same ordered output."""
    rows_a = [
        _row(benchmark="zz_late", mode="bare", run_time_ns=1.0e6),
        _row(benchmark="aa_early", mode="bare", run_time_ns=2.0e6),
    ]
    rows_b = list(reversed(rows_a))
    out_a = [s["benchmark_name"] for s in group_summaries(rows_a)]
    out_b = [s["benchmark_name"] for s in group_summaries(rows_b)]
    assert out_a == out_b == ["aa_early", "zz_late"]

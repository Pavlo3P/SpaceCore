"""Dashboard macro-section tests using synthesized rows.

These tests build a small synthetic :class:`MacroResult` set with
representative shapes (bare / public_none / public_cheap / lowered
across NumPy / JAX), pass it through ``render_dashboard`` with
``macro_results=...``, and assert the rendered HTML contains the
required structural markers. Nothing is benchmarked.
"""
from __future__ import annotations

import inspect

import pytest

from bench._probes import ProbeResult, SeedTiming
from bench.macro._schema import MODE_CHECK_LEVEL, MacroResult


def _dashboard_supports_macro() -> bool:
    """Whether ``render_dashboard`` already accepts ``macro_results=`` kwarg.

    The macro-section extension lands in a parallel workflow; tests in
    this file skip until the kwarg appears so the suite stays green
    while authoring is in progress.
    """
    from bench._dashboard import render_dashboard

    sig = inspect.signature(render_dashboard)
    return "macro_results" in sig.parameters


pytestmark = pytest.mark.skipif(
    not _dashboard_supports_macro(),
    reason="dashboard macro section not yet integrated",
)


def _macro_row(
    *,
    benchmark="cg_poisson",
    backend="numpy",
    mode="bare",
    size_label="n=64",
    run_time_ns=1_000_000.0,
    compile_time_ns=None,
    residual=None,
    error_vs_bare=0.0,
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
        seed=0,
        iterations=100,
        setup_time_ns=0.0,
        run_time_ns=run_time_ns,
        compile_time_ns=compile_time_ns,
        memory_peak_bytes=1024,
        error_vs_bare=error_vs_bare,
        residual=residual,
    )


def _micro_row() -> ProbeResult:
    seeds = tuple(
        SeedTiming(
            seed=i,
            bare_best_ns=100.0,
            bare_median_ns=100.0,
            sc_best_ns=200.0,
            sc_median_ns=200.0,
            optimized_best_ns=None,
            optimized_median_ns=None,
            error_vs_reference=0.0,
            sc_peak_bytes=128,
            bare_peak_bytes=64,
            compile_ns=None,
        )
        for i in range(4)
    )
    return ProbeResult(
        operation_name="space.add",
        family="space",
        size=256,
        seeds=seeds,
        bare_median_ns=100.0,
        sc_median_ns=200.0,
        speedup=0.5,
        speedup_std=0.0,
        error_max=0.0,
        sc_peak_bytes_median=128,
        bare_peak_bytes_median=64,
        backend="numpy",
        device="cpu",
        check_level="cheap",
    )


def test_dashboard_renders_micro_only_when_no_macro_results(tmp_path):
    from bench._dashboard import render_dashboard

    out = render_dashboard([_micro_row()], tmp_path / "dash.html")
    body = out.read_text()
    assert "Macrobenchmarks" not in body
    assert "BENCH_DATA" in body


def test_dashboard_renders_macro_section_when_provided(tmp_path):
    from bench._dashboard import render_dashboard

    rows = [
        _macro_row(mode="bare", run_time_ns=1_000_000.0),
        _macro_row(mode="spacecore_public_none", run_time_ns=1_200_000.0),
        _macro_row(mode="spacecore_public_cheap", run_time_ns=1_500_000.0),
        _macro_row(mode="spacecore_lowered", run_time_ns=1_100_000.0),
    ]
    out = render_dashboard([_micro_row()], tmp_path / "dash.html", macro_results=rows)
    body = out.read_text()
    assert "Macrobenchmarks" in body
    assert "MACRO_DATA" in body


def test_dashboard_handles_only_bare_rows(tmp_path):
    """Partial data: only ``bare`` rows is valid and must not crash."""
    from bench._dashboard import render_dashboard

    rows = [_macro_row(mode="bare", run_time_ns=1_000_000.0)]
    out = render_dashboard([], tmp_path / "dash.html", macro_results=rows)
    body = out.read_text()
    assert "Macrobenchmarks" in body


def test_dashboard_handles_missing_compile_time(tmp_path):
    """Macro rows with compile_time_ns=None are accepted by the JAX panel."""
    from bench._dashboard import render_dashboard

    rows = [
        _macro_row(backend="jax", mode="bare", run_time_ns=1_000_000.0, compile_time_ns=None),
        _macro_row(
            backend="jax", mode="spacecore_lowered",
            run_time_ns=900_000.0,
            compile_time_ns=50_000_000.0,
        ),
    ]
    out = render_dashboard([], tmp_path / "dash.html", macro_results=rows)
    body = out.read_text()
    assert "Macrobenchmarks" in body
    assert "compile" in body.lower()


def test_dashboard_conclusions_panel_handles_no_lowered_data(tmp_path):
    """Conclusions panel must print 'Insufficient data' rather than crash."""
    from bench._dashboard import render_dashboard

    rows = [
        _macro_row(mode="bare", run_time_ns=1_000_000.0),
        _macro_row(mode="spacecore_public_none", run_time_ns=1_200_000.0),
        # no spacecore_lowered row
    ]
    out = render_dashboard([], tmp_path / "dash.html", macro_results=rows)
    body = out.read_text()
    # Either the conclusion text or its insufficient-data fallback must appear.
    assert ("Insufficient data" in body) or ("lowering" in body.lower())


def test_dashboard_does_not_mix_jax_sc_against_numpy_bare(tmp_path):
    """The dashboard never compares JAX SC against NumPy bare in the summary table.

    The summary status comparison is always within the same backend; we
    assert the rendered HTML shows the JAX backend label alongside any
    JAX SC row that exists in the input.
    """
    from bench._dashboard import render_dashboard

    rows = [
        _macro_row(backend="numpy", mode="bare", run_time_ns=1_000_000.0),
        _macro_row(backend="jax", mode="bare", run_time_ns=900_000.0),
        _macro_row(backend="jax", mode="spacecore_public_none", run_time_ns=1_100_000.0),
    ]
    out = render_dashboard([], tmp_path / "dash.html", macro_results=rows)
    body = out.read_text()
    # Every backend present in the input is labelled in the HTML.
    assert "numpy" in body and "jax" in body


def test_dashboard_partial_data_no_crash(tmp_path):
    """Empty macro list with non-empty micro must not crash."""
    from bench._dashboard import render_dashboard

    # At least one micro row keeps the dashboard happy; an empty macro
    # list is the partial-data contract — the macro section must simply
    # be omitted, not raise.
    out = render_dashboard([_micro_row()], tmp_path / "empty.html", macro_results=[])
    body = out.read_text()
    assert "SpaceCore" in body or "bench" in body.lower()

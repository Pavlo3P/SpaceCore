"""Smoke tests for the :mod:`bench` framework.

These tests pin the *structural* invariants of the bench framework: the
probe registry is non-empty, every probe builds a working
:class:`ProbeCase`, every case's ``bare`` / ``sc`` / ``optimized``
callables run to completion, and the verdict + plot + I/O layers stay
on the typed contract.

The tests deliberately do *not* time anything — that is the bench
runner's job. Running a microbenchmark inside pytest would be slow and
flaky.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import bench._operations  # noqa: F401  (registers probes)
from bench._io import load, save
from bench._probes import ProbeResult, SeedTiming, registry
from bench._seeds import SEEDS
from bench._verdict import (
    Status,
    categorize,
    compare_to_baseline,
    make_verdict,
    render_text,
)


# ---------------------------------------------------------------------------
# Registry


def test_seeds_are_the_documented_quartet():
    """Canonical seeds remain ``(0, 1, 2, 3)``."""
    assert SEEDS == (0, 1, 2, 3)


def test_registry_is_nonempty():
    assert len(registry.all()) >= 5


def test_registry_names_are_unique():
    names = [p.name for p in registry.all()]
    assert len(names) == len(set(names))


def test_every_probe_declares_at_least_one_size():
    for p in registry.all():
        assert len(p.sizes) >= 1, f"{p.name}: no sizes"
        for s in p.sizes:
            assert s > 0, f"{p.name}: non-positive size {s}"


def test_every_family_is_recognized():
    allowed = {"space", "linop", "functional"}
    for p in registry.all():
        assert p.family in allowed, f"{p.name}: unknown family {p.family!r}"


# ---------------------------------------------------------------------------
# Probe construction


@pytest.mark.parametrize("probe", registry.all(), ids=lambda p: p.name)
def test_probe_factory_builds_a_runnable_numpy_case(probe):
    """Every probe builds and runs at the smallest size on the NumPy backend."""
    if "numpy" not in probe.backends:
        pytest.skip(f"{probe.name}: numpy not in declared backends")
    size = min(probe.sizes)
    for seed in SEEDS:
        case = (
            probe.factory("numpy", "cpu", seed, size)
            if probe.device_aware
            else probe.factory("numpy", seed, size)
        )
        assert callable(case.bare)
        assert callable(case.sc)
        case.bare()
        case.sc()
        if case.optimized is not None:
            case.optimized()
        if case.reference is not None:
            case.reference()


def test_probe_factory_builds_jax_case_when_available():
    """JAX-eligible probes build successfully when JAX is installed."""
    from tests._helpers import has_jax

    if not has_jax():
        pytest.skip("jax not installed")
    jax_probes = [p for p in registry.all() if "jax" in p.backends]
    assert jax_probes, "expected at least one JAX-eligible probe"
    probe = jax_probes[0]
    case = probe.factory("jax", 0, min(probe.sizes))
    case.bare()
    case.sc()


def test_probe_factory_builds_torch_case_when_available():
    """Torch-eligible probes build successfully when Torch is installed."""
    from tests._helpers import has_torch

    if not has_torch():
        pytest.skip("torch not installed")
    torch_probes = [p for p in registry.all() if "torch" in p.backends]
    assert torch_probes, "expected at least one Torch-eligible probe"
    probe = torch_probes[0]
    case = probe.factory("torch", 0, min(probe.sizes))
    case.bare()
    case.sc()


@pytest.mark.parametrize("name", ["space.add", "space.scale", "linop.dense.apply"])
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_bare_inputs_are_backend_native(name, backend):
    """Bare timings must use the same backend family as SpaceCore."""
    from tests._helpers import has_jax, has_torch

    if backend == "jax" and not has_jax():
        pytest.skip("jax not installed")
    if backend == "torch" and not has_torch():
        pytest.skip("torch not installed")

    probe = registry.get(name)
    case = probe.factory(backend, 0, min(probe.sizes))
    assert case.bare_inputs
    if backend == "numpy":
        assert all(isinstance(value, np.ndarray) for value in case.bare_inputs)
    elif backend == "jax":
        import jax

        assert all(isinstance(value, jax.Array) for value in case.bare_inputs)
    else:
        import torch

        assert all(isinstance(value, torch.Tensor) for value in case.bare_inputs)


def test_none_check_level_skips_membership_checks(monkeypatch):
    import spacecore as sc

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
    space = sc.DenseCoordinateSpace((3,), ctx)
    monkeypatch.setattr(
        space,
        "_check_member",
        lambda value: pytest.fail("membership validation must be skipped"),
    )
    x = ctx.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(space.add(x, x), [2.0, 4.0, 6.0])


def test_checked_space_methods_match_unchecked_cores():
    import spacecore as sc

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
    space = sc.DenseCoordinateSpace((3,), ctx)
    x = ctx.asarray([1.0, 2.0, 3.0])
    y = ctx.asarray([4.0, 5.0, 6.0])
    np.testing.assert_allclose(space.add(x, y), space._add_core(x, y))
    np.testing.assert_allclose(space.scale(2.0, x), space._scale_core(2.0, x))
    np.testing.assert_allclose(space.inner(x, y), space._inner_core(x, y))


def test_checked_dense_linop_methods_match_unchecked_cores():
    import spacecore as sc

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
    space = sc.DenseCoordinateSpace((2,), ctx)
    matrix = ctx.asarray([[2.0, 1.0], [0.0, 3.0]])
    op = sc.DenseLinOp(matrix, space, space, ctx)
    x = ctx.asarray([1.0, 2.0])
    xs = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(op.apply(x), op._apply_core(x))
    np.testing.assert_allclose(op.rapply(x), op._rapply_core(x))
    np.testing.assert_allclose(op.vapply(xs), op._vapply_core(xs))


def test_error_checker_extracts_current_solver_result_payloads():
    """Solver result NamedTuples must compare their payload, not tuple fields."""
    import spacecore as sc
    from bench._run import _error_vs_reference

    assert _error_vs_reference(
        sc.PowerIterationResult(5.0, np.zeros(2), True, 3, 1e-9),
        5.0,
    ) == 0.0
    assert _error_vs_reference(
        sc.CGResult(np.asarray([1.0, 2.0]), True, 3, 1e-9),
        np.asarray([1.0, 2.0]),
    ) == 0.0


# ---------------------------------------------------------------------------
# Verdict / IO


def _mock_result(
    name: str,
    family: str,
    speedup: float,
    *,
    err: float = 0.0,
    check_level: str = "cheap",
) -> ProbeResult:
    # Use bare=100 µs so the regression threshold (20% + 1 µs floor) is meaningful.
    bare = 100_000.0
    sc = bare / speedup if speedup else 1e9
    seeds = tuple(
        SeedTiming(
            seed=s,
            bare_best_ns=bare,
            bare_median_ns=bare,
            sc_best_ns=sc,
            sc_median_ns=sc,
            optimized_best_ns=None,
            optimized_median_ns=None,
            error_vs_reference=err,
            sc_peak_bytes=1024,
            bare_peak_bytes=512,
        )
        for s in SEEDS
    )
    return ProbeResult(
        operation_name=name,
        family=family,
        size=256,
        seeds=seeds,
        bare_median_ns=bare,
        sc_median_ns=sc,
        speedup=speedup,
        speedup_std=0.0,
        error_max=err,
        sc_peak_bytes_median=1024,
        bare_peak_bytes_median=512,
        check_level=check_level,
        abstraction_overhead_ns=sc - bare,
    )


def test_categorize_buckets_speedups_into_known_statuses():
    assert categorize(_mock_result("x", "linop", 1.0)) == Status.WIN
    assert categorize(_mock_result("x", "linop", 0.95)) == Status.WIN
    assert categorize(_mock_result("x", "linop", 0.50)) == Status.NEUTRAL
    assert categorize(_mock_result("x", "linop", 0.25)) == Status.LOSS
    assert categorize(_mock_result("x", "linop", 0.05)) == Status.HEAVY_LOSS


def test_categorize_correctness_failure_dominates_speedup():
    """A correctness mismatch overrides the speedup band."""
    r = _mock_result("x", "linop", 5.0, err=1.0)
    assert categorize(r) == Status.CORRECTNESS_FAILURE


def test_make_verdict_reports_families_and_top_lists():
    results = [
        _mock_result("a", "linop", 5.0),
        _mock_result("b", "linop", 0.5),
        _mock_result("c", "space", 1.0),
    ]
    verdict = make_verdict(results)
    family_names = {f.family for f in verdict.families}
    assert family_names == {"linop", "space"}
    assert verdict.overall_max_speedup == 5.0
    assert verdict.overall_min_speedup == 0.5
    assert len(verdict.top_wins) <= 5
    assert verdict.top_wins[0][0] == "a"


def test_compare_to_baseline_flags_regression():
    current = [_mock_result("op", "linop", 0.10)]  # 10× slower than ref
    baseline = [_mock_result("op", "linop", 1.0)]  # 1× of ref
    verdict, lines = compare_to_baseline(current, baseline)
    assert lines, "expected a regression line"
    assert verdict.statuses["op@256/numpy/cpu/cheap"] == Status.REGRESSION


def test_compare_to_baseline_clean_when_within_threshold():
    """No regression when current and baseline are identical."""
    current = [_mock_result("op", "linop", 0.5)]
    baseline = [_mock_result("op", "linop", 0.5)]
    _, lines = compare_to_baseline(current, baseline)
    assert lines == []


def test_render_text_emits_section_headers(tmp_path):
    results = [_mock_result("op", "linop", 0.5)]
    verdict = make_verdict(results)
    out = render_text(results, verdict)
    assert "SpaceCore benchmark summary" in out
    assert "Family rollup" in out
    assert "Top 5 wins" in out
    assert "Top 5 losses" in out
    assert "Per-case detail" in out


def test_save_and_load_round_trip(tmp_path):
    results = [
        _mock_result("a", "space", 1.0),
        _mock_result("b", "linop", 0.5, check_level="none"),
    ]
    path = tmp_path / "bench.json"
    save(results, path)
    raw = json.loads(path.read_text())
    assert "meta" in raw and "results" in raw
    assert "python" in raw["meta"]
    loaded = load(path)
    assert len(loaded) == 2
    assert loaded[0].operation_name == "a"
    assert loaded[1].family == "linop"
    assert loaded[1].check_level == "none"
    assert loaded[1].seeds[0].seed == 0


# ---------------------------------------------------------------------------
# Plotting


def test_dashboard_writes_a_self_contained_html_file(tmp_path):
    """The Plotly dashboard ships every chart + the data inline."""
    from bench._dashboard import render_dashboard

    # Cover all four performance statuses so the (count>0) status chips and
    # cards actually render: WIN(1.0), NEUTRAL(0.8/0.5), LOSS(0.3), HEAVY_LOSS(0.05).
    results = [
        _mock_result("a", "space", 1.0),
        _mock_result("b", "linop", 0.5, check_level="none"),
        _mock_result("c", "functional", 0.8),
        _mock_result("d", "linop", 0.3),
        _mock_result("e", "space", 0.05),
    ]
    out = render_dashboard(results, tmp_path / "dashboard.html")
    assert out.exists()
    body = out.read_text()
    # The dashboard must embed Plotly (from CDN) and the data + filters.
    assert "plotly" in body.lower()
    assert "BENCH_DATA" in body
    assert "<table" in body
    # Status filter chips render for the statuses present in the data.
    for status in ("WIN", "NEUTRAL", "LOSS", "HEAVY_LOSS"):
        assert f'class="f-status" value="{status}"' in body
    # A status with zero cases (no CORRECTNESS_FAILURE here) gets no chip.
    assert 'class="f-status" value="CORRECTNESS_FAILURE"' not in body
    # Family filter checkboxes must be present.
    for family in ("space", "linop", "functional"):
        assert family in body
    assert 'id="f-check-level"' in body
    assert 'value="all"' in body
    assert 'value="none"' in body
    assert 'value="cheap"' in body
    assert "check_level" in body
    assert "seriesLabel" in body
    assert 'checkLevel: "all"' in body


def test_dashboard_with_baseline_includes_regression_panel(tmp_path):
    """When a baseline is passed, a current-vs-baseline panel renders."""
    from bench._dashboard import render_dashboard

    current = [_mock_result("op", "linop", 0.1)]
    baseline = [_mock_result("op", "linop", 1.0)]
    out = render_dashboard(current, tmp_path / "compare.html", baseline=baseline)
    body = out.read_text()
    # The compare chart references baseline somewhere in the HTML.
    assert "baseline" in body.lower() or "regression" in body.lower()


# ---------------------------------------------------------------------------
# Multi-seed runner


def test_run_probes_smoke_for_one_probe(monkeypatch):
    """Run a single probe at the smallest size on all 4 seeds end to end."""
    from dataclasses import replace
    from bench._run import run_probes

    # Pick one fast probe, restrict to numpy + smallest size.
    probe = registry.get("space.add")
    probe = replace(probe, sizes=(probe.sizes[0],), backends=("numpy",))

    import bench._run as run_module
    monkeypatch.setattr(run_module, "_numbers_for", lambda size: (2, 5, 1))

    results = run_probes((probe,), backends=("numpy",), progress=False)
    assert len(results) == 2
    assert {r.check_level for r in results} == {"none", "cheap"}
    for r in results:
        assert r.operation_name == "space.add"
        assert r.backend == "numpy"
        assert r.compile_ns_median is None
        assert r.abstraction_overhead_ns is not None
        assert r.validation_overhead_ns is not None
        assert len(r.seeds) == 4
        assert {s.seed for s in r.seeds} == set(SEEDS)
        assert r.error_max == 0.0


def test_run_probes_can_filter_to_one_check_level(monkeypatch):
    from dataclasses import replace
    from bench._run import run_probes

    probe = replace(registry.get("space.add"), sizes=(256,), backends=("numpy",))
    import bench._run as run_module

    monkeypatch.setattr(run_module, "_numbers_for", lambda size: (1, 2, 0))
    results = run_probes(
        (probe,), seeds=(0,), backends=("numpy",), check_levels=("none",), progress=False
    )
    assert len(results) == 1
    assert results[0].check_level == "none"


def test_run_probes_max_size_keeps_only_small_sizes(monkeypatch):
    """``max_size`` drops every configured size above the threshold."""
    from dataclasses import replace
    from bench._run import run_probes

    probe = replace(
        registry.get("space.add"), sizes=(256, 4096, 65536), backends=("numpy",)
    )
    import bench._run as run_module

    monkeypatch.setattr(run_module, "_numbers_for", lambda size: (1, 2, 0))
    results = run_probes(
        (probe,), seeds=(0,), backends=("numpy",), max_size=256, progress=False
    )
    # Both check levels at the single surviving size (256), nothing larger.
    assert {r.size for r in results} == {256}


def test_run_probes_max_size_can_eliminate_every_case(monkeypatch):
    """A threshold below the smallest size yields no results (no crash)."""
    from dataclasses import replace
    from bench._run import run_probes

    probe = replace(registry.get("space.add"), sizes=(256, 4096), backends=("numpy",))
    results = run_probes(
        (probe,), seeds=(0,), backends=("numpy",), max_size=4, progress=False
    )
    assert results == []


def test_run_probes_builds_cases_in_each_declared_check_level(monkeypatch):
    from bench._operations import _backend_ctx
    from bench._probes import Probe, ProbeCase
    from bench._run import run_probes

    built_levels = []

    def factory(backend, seed, size):
        ctx = _backend_ctx(backend)
        built_levels.append(ctx.check_level)
        x = ctx.asarray([1.0])
        return ProbeCase(
            bare_label="x + x",
            sc_label="x + x",
            bare=lambda: x + x,
            sc=lambda: x + x,
            reference=lambda: np.asarray([2.0]),
        )

    probe = Probe("test.check_matrix", "space", factory, sizes=(1,))
    import bench._run as run_module

    monkeypatch.setattr(run_module, "_numbers_for", lambda size: (1, 1, 0))
    results = run_probes((probe,), seeds=(0,), backends=("numpy",), progress=False)
    assert built_levels == ["none", "cheap"]
    assert [r.check_level for r in results] == ["none", "cheap"]

"""MacroResult schema validity tests.

These tests pin the public JSON contract documented in
``docs/benchmarks.md``. Every field listed in
:data:`bench.macro._schema.REQUIRED_FIELDS` must round-trip through
``MacroResult.to_dict`` and ``MacroResult.from_dict`` without loss.
"""
from __future__ import annotations

import pytest

from bench.macro._schema import (
    MODE_CHECK_LEVEL,
    REQUIRED_FIELDS,
    RUN_MODES,
    MacroResult,
    validate,
)


def _example(mode: str = "bare") -> MacroResult:
    return MacroResult(
        benchmark_name="example",
        workload="example workload",
        backend="numpy",
        device="cpu",
        mode=mode,
        check_level=MODE_CHECK_LEVEL[mode],
        size_label="n=64",
        size_params={"n": 64, "maxiter": 100},
        seed=0,
        iterations=100,
        setup_time_ns=1234.0,
        run_time_ns=2_345_678.0,
        compile_time_ns=None,
        time_per_iteration_ns=23456.78,
        throughput=42.0,
        memory_peak_bytes=4096,
        error_vs_bare=0.0,
        residual=1.2e-6,
        objective=None,
        notes="hand-built example",
        extra={"matvecs_per_second": 12345.6},
    )


def test_run_modes_match_documented_set():
    assert RUN_MODES == (
        "bare",
        "spacecore_public_none",
        "spacecore_public_cheap",
        "spacecore_lowered",
    )


def test_each_mode_has_a_check_level_mapping():
    assert MODE_CHECK_LEVEL["bare"] is None
    assert MODE_CHECK_LEVEL["spacecore_public_none"] == "none"
    assert MODE_CHECK_LEVEL["spacecore_public_cheap"] == "cheap"
    assert MODE_CHECK_LEVEL["spacecore_lowered"] == "none"


@pytest.mark.parametrize("mode", RUN_MODES)
def test_to_dict_emits_every_required_field(mode):
    payload = _example(mode).to_dict()
    for required in REQUIRED_FIELDS:
        assert required in payload, f"missing field {required!r} for mode {mode!r}"


def test_family_is_macro():
    payload = _example().to_dict()
    assert payload["family"] == "macro"


def test_validate_accepts_a_well_formed_row():
    validate(_example().to_dict())


def test_validate_rejects_missing_field():
    bad = _example().to_dict()
    del bad["run_time_ns"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate(bad)


def test_validate_rejects_non_macro_family():
    bad = _example().to_dict()
    bad["family"] = "micro"
    with pytest.raises(ValueError, match="family must be 'macro'"):
        validate(bad)


def test_validate_rejects_unknown_mode():
    bad = _example().to_dict()
    bad["mode"] = "fancy_new_mode"
    with pytest.raises(ValueError, match="mode must be one of"):
        validate(bad)


@pytest.mark.parametrize("mode", RUN_MODES)
def test_round_trip_preserves_every_field(mode):
    original = _example(mode)
    payload = original.to_dict()
    restored = MacroResult.from_dict(payload)
    assert restored.benchmark_name == original.benchmark_name
    assert restored.workload == original.workload
    assert restored.backend == original.backend
    assert restored.device == original.device
    assert restored.mode == original.mode
    assert restored.check_level == original.check_level
    assert restored.size_label == original.size_label
    assert restored.size_params == original.size_params
    assert restored.seed == original.seed
    assert restored.iterations == original.iterations
    assert restored.setup_time_ns == original.setup_time_ns
    assert restored.run_time_ns == original.run_time_ns
    assert restored.compile_time_ns == original.compile_time_ns
    assert restored.time_per_iteration_ns == original.time_per_iteration_ns
    assert restored.throughput == original.throughput
    assert restored.memory_peak_bytes == original.memory_peak_bytes
    assert restored.error_vs_bare == original.error_vs_bare
    assert restored.residual == original.residual
    assert restored.objective == original.objective
    assert restored.notes == original.notes
    assert restored.extra == original.extra


def test_check_level_is_null_for_bare_mode():
    bare = _example("bare")
    assert bare.check_level is None


def test_check_level_is_none_for_spacecore_public_none():
    row = _example("spacecore_public_none")
    assert row.check_level == "none"


def test_check_level_is_cheap_for_spacecore_public_cheap():
    row = _example("spacecore_public_cheap")
    assert row.check_level == "cheap"


def test_compile_time_can_be_recorded_separately():
    """JAX-style rows can carry compile_time_ns without affecting run_time_ns."""
    row = MacroResult(
        benchmark_name="x",
        workload="x",
        backend="jax",
        device="cpu",
        mode="spacecore_lowered",
        check_level="none",
        size_label="n=64",
        size_params={"n": 64},
        seed=0,
        iterations=100,
        setup_time_ns=0.0,
        run_time_ns=1000.0,
        compile_time_ns=50_000_000.0,
    )
    assert row.compile_time_ns == 50_000_000.0
    assert row.run_time_ns == 1000.0
    validate(row.to_dict())

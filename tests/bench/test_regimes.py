"""Tests for the configuration-regime axis (ADR-023).

These pin the regime *contract*: the regime vocabulary, which families
sweep which regimes, that :func:`benchmark_regime` actually activates the
SpaceCore dispatch mode, and that a real (tiny) run threads the regime
into every result row and pairs ``regime_speedup`` against baseline.
"""
from __future__ import annotations

import pytest

import bench._operations  # noqa: F401  (registers probes)
from bench._probes import registry
from bench._regimes import (
    BASELINE,
    DEFAULT_DISPATCH_REGIMES,
    REGIMES,
    benchmark_regime,
    dispatch_eligible,
    regimes_for,
)
from bench._run import run_probes


# ---------------------------------------------------------------------------
# Vocabulary


def test_regime_vocabulary():
    assert REGIMES == ("baseline", "dispatch", "dispatch_cache", "verify")
    assert BASELINE == "baseline"
    assert DEFAULT_DISPATCH_REGIMES == ("baseline", "dispatch_cache")


def test_only_linop_is_dispatch_eligible():
    assert dispatch_eligible("linop")
    assert not dispatch_eligible("space")
    assert not dispatch_eligible("functional")


def test_regimes_for_non_dispatch_family_is_baseline_only():
    assert regimes_for("space", None) == (BASELINE,)
    assert regimes_for("functional", ("dispatch_cache",)) == (BASELINE,)


def test_regimes_for_linop_defaults_and_requests():
    assert regimes_for("linop", None) == DEFAULT_DISPATCH_REGIMES
    # An explicit request always gets baseline prepended for pairing.
    assert regimes_for("linop", ("dispatch_cache",)) == ("baseline", "dispatch_cache")
    assert regimes_for("linop", ("verify",)) == ("baseline", "verify")
    # Unknown regimes are dropped.
    assert regimes_for("linop", ("bogus",)) == (BASELINE,)


# ---------------------------------------------------------------------------
# Context manager activates the real dispatch mode


@pytest.mark.parametrize(
    "regime,expected",
    [
        ("baseline", "off"),
        ("dispatch", "on"),
        ("dispatch_cache", "on"),
        ("verify", "verify"),
    ],
)
def test_benchmark_regime_sets_dispatch_mode(regime, expected):
    from spacecore.kernels import get_dispatch_mode

    with benchmark_regime(regime):
        assert get_dispatch_mode() == expected
    # Restored on exit.
    assert get_dispatch_mode() == "off"


# ---------------------------------------------------------------------------
# End-to-end: a tiny run carries the regime coordinate


def _smallest_linop_probe():
    for p in registry.all():
        if p.family == "linop" and "numpy" in p.backends:
            return p
    pytest.skip("no numpy linop probe available")


def test_run_threads_regime_and_pairs_speedup():
    probe = _smallest_linop_probe()
    size = min(probe.sizes)
    results = run_probes(
        [probe],
        seeds=(0,),
        backends=("numpy",),
        max_size=size,
        progress=False,
    )
    assert results, "expected at least one result row"

    regimes_seen = {r.regime for r in results}
    # A linop probe sweeps the default regimes.
    assert "baseline" in regimes_seen
    assert "dispatch_cache" in regimes_seen

    for r in results:
        assert r.regime in REGIMES
        if r.regime == BASELINE:
            assert r.regime_speedup == 1.0
        else:
            # Paired against the baseline cell at the same coordinate.
            assert r.regime_speedup is None or r.regime_speedup > 0


def test_non_dispatch_probe_runs_baseline_only():
    space_probe = next(
        (p for p in registry.all() if p.family == "space" and "numpy" in p.backends),
        None,
    )
    if space_probe is None:
        pytest.skip("no numpy space probe available")
    size = min(space_probe.sizes)
    results = run_probes(
        [space_probe],
        seeds=(0,),
        backends=("numpy",),
        regimes=("dispatch_cache", "verify"),  # requested, but ignored for space
        max_size=size,
        progress=False,
    )
    assert results
    assert {r.regime for r in results} == {BASELINE}

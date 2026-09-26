"""Tests for the configuration-regime axis (ADR-023), reduced to the baseline.

The ADR-016 dispatch layer the other regimes toggled was removed in 0.5.0a, so every
probe family sweeps only ``baseline`` and a real (tiny) run carries that coordinate.
"""
from __future__ import annotations

import pytest

import bench._operations  # noqa: F401  (registers probes)
from bench._probes import registry
from bench._regimes import BASELINE, REGIMES, benchmark_regime, regimes_for
from bench._run import run_probes


def test_vocabulary_is_baseline_only():
    assert REGIMES == (BASELINE,) == ("baseline",)


@pytest.mark.parametrize("family", ["linop", "space", "functional"])
def test_every_family_runs_baseline_only(family):
    assert regimes_for(family, None) == (BASELINE,)
    assert regimes_for(family, ("dispatch_cache", "verify")) == (BASELINE,)


def test_benchmark_regime_is_a_plain_context():
    with benchmark_regime(BASELINE):
        pass


def test_run_threads_the_baseline_regime():
    probe = next(
        (p for p in registry.all() if p.family == "linop" and "numpy" in p.backends), None
    )
    if probe is None:
        pytest.skip("no numpy linop probe available")
    results = run_probes(
        [probe], seeds=(0,), backends=("numpy",), max_size=min(probe.sizes), progress=False
    )
    assert results
    for r in results:
        assert r.regime == BASELINE and r.regime_speedup == 1.0

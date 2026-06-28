"""The pure-array-library baseline contract (ADR-023).

Every probe's ``bare`` callable must be the hand-optimal raw-array-library
implementation of the *same* mathematical operation as ``sc`` — and, in
particular, must compute the *same value* as the probe's ``reference``.
This is the gate that keeps a fast-but-wrong baseline from flattering
SpaceCore: a strawman bare that skips work would win the speed comparison
while silently computing the wrong answer.

The runner already records ``bare``'s error per cell; this test pins the
invariant structurally so a bad baseline fails CI rather than quietly
inflating a speedup number.
"""
from __future__ import annotations

import pytest

import bench._operations  # noqa: F401  (registers probes)
from bench._probes import registry
from bench._run import _error_vs_reference
from bench._seeds import SEEDS
from bench._verdict import _FAMILY_TOL


def _build(probe, seed, size):
    return (
        probe.factory("numpy", "cpu", seed, size)
        if probe.device_aware
        else probe.factory("numpy", seed, size)
    )


@pytest.mark.parametrize("probe", registry.all(), ids=lambda p: p.name)
def test_bare_matches_reference(probe):
    """``bare`` reproduces the reference value within the family tolerance."""
    if "numpy" not in probe.backends:
        pytest.skip(f"{probe.name}: numpy not in declared backends")
    tol = _FAMILY_TOL.get(probe.family, 1e-9)
    size = min(probe.sizes)
    for seed in SEEDS:
        case = _build(probe, seed, size)
        if case.reference is None:
            continue
        err = _error_vs_reference(case.bare(), case.reference())
        assert err <= tol, (
            f"{probe.name} seed={seed}: bare deviates from reference by "
            f"{err:.3e} > family tol {tol:.3e} — the baseline computes a "
            f"different value than the reference it is benchmarked against."
        )

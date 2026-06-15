"""Policy enforcement tests for :mod:`spacecore.kernels`.

These tests are the gatekeepers for kernel additions. Every spec must
satisfy the contract: a correctness reference, a benchmark id, a
generic and optimized implementation, and an applicability predicate.

If a contributor adds a new kernel without wiring it into the matching
bench case set, :func:`test_every_kernel_has_a_bench_case_id` fires.
"""
from __future__ import annotations

import pytest

import spacecore.kernels as K
from bench._operations import kernel_benchmark_ids as benchmark_ids


def test_registry_is_nonempty():
    """At least one kernel must be registered; the registry itself is alive."""
    assert len(K.registry) >= 1


def test_every_kernel_has_a_correctness_reference():
    """Every spec must name a pytest node id."""
    for spec in K.registry.all():
        assert spec.correctness_ref, f"{spec.name}: missing correctness_ref"
        assert "::" in spec.correctness_ref, (
            f"{spec.name}: correctness_ref must be a pytest node id, "
            f"got {spec.correctness_ref!r}"
        )


def test_every_kernel_has_a_benchmark_id():
    """Every spec must name a bench case id."""
    for spec in K.registry.all():
        assert spec.benchmark_id, f"{spec.name}: missing benchmark_id"


def test_every_kernel_has_a_bench_case():
    """Every registered ``benchmark_id`` must resolve in ``generator_cases``.

    This is the policy enforcement: adding a kernel without a bench case
    is rejected at test time.
    """
    ids_in_use = set(benchmark_ids())
    for spec in K.registry.all():
        assert spec.benchmark_id in ids_in_use, (
            f"{spec.name}: benchmark_id {spec.benchmark_id!r} has no "
            "matching probe in bench/_operations.py"
        )


def test_kernel_names_are_unique():
    """The registry is a set, not a multiset."""
    names = [spec.name for spec in K.registry.all()]
    assert len(names) == len(set(names))


def test_kernel_spec_rejects_missing_reference():
    """The dataclass refuses to be constructed without a correctness ref."""
    with pytest.raises(K.MissingReferenceError):
        K.KernelSpec(
            name="bad-no-ref",
            generic=lambda: None,
            optimized=lambda: None,
            applicable=lambda: True,
            correctness_ref="",
            benchmark_id="some.bench",
        )


def test_kernel_spec_rejects_missing_benchmark():
    """The dataclass refuses to be constructed without a benchmark id."""
    with pytest.raises(K.MissingBenchmarkError):
        K.KernelSpec(
            name="bad-no-bench",
            generic=lambda: None,
            optimized=lambda: None,
            applicable=lambda: True,
            correctness_ref="tests/kernels/test_x.py::test_x",
            benchmark_id="",
        )

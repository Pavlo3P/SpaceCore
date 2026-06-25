"""Tests for :class:`spacecore.kernels.KernelSpec` — the kernel contract.

Checklist section 9:

* A fully-valid spec builds and exposes all fields read-only.
* ``rtol`` / ``atol`` default to ``1e-12``; ``applicable`` defaults are
  supplied by the caller (the dataclass has no default for it) so the
  construction test exercises an explicit predicate.
* The dataclass is frozen + slots: attribute assignment raises.
* ``__post_init__`` validation:
  - empty ``correctness_ref`` -> :class:`MissingReferenceError`.
  - empty ``benchmark_id`` -> :class:`MissingBenchmarkError`.
  - non-callable ``generic`` / ``optimized`` / ``applicable`` -> ``TypeError``.
"""
from __future__ import annotations

import dataclasses

import pytest

import spacecore.kernels as K


def _generic(*args, **kwargs):
    return None


def _optimized(*args, **kwargs):
    return None


def _applicable(*args, **kwargs):
    return True


def _make_spec(**overrides):
    """Build a valid :class:`KernelSpec`, overriding selected fields."""
    fields = dict(
        name="unit-test-kernel",
        generic=_generic,
        optimized=_optimized,
        applicable=_applicable,
        correctness_ref="tests/kernels/test_x.py::test_x",
        benchmark_id="kernels.unit_test",
    )
    fields.update(overrides)
    return K.KernelSpec(**fields)


# ===========================================================================
# Construction
# ===========================================================================
class TestConstruction:
    def test_fully_valid_spec_builds(self):
        spec = _make_spec()
        assert isinstance(spec, K.KernelSpec)

    def test_all_fields_readable(self):
        spec = _make_spec(notes="why this exists")
        assert spec.name == "unit-test-kernel"
        assert spec.generic is _generic
        assert spec.optimized is _optimized
        assert spec.applicable is _applicable
        assert spec.correctness_ref == "tests/kernels/test_x.py::test_x"
        assert spec.benchmark_id == "kernels.unit_test"
        assert spec.notes == "why this exists"

    def test_default_rtol_atol_are_tight(self):
        spec = _make_spec()
        assert spec.rtol == 1e-12
        assert spec.atol == 1e-12

    def test_default_notes_is_empty(self):
        spec = _make_spec()
        assert spec.notes == ""

    def test_callable_applicable_is_stored(self):
        spec = _make_spec(applicable=_applicable)
        assert spec.applicable(1, 2, ops=None) is True

    def test_frozen_slots_assignment_raises(self):
        spec = _make_spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "mutated"  # type: ignore[misc]


# ===========================================================================
# __post_init__ validation
# ===========================================================================
class TestValidation:
    def test_empty_correctness_ref_raises_missing_reference(self):
        with pytest.raises(K.MissingReferenceError):
            _make_spec(correctness_ref="")

    def test_empty_benchmark_id_raises_missing_benchmark(self):
        with pytest.raises(K.MissingBenchmarkError):
            _make_spec(benchmark_id="")

    def test_non_callable_generic_raises_type_error(self):
        with pytest.raises(TypeError, match="generic must be callable"):
            _make_spec(generic="not-callable")

    def test_non_callable_optimized_raises_type_error(self):
        with pytest.raises(TypeError, match="optimized must be callable"):
            _make_spec(optimized=123)

    def test_non_callable_applicable_raises_type_error(self):
        with pytest.raises(TypeError, match="applicable must be callable"):
            _make_spec(applicable=object())

"""Tests for :class:`spacecore.kernels.KernelRegistry` and the singleton.

Checklist section 9:

* ``register`` stores the spec and returns it; re-registering the
  *identical* object is idempotent; a different spec with the same name
  raises ``ValueError``.
* ``get`` returns the spec; an unknown name raises ``KeyError``;
  ``__contains__`` reports membership.
* ``all`` / ``names`` / ``__iter__`` / ``__len__`` on a controlled
  registry.
* The process-wide singleton ``spacecore.kernels.registry`` contains the
  shipped kernels, exposes them via the modules' ``SPEC`` attributes,
  has unique names, and is nonempty.
"""
from __future__ import annotations

import pytest

import spacecore.kernels as K
from spacecore.kernels import block_diagonal, composed


def _spec(name="unit-test-kernel", **overrides):
    fields = dict(
        name=name,
        generic=lambda *a, **k: None,
        optimized=lambda *a, **k: None,
        applicable=lambda *a, **k: True,
        correctness_ref="tests/kernels/test_x.py::test_x",
        benchmark_id="kernels.unit_test",
    )
    fields.update(overrides)
    return K.KernelSpec(**fields)


# ===========================================================================
# register
# ===========================================================================
class TestRegister:
    def test_register_stores_and_returns_spec(self):
        reg = K.KernelRegistry()
        spec = _spec()
        returned = reg.register(spec)
        assert returned is spec
        assert reg.get("unit-test-kernel") is spec

    def test_reregistering_identical_spec_is_idempotent(self):
        reg = K.KernelRegistry()
        spec = _spec()
        reg.register(spec)
        # No raise; still a single entry.
        assert reg.register(spec) is spec
        assert len(reg) == 1

    def test_duplicate_name_different_spec_raises(self):
        reg = K.KernelRegistry()
        reg.register(_spec(name="dup"))
        with pytest.raises(ValueError, match="name collision"):
            reg.register(_spec(name="dup", benchmark_id="kernels.other"))


# ===========================================================================
# lookup
# ===========================================================================
class TestLookup:
    def test_get_returns_spec(self):
        reg = K.KernelRegistry()
        spec = _spec()
        reg.register(spec)
        assert reg.get("unit-test-kernel") is spec

    def test_get_unknown_name_raises_key_error(self):
        reg = K.KernelRegistry()
        with pytest.raises(KeyError):
            reg.get("does-not-exist")

    def test_contains_reports_membership(self):
        reg = K.KernelRegistry()
        reg.register(_spec(name="present"))
        assert "present" in reg
        assert "absent" not in reg

    def test_contains_non_string_is_false(self):
        reg = K.KernelRegistry()
        assert 123 not in reg


# ===========================================================================
# iteration
# ===========================================================================
class TestIteration:
    def test_all_returns_specs_in_registration_order(self):
        reg = K.KernelRegistry()
        a = _spec(name="a")
        b = _spec(name="b")
        reg.register(a)
        reg.register(b)
        assert reg.all() == (a, b)

    def test_names_returns_names_in_registration_order(self):
        reg = K.KernelRegistry()
        reg.register(_spec(name="a"))
        reg.register(_spec(name="b"))
        assert reg.names() == ("a", "b")

    def test_iter_yields_specs(self):
        reg = K.KernelRegistry()
        a = _spec(name="a")
        b = _spec(name="b")
        reg.register(a)
        reg.register(b)
        assert list(reg) == [a, b]

    def test_len_counts_registered_specs(self):
        reg = K.KernelRegistry()
        assert len(reg) == 0
        reg.register(_spec(name="a"))
        assert len(reg) == 1


# ===========================================================================
# the process-wide singleton
# ===========================================================================
class TestSingleton:
    def test_singleton_contains_shipped_kernels(self):
        assert "block-diagonal-dense-apply" in K.registry
        assert "composed-chain-apply" in K.registry

    def test_block_diagonal_spec_matches_registry(self):
        assert block_diagonal.SPEC is K.registry.get(
            "block-diagonal-dense-apply"
        )

    def test_composed_spec_matches_registry(self):
        assert composed.SPEC is K.registry.get("composed-chain-apply")

    def test_registry_is_nonempty(self):
        assert len(K.registry) >= 1

    def test_kernel_names_are_unique(self):
        names = [spec.name for spec in K.registry.all()]
        assert len(names) == len(set(names))

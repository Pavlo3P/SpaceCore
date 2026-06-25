"""Tests for the ADR-016 optimized-kernel dispatch policy.

Covers the mechanism end to end:

* :class:`KernelSpec` dispatch metadata and dispatch-eligibility.
* :class:`KernelCost` validation.
* The registry's ``dispatch_key`` index and its registration-time
  ambiguity rejection.
* The :func:`dispatch` entry point in ``off`` / ``on`` / ``verify`` modes,
  including the ``check_level="strict"`` → ``verify`` rule.
* The shape-only memory gate: ``no estimate, no fuse``; ``no budget, no
  fuse``; affordability under a known budget.
* The two wired call sites (composed apply, block-diagonal apply) are
  result-identical across modes when nothing is routed, and actually route
  to an eligible spec when one is registered and dispatch is on.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
import spacecore.kernels as K
from spacecore.kernels.specs._dispatch import DispatchVerificationError
from spacecore.kernels.specs._registry import (
    DispatchAmbiguityError,
    KernelRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _spec(name="d-kernel", **overrides):
    fields = dict(
        name=name,
        generic=lambda *a, **k: ("generic", a),
        optimized=lambda *a, **k: ("optimized", a),
        applicable=lambda *a, **k: True,
        correctness_ref="tests/kernels/test_x.py::test_x",
        benchmark_id="kernels.unit_test",
        rtol=0.0,
        atol=0.0,
        dispatch_key="op.family",
        priority=0,
    )
    fields.update(overrides)
    return K.KernelSpec(**fields)


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    """Keep global dispatch mode/budget pristine across tests."""
    mode = K.get_dispatch_mode()
    frac = K.get_memory_budget_fraction()
    yield
    K.set_dispatch_mode(mode)
    K.set_memory_budget_fraction(frac)


@pytest.fixture
def patched_registry(monkeypatch):
    """Swap the singleton the dispatcher reads for an isolated registry."""
    fresh = KernelRegistry()
    monkeypatch.setattr(
        "spacecore.kernels.specs._registry.registry", fresh, raising=True
    )
    return fresh


@pytest.fixture
def strict_ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="strict")


# ---------------------------------------------------------------------------
# KernelSpec dispatch metadata & eligibility
# ---------------------------------------------------------------------------
class TestEligibility:
    def test_defaults_are_explicit_entry_only(self):
        spec = _spec(dispatch_key="")
        assert spec.dispatch_key == ""
        assert spec.priority == 0
        assert spec.cost is None
        assert spec.is_dispatch_eligible is False

    def test_keyed_exact_spec_is_eligible(self):
        assert _spec(dispatch_key="k", rtol=0.0, atol=0.0).is_dispatch_eligible

    def test_loosened_tolerance_is_not_eligible(self):
        assert not _spec(dispatch_key="k", rtol=1e-9, atol=0.0).is_dispatch_eligible
        assert not _spec(dispatch_key="k", rtol=0.0, atol=1e-9).is_dispatch_eligible

    def test_no_key_is_not_eligible_even_when_exact(self):
        assert not _spec(dispatch_key="", rtol=0.0, atol=0.0).is_dispatch_eligible

    def test_non_callable_cost_raises(self):
        with pytest.raises(TypeError, match="cost must be callable"):
            _spec(cost=123)

    def test_non_int_priority_raises(self):
        with pytest.raises(TypeError, match="priority must be an int"):
            _spec(priority=1.5)

    def test_bool_priority_rejected(self):
        with pytest.raises(TypeError, match="priority must be an int"):
            _spec(priority=True)


class TestKernelCost:
    def test_fields(self):
        c = K.KernelCost(peak_bytes=1024, flops=2048)
        assert c.peak_bytes == 1024
        assert c.flops == 2048

    def test_flops_defaults_zero(self):
        assert K.KernelCost(peak_bytes=8).flops == 0

    def test_negative_bytes_rejected(self):
        with pytest.raises(ValueError):
            K.KernelCost(peak_bytes=-1)

    def test_negative_flops_rejected(self):
        with pytest.raises(ValueError):
            K.KernelCost(peak_bytes=0, flops=-1)


# ---------------------------------------------------------------------------
# Registry dispatch index & ambiguity
# ---------------------------------------------------------------------------
class TestRegistryIndex:
    def test_eligible_spec_is_indexed(self):
        reg = KernelRegistry()
        spec = _spec(name="a", dispatch_key="k")
        reg.register(spec)
        assert reg.dispatch_candidates("k") == (spec,)
        assert reg.dispatch_keys() == ("k",)

    def test_non_eligible_spec_not_indexed(self):
        reg = KernelRegistry()
        reg.register(_spec(name="explicit", dispatch_key=""))
        reg.register(_spec(name="loose", dispatch_key="k", rtol=1e-9))
        assert reg.dispatch_candidates("k") == ()
        assert reg.dispatch_keys() == ()

    def test_candidates_sorted_by_descending_priority(self):
        reg = KernelRegistry()
        low = _spec(name="low", dispatch_key="k", priority=1)
        high = _spec(name="high", dispatch_key="k", priority=5)
        mid = _spec(name="mid", dispatch_key="k", priority=3)
        reg.register(low)
        reg.register(high)
        reg.register(mid)
        assert reg.dispatch_candidates("k") == (high, mid, low)

    def test_equal_priority_same_key_raises(self):
        reg = KernelRegistry()
        reg.register(_spec(name="a", dispatch_key="k", priority=2))
        with pytest.raises(DispatchAmbiguityError, match="dispatch ambiguity"):
            reg.register(_spec(name="b", dispatch_key="k", priority=2))

    def test_equal_priority_different_key_is_fine(self):
        reg = KernelRegistry()
        reg.register(_spec(name="a", dispatch_key="k1", priority=2))
        reg.register(_spec(name="b", dispatch_key="k2", priority=2))
        assert reg.dispatch_candidates("k1")[0].name == "a"
        assert reg.dispatch_candidates("k2")[0].name == "b"

    def test_idempotent_reregister_does_not_double_index(self):
        reg = KernelRegistry()
        spec = _spec(name="a", dispatch_key="k")
        reg.register(spec)
        reg.register(spec)
        assert reg.dispatch_candidates("k") == (spec,)

    def test_unknown_key_returns_empty(self):
        assert KernelRegistry().dispatch_candidates("nope") == ()


# ---------------------------------------------------------------------------
# dispatch() modes
# ---------------------------------------------------------------------------
class TestDispatchModes:
    def _generic(self, *args):
        return ("generic", args)

    def test_off_runs_generic_even_when_applicable(self, patched_registry):
        patched_registry.register(_spec(dispatch_key="k"))
        K.set_dispatch_mode("off")
        out = K.dispatch("k", 1, 2, generic=self._generic)
        assert out == ("generic", (1, 2))

    def test_on_routes_to_optimized(self, patched_registry):
        patched_registry.register(
            _spec(dispatch_key="k", optimized=lambda *a: ("opt", a))
        )
        K.set_dispatch_mode("on")
        out = K.dispatch("k", 7, generic=self._generic)
        assert out == ("opt", (7,))

    def test_on_falls_back_when_not_applicable(self, patched_registry):
        patched_registry.register(
            _spec(dispatch_key="k", applicable=lambda *a: False)
        )
        K.set_dispatch_mode("on")
        out = K.dispatch("k", 1, generic=self._generic)
        assert out == ("generic", (1,))

    def test_on_skips_inapplicable_higher_priority(self, patched_registry):
        patched_registry.register(
            _spec(
                name="high",
                dispatch_key="k",
                priority=5,
                applicable=lambda *a: False,
                optimized=lambda *a: ("high", a),
            )
        )
        patched_registry.register(
            _spec(
                name="low",
                dispatch_key="k",
                priority=1,
                applicable=lambda *a: True,
                optimized=lambda *a: ("low", a),
            )
        )
        K.set_dispatch_mode("on")
        out = K.dispatch("k", 9, generic=self._generic)
        assert out == ("low", (9,))

    def test_no_candidates_runs_generic(self, patched_registry):
        K.set_dispatch_mode("on")
        out = K.dispatch("absent", 1, generic=self._generic)
        assert out == ("generic", (1,))

    def test_verify_agreement_returns_optimized(self, patched_registry):
        patched_registry.register(
            _spec(
                dispatch_key="k",
                generic=lambda *a: ("same", a),
                optimized=lambda *a: ("same", a),
            )
        )
        K.set_dispatch_mode("verify")
        out = K.dispatch("k", 3, generic=lambda *a: ("same", a))
        assert out == ("same", (3,))

    def test_verify_mismatch_raises(self, patched_registry):
        patched_registry.register(
            _spec(dispatch_key="k", optimized=lambda *a: ("wrong", a))
        )
        K.set_dispatch_mode("verify")
        with pytest.raises(DispatchVerificationError, match="verify mismatch"):
            K.dispatch("k", 1, generic=lambda *a: ("right", a))

    def test_verify_array_results_agree_without_ctx(self, patched_registry):
        # ctx=None: array agreement must not degrade to an identity check and
        # raise a spurious mismatch on equal-but-distinct arrays.
        patched_registry.register(
            _spec(
                dispatch_key="k",
                generic=lambda *a: np.array([1.0, 2.0, 3.0]),
                optimized=lambda *a: np.array([1.0, 2.0, 3.0]),
            )
        )
        K.set_dispatch_mode("verify")
        out = K.dispatch("k", 1, generic=lambda *a: np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_verify_array_results_disagree_without_ctx_raises(self, patched_registry):
        patched_registry.register(
            _spec(dispatch_key="k", optimized=lambda *a: np.array([9.0, 9.0, 9.0]))
        )
        K.set_dispatch_mode("verify")
        with pytest.raises(DispatchVerificationError):
            K.dispatch("k", 1, generic=lambda *a: np.array([1.0, 2.0, 3.0]))

    def test_applicable_exception_is_skipped_with_warning(self, patched_registry):
        def boom(*a):
            raise RuntimeError("buggy predicate")

        patched_registry.register(
            _spec(dispatch_key="k", applicable=boom, optimized=lambda *a: ("opt", a))
        )
        K.set_dispatch_mode("on")
        with pytest.warns(RuntimeWarning, match="treating it as not applicable"):
            out = K.dispatch("k", 5, generic=lambda *a: ("gen", a))
        assert out == ("gen", (5,))

    def test_verify_compares_against_callsite_generic(self, patched_registry):
        # Optimized matches the call-site generic but not the spec's own
        # generic; verify must compare against the call-site fallback.
        patched_registry.register(
            _spec(
                dispatch_key="k",
                generic=lambda *a: ("spec-generic", a),
                optimized=lambda *a: ("callsite", a),
            )
        )
        K.set_dispatch_mode("verify")
        out = K.dispatch("k", 2, generic=lambda *a: ("callsite", a))
        assert out == ("callsite", (2,))


class TestStrictImpliesVerify:
    def test_should_consult_under_strict_when_off(self, strict_ctx):
        K.set_dispatch_mode("off")
        assert K.should_consult_dispatch(strict_ctx) is True

    def test_effective_mode_strict_is_verify(self, strict_ctx):
        K.set_dispatch_mode("off")
        assert K.effective_mode(strict_ctx) == "verify"
        K.set_dispatch_mode("on")
        assert K.effective_mode(strict_ctx) == "verify"

    def test_strict_routes_through_verify(self, patched_registry, strict_ctx):
        patched_registry.register(
            _spec(dispatch_key="k", optimized=lambda *a: ("bad", a))
        )
        K.set_dispatch_mode("off")  # strict still forces verify
        with pytest.raises(DispatchVerificationError):
            K.dispatch("k", 1, generic=lambda *a: ("good", a), ctx=strict_ctx)


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------
class _FakeOps:
    """Backend stand-in with a controllable free-memory report."""

    def __init__(self, free):
        self._free = free

    def free_memory_bytes(self):
        return self._free

    def is_array(self, x):
        return False


class _Ctx:
    def __init__(self, free, check_level="standard"):
        self.ops = _FakeOps(free)
        self.check_level = check_level


class TestMemoryGate:
    def test_no_cost_always_affordable(self, patched_registry):
        patched_registry.register(
            _spec(dispatch_key="k", optimized=lambda *a: ("opt", a))
        )
        K.set_dispatch_mode("on")
        out = K.dispatch("k", 1, generic=lambda *a: ("gen", a), ctx=_Ctx(free=0))
        assert out == ("opt", (1,))

    def test_no_estimate_no_fuse(self, patched_registry):
        patched_registry.register(
            _spec(
                dispatch_key="k",
                optimized=lambda *a: ("opt", a),
                cost=lambda *a: None,
            )
        )
        K.set_dispatch_mode("on")
        out = K.dispatch("k", 1, generic=lambda *a: ("gen", a), ctx=_Ctx(free=10**9))
        assert out == ("gen", (1,))

    def test_unknown_budget_skips_cost_carrying_spec(self, patched_registry):
        patched_registry.register(
            _spec(
                dispatch_key="k",
                optimized=lambda *a: ("opt", a),
                cost=lambda *a: K.KernelCost(peak_bytes=8),
            )
        )
        K.set_dispatch_mode("on")
        out = K.dispatch("k", 1, generic=lambda *a: ("gen", a), ctx=_Ctx(free=None))
        assert out == ("gen", (1,))

    def test_over_budget_falls_through(self, patched_registry):
        patched_registry.register(
            _spec(
                dispatch_key="k",
                optimized=lambda *a: ("opt", a),
                cost=lambda *a: K.KernelCost(peak_bytes=2_000),
            )
        )
        K.set_dispatch_mode("on")
        K.set_memory_budget_fraction(1.0)
        out = K.dispatch("k", 1, generic=lambda *a: ("gen", a), ctx=_Ctx(free=1_000))
        assert out == ("gen", (1,))

    def test_within_budget_is_selected(self, patched_registry):
        patched_registry.register(
            _spec(
                dispatch_key="k",
                optimized=lambda *a: ("opt", a),
                cost=lambda *a: K.KernelCost(peak_bytes=500),
            )
        )
        K.set_dispatch_mode("on")
        K.set_memory_budget_fraction(1.0)
        out = K.dispatch("k", 1, generic=lambda *a: ("gen", a), ctx=_Ctx(free=1_000))
        assert out == ("opt", (1,))

    def test_budget_fraction_applies(self, patched_registry):
        patched_registry.register(
            _spec(
                dispatch_key="k",
                optimized=lambda *a: ("opt", a),
                cost=lambda *a: K.KernelCost(peak_bytes=600),
            )
        )
        K.set_dispatch_mode("on")
        K.set_memory_budget_fraction(0.5)  # budget = 500 < 600 → skip
        out = K.dispatch("k", 1, generic=lambda *a: ("gen", a), ctx=_Ctx(free=1_000))
        assert out == ("gen", (1,))


# ---------------------------------------------------------------------------
# Mode context manager / configuration
# ---------------------------------------------------------------------------
class TestModeConfig:
    def test_context_manager_restores(self):
        K.set_dispatch_mode("off")
        with K.dispatch_mode("on"):
            assert K.get_dispatch_mode() == "on"
        assert K.get_dispatch_mode() == "off"

    def test_nested_context_managers(self):
        K.set_dispatch_mode("off")
        with K.dispatch_mode("on"):
            with K.dispatch_mode("verify"):
                assert K.get_dispatch_mode() == "verify"
            assert K.get_dispatch_mode() == "on"
        assert K.get_dispatch_mode() == "off"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="Unknown dispatch mode"):
            K.set_dispatch_mode("sometimes")
        with pytest.raises(ValueError, match="Unknown dispatch mode"):
            with K.dispatch_mode("maybe"):
                pass

    def test_should_consult_off_by_default(self):
        K.set_dispatch_mode("off")
        assert K.should_consult_dispatch() is False
        assert K.should_consult_dispatch(None) is False

    def test_budget_fraction_bounds(self):
        with pytest.raises(ValueError):
            K.set_memory_budget_fraction(0.0)
        with pytest.raises(ValueError):
            K.set_memory_budget_fraction(1.5)


# ---------------------------------------------------------------------------
# Wired call sites: identity when nothing routes, routing when one does
# ---------------------------------------------------------------------------
@pytest.fixture
def composed_op(numpy_ctx):
    v = sc.DenseVectorSpace((2,), ctx=numpy_ctx)
    A = sc.DenseLinOp(
        numpy_ctx.asarray(np.array([[1.0, 2.0], [0.0, 1.0]])), v, v, numpy_ctx
    )
    B = sc.DenseLinOp(
        numpy_ctx.asarray(np.array([[0.0, 1.0], [1.0, 0.0]])), v, v, numpy_ctx
    )
    x = numpy_ctx.asarray(np.array([3.0, 4.0]))
    return A @ B, x, (A, B)


class TestComposedCallSite:
    def test_modes_are_result_identical_when_nothing_routes(self, composed_op):
        op, x, _ = composed_op
        baseline = np.asarray(op.apply(x))
        with K.dispatch_mode("on"):
            assert np.array_equal(np.asarray(op.apply(x)), baseline)
        with K.dispatch_mode("verify"):
            assert np.array_equal(np.asarray(op.apply(x)), baseline)

    def test_dispatch_routes_at_composed_site(
        self, composed_op, patched_registry, numpy_ctx
    ):
        op, x, _ = composed_op
        sentinel = numpy_ctx.asarray(np.array([42.0, 42.0]))
        patched_registry.register(
            K.KernelSpec(
                name="composed-test-route",
                generic=lambda chain, xx: xx,
                optimized=lambda chain, xx: sentinel,
                applicable=lambda chain, xx: True,
                correctness_ref="tests/kernels/test_kernel_dispatch.py::route",
                benchmark_id="kernels.unit_test",
                rtol=0.0,
                atol=0.0,
                dispatch_key="linop.composed.apply",
            )
        )
        with K.dispatch_mode("on"):
            assert np.array_equal(np.asarray(op.apply(x)), np.asarray(sentinel))


class TestBlockDiagonalCallSite:
    @pytest.fixture
    def block_op(self, numpy_ctx):
        v = sc.DenseVectorSpace((2,), ctx=numpy_ctx)
        A = sc.DenseLinOp(
            numpy_ctx.asarray(np.array([[1.0, 2.0], [0.0, 1.0]])), v, v, numpy_ctx
        )
        B = sc.DenseLinOp(
            numpy_ctx.asarray(np.array([[0.0, 1.0], [1.0, 0.0]])), v, v, numpy_ctx
        )
        from spacecore.linop.tree import BlockDiagonalLinOp

        op = BlockDiagonalLinOp((A, B))
        xb = (
            numpy_ctx.asarray(np.array([1.0, 2.0])),
            numpy_ctx.asarray(np.array([3.0, 4.0])),
        )
        return op, xb

    def test_modes_are_result_identical_when_nothing_routes(self, block_op):
        op, xb = block_op
        baseline = [np.asarray(z) for z in op.apply(xb)]
        with K.dispatch_mode("on"):
            got = [np.asarray(z) for z in op.apply(xb)]
        for a, b in zip(got, baseline):
            assert np.array_equal(a, b)

    def test_dispatch_routes_at_block_site(
        self, block_op, patched_registry, numpy_ctx
    ):
        op, xb = block_op
        sentinel = (
            numpy_ctx.asarray(np.array([7.0, 7.0])),
            numpy_ctx.asarray(np.array([8.0, 8.0])),
        )
        patched_registry.register(
            K.KernelSpec(
                name="block-test-route",
                generic=lambda parts, xs: tuple(p(x) for p, x in zip(parts, xs)),
                optimized=lambda parts, xs: sentinel,
                applicable=lambda parts, xs: True,
                correctness_ref="tests/kernels/test_kernel_dispatch.py::route",
                benchmark_id="kernels.unit_test",
                rtol=0.0,
                atol=0.0,
                dispatch_key="linop.block_diagonal.apply",
            )
        )
        with K.dispatch_mode("on"):
            got = op.apply(xb)
        for a, b in zip(got, sentinel):
            assert np.array_equal(np.asarray(a), np.asarray(b))


class TestShippedSpecsRemainExplicitOnly:
    def test_original_catalog_specs_are_not_dispatch_eligible(self):
        # The 0.4.0 catalog kernels are retained as explicit-entry kernels;
        # their signatures predate the call-site dispatch contracts.
        assert not K.registry.get("composed-chain-apply").is_dispatch_eligible
        assert not K.registry.get("block-diagonal-dense-apply").is_dispatch_eligible

    def test_algebraic_specs_populate_the_dispatch_index(self):
        keys = K.registry.dispatch_keys()
        assert "linop.composed.apply" in keys
        assert "linop.block_diagonal.apply" in keys
        composed = [s.name for s in K.registry.dispatch_candidates("linop.composed.apply")]
        # Zero annihilation outranks identity elision (priority 20 > 10).
        assert composed == ["composed-zero-annihilation", "composed-identity-elision"]
        block = [s.name for s in K.registry.dispatch_candidates("linop.block_diagonal.apply")]
        assert block == ["block-diagonal-uniform-dense-batched"]

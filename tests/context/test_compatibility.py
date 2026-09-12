"""Tests for context compatibility / inference helpers.

Book justification: the resolver relations and inference are pinned as
contracts (Hunt & Thomas, *The Pragmatic Programmer*, tip 37). The synthetic
``_OtherFamilyOps`` is a Test-Specific Subclass standing in for a second backend
family, so cross-family behavior is covered without depending on an installed
optional backend — keeping the suite Repeatable and free of Erratic Tests
(Meszaros, *xUnit Test Patterns*).

Checklist section 2: context compatibility and inference.

Covers the free helpers that gate operator algebra and context resolution.
(The ``Context.same_math`` / ``same_backend`` relation contracts live in
``test_context_contracts.py``; this file covers the resolver helpers.)

* ``Contextual.are_compatible_values`` / ``are_compatible_ops`` — the
  family-mismatch logic for raw values and raw ``BackendOps``.
* ``Contextual.infer_context`` / ``infer_contexts`` — the ``.ctx`` fast
  path, ``is_array`` matching, the ``get_dtype`` fallback, and the
  no-match ``None`` branch.
* ``Contextual.ctx_from_ops`` — dtype sanitization for a raw ``BackendOps``
  instance. Validation policy is not part of the context: the ambient level
  seeds the bound object instead.
* ``normalize_check_level`` — the deprecated ``enable_checks`` legacy shim,
  now the only surviving home of the Boolean switch, and the
  ``DeprecationWarning`` it emits under ``warn_legacy=True``.

References are independent: NumPy dtypes, explicit family strings, and the
source contracts read from ``_state.py`` / ``_bound.py`` / ``_check_policy.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore._check_policy import normalize_check_level
from spacecore.contextual._contextual import Contextual
from spacecore.backend import OpsRegistry


# A NumpyOps subclass with a distinct family. Backend equality and the
# compatibility helpers key off ``family``, so this stands in for a
# genuinely different ops family without needing an optional backend
# (jax/torch/cupy) installed.
class _OtherFamilyOps(sc.NumpyOps):
    _family = "other_family"


# ===========================================================================
# are_compatible_ops — raw BackendOps family logic
# ===========================================================================
class TestAreCompatibleOps:
    def setup_method(self):
        self.state = Contextual()

    def test_empty_is_compatible(self):
        assert self.state.are_compatible_ops() is True

    def test_single_is_compatible(self):
        assert self.state.are_compatible_ops(sc.NumpyOps()) is True

    def test_same_family_is_compatible(self):
        assert self.state.are_compatible_ops(sc.NumpyOps(), sc.NumpyOps()) is True

    def test_cross_family_is_incompatible(self):
        assert (
            self.state.are_compatible_ops(sc.NumpyOps(), _OtherFamilyOps()) is False
        )


# ===========================================================================
# are_compatible_values — raw values, via inferred contexts
# ===========================================================================
class TestAreCompatibleValues:
    def setup_method(self):
        self.state = Contextual()

    def test_same_backend_arrays_are_compatible(self):
        a = np.zeros(3, dtype=np.float64)
        b = np.ones(2, dtype=np.float32)
        assert self.state.are_compatible_values(a, b) is True

    def test_uninferrable_values_filtered_out_are_compatible(self):
        """Non-array values infer to ``None`` and drop out, leaving the
        single numpy array — trivially compatible."""
        a = np.zeros(3)
        assert self.state.are_compatible_values(a, object(), 5) is True

    def test_cross_family_values_are_incompatible(self):
        """A numpy array and a bound object on a distinct ops family infer to
        contexts in different families and are incompatible."""
        np_array = np.zeros(3)
        other_ctx = sc.Context(_OtherFamilyOps(), dtype=np.float64)
        bound = sc.DenseCoordinateSpace((2,), other_ctx)
        assert self.state.are_compatible_values(np_array, bound) is False


# ===========================================================================
# infer_context — primary branches
# ===========================================================================
class TestInferContext:
    def setup_method(self):
        self.state = Contextual()

    def test_context_argument_returns_itself(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
        assert self.state.infer_context(ctx) is ctx

    def test_ctx_fast_path(self):
        """An object exposing ``.ctx`` resolves via that attribute directly."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
        bound = sc.DenseCoordinateSpace((2,), ctx, check_level="cheap")
        out = self.state.infer_context(bound)
        assert out == ctx

    def test_is_array_match_infers_dtype(self):
        """A raw backend array is matched by ``is_array`` and its dtype read
        via ``get_dtype``."""
        x = np.zeros(4, dtype=np.float32)
        out = self.state.infer_context(x)
        assert out is not None
        assert out.ops.family == "numpy"
        assert out.dtype == np.dtype(np.float32)

    def test_get_dtype_exception_falls_back_to_x_dtype(self):
        """When ``get_dtype`` rejects the object, inference falls back to the
        object's own ``.dtype`` attribute.

        We pass a numpy 0-d scalar-like array subclass that ``is_array``
        accepts but whose ``get_dtype`` path we force to fail. Concretely we
        use an object that lies about being a backend array."""

        captured = {}

        class _FakeOps(sc.NumpyOps):
            _family = "fake_only"

            def is_array(self, x):  # noqa: D401 - test stub
                return isinstance(x, _FakeArray)

            def get_dtype(self, x):
                raise RuntimeError("boom")

        class _FakeArray:
            dtype = np.dtype(np.float32)

        # Resolve against a registry holding only the fake backend, so it is
        # the sole match for the fake array. Injecting a registry is the
        # supported seam; this used to overwrite private state.
        state = Contextual(ops_registry=OpsRegistry([_FakeOps]))
        captured["x"] = _FakeArray()
        out = state.infer_context(captured["x"])
        assert out is not None
        assert out.dtype == np.dtype(np.float32)

    def test_no_match_returns_none(self):
        assert self.state.infer_context(object()) is None
        assert self.state.infer_context("not an array") is None


# ===========================================================================
# infer_contexts — None-filtering wrapper
# ===========================================================================
class TestInferContexts:
    def setup_method(self):
        self.state = Contextual()

    def test_filters_none_results(self):
        x = np.zeros(3)
        out = self.state.infer_contexts([x, object(), 7])
        assert len(out) == 1
        assert out[0].ops.family == "numpy"

    def test_empty_iterable_yields_empty_tuple(self):
        assert self.state.infer_contexts([]) == ()

    def test_all_uninferrable_yields_empty_tuple(self):
        assert self.state.infer_contexts([object(), "x", 1]) == ()


# ===========================================================================
# ctx_from_ops — dtype sanitization; policy lives on the bound object
# ===========================================================================
class TestCtxFromOps:
    def setup_method(self):
        self.state = Contextual()

    def test_default_dtype_per_family(self):
        ops = sc.NumpyOps()
        out = self.state.ctx_from_ops(ops)
        assert out.dtype == ops.sanitize_dtype(None)

    def test_explicit_dtype_is_sanitized(self):
        ops = sc.NumpyOps()
        out = self.state.ctx_from_ops(ops, dtype=np.float32)
        assert out.dtype == np.dtype(np.float32)

    def test_baseline_check_level_is_standard(self):
        """``Contextual._baseline_check_level`` is 'standard', and it is what a
        bound object built on a resolver-made context inherits."""
        assert self.state.get_check_level() == "standard"
        out = self.state.ctx_from_ops(sc.NumpyOps())
        assert sc.DenseCoordinateSpace((2,), out).check_level == "standard"

    def test_explicit_check_level_is_honored(self):
        out = self.state.ctx_from_ops(sc.NumpyOps())
        bound = sc.DenseCoordinateSpace((2,), out, check_level="strict")
        assert bound.check_level == "strict"

    def test_returned_ops_match_input(self):
        ops = sc.NumpyOps()
        out = self.state.ctx_from_ops(ops)
        assert out.ops.family == "numpy"


# ===========================================================================
# normalize_check_level — legacy enable_checks shim + DeprecationWarning
# ===========================================================================
class TestEnableChecksShim:
    def test_enable_checks_true_resolves_standard(self):
        assert normalize_check_level(enable_checks=True) == "standard"

    def test_enable_checks_false_resolves_none(self):
        assert normalize_check_level(enable_checks=False) == "none"

    def test_enable_checks_emits_deprecation_warning_when_requested(self):
        """The shim warns only under ``warn_legacy=True``; no production caller
        passes it, so the warning is opt-in for callers still bridging the
        Boolean switch."""
        with pytest.warns(DeprecationWarning) as caught:
            normalize_check_level(enable_checks=True, warn_legacy=True)
        assert "enable_checks is deprecated" in str(caught[0].message)

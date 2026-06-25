"""Tests for the free-function API in :mod:`spacecore._contextual._state`.

The functions covered:

* :func:`spacecore.set_context` / :func:`spacecore.get_context` — the
  process-wide default context with try/finally-safe nesting.
* :func:`spacecore.register_ops` — register a new ``BackendOps`` subclass
  and reject duplicate-family registration.
* :func:`spacecore.normalize_ops` — accept ``BackendOps`` instance, class,
  ``BackendFamily``, or family string.
* :func:`spacecore.normalize_context` — accept ``Context``, family string,
  ``BackendFamily``, or ``None``.
* :func:`spacecore.resolve_context_priority` — explicit > bound > active
  default precedence.
"""
from __future__ import annotations

from typing import Type

import numpy as np
import pytest

import spacecore as sc
from spacecore.backend import BackendFamily
from spacecore._contextual import UnknownBackendError

from tests._helpers import has_cupy, has_jax, has_torch


_OPTIONAL_BACKEND_PROBES = {"jax": has_jax, "torch": has_torch, "cupy": has_cupy}


# ===========================================================================
# set_context / get_context
# ===========================================================================
class TestSetGetContext:
    def test_get_context_returns_a_context(self):
        ctx = sc.get_context()
        assert isinstance(ctx, sc.Context)

    def test_default_context_is_numpy(self):
        ctx = sc.get_context()
        assert ctx.ops.family == "numpy"

    def test_set_context_with_context_object(self, preserve_default_context):
        target = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
        sc.set_context(target)
        assert sc.get_context() == target

    def test_set_context_with_family_string(self, preserve_default_context):
        sc.set_context("numpy")
        assert sc.get_context().ops.family == "numpy"

    def test_set_context_with_backend_family_enum(self, preserve_default_context):
        sc.set_context(BackendFamily.numpy)
        assert sc.get_context().ops.family == "numpy"

    def test_set_context_with_none_is_a_noop(self, preserve_default_context):
        """``set_context(None)`` returns the existing default unchanged.

        ``normalize_context(None)`` short-circuits to ``default_ctx``, so
        the assignment ``default_ctx = default_ctx`` is a no-op.
        """
        target = sc.Context(sc.NumpyOps(), dtype=np.float32)
        sc.set_context(target)
        sc.set_context(None)
        assert sc.get_context() == target

    def test_set_context_nesting_with_try_finally(self, preserve_default_context):
        """Two-deep nesting restores the previous default at each level."""
        a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        original = sc.get_context()
        sc.set_context(a)
        try:
            assert sc.get_context() == a
            sc.set_context(b)
            try:
                assert sc.get_context() == b
            finally:
                sc.set_context(a)
            assert sc.get_context() == a
        finally:
            sc.set_context(original)
        assert sc.get_context() == original

    @pytest.mark.parametrize("family", ["jax", "torch", "cupy"])
    def test_set_context_with_optional_backend(self, preserve_default_context, family):
        if not _OPTIONAL_BACKEND_PROBES[family]():
            pytest.skip(f"{family} is not installed")
        sc.set_context(family)
        assert sc.get_context().ops.family == family

    def test_set_context_rejects_unknown_backend(self, preserve_default_context):
        with pytest.raises(UnknownBackendError, match="(?i)unknown backend"):
            sc.set_context("definitely_not_a_backend")


# ===========================================================================
# normalize_ops
# ===========================================================================
class TestNormalizeOps:
    def test_accepts_instance(self):
        ops = sc.NumpyOps()
        out = sc.normalize_ops(ops)
        assert out is ops  # pass-through for instances

    def test_accepts_family_string(self):
        out = sc.normalize_ops("numpy")
        assert out.family == "numpy"
        assert isinstance(out, sc.NumpyOps)

    def test_accepts_backend_family_enum(self):
        out = sc.normalize_ops(BackendFamily.numpy)
        assert out.family == "numpy"

    def test_accepts_backend_ops_class(self):
        """Passing the class itself resolves to an instance of that family."""
        out = sc.normalize_ops(sc.NumpyOps)
        assert isinstance(out, sc.NumpyOps)

    def test_accepts_context_object(self):
        """Passing a ``Context`` returns its ``ops`` instance."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        out = sc.normalize_ops(ctx)
        assert out.family == "numpy"

    def test_pytorch_alias_resolves_to_torch(self):
        if not has_torch():
            pytest.skip("torch is not installed")
        out = sc.normalize_ops("pytorch")
        assert out.family == "torch"

    def test_unknown_string_raises(self):
        with pytest.raises(UnknownBackendError, match="(?i)unknown backend"):
            sc.normalize_ops("definitely_not_a_backend")


# ===========================================================================
# normalize_context
# ===========================================================================
class TestNormalizeContext:
    def test_none_returns_active_default(self, preserve_default_context):
        explicit = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
        sc.set_context(explicit)
        out = sc.normalize_context(None)
        assert out == explicit

    def test_context_returns_equal_context(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        out = sc.normalize_context(ctx)
        assert out == ctx

    def test_family_string_constructs_default_dtype_context(self):
        out = sc.normalize_context("numpy")
        assert out.ops.family == "numpy"
        assert out.dtype == sc.NumpyOps().sanitize_dtype(None)

    def test_family_string_with_explicit_dtype(self):
        out = sc.normalize_context("numpy", dtype=np.float32)
        assert out.dtype == np.dtype(np.float32)

    def test_family_string_with_check_level(self):
        out = sc.normalize_context("numpy", check_level="strict")
        assert out.check_level == "strict"

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError):
            sc.normalize_context(42)

    def test_warns_when_context_provided_with_dtype_override(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        with pytest.warns(UserWarning, match="ignored"):
            sc.normalize_context(ctx, dtype=np.float32)

    def test_warns_when_none_provided_with_dtype_override(self, preserve_default_context):
        with pytest.warns(UserWarning, match="ignored"):
            sc.normalize_context(None, dtype=np.float32)

    def test_rejects_both_check_level_and_enable_checks(self):
        with pytest.raises(TypeError, match="either check_level or enable_checks"):
            sc.normalize_context("numpy", check_level="strict", enable_checks=True)


# ===========================================================================
# register_ops
# ===========================================================================


def _make_ephemeral_backend(family_name: str) -> Type[sc.BackendOps]:
    """Build a minimal NumpyOps subclass with a fresh family name.

    Used by ``register_ops`` tests so each test gets a unique family that
    we then clean up by yanking it back out of the registry. Subclassing
    ``NumpyOps`` rather than ``BackendOps`` avoids re-implementing the
    full ABC contract.
    """

    class _EphemeralOps(sc.NumpyOps):
        _family = family_name

    _EphemeralOps.__name__ = f"_Ephemeral_{family_name}_Ops"
    return _EphemeralOps


class TestRegisterOps:
    def test_register_adds_new_family(self):
        from spacecore._contextual._state import _state

        family = "test_register_adds_new_family_one"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            assert family in _state().available_ops
            assert _state().available_ops[family] is cls
        finally:
            _state().available_ops.pop(family, None)

    def test_registered_family_is_usable_via_set_context(self, preserve_default_context):
        from spacecore._contextual._state import _state

        family = "test_registered_family_is_usable"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            sc.set_context(family)
            assert sc.get_context().ops.family == family
        finally:
            _state().available_ops.pop(family, None)

    def test_duplicate_registration_raises_context_conflict_error(self):
        from spacecore._contextual import ContextConflictError
        from spacecore._contextual._state import _state

        family = "test_duplicate_registration_raises"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            with pytest.raises(ContextConflictError, match="already registered"):
                sc.register_ops(cls)
        finally:
            _state().available_ops.pop(family, None)

    def test_rejects_non_class_argument(self):
        with pytest.raises(TypeError, match="Expected type"):
            sc.register_ops(sc.NumpyOps())  # instance, not class

    def test_rejects_non_backend_subclass(self):
        class NotABackend:
            pass

        with pytest.raises(TypeError, match="Expected type"):
            sc.register_ops(NotABackend)  # type: ignore[arg-type]

    def test_returns_the_registered_class(self):
        from spacecore._contextual._state import _state

        family = "test_returns_the_registered_class"
        cls = _make_ephemeral_backend(family)
        try:
            returned = sc.register_ops(cls)
            assert returned is cls
        finally:
            _state().available_ops.pop(family, None)


# ===========================================================================
# resolve_context_priority — explicit > inferred > default
# ===========================================================================
class TestResolveContextPriority:
    def test_default_used_when_no_inputs(self, preserve_default_context):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
        sc.set_context(ctx)
        out = sc.resolve_context_priority(None)
        assert out == ctx

    def test_explicit_overrides_inferred(self, preserve_default_context):
        sc.set_context(sc.Context(sc.NumpyOps(), dtype=np.float16))
        inferred = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
        explicit = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        X = sc.DenseCoordinateSpace((2,), inferred)
        out = sc.resolve_context_priority(explicit, X)
        assert out == explicit

    def test_inferred_used_when_explicit_is_none(self, preserve_default_context):
        sc.set_context(sc.Context(sc.NumpyOps(), dtype=np.float16))
        inferred = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
        X = sc.DenseCoordinateSpace((2,), inferred)
        out = sc.resolve_context_priority(None, X)
        assert out.dtype == inferred.dtype
        assert out.check_level == inferred.check_level

    def test_default_used_when_no_inferred(self, preserve_default_context):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
        sc.set_context(ctx)
        out = sc.resolve_context_priority(None)
        assert out == ctx

    def test_compatible_inferred_dtypes_are_promoted(self, preserve_default_context):
        """Two compatible inferred contexts (f32 and f64 on numpy) promote
        to the wider precision."""
        a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        Xa = sc.DenseCoordinateSpace((2,), a)
        Xb = sc.DenseCoordinateSpace((3,), b)
        out = sc.resolve_context_priority(None, Xa, Xb)
        assert out.dtype == np.dtype(np.float64)

    def test_minimum_check_level_among_inferred_contexts(self, preserve_default_context):
        strict = sc.Context(sc.NumpyOps(), check_level="strict")
        cheap = sc.Context(sc.NumpyOps(), check_level="cheap")
        Xs = sc.DenseCoordinateSpace((2,), strict)
        Xc = sc.DenseCoordinateSpace((3,), cheap)
        out = sc.resolve_context_priority(None, Xs, Xc)
        assert out.check_level == "cheap"

    def test_incompatible_inferred_contexts_raise(self):
        if not has_jax():
            pytest.skip("jax is not installed")
        np_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
        # Use a dtype JAX honors this session for the JAX side.
        jx_dt = np.float64 if sc.JaxOps().sanitize_dtype(None) == np.float64 else np.float32
        jx_ctx = sc.Context(sc.JaxOps(), dtype=jx_dt)
        Xn = sc.DenseCoordinateSpace((2,), np_ctx)
        Xj = sc.DenseCoordinateSpace((3,), jx_ctx)
        with pytest.raises(ValueError, match="(?i)incompatible inferred"):
            sc.resolve_context_priority(None, Xn, Xj)

    def test_accepts_family_string_as_explicit(self):
        out = sc.resolve_context_priority("numpy")
        assert out.ops.family == "numpy"

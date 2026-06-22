"""Tests for the :mod:`spacecore._contextual._policies` error hierarchy.

Four exception types live in this module:

* :class:`spacecore._contextual.ContextError` — base, subclass of
  ``RuntimeError``;
* :class:`spacecore._contextual.ContextInferenceError` — context cannot be
  inferred from input (typically: ambiguous backend match);
* :class:`spacecore._contextual.ContextConflictError` — contradictory
  registrations or contexts (typically: duplicate ``register_ops``);
* :class:`spacecore._contextual.UnknownBackendError` — a family name that
  was never registered.

Each test pins both the hierarchy (``isinstance`` relationship) and the
condition under which the corresponding error fires.
"""
from __future__ import annotations

from typing import Type

import numpy as np
import pytest

import spacecore as sc
from spacecore._contextual import (
    ContextConflictError,
    ContextError,
    ContextInferenceError,
    UnknownBackendError,
    enforce_convert_policy,
)


# ===========================================================================
# Hierarchy
# ===========================================================================
class TestHierarchy:
    def test_context_error_subclasses_runtime_error(self):
        assert issubclass(ContextError, RuntimeError)

    def test_inference_error_subclasses_context_error(self):
        assert issubclass(ContextInferenceError, ContextError)

    def test_conflict_error_subclasses_context_error(self):
        assert issubclass(ContextConflictError, ContextError)

    def test_unknown_backend_error_subclasses_context_error(self):
        assert issubclass(UnknownBackendError, ContextError)

    def test_each_error_is_instantiable_with_message(self):
        for cls in (ContextError, ContextInferenceError,
                    ContextConflictError, UnknownBackendError):
            inst = cls("test message")
            assert str(inst) == "test message"


# ===========================================================================
# UnknownBackendError: unregistered family
# ===========================================================================
class TestUnknownBackendError:
    def test_normalize_ops_unknown_string(self):
        with pytest.raises(UnknownBackendError, match="(?i)unknown backend"):
            sc.normalize_ops("definitely_not_a_real_backend")

    def test_set_context_unknown_string(self, preserve_default_context):
        with pytest.raises(UnknownBackendError):
            sc.set_context("definitely_not_a_real_backend")

    def test_normalize_context_unknown_string(self):
        with pytest.raises(UnknownBackendError):
            sc.normalize_context("definitely_not_a_real_backend")

    def test_message_lists_available_families(self):
        with pytest.raises(UnknownBackendError) as exc_info:
            sc.normalize_ops("xyz")
        # The error enumerates the registered families so the user can
        # see what they got wrong.
        assert "numpy" in str(exc_info.value)


# ===========================================================================
# ContextConflictError: duplicate register_ops
# ===========================================================================
def _make_ephemeral_backend(family_name: str) -> Type[sc.BackendOps]:
    """Throwaway ``NumpyOps`` subclass with a unique family name."""

    class _EphemeralOps(sc.NumpyOps):
        _family = family_name

    _EphemeralOps.__name__ = f"_Ephemeral_{family_name}_Ops"
    return _EphemeralOps


class TestContextConflictError:
    def test_duplicate_register_ops_raises(self):
        from spacecore._contextual._state import _state

        family = "test_duplicate_register_ops_raises"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            with pytest.raises(ContextConflictError, match="already registered"):
                sc.register_ops(cls)
        finally:
            _state().available_ops.pop(family, None)

    def test_duplicate_message_includes_family(self):
        from spacecore._contextual._state import _state

        family = "test_duplicate_message_includes_family"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            with pytest.raises(ContextConflictError) as exc_info:
                sc.register_ops(cls)
            assert family in str(exc_info.value)
        finally:
            _state().available_ops.pop(family, None)


# ===========================================================================
# ContextInferenceError: ambiguous backend inference
# ===========================================================================
class TestContextInferenceError:
    def test_ambiguous_inference_when_two_backends_claim_an_object(self):
        """Two registered backends both report ``is_array(x) == True``.

        We construct the situation by registering a permissive ``NumpyOps``
        subclass that also claims ``np.ndarray`` as its dense type. With
        both the real numpy and our impostor in the registry, inferring a
        backend for a plain numpy array becomes ambiguous.
        """
        from spacecore._contextual._state import _state

        family = "test_ambiguous_inference_two_claimants"

        class _ClaimsNdarrayOps(sc.NumpyOps):
            _family = family

        _ClaimsNdarrayOps.__name__ = f"_Ephemeral_{family}_Ops"

        x = np.asarray([1.0, 2.0, 3.0])
        try:
            sc.register_ops(_ClaimsNdarrayOps)
            with pytest.raises(ContextInferenceError, match="(?i)ambiguous"):
                _state().infer_context(x)
        finally:
            _state().available_ops.pop(family, None)


# ===========================================================================
# enforce_convert_policy — accept / reject
# ===========================================================================
class TestEnforceConvertPolicy:
    def test_accepts_valid_target_context(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        x = ctx.asarray([1.0, 2.0])
        out_x, out_ctx = enforce_convert_policy(x, ctx)
        assert out_x is x
        assert out_ctx == ctx

    def test_accepts_family_string_target(self):
        x = sc.NumpyOps().asarray([1.0, 2.0])
        _, out_ctx = enforce_convert_policy(x, "numpy")
        assert out_ctx.ops.family == "numpy"

    def test_accepts_none_target_falls_back_to_default(self, preserve_default_context):
        default = sc.Context(sc.NumpyOps(), dtype=np.float32)
        sc.set_context(default)
        x = default.asarray([1.0, 2.0])
        out_x, out_ctx = enforce_convert_policy(x, None)
        assert out_x is x
        assert out_ctx == default

    def test_rejects_unknown_family_target(self):
        x = sc.NumpyOps().asarray([1.0, 2.0])
        with pytest.raises(UnknownBackendError):
            enforce_convert_policy(x, "not_a_real_backend")

    def test_rejects_invalid_target_type(self):
        x = sc.NumpyOps().asarray([1.0, 2.0])
        with pytest.raises(TypeError):
            enforce_convert_policy(x, 42)

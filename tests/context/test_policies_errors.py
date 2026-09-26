"""Tests for the context/backend error hierarchy (:mod:`spacecore._errors`).

The types are defined at the top level because both :mod:`spacecore.backend`
(the ops registry) and :mod:`spacecore.contextual` raise them; they remain
public API of ``spacecore.contextual``, which is how these tests import them.


Book justification: exception design deserves deliberate tests — each error is
raised at its real trigger, sits at the correct (base-catchable) place in the
hierarchy, and carries a useful message (Ousterhout, *A Philosophy of Software
Design*: exceptions add complexity and must be tested and kept a small, stable
set; Myers & Stylos, *Improving API Usability*: help users recognize, diagnose,
and recover from errors).

Four exception types live in this module:

* :class:`spacecore.contextual.ContextError` — base, subclass of
  ``RuntimeError``;
* :class:`spacecore.contextual.ContextInferenceError` — context cannot be
  inferred from input (typically: ambiguous backend match);
* :class:`spacecore.contextual.ContextConflictError` — contradictory
  registrations or contexts (typically: duplicate ``register_ops``);
* :class:`spacecore.contextual.UnknownBackendError` — a family name that
  was never registered.

Each test pins both the hierarchy (``isinstance`` relationship) and the
condition under which the corresponding error fires.
"""
from __future__ import annotations

from typing import Type

import numpy as np
import pytest

import spacecore as sc
from spacecore.backend import ops_registry
from spacecore.contextual import (
    ContextConflictError,
    ContextError,
    ContextInferenceError,
    UnknownBackendError,
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

        family = "test_duplicate_register_ops_raises"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            with pytest.raises(ContextConflictError, match="already registered"):
                sc.register_ops(cls)
        finally:
            ops_registry.unregister(family)

    def test_duplicate_message_includes_family(self):

        family = "test_duplicate_message_includes_family"
        cls = _make_ephemeral_backend(family)
        try:
            sc.register_ops(cls)
            with pytest.raises(ContextConflictError) as exc_info:
                sc.register_ops(cls)
            assert family in str(exc_info.value)
        finally:
            ops_registry.unregister(family)


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

        family = "test_ambiguous_inference_two_claimants"

        class _ClaimsNdarrayOps(sc.NumpyOps):
            _family = family

        _ClaimsNdarrayOps.__name__ = f"_Ephemeral_{family}_Ops"

        from spacecore.contextual._state import _state

        x = np.asarray([1.0, 2.0, 3.0])
        try:
            sc.register_ops(_ClaimsNdarrayOps)
            with pytest.raises(ContextInferenceError, match="(?i)ambiguous"):
                _state().infer_context(x)
        finally:
            ops_registry.unregister(family)



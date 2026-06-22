"""Tests for :class:`spacecore._contextual.ContextBound`.

``ContextBound`` is the abstract base of every object that lives in a
SpaceCore ``Context`` — spaces, linear operators, functionals. The tests
here pin the base-class contract using a small concrete subclass; they do
not exercise any concrete subclass implementation (those live in
``tests/spaces``, ``tests/linops``, ``tests/functional``).

Checklist (per-object section 2):

* ``ctx`` property returns the bound ``Context``
* ``ops`` property delegates to ``ctx.ops``
* ``convert(new_ctx)`` is idempotent for the same context and dispatches to
  ``_convert`` otherwise
* the subclass ``_convert`` hook is invoked
"""
from __future__ import annotations

import abc
from typing import Self

import numpy as np
import pytest

import spacecore as sc
from spacecore._contextual import ContextBound


class _ToyBound(ContextBound):
    """Minimal concrete ``ContextBound`` subclass for the base-class tests.

    Records every ``_convert`` invocation so the test can confirm dispatch.
    """

    def __init__(self, ctx: sc.Context | str | None = None) -> None:
        super().__init__(ctx)
        self._convert_calls: list[sc.Context] = []

    def _convert(self, new_ctx: sc.Context) -> Self:
        self._convert_calls.append(new_ctx)
        new = _ToyBound(new_ctx)
        return new


class _BareBound(ContextBound):
    """``ContextBound`` subclass that does not override ``_convert``."""


# ---------------------------------------------------------------------------
# Identity properties: ctx / ops / dtype / check_level
# ---------------------------------------------------------------------------
class TestProperties:
    def test_ctx_property_returns_bound_context(self):
        """``bound.ctx`` equals the supplied Context.

        Note: ``ContextBound.__init__`` normalizes the input through
        :func:`spacecore.normalize_context`, which returns a fresh
        ``Context`` even when handed one. The check is equality, not
        identity.
        """
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _ToyBound(ctx)
        assert bound.ctx == ctx

    def test_ops_property_delegates_to_ctx(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _ToyBound(ctx)
        assert bound.ops == ctx.ops
        assert bound.ops.family == ctx.ops.family

    def test_dtype_property_delegates_to_ctx(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
        bound = _ToyBound(ctx)
        assert bound.dtype == ctx.dtype

    def test_check_level_property_delegates_to_ctx(self):
        ctx = sc.Context(sc.NumpyOps(), check_level="cheap")
        bound = _ToyBound(ctx)
        assert bound.check_level == "cheap"

    def test_default_init_uses_active_default_context(self, preserve_default_context):
        explicit = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
        sc.set_context(explicit)
        bound = _ToyBound()
        assert bound.ctx == explicit

    def test_init_from_family_string(self):
        bound = _ToyBound("numpy")
        assert bound.ctx.ops.family == "numpy"


# ---------------------------------------------------------------------------
# _checks_at_least: thin wrapper over the check_level dispatch
# ---------------------------------------------------------------------------
class TestChecksAtLeast:
    @pytest.mark.parametrize("current,required,expected", [
        ("none", "none", True),
        ("none", "cheap", False),
        ("none", "standard", False),
        ("none", "strict", False),
        ("cheap", "none", True),
        ("cheap", "cheap", True),
        ("cheap", "standard", False),
        ("cheap", "strict", False),
        ("standard", "cheap", True),
        ("standard", "standard", True),
        ("standard", "strict", False),
        ("strict", "none", True),
        ("strict", "cheap", True),
        ("strict", "standard", True),
        ("strict", "strict", True),
    ])
    def test_truth_table(self, current, required, expected):
        ctx = sc.Context(sc.NumpyOps(), check_level=current)
        bound = _ToyBound(ctx)
        assert bound._checks_at_least(required) is expected

    def test_enable_checks_property_is_legacy_view(self):
        """``ContextBound._enable_checks`` is the legacy bool view of
        ``check_level``: True for anything other than 'none'."""
        for level in ("cheap", "standard", "strict"):
            ctx = sc.Context(sc.NumpyOps(), check_level=level)
            assert _ToyBound(ctx)._enable_checks is True
        ctx = sc.Context(sc.NumpyOps(), check_level="none")
        assert _ToyBound(ctx)._enable_checks is False


# ---------------------------------------------------------------------------
# convert(): idempotency, dispatch, target resolution
# ---------------------------------------------------------------------------
class TestConvert:
    def test_convert_to_same_context_is_identity(self):
        """``convert(same_ctx)`` short-circuits without calling ``_convert``."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _ToyBound(ctx)
        out = bound.convert(ctx)
        assert out is bound
        assert bound._convert_calls == []

    def test_convert_with_none_uses_default_context(self, preserve_default_context):
        """``convert(None)`` resolves through ``normalize_context`` =
        the active default context. When already in that context, returns self."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        sc.set_context(ctx)
        bound = _ToyBound(ctx)
        out = bound.convert(None)
        assert out is bound

    def test_convert_to_different_dtype_dispatches_to__convert(self):
        ctx_a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        ctx_b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _ToyBound(ctx_a)
        out = bound.convert(ctx_b)
        assert out is not bound
        assert bound._convert_calls == [ctx_b]
        assert isinstance(out, _ToyBound)
        assert out.ctx == ctx_b

    def test_convert_accepts_family_string(self):
        """``convert("numpy")`` resolves the string through ``normalize_context``.

        Even though the resulting context is structurally a numpy context,
        check_level differences make it distinct enough to dispatch.
        """
        ctx_strict = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="strict")
        bound = _ToyBound(ctx_strict)
        out = bound.convert("numpy")
        # The default check_level is 'standard' so the contexts differ.
        assert out is not bound
        assert bound._convert_calls and bound._convert_calls[0].ops.family == "numpy"

    def test_convert_round_trip_returns_to_original_ctx(self):
        ctx_a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        ctx_b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _ToyBound(ctx_a)
        mid = bound.convert(ctx_b)
        back = mid.convert(ctx_a)
        assert back.ctx == ctx_a

    def test_default__convert_raises_not_implemented(self):
        """Subclasses that don't override ``_convert`` must raise when invoked."""
        ctx_a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        ctx_b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _BareBound(ctx_a)
        with pytest.raises(NotImplementedError):
            bound.convert(ctx_b)


# ---------------------------------------------------------------------------
# Abstract-base behavior
# ---------------------------------------------------------------------------
class TestAbstractBase:
    def test_context_bound_uses_abcmeta(self):
        """``ContextBound`` is built with ``abc.ABCMeta`` as its metaclass."""
        assert isinstance(ContextBound, abc.ABCMeta)

    def test_bare_subclass_convert_hook_raises_not_implemented(self):
        """The real hook contract: a subclass that does not override
        ``_convert`` raises ``NotImplementedError`` when ``convert`` dispatches
        to a different context.

        ``ContextBound`` has no abstract methods, so instantiation itself is
        allowed; the contract is enforced lazily at the ``_convert`` call site
        (see ``ContextBound._convert`` in ``_bound.py``).
        """
        bound = _BareBound(sc.Context(sc.NumpyOps(), dtype=np.float32))
        target = sc.Context(sc.NumpyOps(), dtype=np.float64)
        with pytest.raises(NotImplementedError):
            bound.convert(target)

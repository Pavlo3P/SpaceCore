"""Tests for :class:`spacecore.contextual.ContextBound`.

Book justification: the abstract base's contract is exercised through a minimal
Test-Specific Subclass rather than a production subclass, isolating the SUT from
concrete implementations (Meszaros, *xUnit Test Patterns*). Property delegation
and ``convert`` dispatch are pinned as a contract (Hunt & Thomas, *The Pragmatic
Programmer*, tip 37), and the ``_checks_at_least`` truth table is a Parameterized
Test to keep one reported concern per case (avoid Assertion Roulette).

``ContextBound`` is the abstract base of every object that lives in a
SpaceCore ``Context`` — spaces, linear operators, functionals. The tests
here pin the base-class contract using a small concrete subclass; they do
not exercise any concrete subclass implementation (those live in
``tests/spaces``, ``tests/linops``, ``tests/functional``).

Checklist (per-object section 2):

* ``ctx`` property returns the bound ``Context``
* ``ops`` property delegates to ``ctx.ops``
* ``check_level`` is a property of the *object*, seeded at construction and
  independent of the backend/dtype context it is bound to
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
from spacecore.contextual import ContextBound


class _ToyBound(ContextBound):
    """Minimal concrete ``ContextBound`` subclass for the base-class tests.

    Records every ``_convert`` invocation so the test can confirm dispatch.
    """

    def __init__(
        self,
        ctx: sc.Context | str | None = None,
        check_level: sc.CheckLevel | None = None,
    ) -> None:
        super().__init__(ctx, check_level)
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

    def test_check_level_property_is_seeded_from_the_constructor(self):
        ctx = sc.Context(sc.NumpyOps())
        bound = _ToyBound(ctx, check_level="cheap")
        assert bound.check_level == "cheap"

    def test_check_level_defaults_to_the_ambient_level(self):
        ctx = sc.Context(sc.NumpyOps())
        with sc.use_check_level("strict"):
            bound = _ToyBound(ctx)
        assert bound.check_level == "strict"

    def test_check_level_is_independent_of_the_context(self):
        """Two objects on one context may validate at different strictness, and
        they still share a math context."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        cheap = _ToyBound(ctx, check_level="cheap")
        strict = _ToyBound(ctx, check_level="strict")
        assert cheap.check_level == "cheap"
        assert strict.check_level == "strict"
        assert cheap.same_math(strict) is True

    def test_default_init_uses_active_default_context(self, preserve_default_context):
        explicit = sc.Context(sc.NumpyOps(), dtype=np.float32)
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
        ctx = sc.Context(sc.NumpyOps())
        bound = _ToyBound(ctx, check_level=current)
        assert bound._checks_at_least(required) is expected

    def test_enable_checks_property_is_legacy_view(self):
        """``ContextBound._enable_checks`` is the legacy bool view of
        ``check_level``: True for anything other than 'none'."""
        ctx = sc.Context(sc.NumpyOps())
        for level in ("cheap", "standard", "strict"):
            assert _ToyBound(ctx, check_level=level)._enable_checks is True
        assert _ToyBound(ctx, check_level="none")._enable_checks is False


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

        The family string carries no dtype, so it resolves to the backend's
        default (float64) — distinct from the float32 context below, hence a
        real dispatch rather than the identity short-circuit.
        """
        ctx_f32 = sc.Context(sc.NumpyOps(), dtype=np.float32)
        bound = _ToyBound(ctx_f32)
        out = bound.convert("numpy")
        assert out is not bound
        assert bound._convert_calls and bound._convert_calls[0].ops.family == "numpy"
        assert out.dtype == sc.NumpyOps().sanitize_dtype(None)

    def test_convert_preserves_the_objects_check_level(self):
        """``check_level`` is a property of the object, not of the context, so
        it survives a conversion onto a different context."""
        ctx_a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        ctx_b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        bound = _ToyBound(ctx_a, check_level="strict")
        out = bound.convert(ctx_b)
        assert out.check_level == "strict"

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

"""Tests for :class:`spacecore.Space` — the abstract base of all spaces.

Checklist item 1:

* ``Space.field`` derivation from ``ctx.dtype``
* ``check_member`` dispatch at every check level
* ``_convert`` hook contract (idempotency when ctx matches, dispatch otherwise)
* ``__eq__`` shallow contract — base ``Space.__eq__`` returns NotImplemented
* ``member_checks`` property — composition of instance ``_local_checks`` +
  any class-level ``member_checks``
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import spacecore as sc


class _FiniteSetSpace(sc.Space):
    """Minimal concrete ``Space`` for the base-class contract tests."""

    def __init__(self, values: set[Any], ctx=None) -> None:
        super().__init__(ctx)
        self.values = values

    def _check_member(self, x: Any) -> None:
        if x not in self.values:
            raise ValueError("not a member")

    def _convert(self, new_ctx) -> "_FiniteSetSpace":
        return _FiniteSetSpace(self.values, new_ctx)


# ===========================================================================
# field: derivation from ctx.dtype
# ===========================================================================
class TestField:
    @pytest.mark.parametrize("dtype, expected", [
        (np.float32, "real"),
        (np.float64, "real"),
        (np.complex64, "complex"),
        (np.complex128, "complex"),
    ])
    def test_field_derived_from_dtype(self, dtype, expected):
        ctx = sc.Context(sc.NumpyOps(), dtype=dtype)
        space = sc.DenseCoordinateSpace((2,), ctx)
        assert space.field == expected

    def test_field_updates_on_convert(self, numpy_ctx, numpy_complex_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        assert space.field == "real"
        converted = space.convert(numpy_complex_ctx)
        assert converted.field == "complex"


# ===========================================================================
# check_member: dispatch at every check level
# ===========================================================================
class TestCheckMember:
    def test_none_skips_membership(self):
        ctx = sc.Context(sc.NumpyOps(), check_level="none")
        space = _FiniteSetSpace({"a", "b"}, ctx)
        # "c" is not a member but ``none`` skips ``_check_member``.
        space.check_member("c")

    def test_membership_runs_at_standard(self):
        ctx = sc.Context(sc.NumpyOps(), check_level="standard")
        space = _FiniteSetSpace({"a", "b"}, ctx)
        space.check_member("a")
        with pytest.raises(ValueError, match="not a member"):
            space.check_member("c")

    def test_membership_runs_at_strict(self):
        ctx = sc.Context(sc.NumpyOps(), check_level="strict")
        space = _FiniteSetSpace({"a", "b"}, ctx)
        with pytest.raises(ValueError, match="not a member"):
            space.check_member("c")


# ===========================================================================
# _convert: hook contract
# ===========================================================================
class TestConvert:
    def test_convert_to_same_ctx_returns_self(self, numpy_ctx):
        space = _FiniteSetSpace({"a", "b"}, numpy_ctx)
        out = space.convert(numpy_ctx)
        assert out is space

    def test_convert_to_different_ctx_dispatches_to__convert(self, numpy_ctx, numpy_f32_ctx):
        space = _FiniteSetSpace({"a", "b"}, numpy_ctx)
        out = space.convert(numpy_f32_ctx)
        assert out is not space
        assert isinstance(out, _FiniteSetSpace)
        assert out.values == space.values
        assert out.ctx == numpy_f32_ctx

    def test_convert_round_trip_preserves_state(self, numpy_ctx, numpy_f32_ctx):
        space = _FiniteSetSpace({"a", "b"}, numpy_ctx)
        roundtrip = space.convert(numpy_f32_ctx).convert(numpy_ctx)
        assert roundtrip.ctx == space.ctx
        assert roundtrip.values == space.values

    def test_convert_accepts_family_string(self):
        space = _FiniteSetSpace({"a"}, sc.Context(sc.NumpyOps(), check_level="strict"))
        out = space.convert("numpy")
        # 'numpy' resolves through normalize_context; default check_level differs.
        assert isinstance(out, _FiniteSetSpace)


# ===========================================================================
# __eq__: shallow contract
# ===========================================================================
class TestEquality:
    def test_base_space_eq_returns_not_implemented_for_non_space(self, numpy_ctx):
        space = _FiniteSetSpace({"a"}, numpy_ctx)
        # ``Space.__eq__`` only defines equality with other ``Space`` instances.
        # Comparison with non-Space returns False without raising.
        assert (space == "foo") is False
        assert (space == 42) is False

    def test_eq_is_symmetric_for_distinct_space_types(self, numpy_ctx):
        """``DenseCoordinateSpace`` ≠ ``HermitianSpace`` from either side."""
        a = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        b = sc.HermitianSpace(2, ctx=numpy_ctx)
        assert (a == b) is False
        assert (b == a) is False
        assert (a == b) == (b == a)


# ===========================================================================
# member_checks: instance + class composition
# ===========================================================================
class TestMemberChecks:
    def test_default_member_checks_returns_tuple(self, numpy_ctx):
        """``Space.member_checks()`` returns a tuple of ``SpaceCheck``."""
        space = _FiniteSetSpace({"a"}, numpy_ctx)
        assert isinstance(space.member_checks(), tuple)

    def test_dense_coordinate_space_has_class_level_checks(self, numpy_ctx):
        """``DenseCoordinateSpace._local_checks`` adds shape+dtype+backend."""
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        checks = space.member_checks()
        assert len(checks) > 0
        # The check tuple includes shape + dtype + backend in some order.
        kinds = [type(c).__name__ for c in checks]
        assert "ShapeCheck" in kinds

"""Tests for the ``spacecore.space.checks`` package.

Checklist items 22–30:

22. :class:`SpaceValidationError` — subclass of ``ValueError`` and ``TypeError``
    (message format, error-class identity).
23. :class:`SpaceCheck` abstract — ``is_valid`` / ``error_message`` / ``validate``
    contract; ``__call__`` short-circuit to ``validate``.
24. :class:`BackendCheck` — accepts matching family, rejects others.
25. :class:`DTypeCheck` — accepts exact dtype, rejects mismatch.
26. :class:`FieldCheck` — accepts real/complex consistent with ``Space.field``.
27. :class:`ShapeCheck` — accepts shape; ``allow_leading=True`` for batched.
28. :class:`HermitianCheck` — accepts Hermitian, rejects non-Hermitian; tunable
    ``atol`` / ``rtol`` / ``enforce`` knobs.
29. :class:`SquareMatrixCheck` — accepts square, rejects rectangular.
30. ``_run_checks`` — orders failures and surfaces the first; honors
    ``minimum_level`` per check (gap-fill).

Plus the tree-leaf-path check tests (``_TreeStructureCheck`` / ``_TreeLeafCheck``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import spacecore as sc
from spacecore.space.checks._base import _run_checks


# ===========================================================================
# 22. SpaceValidationError — hierarchy + message
# ===========================================================================
class TestSpaceValidationError:
    def test_subclasses_both_value_and_type_error(self):
        """``SpaceValidationError`` is catchable as either parent."""
        assert issubclass(sc.SpaceValidationError, ValueError)
        assert issubclass(sc.SpaceValidationError, TypeError)

    def test_message_round_trip(self):
        err = sc.SpaceValidationError("custom message")
        assert str(err) == "custom message"


# ===========================================================================
# 23. SpaceCheck abstract base — validate / __call__ / error_message
# ===========================================================================
@dataclass(frozen=True)
class _AlwaysFailCheck(sc.SpaceCheck):
    name: str = "always_fail"

    def is_valid(self, space, x):  # noqa: ARG002
        return False

    def error_message(self, space, x):  # noqa: ARG002
        return "always fails"


@dataclass(frozen=True)
class _AlwaysPassCheck(sc.SpaceCheck):
    name: str = "always_pass"

    def is_valid(self, space, x):  # noqa: ARG002
        return True

    def error_message(self, space, x):  # noqa: ARG002
        return "never raised"


class TestSpaceCheckAbstract:
    def test_check_call_raises_on_invalid(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(sc.SpaceValidationError, match="always fails"):
            _AlwaysFailCheck()(space, numpy_ctx.asarray([1.0, 2.0]))

    def test_check_call_silent_on_valid(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        _AlwaysPassCheck()(space, numpy_ctx.asarray([1.0, 2.0]))  # no raise

    def test_validate_returns_bool(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        assert _AlwaysFailCheck().validate(space, None, allow_leading=False) is False
        assert _AlwaysPassCheck().validate(space, None, allow_leading=False) is True


# ===========================================================================
# 24. BackendCheck — family-aware accept/reject
# ===========================================================================
class TestBackendCheck:
    def test_accepts_native_dense_array(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        sc.BackendCheck()(space, numpy_ctx.asarray([1.0, 2.0]))  # no raise

    def test_rejects_plain_list(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(sc.SpaceValidationError, match="Expected dense array for numpy"):
            sc.BackendCheck()(space, [1.0, 2.0])

    def test_minimum_level_is_cheap(self):
        assert sc.BackendCheck.minimum_level == "cheap"


# ===========================================================================
# 25. DTypeCheck — exact-match
# ===========================================================================
class TestDTypeCheck:
    def test_accepts_matching_dtype(self, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        sc.DTypeCheck()(space, numpy_f32_ctx.asarray([1.0, 2.0]))

    def test_rejects_mismatched_dtype(self, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        with pytest.raises(sc.SpaceValidationError,
                           match="Expected dtype float32, got float64"):
            sc.DTypeCheck()(space, np.asarray([1.0, 2.0], dtype=np.float64))


# ===========================================================================
# 26. FieldCheck — real/complex consistent with Space.field
# ===========================================================================
class TestFieldCheck:
    def test_real_field_accepts_real_input(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        assert space.field == "real"
        sc.FieldCheck()(space, numpy_ctx.asarray([1.0, 2.0]))

    def test_real_field_rejects_complex_input(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(sc.SpaceValidationError, match="real scalar field"):
            sc.FieldCheck()(space, np.asarray([1 + 0j, 2 + 0j], dtype=np.complex128))

    def test_complex_field_accepts_complex_input(self, numpy_complex_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_complex_ctx)
        sc.FieldCheck()(space, numpy_complex_ctx.asarray([1 + 0j, 2 + 1j]))

    def test_complex_field_accepts_real_input(self, numpy_complex_ctx):
        """A complex-field space happily takes a real input — it just broadens."""
        space = sc.DenseCoordinateSpace((2,), numpy_complex_ctx)
        sc.FieldCheck()(space, np.asarray([1.0, 2.0], dtype=np.float64))


# ===========================================================================
# 27. ShapeCheck — accept/reject, allow_leading for batched
# ===========================================================================
class TestShapeCheck:
    def test_accepts_canonical_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        sc.ShapeCheck()(space, numpy_ctx.asarray([1.0, 2.0, 3.0]))

    def test_rejects_wrong_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        with pytest.raises(sc.SpaceValidationError,
                           match=r"Expected shape \(2,\), got \(3,\)"):
            sc.ShapeCheck()(space, numpy_ctx.asarray([1.0, 2.0, 3.0]))

    def test_allow_leading_accepts_batched(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        # Batched: leading axis is the batch axis, trailing matches shape.
        batched = numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert sc.ShapeCheck().validate(space, batched, allow_leading=True) is True

    def test_allow_leading_false_rejects_batched(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        batched = numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert sc.ShapeCheck().validate(space, batched, allow_leading=False) is False

    def test_batched_message_mentions_trailing_shape(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        bad = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
        msg = sc.ShapeCheck().validation_message(space, bad, allow_leading=True)
        assert "trailing shape" in msg


# ===========================================================================
# 28. HermitianCheck — tolerances and enforce flag
# ===========================================================================
class TestHermitianCheck:
    def test_accepts_hermitian(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        H = numpy_ctx.asarray([[1.0, 2.0], [2.0, 3.0]])
        sc.HermitianCheck()(space, H)

    def test_rejects_non_hermitian(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        bad = numpy_ctx.asarray([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(sc.SpaceValidationError, match="not Hermitian"):
            sc.HermitianCheck()(space, bad)

    def test_atol_loosens_tolerance(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        almost = numpy_ctx.asarray([[1.0, 1.0], [1.0 + 1e-5, 1.0]])
        assert not sc.HermitianCheck(atol=0.0, rtol=0.0).is_valid(space, almost)
        assert sc.HermitianCheck(atol=1e-4, rtol=0.0).is_valid(space, almost)

    def test_enforce_false_short_circuits(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        bad = numpy_ctx.asarray([[1.0, 2.0], [0.0, 1.0]])
        assert sc.HermitianCheck(enforce=False).is_valid(space, bad) is True


# ===========================================================================
# 29. SquareMatrixCheck — square/rectangular
# ===========================================================================
class TestSquareMatrixCheck:
    def test_accepts_square(self, numpy_ctx):
        space = sc.HermitianSpace(3, ctx=numpy_ctx)
        sc.SquareMatrixCheck()(space, numpy_ctx.asarray(np.zeros((3, 3))))

    def test_rejects_rectangular(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        with pytest.raises(sc.SpaceValidationError,
                           match=r"Expected square matrix, got shape \(2, 3\)"):
            sc.SquareMatrixCheck()(space, numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


# ===========================================================================
# 30. _run_checks — orders failures and short-circuits on the first
# (Gap-fill)
# ===========================================================================
@dataclass(frozen=True)
class _CountingFailCheck(sc.SpaceCheck):
    """A failing check that records how many times its ``is_valid`` was called."""

    name: str = "counting_fail"
    counter: list = None  # type: ignore[assignment]

    def is_valid(self, space, x):  # noqa: ARG002
        if self.counter is not None:
            self.counter.append(self.name)
        return False

    def error_message(self, space, x):  # noqa: ARG002
        return f"{self.name} failed"


class _OrderedFailureSpace(sc.DenseCoordinateSpace):
    """Custom space that registers two failing checks in a known order."""

    def __init__(self, shape, ctx, *, counter):
        super().__init__(shape, ctx)
        self._counter = counter

    def _local_checks(self):
        first = _CountingFailCheck("first_fail", self._counter)
        second = _CountingFailCheck("second_fail", self._counter)
        return (first, second)


class TestRunChecksOrdering:
    def test_first_failure_short_circuits(self, numpy_ctx):
        """``_run_checks`` raises on the first failing check; later checks
        are never invoked."""
        counter: list[str] = []
        space = _OrderedFailureSpace((2,), numpy_ctx, counter=counter)
        x = numpy_ctx.asarray([1.0, 2.0])
        # The shape/dtype/backend class-level checks pass; only the two
        # _local_checks fail.
        with pytest.raises(sc.SpaceValidationError, match="first_fail failed"):
            _run_checks(space, x, allow_leading=False)
        assert counter == ["first_fail"]

    def test_passing_checks_run_in_full(self, numpy_ctx):
        """When every check passes, no exception, all called."""
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0])
        _run_checks(space, x, allow_leading=False)

    def test_below_minimum_level_check_is_skipped(self):
        """A check whose ``minimum_level`` exceeds the active level is skipped."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")

        @dataclass(frozen=True)
        class _StrictOnlyAlwaysFail(sc.SpaceCheck):
            name: str = "strict_only_fail"
            from typing import ClassVar as _CV
            minimum_level: _CV[str] = "strict"  # type: ignore[assignment]

            def is_valid(self, space, x):  # noqa: ARG002
                return False

            def error_message(self, space, x):  # noqa: ARG002
                return "should never raise"

        class _SpaceWithStrictCheck(sc.DenseCoordinateSpace):
            def _local_checks(self):
                return (_StrictOnlyAlwaysFail(),)

        space = _SpaceWithStrictCheck((2,), ctx)
        # check_level=cheap < strict ⇒ the strict-only check is skipped.
        _run_checks(space, ctx.asarray([1.0, 2.0]), allow_leading=False)


# ===========================================================================
# _TreeStructureCheck / _TreeLeafCheck — pytree-structured spaces
# ===========================================================================
class TestTreeChecks:
    def test_tree_structure_check_rejects_mismatched_arity(self, numpy_ctx):
        product = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((2,), numpy_ctx),
             sc.DenseCoordinateSpace((3,), numpy_ctx)),
            numpy_ctx,
        )
        with pytest.raises(sc.SpaceValidationError, match="structure mismatch"):
            product.check_member((numpy_ctx.asarray([1.0, 2.0]),))

    def test_tree_leaf_check_includes_leaf_path(self, numpy_ctx):
        product = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((2,), numpy_ctx),
             sc.DenseCoordinateSpace((3,), numpy_ctx)),
            numpy_ctx,
        )
        with pytest.raises(sc.SpaceValidationError, match=r"\$\[1\].*Expected shape \(3,\)"):
            product.check_member(
                (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0, 4.0]))
            )


# ===========================================================================
# Instance _local_checks + subclass class-level ``checks`` composition
# ===========================================================================
@dataclass(frozen=True)
class _RejectFirstEntryCheck(sc.SpaceCheck):
    value: float = 0.0

    def is_valid(self, space, x):  # noqa: ARG002
        return bool(x[0] != self.value)

    def error_message(self, space, x):  # noqa: ARG002
        return f"First entry must not be {self.value}."


class TestCheckComposition:
    def test_subclass_class_level_checks_appended(self, numpy_ctx):
        class _ParentSpace(sc.DenseCoordinateSpace):
            checks = (_RejectFirstEntryCheck("parent_reject", 1.0),)

        class _ChildSpace(_ParentSpace):
            checks = (_RejectFirstEntryCheck("child_reject", 2.0),)

        space = _ChildSpace((1,), numpy_ctx)
        names = [check.name for check in space.member_checks()]
        assert "parent_reject" in names
        assert "child_reject" in names

        with pytest.raises(ValueError, match="1.0"):
            space.check_member(numpy_ctx.asarray([1.0]))
        with pytest.raises(ValueError, match="2.0"):
            space.check_member(numpy_ctx.asarray([2.0]))

    def test_instance_local_checks_extend_inherited(self, numpy_ctx):
        class _Parameterized(sc.DenseCoordinateSpace):
            def __init__(self, shape, reject_value, ctx=None):
                super().__init__(shape, ctx)
                self.reject_value = reject_value

            def _local_checks(self):
                return (_RejectFirstEntryCheck("instance_reject", self.reject_value),)

        space = _Parameterized((1,), 3.0, numpy_ctx)
        assert "instance_reject" in [c.name for c in space.member_checks()]
        with pytest.raises(ValueError, match="3.0"):
            space.check_member(numpy_ctx.asarray([3.0]))

    def test_disabled_context_skips_local_checks(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")

        class _AlwaysRejecting(sc.DenseCoordinateSpace):
            checks = (_RejectFirstEntryCheck("child_reject", 0.0),)

        space = _AlwaysRejecting((1,), ctx)
        # check_level=none silences class-level checks too.
        space.check_member(ctx.asarray([0.0]))


# ===========================================================================
# HermitianSpace knob round-trip
# ===========================================================================
class TestHermitianSpaceKnobs:
    def test_atol_rtol_enforce_propagate_to_member_check(self, numpy_ctx):
        loose = sc.HermitianSpace(2, atol=1e-4, rtol=0.0, ctx=numpy_ctx)
        disabled = sc.HermitianSpace(2, enforce_herm=False, ctx=numpy_ctx)
        loose_check = next(
            c for c in loose.member_checks() if isinstance(c, sc.HermitianCheck)
        )
        disabled_check = next(
            c for c in disabled.member_checks() if isinstance(c, sc.HermitianCheck)
        )
        assert loose_check.atol == 1e-4
        assert loose_check.rtol == 0.0
        assert loose_check.enforce is True
        assert disabled_check.enforce is False

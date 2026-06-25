"""Tests for :func:`spacecore.checked_method`.

The decorator wraps a method so that selected positional arguments are
validated against an input space and the return value against an output
space, gated by the receiver's ``check_level`` policy.

Checklist section 3:

* Input/output validation routes each checked argument and the result
  through the relevant space's ``_check_member`` and raises on invalid
  members (``in_space``/``out_space``/``"self"`` targets, ``value``/``grad``
  variants).
* Enable/disable: a receiver with ``check_level == "none"`` (or the legacy
  ``_enable_checks == False`` fallback) skips all validation, while
  ``"cheap"``/``"standard"``/``"strict"`` take the validated path.
* Metadata: ``functools.wraps`` preserves ``__name__``, ``__doc__``, and
  exposes ``__wrapped__``.
* Argument positions: ``arg_positions`` selects multiple positional args,
  the deprecated ``arg_pos`` alias still selects a single one, and supplying
  both raises ``TypeError``.
* Batched: ``in_batched``/``out_batched`` validate leading-axis batches via
  :func:`spacecore._batching._check_batched`.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore import checked_method


class _RecordingSpace:
    def __init__(self, valid):
        self.valid = valid
        self.calls = []

    def _check_member(self, value):
        self.calls.append(value)
        if value != self.valid:
            raise ValueError(f"invalid member: {value!r}")


class _CheckedDemo:
    """Legacy receiver exposing only ``_enable_checks`` (real fallback path)."""

    def __init__(self, enable_checks=True):
        self._enable_checks = enable_checks
        self.dom = _RecordingSpace("x")
        self.cod = _RecordingSpace("y")
        self.space = _RecordingSpace("z")
        self.apply_result = "y"
        self.rapply_result = "x"
        self.value_result = 1.0
        self.grad_result = "z"

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x):
        """Apply docstring."""
        return self.apply_result

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y):
        return self.rapply_result

    @checked_method(in_space="space")
    def value(self, x):
        return self.value_result

    @checked_method(in_space="space", out_space="space")
    def grad(self, x):
        return self.grad_result

    @checked_method(in_space="self", arg_positions=(0, 1))
    def combine(self, x, y):
        return "combined"

    @checked_method(in_space="self", arg_pos=0)
    def legacy_single_arg(self, x):
        return "legacy"

    def _check_member(self, value):
        self.space._check_member(value)


class _ModernDemo:
    """Modern receiver exposing the ``check_level`` attribute path."""

    def __init__(self, check_level):
        self.check_level = check_level
        self.dom = _RecordingSpace("x")
        self.cod = _RecordingSpace("y")
        self.apply_result = "y"

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x):
        return self.apply_result


class _BatchedDemo:
    """Receiver using real spaces so the ``_check_batched`` branch fires."""

    def __init__(self, check_level="cheap"):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)
        self.ctx = ctx
        self.dom = sc.DenseCoordinateSpace((3,), ctx)
        self.cod = sc.DenseCoordinateSpace((3,), ctx)
        # The decorator reads ``self.check_level`` for gating.
        self.check_level = check_level
        self.out_result = ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @checked_method(in_space="dom", out_space="cod", in_batched=True, out_batched=True)
    def vapply(self, xs):
        return self.out_result


# ===========================================================================
# Input / output validation
# ===========================================================================
class TestInputOutputValidation:
    def test_validates_apply_input_and_output(self):
        demo = _CheckedDemo()

        assert demo.apply("x") == "y"
        assert demo.dom.calls == ["x"]
        assert demo.cod.calls == ["y"]

    def test_validates_rapply_input_and_output(self):
        demo = _CheckedDemo()

        assert demo.rapply("y") == "x"
        assert demo.cod.calls == ["y"]
        assert demo.dom.calls == ["x"]

    def test_validates_value_input(self):
        demo = _CheckedDemo()

        assert demo.value("z") == 1.0
        assert demo.space.calls == ["z"]

    def test_validates_grad_input_and_output(self):
        demo = _CheckedDemo()

        assert demo.grad("z") == "z"
        assert demo.space.calls == ["z", "z"]

    def test_invalid_input_raises_when_enabled(self):
        demo = _CheckedDemo(enable_checks=True)

        with pytest.raises(ValueError, match="invalid member"):
            demo.apply("bad")

    def test_invalid_output_raises_when_enabled(self):
        demo = _CheckedDemo(enable_checks=True)
        demo.apply_result = "bad"

        with pytest.raises(ValueError, match="invalid member"):
            demo.apply("x")


# ===========================================================================
# Enable / disable
# ===========================================================================
class TestEnableDisable:
    def test_skips_checks_when_legacy_disabled(self):
        demo = _CheckedDemo(enable_checks=False)
        demo.apply_result = "bad"

        assert demo.apply("bad") == "bad"
        assert demo.dom.calls == []
        assert demo.cod.calls == []

    @pytest.mark.parametrize("level", ["cheap", "standard", "strict"])
    def test_modern_check_level_takes_validated_path(self, level):
        demo = _ModernDemo(check_level=level)

        assert demo.apply("x") == "y"
        assert demo.dom.calls == ["x"]
        assert demo.cod.calls == ["y"]

    @pytest.mark.parametrize("level", ["cheap", "standard", "strict"])
    def test_modern_check_level_raises_on_invalid(self, level):
        demo = _ModernDemo(check_level=level)

        with pytest.raises(ValueError, match="invalid member"):
            demo.apply("bad")

    def test_modern_check_level_none_skips_validation(self):
        demo = _ModernDemo(check_level="none")
        demo.apply_result = "bad"

        assert demo.apply("bad") == "bad"
        assert demo.dom.calls == []
        assert demo.cod.calls == []


# ===========================================================================
# Metadata
# ===========================================================================
class TestMetadata:
    def test_preserves_metadata(self):
        assert _CheckedDemo.apply.__name__ == "apply"
        assert _CheckedDemo.apply.__doc__ == "Apply docstring."
        assert _CheckedDemo.apply.__wrapped__ is not None


# ===========================================================================
# Argument positions
# ===========================================================================
class TestArgPositions:
    def test_supports_self_target_and_multiple_input_args(self):
        demo = _CheckedDemo()

        assert demo.combine("z", "z") == "combined"
        assert demo.space.calls == ["z", "z"]

    def test_arg_pos_alias_still_works(self):
        demo = _CheckedDemo()

        assert demo.legacy_single_arg("z") == "legacy"
        assert demo.space.calls == ["z"]

    def test_rejects_arg_pos_and_arg_positions_together(self):
        with pytest.raises(TypeError, match="arg_pos"):
            checked_method(in_space="space", arg_pos=0, arg_positions=(0,))


# ===========================================================================
# Batched
# ===========================================================================
class TestBatched:
    def test_batched_input_and_output_pass_for_valid_batch(self):
        demo = _BatchedDemo()

        out = demo.vapply(demo.ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(out, demo.out_result)

    def test_batched_input_validation_rejects_wrong_trailing_shape(self):
        demo = _BatchedDemo()

        with pytest.raises(sc.SpaceValidationError, match="trailing shape"):
            demo.vapply(demo.ctx.asarray([[1.0, 2.0], [3.0, 4.0]]))

    def test_batched_output_validation_rejects_wrong_trailing_shape(self):
        demo = _BatchedDemo()
        demo.out_result = demo.ctx.asarray([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(sc.SpaceValidationError, match="trailing shape"):
            demo.vapply(demo.ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_batched_skipped_when_check_level_none(self):
        demo = _BatchedDemo(check_level="none")
        bad = demo.ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
        demo.out_result = bad

        assert demo.vapply(bad) is bad

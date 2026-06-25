"""Tests for the algebra-factory functions in :mod:`spacecore.linop._algebra`.

Checklist item 3:

* Private scalar helpers — ``is_scalar_like``, ``_conjugate_scalar``,
  ``_scalar_equal``, ``_is_zero_scalar``, ``_is_one_scalar`` truth tables.
* ``make_sum`` — non-empty requirement, flattens nested ``SumLinOp``, drops
  ``ZeroLinOp`` terms, returns ``ZeroLinOp`` for all-zero, returns the
  single survivor when one term remains, raises on mismatched
  domain/codomain or context.
* ``make_scaled`` — ``0*A → ZeroLinOp``, ``1*A → A``, ``-1*A → ScaledLinOp(-1, A)``,
  nested scaling folds scalars, rejects non-scalar values.
* ``make_composed`` — ``I @ A == A``, ``A @ I == A``, ``Zero @ A == Zero``,
  ``A @ Zero == Zero``, rejects mismatched middle space or context.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.linop._algebra import (
    _conjugate_scalar,
    _is_one_scalar,
    _is_zero_scalar,
    _scalar_equal,
    is_scalar_like,
    make_composed,
    make_scaled,
    make_sum,
)


# ===========================================================================
# Private scalar helpers — truth tables
# ===========================================================================
class TestIsScalarLike:
    @pytest.mark.parametrize("value, expected", [
        (1, True),
        (1.5, True),
        (1 + 2j, True),
        (np.float64(3.14), True),
        (np.asarray(2.0), True),       # 0-d array
        (np.asarray([1.0, 2.0]), False),  # 1-d array
        ([1.0], False),
        ("scalar", False),
    ])
    def test_truth_table(self, value, expected):
        assert is_scalar_like(value) is expected


class TestConjugateScalar:
    def test_complex_returns_conjugate(self):
        assert _conjugate_scalar(2 + 3j) == 2 - 3j

    def test_real_returns_unchanged(self):
        assert _conjugate_scalar(3.14) == 3.14

    def test_numpy_complex(self):
        assert _conjugate_scalar(np.complex128(1 + 1j)) == np.complex128(1 - 1j)


class TestScalarEqual:
    @pytest.mark.parametrize("value, target, expected", [
        (0, 0, True),
        (0.0, 0, True),
        (1, 1, True),
        (2, 1, False),
        (np.float64(0.0), 0, True),
    ])
    def test_truth_table(self, value, target, expected):
        assert _scalar_equal(value, target) is expected

    def test_returns_false_on_exception(self):
        """``_scalar_equal`` swallows exceptions and returns False."""
        class _Bad:
            def __eq__(self, other):
                raise RuntimeError("boom")

        assert _scalar_equal(_Bad(), 0) is False


class TestIsZeroScalar:
    @pytest.mark.parametrize("value, expected", [
        (0, True),
        (0.0, True),
        (1, False),
        (np.float64(0.0), True),
    ])
    def test_truth_table(self, value, expected):
        assert _is_zero_scalar(value) is expected


class TestIsOneScalar:
    @pytest.mark.parametrize("value, expected", [
        (1, True),
        (1.0, True),
        (0, False),
        (2, False),
        (np.float64(1.0), True),
    ])
    def test_truth_table(self, value, expected):
        assert _is_one_scalar(value) is expected


# ===========================================================================
# make_sum
# ===========================================================================
class TestMakeSum:
    def _id(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        return sc.IdentityLinOp(X, numpy_ctx)

    def _zero(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        return sc.ZeroLinOp(X, X, numpy_ctx)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="nonempty"):
            make_sum(())

    def test_single_nonzero_term_returned_directly(self, numpy_ctx):
        op = self._id(numpy_ctx)
        assert make_sum((op,)) is op

    def test_all_zero_terms_returns_zero(self, numpy_ctx):
        z = self._zero(numpy_ctx)
        out = make_sum((z, z))
        assert isinstance(out, sc.ZeroLinOp)

    def test_drops_zero_terms(self, numpy_ctx):
        op = self._id(numpy_ctx)
        z = self._zero(numpy_ctx)
        out = make_sum((op, z))
        # Only the non-zero term remains; not a SumLinOp.
        assert out is op

    def test_flattens_nested_sums(self, numpy_ctx):
        a = self._id(numpy_ctx)
        b = self._id(numpy_ctx)
        c = self._id(numpy_ctx)
        nested = make_sum((make_sum((a, b)), c))
        # The flattened form is a SumLinOp with 3 parts.
        assert isinstance(nested, sc.SumLinOp)
        assert len(nested.parts) == 3

    def test_mismatched_domain_raises(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op2 = sc.IdentityLinOp(X, numpy_ctx)
        op3 = sc.IdentityLinOp(Y, numpy_ctx)
        with pytest.raises(ValueError, match="same domain and codomain"):
            make_sum((op2, op3))


# ===========================================================================
# make_scaled
# ===========================================================================
class TestMakeScaled:
    def _id(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        return sc.IdentityLinOp(X, numpy_ctx)

    def test_zero_scalar_returns_zero(self, numpy_ctx):
        op = self._id(numpy_ctx)
        out = make_scaled(0.0, op)
        assert isinstance(out, sc.ZeroLinOp)

    def test_one_scalar_returns_original(self, numpy_ctx):
        op = self._id(numpy_ctx)
        assert make_scaled(1.0, op) is op

    def test_minus_one_yields_scaled_linop(self, numpy_ctx):
        op = self._id(numpy_ctx)
        out = make_scaled(-1, op)
        assert isinstance(out, sc.ScaledLinOp)
        assert out.scalar == -1

    def test_nested_scaling_folds(self, numpy_ctx):
        op = self._id(numpy_ctx)
        out = make_scaled(2.0, make_scaled(3.0, op))
        # Nested ScaledLinOp folds: 2 * (3 * op) → 6 * op
        assert isinstance(out, sc.ScaledLinOp)
        assert out.scalar == 6.0

    def test_zero_op_returns_self(self, numpy_ctx):
        """Scaling a ZeroLinOp returns the same Zero (no wrapping)."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        z = sc.ZeroLinOp(X, X, numpy_ctx)
        out = make_scaled(5.0, z)
        assert out is z

    def test_non_scalar_value_raises(self, numpy_ctx):
        op = self._id(numpy_ctx)
        with pytest.raises(TypeError, match="scalar must be scalar-like"):
            make_scaled([1.0, 2.0], op)


# ===========================================================================
# make_composed
# ===========================================================================
class TestMakeComposed:
    def _id(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        return sc.IdentityLinOp(X, numpy_ctx)

    def _zero(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        return sc.ZeroLinOp(X, X, numpy_ctx)

    def test_identity_left_with_non_identity_right_returns_right(self, numpy_ctx):
        """``I @ A`` simplifies to ``A`` when ``A`` is not itself Identity."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        identity = sc.IdentityLinOp(X, numpy_ctx)
        dense = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), X, X, numpy_ctx,
        )
        out = make_composed(identity, dense)
        assert out is dense

    def test_identity_right_with_non_identity_left_returns_left(self, numpy_ctx):
        """``A @ I`` simplifies to ``A`` when ``A`` is not itself Identity."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        identity = sc.IdentityLinOp(X, numpy_ctx)
        dense = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), X, X, numpy_ctx,
        )
        out = make_composed(dense, identity)
        assert out is dense

    def test_identity_at_identity_simplifies_to_an_identity(self, numpy_ctx):
        """``I @ I`` returns one of the operands (an IdentityLinOp)."""
        left = self._id(numpy_ctx)
        right = self._id(numpy_ctx)
        out = make_composed(left, right)
        # The canonical form returns one of them; the type is preserved.
        assert isinstance(out, sc.IdentityLinOp)

    def test_zero_left_yields_zero(self, numpy_ctx):
        op = self._id(numpy_ctx)
        z = self._zero(numpy_ctx)
        out = make_composed(z, op)
        assert isinstance(out, sc.ZeroLinOp)

    def test_zero_right_yields_zero(self, numpy_ctx):
        op = self._id(numpy_ctx)
        z = self._zero(numpy_ctx)
        out = make_composed(op, z)
        assert isinstance(out, sc.ZeroLinOp)

    def test_mismatched_middle_space_raises(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Y, X, numpy_ctx,
        )
        B = sc.IdentityLinOp(X, numpy_ctx)
        # A: Y -> X, B: X -> X. A @ B has right.codomain (X) == left.domain (Y)? No.
        with pytest.raises(ValueError, match="right.codomain == left.domain"):
            make_composed(A, B)

    def test_rejects_non_linop_operand(self, numpy_ctx):
        op = self._id(numpy_ctx)
        with pytest.raises(TypeError, match="must be a LinOp"):
            make_composed(op, "not a linop")
        with pytest.raises(TypeError, match="must be a LinOp"):
            make_composed("not a linop", op)


# ===========================================================================
# Context mismatch on factories
# ===========================================================================
class TestContextMismatch:
    def test_make_sum_rejects_different_dtype(self, numpy_ctx, numpy_f32_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Xf = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        a = sc.IdentityLinOp(X, numpy_ctx)
        b = sc.IdentityLinOp(Xf, numpy_f32_ctx)
        with pytest.raises(ValueError, match="same ctx"):
            make_sum((a, b))

    def test_make_composed_rejects_different_dtype(self, numpy_ctx, numpy_f32_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Xf = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        a = sc.IdentityLinOp(X, numpy_ctx)
        b = sc.IdentityLinOp(Xf, numpy_f32_ctx)
        with pytest.raises(ValueError, match="same ctx"):
            make_composed(a, b)

    def test_make_sum_ignores_check_level_when_dtype_matches(self):
        """Differing ``check_level`` does not block algebra when dtype matches.

        (Folded from test_algebra.py::test_factories_ignore_enable_checks_when_context_dtype_matches.)
        """
        checked = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
        unchecked = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        X_checked = sc.DenseCoordinateSpace((2,), checked)
        X_unchecked = sc.DenseCoordinateSpace((2,), unchecked)
        A = sc.DenseLinOp(
            checked.asarray([[1.0, 0.0], [0.0, 1.0]]), X_checked, X_checked, checked,
        )
        B = sc.DenseLinOp(
            unchecked.asarray([[2.0, 0.0], [0.0, 3.0]]), X_unchecked, X_unchecked, unchecked,
        )
        assert isinstance(make_sum((A, B)), sc.SumLinOp)
        assert isinstance(make_composed(A, B), sc.ComposedLinOp)


# ===========================================================================
# Geometry compatibility on factories
# ===========================================================================
class TestGeometryMismatch:
    """Folded from test_algebra.py::test_factories_reject_matching_shapes_with_different_geometry."""

    def test_rejects_matching_shapes_with_different_geometry(self, numpy_ctx):
        euclidean = sc.DenseCoordinateSpace((2,), numpy_ctx)
        weighted_a = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        differently_weighted = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 4.0])),
        )
        A = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), euclidean, euclidean, numpy_ctx,
        )
        B = sc.DenseLinOp(
            numpy_ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), weighted_a, weighted_a, numpy_ctx,
        )
        C = sc.DenseLinOp(
            numpy_ctx.asarray([[4.0, 0.0], [0.0, 5.0]]),
            differently_weighted, differently_weighted, numpy_ctx,
        )
        with pytest.raises(ValueError, match="same domain and codomain"):
            make_sum((A, B))
        with pytest.raises(ValueError, match="right.codomain == left.domain"):
            make_composed(A, B)
        with pytest.raises(ValueError, match="same domain and codomain"):
            make_sum((B, C))

    def test_accepts_matching_shapes_with_identical_geometry(self, numpy_ctx):
        weighted_a = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        weighted_b = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        B = sc.DenseLinOp(
            numpy_ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), weighted_a, weighted_a, numpy_ctx,
        )
        B_same = sc.DenseLinOp(
            numpy_ctx.asarray([[0.5, 0.0], [0.0, 0.25]]), weighted_b, weighted_b, numpy_ctx,
        )
        assert isinstance(make_sum((B, B_same)), sc.SumLinOp)
        assert isinstance(make_composed(B, B_same), sc.ComposedLinOp)

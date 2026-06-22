"""Tests for :mod:`spacecore.linalg._utils` — solver building blocks.

Checklist section 8, Internal helpers (``_utils.py``):

* Validation: ``require_linop``, ``require_square``, ``check_maxiter``,
  ``check_interval``, ``require_strict_cg_preconditions``.
* Iteration bookkeeping: ``default_maxiter``, ``should_check_iteration``,
  ``threshold``, ``is_converged``.
* Numerics: ``real_inner``, ``safe_inverse_nonneg``, ``normalize``,
  ``default_initial_vector``.
* Formatting: ``summarize_value``, ``result_repr``.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from spacecore.linalg import _utils
from tests._helpers import to_numpy


# ===========================================================================
# require_linop / require_square
# ===========================================================================
class TestRequire:
    def test_require_linop_returns_linop(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(space, numpy_ctx)
        assert _utils.require_linop(op) is op

    def test_require_linop_rejects_non_linop(self):
        with pytest.raises(TypeError, match="A must be a LinOp"):
            _utils.require_linop(np.eye(2))

    def test_require_square_accepts_square(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        _utils.require_square(sc.IdentityLinOp(space, numpy_ctx), "cg")  # no raise

    def test_require_square_rejects_rectangular(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(numpy_ctx.asarray(np.ones((3, 2))), X, Y, numpy_ctx)
        with pytest.raises(ValueError, match="square LinOp"):
            _utils.require_square(op, "cg")


# ===========================================================================
# maxiter / interval validation
# ===========================================================================
class TestIterationParams:
    def test_default_maxiter_is_domain_size(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 3), numpy_ctx)
        op = sc.IdentityLinOp(space, numpy_ctx)
        assert _utils.default_maxiter(op) == 6

    def test_check_maxiter_none_uses_default(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        op = sc.IdentityLinOp(space, numpy_ctx)
        assert _utils.check_maxiter(None, op) == 4

    def test_check_maxiter_rejects_negative(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        op = sc.IdentityLinOp(space, numpy_ctx)
        with pytest.raises(ValueError, match="maxiter must be nonnegative"):
            _utils.check_maxiter(-1, op)

    def test_check_interval_rejects_non_positive(self):
        assert _utils.check_interval(3) == 3
        with pytest.raises(ValueError, match="check_every must be positive"):
            _utils.check_interval(0)


# ===========================================================================
# should_check_iteration / threshold / is_converged
# ===========================================================================
class TestConvergenceBookkeeping:
    def test_should_check_iteration_on_interval_and_final(self):
        assert bool(_utils.should_check_iteration(0, 100, 64)) is True
        assert bool(_utils.should_check_iteration(64, 100, 64)) is True
        assert bool(_utils.should_check_iteration(5, 100, 64)) is False
        # Always checks on/after the final iteration.
        assert bool(_utils.should_check_iteration(100, 100, 64)) is True

    def test_threshold_combines_absolute_and_relative(self):
        assert _utils.threshold(10.0, 1e-3, 0.5) == pytest.approx(0.51)

    def test_threshold_clamps_negative_tolerances(self):
        assert _utils.threshold(10.0, -1.0, -2.0) == 0.0

    def test_is_converged_is_leq(self):
        assert bool(_utils.is_converged(0.5, 1.0)) is True
        assert bool(_utils.is_converged(1.5, 1.0)) is False


# ===========================================================================
# numerics
# ===========================================================================
class TestNumerics:
    def test_real_inner_returns_real_part(self, numpy_complex_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1.0 + 1.0j, 2.0 - 1.0j])
        y = numpy_complex_ctx.asarray([1.0 - 1.0j, 0.0 + 2.0j])
        out = _utils.real_inner(space, x, y)
        assert np.isreal(to_numpy(out))
        np.testing.assert_allclose(to_numpy(out), np.real(to_numpy(space.inner(x, y))))

    def test_safe_inverse_nonneg_only_inverts_positive(self):
        ops = sc.NumpyOps()
        values = np.asarray([-2.0, 0.0, 4.0])
        np.testing.assert_allclose(
            to_numpy(_utils.safe_inverse_nonneg(ops, values)), [0.0, 0.0, 0.25]
        )

    def test_normalize_returns_unit_and_norm(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        unit, norm = _utils.normalize(space, numpy_ctx.asarray([3.0, 4.0]))
        np.testing.assert_allclose(to_numpy(norm), 5.0)
        np.testing.assert_allclose(to_numpy(unit), [0.6, 0.8])
        np.testing.assert_allclose(to_numpy(space.norm(unit)), 1.0)

    def test_default_initial_vector_is_unit_in_geometry(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 5.0, 11.0])
        space = sc.DenseCoordinateSpace(
            (3,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights)
        )
        op = sc.IdentityLinOp(space, numpy_ctx)
        v = _utils.default_initial_vector(op)
        np.testing.assert_allclose(to_numpy(space.norm(v)), 1.0)


# ===========================================================================
# formatting
# ===========================================================================
class TestFormatting:
    def test_summarize_value_scalar(self):
        assert _utils.summarize_value(np.asarray(1.5)) == "1.5"

    def test_summarize_value_array(self):
        text = _utils.summarize_value(np.zeros((3, 4)))
        assert text.startswith("<array shape=(3, 4)")

    def test_summarize_value_tuple(self):
        text = _utils.summarize_value((np.asarray(1.0), np.zeros((2,))))
        assert text.startswith("(") and "<array shape=(2,)" in text

    def test_summarize_value_bool_scalar(self):
        assert _utils.summarize_value(np.asarray(True)) == "True"

    def test_result_repr_formats_named_fields(self):
        text = _utils.result_repr("Demo", {"a": 1, "b": np.zeros((5,))})
        assert text.startswith("Demo(a=")
        assert "b=<array shape=(5,)" in text


# ===========================================================================
# require_strict_cg_preconditions
# ===========================================================================
class TestStrictCGPreconditions:
    def _dense(self, ctx, matrix):
        space = sc.DenseCoordinateSpace((matrix.shape[0],), ctx)
        return sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)

    def test_noop_when_not_strict(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
        op = self._dense(ctx, np.asarray([[1.0, 2.0], [0.0, 3.0]]))  # non-Hermitian
        _utils.require_strict_cg_preconditions(op)  # no raise: not strict

    def test_strict_accepts_spd(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="strict")
        op = self._dense(ctx, np.asarray([[4.0, 1.0], [1.0, 3.0]]))
        _utils.require_strict_cg_preconditions(op)  # no raise

    def test_strict_rejects_non_hermitian(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="strict")
        op = self._dense(ctx, np.asarray([[1.0, 2.0], [0.0, 3.0]]))
        with pytest.raises(ValueError, match="Hermitian"):
            _utils.require_strict_cg_preconditions(op)

    def test_strict_rejects_nonpositive_curvature(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="strict")
        # Hermitian but indefinite: probe vector lands on zero curvature.
        op = sc.DiagonalLinOp(ctx.asarray([1.0, -1.0]), sc.DenseCoordinateSpace((2,), ctx), ctx)
        with pytest.raises(ValueError, match="positive curvature"):
            _utils.require_strict_cg_preconditions(op)

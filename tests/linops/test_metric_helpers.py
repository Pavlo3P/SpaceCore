"""Tests for :mod:`spacecore.linop._metric` — metric-adjoint helpers.

Checklist item 2:

* ``space_has_riesz_maps`` — true on weighted spaces with Riesz maps,
  false on plain DenseCoordinateSpace lacking them (well, all do via
  EuclideanInnerProduct, so the test verifies the predicate's contract on
  weighted vs Euclidean).
* ``_requires_euclidean_or_riesz`` — accepts Euclidean / weighted-with-Riesz
  spaces; raises with informative message on spaces lacking metric machinery.
* ``_metric_is_hermitian_by_basis`` — returns ``None`` on non-square ops;
  returns ``True`` / ``False`` based on basis test on small square ops;
  returns ``None`` on large ops (perf guard).
* ``metric_rapply`` — equals ``euclidean_rapply`` on Euclidean spaces;
  applies ``R_X^{-1} ∘ A† ∘ R_Y`` on weighted.
* ``metric_rvapply`` — fast path on weighted; fallback warning when batched
  Riesz unavailable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spacecore as sc
from spacecore.linop._metric import (
    _metric_is_hermitian_by_basis,
    _requires_euclidean_or_riesz,
    metric_rapply,
    metric_rvapply,
    space_has_riesz_maps,
)


# ===========================================================================
# space_has_riesz_maps
# ===========================================================================
class TestSpaceHasRieszMaps:
    def test_weighted_dense_coordinate_has_riesz(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        space = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        assert space_has_riesz_maps(space) is True

    def test_euclidean_dense_coordinate_returns_false(self, numpy_ctx):
        """Euclidean spaces don't need Riesz maps; the helper returns False
        (callers handle Euclidean explicitly before invoking this predicate)."""
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        assert space_has_riesz_maps(space) is False

    def test_tree_of_weighted_leaves_returns_true(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        leaf = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        tree = sc.TreeSpace.from_leaf_spaces((leaf, leaf), numpy_ctx)
        assert space_has_riesz_maps(tree) is True


# ===========================================================================
# _requires_euclidean_or_riesz
# ===========================================================================
class TestRequiresEuclideanOrRiesz:
    def test_accepts_euclidean_pair(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        # No raise.
        _requires_euclidean_or_riesz(X, Y, "test_op")

    def test_accepts_weighted_with_riesz(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        X = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        # No raise.
        _requires_euclidean_or_riesz(X, Y, "test_op")

    def test_message_names_role_and_opname(self, numpy_ctx):
        """Custom non-Euclidean space without Riesz maps → informative raise."""

        class _CustomGeometry(sc.InnerProduct):
            @property
            def is_euclidean(self) -> bool:
                return False

            def inner(self, ops, x, y):
                return ops.vdot(x, y)

        class _CustomInnerSpace(sc.DenseCoordinateSpace):
            pass

        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        X = sc.DenseCoordinateSpace((2,), ctx)
        # Build a space whose geometry advertises non-Euclidean but lacks
        # the inherited Riesz machinery.
        bad = _CustomInnerSpace((2,), ctx, geometry=_CustomGeometry())
        with pytest.raises(TypeError, match="(?i)non-euclidean.*requires Riesz"):
            _requires_euclidean_or_riesz(bad, X, "my_op")


# ===========================================================================
# _metric_is_hermitian_by_basis
# ===========================================================================
class TestMetricIsHermitianByBasis:
    def test_returns_false_on_non_square(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, numpy_ctx,
        )
        assert _metric_is_hermitian_by_basis(op) is False

    def test_returns_true_on_symmetric_dense(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = numpy_ctx.asarray([[1.0, 2.0], [2.0, 3.0]])
        op = sc.DenseLinOp(A, X, X, numpy_ctx)
        assert _metric_is_hermitian_by_basis(op) is True

    def test_returns_false_on_asymmetric_dense(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
        op = sc.DenseLinOp(A, X, X, numpy_ctx)
        assert _metric_is_hermitian_by_basis(op) is False

    def test_returns_none_on_large_square_op_perf_guard(self, numpy_ctx):
        """Above the basis-check size cap the helper returns ``None`` without
        applying the operator (perf guard, folded from test_adjoint_identity.py)."""
        n = 1025
        space = sc.DenseCoordinateSpace(
            (n,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray(np.ones(n))),
        )
        op = sc.DiagonalLinOp(numpy_ctx.asarray(np.ones(n)), space, numpy_ctx)

        def fail_apply(_x):
            raise AssertionError("perf guard must not apply large metric operators")

        op.apply = fail_apply
        assert _metric_is_hermitian_by_basis(op) is None


# ===========================================================================
# metric_rapply: equals euclidean_rapply on Euclidean; R_X^{-1} A† R_Y otherwise
# ===========================================================================
class TestMetricRapply:
    def test_euclidean_branch_calls_euclidean_rapply_directly(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        y = numpy_ctx.asarray([1.0, 2.0, 3.0])
        called = []

        def euclid_rapply(z):
            called.append(z)
            return z * 2.0

        out = metric_rapply(X, Y, euclid_rapply, y)
        assert len(called) == 1
        np.testing.assert_allclose(out, [2.0, 4.0, 6.0])

    def test_weighted_branch_wraps_with_riesz_pair(self, numpy_ctx):
        """metric_rapply(weighted) = R_X^{-1}(euclid_rapply(R_Y(y)))."""
        weights_x = numpy_ctx.asarray([2.0, 3.0])
        weights_y = numpy_ctx.asarray([5.0, 7.0])
        X = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights_x),
        )
        Y = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights_y),
        )
        y = numpy_ctx.asarray([4.0, 6.0])

        def euclid_rapply(z):
            # Identity acts as A† on the Euclidean coordinate level.
            return z

        out = metric_rapply(X, Y, euclid_rapply, y)
        # R_Y(y) = [5*4, 7*6] = [20, 42]
        # euclid_rapply == identity ⇒ [20, 42]
        # R_X^{-1}(...) = [20/2, 42/3] = [10, 14]
        np.testing.assert_allclose(out, [10.0, 14.0])


# ===========================================================================
# metric_rvapply: warning + vmap fallback when batched Riesz unavailable
# ===========================================================================
class TestMetricRvapply:
    def test_euclidean_branch_calls_rvapply_directly(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        ys = numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def euclid_rapply(y):  # noqa: ARG001
            raise AssertionError("should not be called on Euclidean fast path")

        def euclid_rvapply(zs):
            return zs * 2.0

        out = metric_rvapply(
            X, X, euclid_rapply, euclid_rvapply, ys,
            opname="test", ops=numpy_ctx.ops,
        )
        np.testing.assert_allclose(out, ys * 2.0)

    def test_weighted_fast_path_no_warning(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        X = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        ys = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]])

        def euclid_rapply(y):  # noqa: ARG001
            raise AssertionError("should not be called on fast path")

        def euclid_rvapply(zs):
            return zs

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            out = metric_rvapply(
                X, X, euclid_rapply, euclid_rvapply, ys,
                opname="test_op", ops=numpy_ctx.ops,
            )
        # No metric-batch-fallback warnings on the fast path.
        assert not any("could not use batched Riesz maps" in str(w.message)
                       for w in recorded)
        # R_X(ys) = ys * weights, identity, R_X^{-1}(...) = ys.
        np.testing.assert_allclose(out, ys)

    def test_fallback_warns_and_uses_per_element(self, numpy_ctx):
        """When batched Riesz raises a documented fallback error, the helper
        warns and falls back to vmap(metric_rapply)."""
        weights = numpy_ctx.asarray([2.0, 3.0])
        X = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        ys = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]])

        def euclid_rapply(y):
            return y * 1.0

        def euclid_rvapply(zs):  # noqa: ARG001
            raise NotImplementedError("simulated unavailable")

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            out = metric_rvapply(
                X, X, euclid_rapply, euclid_rvapply, ys,
                opname="my_op", ops=numpy_ctx.ops,
            )
        assert any("could not use batched Riesz maps" in str(w.message)
                   for w in recorded)
        # The fallback applies metric_rapply per row; R_X then R_X^{-1} on
        # identity-style euclid_rapply round-trips to ys.
        np.testing.assert_allclose(out, ys)

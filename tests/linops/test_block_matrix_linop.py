"""Tests for :class:`spacecore.BlockMatrixLinOp` — rectangular block matrices.

Checklist section 6:

* ``BlockMatrixLinOp(block_rows)`` infers ``TreeSpace`` domain/codomain and
  exposes ``domain.arity`` / ``codomain.arity`` from the block layout.
* ``apply`` sums each output row; ``rapply`` sums each transposed column.
* ``H`` is a structural adjoint (transposed layout, per-block ``A_ij.H``) with
  ``A.H.H is A``, and ``H.apply`` matches ``rapply``.
* Non-Euclidean (weighted) metrics satisfy the adjoint identity
  ``<A x, y>_cod == <x, A* y>_dom``.
* Batched ``vapply`` / ``rvapply`` match the leafwise per-row loops.
* Operator algebra: ``A + A``, scalar ``c * A``, and the normal operator
  ``A.H @ A``.
* Construction rejects non-rectangular layouts, non-LinOp blocks, mismatched
  row codomains / column domains, and blocks with a divergent check policy.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.linop import BlockMatrixLinOp
from tests._helpers import to_numpy


def _weighted_space(weights, ctx):
    return sc.DenseCoordinateSpace(
        tuple(np.asarray(weights).shape), ctx, sc.WeightedInnerProduct(ctx.asarray(weights))
    )


def _blocks(ctx, *, weighted=False):
    make = _weighted_space if weighted else lambda shape, c: sc.DenseCoordinateSpace(shape, c)
    if weighted:
        x0, x1 = make([2.0, 5.0], ctx), make([3.0], ctx)
        y0, y1 = make([7.0], ctx), make([11.0, 13.0], ctx)
    else:
        x0, x1 = make((2,), ctx), make((1,), ctx)
        y0, y1 = make((1,), ctx), make((2,), ctx)
    return (
        (
            sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x0, y0, ctx),
            sc.DenseLinOp(ctx.asarray([[4.0]]), x1, y0, ctx),
        ),
        (
            sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 2.0]]), x0, y1, ctx),
            sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x1, y1, ctx),
        ),
    )


def _assert_tuple_allclose(actual, expected):
    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(to_numpy(actual_leaf), expected_leaf)


# ===========================================================================
# apply / rapply inference and structural adjoint
# ===========================================================================
class TestApplyRapplyAdjoint:
    def test_apply_rapply_inference_and_structural_adjoint(self, numpy_ctx):
        op = sc.BlockMatrixLinOp(_blocks(numpy_ctx))
        x = (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0]))
        y = (numpy_ctx.asarray([2.0]), numpy_ctx.asarray([1.0, -2.0]))

        assert BlockMatrixLinOp is sc.BlockMatrixLinOp
        assert op.domain.arity == 2
        assert op.codomain.arity == 2
        _assert_tuple_allclose(op.apply(x), ([17.0], [10.0, 1.0]))
        _assert_tuple_allclose(op.rapply(y), ([3.0, 0.0], [13.0]))
        assert isinstance(op.H, sc.BlockMatrixLinOp)
        assert op.H.H is op
        _assert_tuple_allclose(op.H.apply(y), ([3.0, 0.0], [13.0]))


# ===========================================================================
# Non-Euclidean metric adjoint identity
# ===========================================================================
class TestMetricAdjoint:
    def test_non_euclidean_metric_adjoint_identity(self, numpy_ctx):
        op = sc.BlockMatrixLinOp(_blocks(numpy_ctx, weighted=True))
        x = (numpy_ctx.asarray([1.0, -2.0]), numpy_ctx.asarray([3.0]))
        y = (numpy_ctx.asarray([0.5]), numpy_ctx.asarray([2.0, -1.0]))

        np.testing.assert_allclose(
            to_numpy(op.codomain.inner(op.apply(x), y)),
            to_numpy(op.domain.inner(x, op.rapply(y))),
        )


# ===========================================================================
# Batched apply / rapply and operator algebra
# ===========================================================================
class TestBatchingAndAlgebra:
    def test_batched_apply_rapply_and_algebra(self, numpy_ctx):
        op = sc.BlockMatrixLinOp(_blocks(numpy_ctx))
        xs = (
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
            numpy_ctx.asarray([[3.0], [-1.0]]),
        )
        ys = (
            numpy_ctx.asarray([[2.0], [-1.0]]),
            numpy_ctx.asarray([[1.0, -2.0], [3.0, 4.0]]),
        )

        expected_y = tuple(
            np.stack([to_numpy(op.apply((xs[0][i], xs[1][i]))[j]) for i in range(2)])
            for j in range(2)
        )
        expected_x = tuple(
            np.stack([to_numpy(op.rapply((ys[0][i], ys[1][i]))[j]) for i in range(2)])
            for j in range(2)
        )
        _assert_tuple_allclose(op.vapply(xs), expected_y)
        _assert_tuple_allclose(op.rvapply(ys), expected_x)

        _assert_tuple_allclose((op + op).apply((xs[0][0], xs[1][0])), ([34.0], [20.0, 2.0]))
        _assert_tuple_allclose((0.5 * op).apply((xs[0][0], xs[1][0])), ([8.5], [5.0, 0.5]))
        normal = op.H @ op
        _assert_tuple_allclose(
            normal.apply((xs[0][0], xs[1][0])),
            op.rapply(op.apply((xs[0][0], xs[1][0]))),
        )


# ===========================================================================
# Block-layout and context validation
# ===========================================================================
class TestValidation:
    def test_rejects_invalid_block_layouts_and_contexts(self, numpy_ctx):
        rows = _blocks(numpy_ctx)
        with pytest.raises(ValueError, match="rectangular"):
            sc.BlockMatrixLinOp((rows[0], rows[1][:1]))
        with pytest.raises(TypeError, match="every block"):
            sc.BlockMatrixLinOp(((rows[0][0], object()),))

        wrong_cod = sc.DenseCoordinateSpace((2,), numpy_ctx)
        incompatible_row = sc.IdentityLinOp(wrong_cod)
        with pytest.raises(ValueError, match="row 0"):
            sc.BlockMatrixLinOp(((rows[0][0], incompatible_row),))

        wrong_dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        incompatible_column = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), wrong_dom, rows[1][0].codomain, numpy_ctx
        )
        with pytest.raises(ValueError, match="column 1"):
            sc.BlockMatrixLinOp((rows[0], (rows[1][0], incompatible_column)))

        other_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
        other_space = sc.DenseCoordinateSpace((1,), other_ctx)
        with pytest.raises(ValueError, match="check policy"):
            sc.BlockMatrixLinOp(((rows[0][0], sc.IdentityLinOp(other_space)),))

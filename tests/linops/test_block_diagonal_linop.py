"""Tests for :class:`spacecore.BlockDiagonalLinOp` — direct-product blocks.

Checklist section 6:

* ``BlockDiagonalLinOp(blocks)`` infers matching ``TreeSpace`` domain/codomain
  with a shared treedef and preserves the block tree structure under ``apply``.
* The legacy four-argument ``(dom, cod, blocks, ctx)`` form accepts explicit
  structured-tree (NamedTuple template) layouts and round-trips them through
  ``apply`` / ``rapply`` / ``H``.
* ``from_operators`` keeps a tuple default as a tuple element structure.
* Non-Euclidean (weighted) metrics satisfy ``<A x, y>_cod == <x, A* y>_dom``;
  ``H`` is a structural adjoint with ``A.H.H is A`` and ``H.apply == rapply``.
* Batched ``vapply`` / ``rvapply`` match leafwise loops for two and three
  components, preserve NamedTuple structure, and dispatch through the owning
  space's ``add_batch``.
* Operator algebra: ``A + A``, scalar ``c * A``, and ``A @ A``.
* Construction rejects empty leaf tuples and non-LinOp blocks; batch checks
  reject a wrong tuple layout (bad structure or trailing shape); blocks with a
  divergent check policy are rejected.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import jax_real_dtype, to_numpy


class State(NamedTuple):
    a: object
    b: object


def _ctx(kind: str):
    if kind == "jax":
        pytest.importorskip("jax")
        return sc.Context(sc.JaxOps(), dtype=jax_real_dtype())
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _weighted_space(weights, ctx):
    return sc.DenseCoordinateSpace(
        tuple(np.asarray(weights).shape), ctx, sc.WeightedInnerProduct(ctx.asarray(weights))
    )


def _assert_tree_allclose(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_allclose(actual[key], expected[key])
    elif isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected):
            _assert_tree_allclose(actual_leaf, expected_leaf)
    else:
        np.testing.assert_allclose(to_numpy(actual), expected)


def _assert_allclose(actual, expected):
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-7, atol=1e-7)


def _assert_state_allclose(actual, expected_a, expected_b):
    assert isinstance(actual, State)
    _assert_allclose(actual.a, expected_a)
    _assert_allclose(actual.b, expected_b)


def _block_parts(ctx):
    x1 = sc.DenseCoordinateSpace((2,), ctx)
    x2 = sc.DenseCoordinateSpace((3,), ctx)
    y1 = sc.DenseCoordinateSpace((2,), ctx)
    y2 = sc.DenseCoordinateSpace((1,), ctx)
    return (
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), x1, y1, ctx),
        sc.DenseLinOp(ctx.asarray([[5.0, 6.0, 7.0]]), x2, y2, ctx),
    )


def _structured_tree(spaces, template, ctx):
    return sc.TreeSpace.from_template(template, tuple(spaces), ctx=ctx)


def _slice_tree_batch(xs, i):
    return tuple(_slice_tree_batch(xi, i) if isinstance(xi, tuple) else xi[i] for xi in xs)


def _stack_tree_rows(rows):
    if isinstance(rows[0], tuple):
        return tuple(
            _stack_tree_rows(tuple(row[i] for row in rows)) for i in range(len(rows[0]))
        )
    return np.stack([to_numpy(row) for row in rows], axis=0)


def _assert_vapply_loop(op, xs):
    rows = tuple(op.apply(_slice_tree_batch(xs, i)) for i in range(xs[0].shape[0]))
    _assert_tree_allclose(op.vapply(xs), _stack_tree_rows(rows))


def _assert_rvapply_loop(op, ys):
    rows = tuple(op.rapply(_slice_tree_batch(ys, i)) for i in range(ys[0].shape[0]))
    _assert_tree_allclose(op.rvapply(ys), _stack_tree_rows(rows))


# ===========================================================================
# TreeSpace inference and block-tree structure preservation
# ===========================================================================
class TestInferenceAndStructure:
    def test_infers_tree_spaces_and_preserves_block_tree_structure(self, numpy_ctx):
        x0 = sc.DenseCoordinateSpace((2,), numpy_ctx)
        x1 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        y0 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        y1 = sc.DenseCoordinateSpace((2,), numpy_ctx)
        blocks = {
            "left": sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0]]), x0, y0, numpy_ctx),
            "right": sc.DenseLinOp(numpy_ctx.asarray([[3.0], [-1.0]]), x1, y1, numpy_ctx),
        }

        op = sc.BlockDiagonalLinOp(blocks)

        assert isinstance(op.domain, sc.TreeSpace)
        assert isinstance(op.codomain, sc.TreeSpace)
        assert op.domain.treedef == op.codomain.treedef
        result = op.apply(
            {"left": numpy_ctx.asarray([2.0, 4.0]), "right": numpy_ctx.asarray([5.0])}
        )
        _assert_tree_allclose(result, {"left": [10.0], "right": [15.0, -5.0]})

    def test_tuple_default_remains_tuple(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py.
        op = sc.BlockDiagonalLinOp.from_operators(_block_parts(numpy_ctx))

        y = op.apply((numpy_ctx.asarray([10.0, 20.0]), numpy_ctx.asarray([1.0, 2.0, 3.0])))
        assert isinstance(y, tuple)
        _assert_allclose(y[0], [50.0, 110.0])
        _assert_allclose(y[1], [38.0])

        x = op.rapply((numpy_ctx.asarray([2.0, -1.0]), numpy_ctx.asarray([3.0])))
        assert isinstance(x, tuple)
        _assert_allclose(x[0], [-1.0, 0.0])
        _assert_allclose(x[1], [15.0, 18.0, 21.0])


# ===========================================================================
# Structured-tree (NamedTuple template) layouts via the legacy 4-arg form
# ===========================================================================
class TestStructuredTreeLayout:
    @pytest.mark.parametrize("kind", ["numpy", "jax"])
    def test_accepts_and_returns_structured_tree_elements(self, kind):
        # Folded from tests/linops/test_tree_structure.py.
        ctx = _ctx(kind)
        parts = _block_parts(ctx)
        dom = _structured_tree(
            (parts[0].domain, parts[1].domain),
            State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0, 0.0])),
            ctx,
        )
        cod = _structured_tree(
            (parts[0].codomain, parts[1].codomain),
            State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0])),
            ctx,
        )
        op = sc.BlockDiagonalLinOp(dom, cod, parts, ctx)

        x = State(ctx.asarray([10.0, 20.0]), ctx.asarray([1.0, 2.0, 3.0]))
        y = op.apply(x)
        _assert_state_allclose(y, [50.0, 110.0], [38.0])

        xr = op.rapply(State(ctx.asarray([2.0, -1.0]), ctx.asarray([3.0])))
        _assert_state_allclose(xr, [-1.0, 0.0], [15.0, 18.0, 21.0])

        _assert_state_allclose(op.H.apply(y), [380.0, 540.0], [190.0, 228.0, 266.0])

    @pytest.mark.parametrize("kind", ["numpy", "jax"])
    def test_batch_paths_preserve_namedtuple_structure(self, kind):
        # Folded from tests/linops/test_tree_structure.py.
        ctx = _ctx(kind)
        parts = _block_parts(ctx)
        dom = _structured_tree(
            (parts[0].domain, parts[1].domain),
            State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0, 0.0])),
            ctx,
        )
        cod = _structured_tree(
            (parts[0].codomain, parts[1].codomain),
            State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0])),
            ctx,
        )
        op = sc.BlockDiagonalLinOp(dom, cod, parts, ctx)

        xb = State(
            ctx.asarray([[10.0, 20.0], [1.0, 2.0]]),
            ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        _assert_state_allclose(op.vapply(xb), [[50.0, 110.0], [5.0, 11.0]], [[38.0], [92.0]])

        yb = State(ctx.asarray([[2.0, -1.0], [1.0, 1.0]]), ctx.asarray([[3.0], [2.0]]))
        _assert_state_allclose(
            op.rvapply(yb),
            [[-1.0, 0.0], [4.0, 6.0]],
            [[15.0, 18.0, 21.0], [10.0, 12.0, 14.0]],
        )


# ===========================================================================
# Metric adjoint and structural double adjoint
# ===========================================================================
class TestMetricAdjoint:
    def test_metric_adjoint_and_structural_double_adjoint(self, numpy_ctx):
        x0 = _weighted_space([2.0, 5.0], numpy_ctx)
        x1 = _weighted_space([3.0], numpy_ctx)
        y0 = _weighted_space([7.0], numpy_ctx)
        y1 = _weighted_space([11.0, 13.0], numpy_ctx)
        blocks = (
            sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0]]), x0, y0, numpy_ctx),
            sc.DenseLinOp(numpy_ctx.asarray([[3.0], [-1.0]]), x1, y1, numpy_ctx),
        )
        op = sc.BlockDiagonalLinOp(blocks)
        x = (numpy_ctx.asarray([2.0, -1.0]), numpy_ctx.asarray([4.0]))
        y = (numpy_ctx.asarray([3.0]), numpy_ctx.asarray([1.0, -2.0]))

        np.testing.assert_allclose(
            to_numpy(op.codomain.inner(op.apply(x), y)),
            to_numpy(op.domain.inner(x, op.rapply(y))),
        )
        assert isinstance(op.H, sc.BlockDiagonalLinOp)
        assert op.H.H is op
        _assert_tree_allclose(op.H.apply(y), op.rapply(y))


# ===========================================================================
# Batched apply / rapply match leafwise loops; add_batch dispatch
# ===========================================================================
class TestBatching:
    def test_batched_apply_and_rapply_match_leafwise_loops(self, numpy_ctx):
        x0 = sc.DenseCoordinateSpace((2,), numpy_ctx)
        x1 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        y0 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        y1 = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.BlockDiagonalLinOp(
            (
                sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0]]), x0, y0, numpy_ctx),
                sc.DenseLinOp(numpy_ctx.asarray([[3.0], [-1.0]]), x1, y1, numpy_ctx),
            )
        )
        xs = (
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
            numpy_ctx.asarray([[5.0], [6.0]]),
        )
        ys = (
            numpy_ctx.asarray([[2.0], [-1.0]]),
            numpy_ctx.asarray([[3.0, 4.0], [5.0, 6.0]]),
        )

        expected_y = tuple(
            np.stack([to_numpy(op.apply((xs[0][i], xs[1][i]))[j]) for i in range(2)])
            for j in range(2)
        )
        expected_x = tuple(
            np.stack([to_numpy(op.rapply((ys[0][i], ys[1][i]))[j]) for i in range(2)])
            for j in range(2)
        )
        _assert_tree_allclose(op.vapply(xs), expected_y)
        _assert_tree_allclose(op.rvapply(ys), expected_x)

    def test_vapply_and_rvapply_match_loops_for_two_and_three_components(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py.
        x1, x2, x3 = (
            sc.DenseCoordinateSpace((2,), numpy_ctx),
            sc.DenseCoordinateSpace((1,), numpy_ctx),
            sc.DenseCoordinateSpace((2,), numpy_ctx),
        )
        y1, y2, y3 = (
            sc.DenseCoordinateSpace((1,), numpy_ctx),
            sc.DenseCoordinateSpace((2,), numpy_ctx),
            sc.DenseCoordinateSpace((2,), numpy_ctx),
        )
        parts = (
            sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0]]), x1, y1, numpy_ctx),
            sc.DenseLinOp(numpy_ctx.asarray([[3.0], [-1.0]]), x2, y2, numpy_ctx),
            sc.DenseLinOp(numpy_ctx.asarray([[2.0, 0.0], [0.5, 4.0]]), x3, y3, numpy_ctx),
        )

        op2 = sc.BlockDiagonalLinOp.from_operators(parts[:2])
        xs2 = (
            numpy_ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]),
            numpy_ctx.asarray([[3.0], [4.0]]),
        )
        ys2 = (
            numpy_ctx.asarray([[5.0], [-2.0]]),
            numpy_ctx.asarray([[1.0, 2.0], [0.5, -1.0]]),
        )
        _assert_vapply_loop(op2, xs2)
        _assert_rvapply_loop(op2, ys2)

        op3 = sc.BlockDiagonalLinOp.from_operators(parts)
        xs3 = xs2 + (numpy_ctx.asarray([[2.0, -3.0], [0.25, 1.5]]),)
        ys3 = ys2 + (numpy_ctx.asarray([[1.0, -1.0], [2.0, 0.5]]),)
        _assert_vapply_loop(op3, xs3)
        _assert_rvapply_loop(op3, ys3)

    def test_rvapply_dispatches_through_space_add_batch(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py: add_batch
        # dispatch via a CountingVectorSpace (exercised on the dual SumToSingle
        # accumulation path that shares BlockDiagonal's tree machinery).
        ctx = numpy_ctx

        class CountingVectorSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx, counter):
                self.counter = counter
                super().__init__(shape, ctx)

            def add_batch(self, x, y):
                self.counter["calls"] += 1
                return super().add_batch(x, y)

            def _convert(self, new_ctx):
                return CountingVectorSpace(self.shape, new_ctx, self.counter)

        counter = {"calls": 0}
        shared = CountingVectorSpace((2,), ctx, counter)
        dom1, dom2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((1,), ctx)
        B1 = sc.DenseLinOp(ctx.asarray([[1.0], [2.0]]), dom1, shared, ctx)
        B2 = sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), dom2, shared, ctx)
        summed = sc.SumToSingleLinOp.from_operators((B1, B2))
        xs = (ctx.asarray([[1.0], [2.0]]), ctx.asarray([[3.0], [4.0]]))

        summed.vapply(xs)
        assert counter["calls"] == 1


# ===========================================================================
# Operator algebra and block / context validation
# ===========================================================================
class TestAlgebraAndValidation:
    def test_algebra_and_block_validation(self, numpy_ctx):
        x = sc.DenseCoordinateSpace((1,), numpy_ctx)
        block = sc.DenseLinOp(numpy_ctx.asarray([[2.0]]), x, x, numpy_ctx)
        op = sc.BlockDiagonalLinOp((block, block))
        value = (numpy_ctx.asarray([3.0]), numpy_ctx.asarray([4.0]))

        _assert_tree_allclose((op + op).apply(value), ([12.0], [16.0]))
        _assert_tree_allclose((3.0 * op).apply(value), ([18.0], [24.0]))
        _assert_tree_allclose((op @ op).apply(value), ([12.0], [16.0]))

        with pytest.raises(TypeError, match="every block"):
            sc.BlockDiagonalLinOp((block, object()))
        other_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
        other_x = sc.DenseCoordinateSpace((1,), other_ctx)
        other = sc.IdentityLinOp(other_x)
        with pytest.raises(ValueError, match="check policy"):
            sc.BlockDiagonalLinOp((block, other))

    def test_from_empty_operators_raises(self):
        """``from_operators(())`` rejects an empty leaf tuple."""
        with pytest.raises(Exception):
            sc.BlockDiagonalLinOp.from_operators(())

    def test_batch_checks_reject_wrong_tuple_layout(self):
        # Folded from tests/linops/test_tree_linop_batching.py.
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
        x1, x2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)
        y1, y2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx)
        A1 = sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x1, y1, ctx)
        A2 = sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x2, y2, ctx)
        op = sc.BlockDiagonalLinOp.from_operators((A1, A2))

        with pytest.raises(ValueError, match="structure"):
            op.vapply((ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]),))
        with pytest.raises(ValueError, match="trailing shape"):
            op.vapply(
                (ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]), ctx.asarray([[3.0, 4.0]]))
            )

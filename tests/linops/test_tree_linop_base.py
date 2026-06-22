"""Tests for :class:`spacecore.TreeLinOp` — abstract base for tree-shaped ops.

Checklist item 14:

* ``TreeLinOp`` is abstract — direct instantiation requires implementing
  ``_check_layout`` and ``from_operators``.
* The constructor stores the ``parts`` tuple and rejects empty ``parts``.
* ``parts`` is exposed as a tuple of converted operators.
* ``__eq__`` compares layout (domain, codomain, and parts).
* ``tree_flatten`` / ``tree_unflatten`` round-trip preserves the layout.

Concrete subclass coverage lives in the per-subclass files
(``test_block_diagonal_linop.py``, ``test_block_matrix_linop.py``,
``test_stacked_linop.py``, ``test_sum_to_single_linop.py``).
"""
from __future__ import annotations

import pytest

import spacecore as sc


# ===========================================================================
# Empty parts → ValueError
# ===========================================================================
class TestEmptyParts:
    def test_rejects_empty_parts_via_concrete_subclass(self, numpy_ctx):
        """Every TreeLinOp subclass forwards through the base constructor
        which raises on empty ``parts``."""
        with pytest.raises(ValueError, match="at least one block"):
            sc.BlockDiagonalLinOp.from_operators(())
        with pytest.raises(ValueError, match="Parts must be non-empty"):
            sc.StackedLinOp.from_operators(())
        with pytest.raises(ValueError, match="Parts must be non-empty"):
            sc.SumToSingleLinOp.from_operators(())


# ===========================================================================
# parts tuple exposure
# ===========================================================================
class TestParts:
    def test_parts_are_tuple(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        blocks = (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(X, numpy_ctx))
        op = sc.BlockDiagonalLinOp.from_operators(blocks)
        assert isinstance(op.parts, tuple)
        assert len(op.parts) == 2


# ===========================================================================
# __eq__: layout comparison
# ===========================================================================
class TestEquality:
    def test_same_layout_equal(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        a = sc.BlockDiagonalLinOp.from_operators(
            (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(X, numpy_ctx))
        )
        b = sc.BlockDiagonalLinOp.from_operators(
            (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(X, numpy_ctx))
        )
        assert a == b

    def test_different_part_count_not_equal(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        a = sc.BlockDiagonalLinOp.from_operators(
            (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(Y, numpy_ctx))
        )
        b = sc.BlockDiagonalLinOp.from_operators((sc.IdentityLinOp(X, numpy_ctx),))
        assert a != b


# ===========================================================================
# Pytree round-trip
# ===========================================================================
class TestPytreeRoundTrip:
    def test_block_diagonal_round_trip(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.BlockDiagonalLinOp.from_operators(
            (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(X, numpy_ctx))
        )
        children, aux = op.tree_flatten()
        rebuilt = sc.BlockDiagonalLinOp.tree_unflatten(aux, children)
        assert rebuilt == op

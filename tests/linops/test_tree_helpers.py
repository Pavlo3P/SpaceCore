"""Tests for the private tree-helper utilities in :mod:`spacecore.linop.tree._block`.

Checklist items 19 and 20 (gap-fill — these private helpers had no dedicated
unit tests; they were only exercised via the BlockDiagonalLinOp /
BlockMatrixLinOp constructors).

* ``_validate_blocks`` — accepts a well-formed nonempty sequence; rejects
  empty tuples, non-LinOp entries, and entries with mismatched contexts.
* ``_sum_values`` — sums via ``space.add`` (unbatched) or ``space.add_batch``
  (batched); single-element sequence returns that element unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.linop.tree._block import _sum_values, _validate_blocks


# ===========================================================================
# _validate_blocks
# ===========================================================================
class TestValidateBlocks:
    def test_accepts_well_formed_blocks(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        blocks = (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(X, numpy_ctx))
        out = _validate_blocks(blocks, "TestOwner")
        assert out == blocks

    def test_rejects_empty_sequence(self):
        with pytest.raises(ValueError, match="at least one block"):
            _validate_blocks((), "TestOwner")

    def test_rejects_non_linop_entry(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        good = sc.IdentityLinOp(X, numpy_ctx)
        bad = object()
        with pytest.raises(TypeError, match="every block to be a LinOp"):
            _validate_blocks((good, bad), "TestOwner")

    def test_rejects_none_entry(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        good = sc.IdentityLinOp(X, numpy_ctx)
        with pytest.raises(TypeError, match="every block to be a LinOp"):
            _validate_blocks((good, None), "TestOwner")

    def test_rejects_blocks_with_different_dtype(self, numpy_ctx, numpy_f32_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Xf = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        with pytest.raises(ValueError, match="same mathematical context"):
            _validate_blocks(
                (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(Xf, numpy_f32_ctx)),
                "TestOwner",
            )

    def test_rejects_blocks_with_different_check_level(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cheap_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
        Xc = sc.DenseCoordinateSpace((2,), cheap_ctx)
        with pytest.raises(ValueError, match="same check policy"):
            _validate_blocks(
                (sc.IdentityLinOp(X, numpy_ctx), sc.IdentityLinOp(Xc, cheap_ctx)),
                "TestOwner",
            )

    def test_owner_name_in_message(self):
        with pytest.raises(ValueError, match="MyOwner"):
            _validate_blocks((), "MyOwner")


# ===========================================================================
# _sum_values
# ===========================================================================
class TestSumValues:
    def test_unbatched_uses_space_add(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        values = (
            numpy_ctx.asarray([1.0, 2.0, 3.0]),
            numpy_ctx.asarray([10.0, 20.0, 30.0]),
            numpy_ctx.asarray([100.0, 200.0, 300.0]),
        )
        out = _sum_values(space, values, batched=False)
        np.testing.assert_allclose(out, [111.0, 222.0, 333.0])

    def test_batched_uses_space_add_batch(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        values = (
            numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            numpy_ctx.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        )
        out = _sum_values(space, values, batched=True)
        np.testing.assert_allclose(out, [[11, 22, 33], [44, 55, 66]])

    def test_single_element_returns_unchanged(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        # No `add` is invoked when there's only one element.
        np.testing.assert_allclose(_sum_values(space, (x,), batched=False), x)
        np.testing.assert_allclose(_sum_values(space, (x,), batched=True), x)

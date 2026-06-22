"""Tests that each optimized kernel matches its generic reference.

Checklist section 9:

* ``block-diagonal-dense-apply``: optimized == generic across a
  block-count / block-size grid, and both equal an independent NumPy
  ground truth (per-block ``matrix @ leaf``). Applicability rejects
  mismatched and empty sequences.
* ``composed-chain-apply``: optimized == generic across a chain-length /
  size grid, and both equal an independent NumPy ground truth
  (sequential ``matrix @ x``, rightmost applied first). An empty chain
  returns the input unchanged.

The ``correctness_ref`` field on each :class:`KernelSpec` points back at
the per-kernel parity test below. These tests carry their own NumPy
ground truth so they have no dependency on the (removed) bench harness.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy
from spacecore.kernels.block_diagonal import (
    block_diagonal_dense_apply_applicable,
    block_diagonal_dense_apply_generic,
    block_diagonal_dense_apply_optimized,
)
from spacecore.kernels.composed import (
    composed_chain_apply_applicable,
    composed_chain_apply_generic,
    composed_chain_apply_optimized,
)


# ===========================================================================
# block-diagonal-dense-apply
# ===========================================================================
class TestBlockDiagonalDenseApply:
    @pytest.mark.parametrize(
        "block_count,block_size",
        [(1, 4), (2, 8), (4, 32), (8, 16), (16, 64)],
    )
    def test_matches_generic(self, numpy_ctx, block_count, block_size):
        """Optimized == generic == NumPy ground truth across the grid."""
        ops = numpy_ctx.ops
        rng = np.random.default_rng(seed=block_count * 1000 + block_size)
        matrices = tuple(
            numpy_ctx.asarray(rng.standard_normal((block_size, block_size)))
            for _ in range(block_count)
        )
        leaves = tuple(
            numpy_ctx.asarray(rng.standard_normal(block_size))
            for _ in range(block_count)
        )
        assert block_diagonal_dense_apply_applicable(matrices, leaves, ops)

        generic = block_diagonal_dense_apply_generic(matrices, leaves, ops)
        optimized = block_diagonal_dense_apply_optimized(matrices, leaves, ops)
        # Independent NumPy ground truth: per-block matrix @ leaf.
        reference = tuple(
            to_numpy(m) @ to_numpy(x) for m, x in zip(matrices, leaves)
        )

        assert len(generic) == len(optimized) == block_count
        for g, o, r in zip(generic, optimized, reference):
            np.testing.assert_array_equal(to_numpy(g), to_numpy(o))
            np.testing.assert_allclose(to_numpy(o), r, rtol=1e-12, atol=1e-12)

    def test_rejects_mismatched_lengths(self, numpy_ctx):
        """Applicability returns False on shape-mismatched inputs."""
        matrix = numpy_ctx.asarray(np.eye(4))
        leaf = numpy_ctx.asarray(np.zeros(4))
        assert not block_diagonal_dense_apply_applicable(
            (matrix, matrix), (leaf,), numpy_ctx.ops
        )

    def test_rejects_empty_inputs(self, numpy_ctx):
        """Applicability returns False for empty sequences."""
        assert not block_diagonal_dense_apply_applicable(
            (), (), numpy_ctx.ops
        )


# ===========================================================================
# composed-chain-apply
# ===========================================================================
class TestComposedChainApply:
    @pytest.mark.parametrize("chain_length", [1, 2, 4, 8])
    @pytest.mark.parametrize("n", [16, 64, 128])
    def test_matches_generic(self, numpy_ctx, chain_length, n):
        """Optimized == generic == NumPy ground truth across the grid."""
        rng = np.random.default_rng(seed=chain_length * 100 + n)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        arrays = [rng.standard_normal((n, n)) for _ in range(chain_length)]
        linops = tuple(
            sc.DenseLinOp(numpy_ctx.asarray(a), space, space, numpy_ctx)
            for a in arrays
        )
        x = rng.standard_normal(n)
        x_sc = numpy_ctx.asarray(x)

        assert composed_chain_apply_applicable(linops, x_sc)
        generic = composed_chain_apply_generic(linops, x_sc)
        optimized = composed_chain_apply_optimized(linops, x_sc)

        # Independent NumPy ground truth: apply rightmost operator first.
        reference = x
        for a in reversed(arrays):
            reference = a @ reference

        np.testing.assert_array_equal(to_numpy(generic), to_numpy(optimized))
        np.testing.assert_allclose(
            to_numpy(optimized), reference, rtol=1e-12, atol=1e-12
        )

    def test_empty_chain_returns_input_identity(self, numpy_ctx):
        """An empty chain returns the input unchanged (matches generic)."""
        x = numpy_ctx.asarray(np.asarray([1.0, 2.0, 3.0]))
        assert composed_chain_apply_optimized((), x) is x
        assert composed_chain_apply_generic((), x) is x

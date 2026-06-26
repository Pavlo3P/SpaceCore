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
from spacecore.kernels.specs.block_diagonal import (
    block_diagonal_dense_apply_applicable,
    block_diagonal_dense_apply_generic,
    block_diagonal_dense_apply_optimized,
)
from spacecore.kernels.specs.composed import (
    composed_chain_apply_applicable,
    composed_chain_apply_generic,
    composed_chain_apply_optimized,
)
from spacecore.kernels.specs.composed_simplify import (
    composed_chain_apply_generic as composed_simplify_generic,
    composed_identity_applicable,
    composed_identity_optimized,
    composed_zero_applicable,
    composed_zero_optimized,
)
from spacecore.kernels.specs.block_batched import (
    block_batched_applicable,
    block_batched_cost,
    block_batched_optimized,
    block_batched_rapply_applicable,
    block_batched_rapply_cost,
    block_batched_rapply_optimized,
    block_batched_rvapply_applicable,
    block_batched_rvapply_cost,
    block_batched_rvapply_optimized,
    block_batched_vapply_applicable,
    block_batched_vapply_cost,
    block_batched_vapply_optimized,
    block_diagonal_apply_generic,
    block_diagonal_rapply_generic,
    block_diagonal_rvapply_generic,
    block_diagonal_vapply_generic,
)
from spacecore.kernels.specs.stacked_batched import (
    stacked_apply_applicable,
    stacked_apply_cost,
    stacked_apply_generic,
    stacked_apply_optimized,
    sum_to_single_rapply_applicable,
    sum_to_single_rapply_cost,
    sum_to_single_rapply_generic,
    sum_to_single_rapply_optimized,
)
from spacecore.linop._algebra import IdentityLinOp, ZeroLinOp


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


# ===========================================================================
# composed-zero-annihilation  (ADR-016 dispatch spec)
# ===========================================================================
def _dense(numpy_ctx, space, array):
    return sc.DenseLinOp(numpy_ctx.asarray(array), space, space, numpy_ctx)


class TestComposedZeroAnnihilation:
    @pytest.mark.parametrize("zero_pos", [0, 1, 2])
    @pytest.mark.parametrize("n", [4, 16])
    def test_matches_generic(self, numpy_ctx, zero_pos, n):
        """optimized == generic == zeros for any chain with a zero leaf."""
        rng = np.random.default_rng(seed=zero_pos * 10 + n)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        leaves = [
            _dense(numpy_ctx, space, rng.standard_normal((n, n))) for _ in range(3)
        ]
        leaves[zero_pos] = ZeroLinOp(space, space, numpy_ctx)
        chain = tuple(leaves)
        x = numpy_ctx.asarray(rng.standard_normal(n))

        assert composed_zero_applicable(chain, x)
        optimized = composed_zero_optimized(chain, x)
        generic = composed_simplify_generic(chain, x)

        # Exact agreement with the generic chain and with a zero ground truth.
        np.testing.assert_array_equal(to_numpy(optimized), to_numpy(generic))
        np.testing.assert_array_equal(to_numpy(optimized), np.zeros(n))

    def test_not_applicable_without_zero(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        chain = (_dense(numpy_ctx, space, np.eye(4)),)
        x = numpy_ctx.asarray(np.ones(4))
        assert not composed_zero_applicable(chain, x)


# ===========================================================================
# composed-identity-elision  (ADR-016 dispatch spec)
# ===========================================================================
class TestComposedIdentityElision:
    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_matches_generic(self, numpy_ctx, n):
        """optimized (skip identities) == generic == hand-computed product."""
        rng = np.random.default_rng(seed=n)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        a0 = rng.standard_normal((n, n))
        a1 = rng.standard_normal((n, n))
        # Chain in application order: a0, I, a1, I  (identities interspersed).
        chain = (
            _dense(numpy_ctx, space, a0),
            IdentityLinOp(space, numpy_ctx),
            _dense(numpy_ctx, space, a1),
            IdentityLinOp(space, numpy_ctx),
        )
        x = rng.standard_normal(n)
        x_sc = numpy_ctx.asarray(x)

        assert composed_identity_applicable(chain, x_sc)
        optimized = composed_identity_optimized(chain, x_sc)
        generic = composed_simplify_generic(chain, x_sc)

        # Ground truth: apply only the dense leaves in order (a0 first, then a1).
        reference = a1 @ (a0 @ x)
        np.testing.assert_array_equal(to_numpy(optimized), to_numpy(generic))
        np.testing.assert_allclose(to_numpy(optimized), reference, rtol=1e-12, atol=1e-12)

    def test_not_applicable_without_identity(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        chain = (_dense(numpy_ctx, space, np.eye(4)),)
        x = numpy_ctx.asarray(np.ones(4))
        assert not composed_identity_applicable(chain, x)


# ===========================================================================
# block-diagonal-uniform-dense-batched  (ADR-016 dispatch spec)
# ===========================================================================
class TestBlockDiagonalUniformBatched:
    @pytest.mark.parametrize("block_count,n", [(2, 4), (4, 16), (8, 8)])
    def test_matches_generic(self, numpy_ctx, block_count, n):
        """Batched matmul == per-block generic == NumPy ground truth, exactly."""
        rng = np.random.default_rng(seed=block_count * 100 + n)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        mats = [rng.standard_normal((n, n)) for _ in range(block_count)]
        parts = tuple(_dense(numpy_ctx, space, m) for m in mats)
        x_parts = tuple(numpy_ctx.asarray(rng.standard_normal(n)) for _ in range(block_count))

        assert block_batched_applicable(parts, x_parts)
        optimized = block_batched_optimized(parts, x_parts)
        generic = block_diagonal_apply_generic(parts, x_parts)
        reference = tuple(m @ to_numpy(x) for m, x in zip(mats, x_parts))

        assert len(optimized) == len(generic) == block_count
        for o, g, r in zip(optimized, generic, reference):
            np.testing.assert_array_equal(to_numpy(o), to_numpy(g))
            np.testing.assert_array_equal(to_numpy(o), r)

    def test_cost_is_shape_only_and_positive(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((8,), numpy_ctx)
        parts = tuple(_dense(numpy_ctx, space, np.eye(8)) for _ in range(4))
        x_parts = tuple(numpy_ctx.asarray(np.zeros(8)) for _ in range(4))
        cost = block_batched_cost(parts, x_parts)
        assert cost is not None
        # 4*(8*8) + 4*8 + 4*8 = 256 + 32 + 32 = 320 elements * 8 bytes.
        assert cost.peak_bytes == 320 * 8
        assert cost.flops > 0

    def test_not_applicable_for_single_block(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = (_dense(numpy_ctx, space, np.eye(4)),)
        x_parts = (numpy_ctx.asarray(np.ones(4)),)
        assert not block_batched_applicable(parts, x_parts)
        assert block_batched_cost(parts, x_parts) is None

    def test_not_applicable_for_nonuniform_shapes(self, numpy_ctx):
        s4 = sc.DenseCoordinateSpace((4,), numpy_ctx)
        s8 = sc.DenseCoordinateSpace((8,), numpy_ctx)
        parts = (
            _dense(numpy_ctx, s4, np.eye(4)),
            _dense(numpy_ctx, s8, np.eye(8)),
        )
        x_parts = (numpy_ctx.asarray(np.ones(4)), numpy_ctx.asarray(np.ones(8)))
        assert not block_batched_applicable(parts, x_parts)

    def test_not_applicable_for_heterogeneous_operand_dtypes(self, numpy_ctx):
        # At check_level="none" a tree element may carry mixed-dtype leaves;
        # stacking would promote them and compute a block in the wrong
        # precision, so the spec must fall back to the exact per-block loop.
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = (_dense(numpy_ctx, space, np.eye(4)), _dense(numpy_ctx, space, np.eye(4)))
        x_parts = (
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float64),
        )
        assert not block_batched_applicable(parts, x_parts)


# ===========================================================================
# block-diagonal-uniform-dense-batched-rapply  (ADR-016 dispatch spec)
# ===========================================================================
class TestBlockDiagonalUniformRapply:
    @pytest.mark.parametrize("block_count,n", [(2, 4), (4, 16), (8, 8)])
    def test_matches_generic(self, numpy_ctx, block_count, n):
        """Batched adjoint matmul == per-block generic == NumPy ground truth, exactly."""
        rng = np.random.default_rng(seed=block_count * 100 + n + 7)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        mats = [rng.standard_normal((n, n)) for _ in range(block_count)]
        parts = tuple(_dense(numpy_ctx, space, m) for m in mats)
        y_parts = tuple(numpy_ctx.asarray(rng.standard_normal(n)) for _ in range(block_count))

        assert block_batched_rapply_applicable(parts, y_parts)
        optimized = block_batched_rapply_optimized(parts, y_parts)
        generic = block_diagonal_rapply_generic(parts, y_parts)
        # Independent NumPy ground truth: per-block conjugate-transpose @ y.
        reference = tuple(to_numpy(m).conj().T @ to_numpy(y) for m, y in zip(mats, y_parts))

        assert len(optimized) == len(generic) == block_count
        for o, g, r in zip(optimized, generic, reference):
            np.testing.assert_array_equal(to_numpy(o), to_numpy(g))
            np.testing.assert_array_equal(to_numpy(o), r)

    def test_cost_is_shape_only_and_positive(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((8,), numpy_ctx)
        parts = tuple(_dense(numpy_ctx, space, np.eye(8)) for _ in range(4))
        y_parts = tuple(numpy_ctx.asarray(np.zeros(8)) for _ in range(4))
        cost = block_batched_rapply_cost(parts, y_parts)
        assert cost is not None
        # 4*(8*8) + 4*8 + 4*8 = 320 elements * 8 bytes.
        assert cost.peak_bytes == 320 * 8
        assert cost.flops > 0

    def test_not_applicable_for_single_block(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = (_dense(numpy_ctx, space, np.eye(4)),)
        y_parts = (numpy_ctx.asarray(np.ones(4)),)
        assert not block_batched_rapply_applicable(parts, y_parts)
        assert block_batched_rapply_cost(parts, y_parts) is None

    def test_not_applicable_for_nonuniform_shapes(self, numpy_ctx):
        s4 = sc.DenseCoordinateSpace((4,), numpy_ctx)
        s8 = sc.DenseCoordinateSpace((8,), numpy_ctx)
        parts = (_dense(numpy_ctx, s4, np.eye(4)), _dense(numpy_ctx, s8, np.eye(8)))
        y_parts = (numpy_ctx.asarray(np.ones(4)), numpy_ctx.asarray(np.ones(8)))
        assert not block_batched_rapply_applicable(parts, y_parts)


# ===========================================================================
# block-diagonal-uniform-dense-batched-vapply / -rvapply  (ADR-016 dispatch)
# ===========================================================================
class TestBlockDiagonalUniformVapply:
    @pytest.mark.parametrize("block_count,n,N", [(2, 4, 3), (4, 16, 5), (8, 8, 2)])
    def test_matches_generic(self, numpy_ctx, block_count, n, N):
        """Batched vapply (X @ A2T) == per-block generic == NumPy ground truth, exactly."""
        rng = np.random.default_rng(seed=block_count * 100 + n + N)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        mats = [rng.standard_normal((n, n)) for _ in range(block_count)]
        parts = tuple(_dense(numpy_ctx, space, m) for m in mats)
        x_parts = tuple(
            numpy_ctx.asarray(rng.standard_normal((N, n))) for _ in range(block_count)
        )

        assert block_batched_vapply_applicable(parts, x_parts)
        optimized = block_batched_vapply_optimized(parts, x_parts)
        generic = block_diagonal_vapply_generic(parts, x_parts)
        # Ground truth in the core's transpose-on-right orientation: X @ A2.T.
        reference = tuple(to_numpy(x) @ to_numpy(m).T for m, x in zip(mats, x_parts))

        assert len(optimized) == len(generic) == block_count
        for o, g, r in zip(optimized, generic, reference):
            np.testing.assert_array_equal(to_numpy(o), to_numpy(g))
            np.testing.assert_array_equal(to_numpy(o), r)

    def test_cost_is_shape_only_and_positive(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((8,), numpy_ctx)
        parts = tuple(_dense(numpy_ctx, space, np.eye(8)) for _ in range(4))
        x_parts = tuple(numpy_ctx.asarray(np.zeros((3, 8))) for _ in range(4))
        cost = block_batched_vapply_cost(parts, x_parts)
        assert cost is not None
        # mats 4*(8*8) + batch 4*3*8 + out 4*3*8 = 256 + 96 + 96 = 448 elems * 8 bytes.
        assert cost.peak_bytes == 448 * 8
        assert cost.flops > 0

    def test_not_applicable_for_single_block(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = (_dense(numpy_ctx, space, np.eye(4)),)
        x_parts = (numpy_ctx.asarray(np.ones((2, 4))),)
        assert not block_batched_vapply_applicable(parts, x_parts)


class TestBlockDiagonalUniformRvapply:
    @pytest.mark.parametrize("block_count,n,N", [(2, 4, 3), (4, 16, 5), (8, 8, 2)])
    def test_matches_generic(self, numpy_ctx, block_count, n, N):
        """Batched rvapply (Y @ A2H.T) == per-block generic == NumPy ground truth, exactly."""
        rng = np.random.default_rng(seed=block_count * 100 + n + N + 11)
        space = sc.DenseCoordinateSpace((n,), numpy_ctx)
        mats = [rng.standard_normal((n, n)) for _ in range(block_count)]
        parts = tuple(_dense(numpy_ctx, space, m) for m in mats)
        y_parts = tuple(
            numpy_ctx.asarray(rng.standard_normal((N, n))) for _ in range(block_count)
        )

        assert block_batched_rvapply_applicable(parts, y_parts)
        optimized = block_batched_rvapply_optimized(parts, y_parts)
        generic = block_diagonal_rvapply_generic(parts, y_parts)
        # Ground truth: Y @ conj(A2).  A2H.T == conj(A2); for real mats this is Y @ A2.
        reference = tuple(to_numpy(y) @ to_numpy(m).conj() for m, y in zip(mats, y_parts))

        assert len(optimized) == len(generic) == block_count
        for o, g, r in zip(optimized, generic, reference):
            np.testing.assert_array_equal(to_numpy(o), to_numpy(g))
            np.testing.assert_array_equal(to_numpy(o), r)

    def test_not_applicable_for_single_block(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = (_dense(numpy_ctx, space, np.eye(4)),)
        y_parts = (numpy_ctx.asarray(np.ones((2, 4))),)
        assert not block_batched_rvapply_applicable(parts, y_parts)
        assert block_batched_rvapply_cost(parts, y_parts) is None


# ===========================================================================
# stacked-uniform-dense-batched-apply  (ADR-016 dispatch spec)
# ===========================================================================
def _rect_dense(numpy_ctx, X, Y, array):
    return sc.DenseLinOp(numpy_ctx.asarray(array), X, Y, numpy_ctx)


class TestStackedUniformApply:
    @pytest.mark.parametrize("part_count,m,n", [(2, 4, 4), (4, 8, 6), (8, 3, 5)])
    def test_matches_generic(self, numpy_ctx, part_count, m, n):
        """Broadcast apply (M_k @ x) == per-component generic == NumPy ground truth."""
        rng = np.random.default_rng(seed=part_count * 100 + m * 10 + n)
        X = sc.DenseCoordinateSpace((n,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((m,), numpy_ctx)
        mats = [rng.standard_normal((m, n)) for _ in range(part_count)]
        parts = tuple(_rect_dense(numpy_ctx, X, Y, mat) for mat in mats)
        x = numpy_ctx.asarray(rng.standard_normal(n))

        assert stacked_apply_applicable(parts, x)
        optimized = stacked_apply_optimized(parts, x)
        generic = stacked_apply_generic(parts, x)
        reference = tuple(to_numpy(mat) @ to_numpy(x) for mat in mats)

        assert len(optimized) == len(generic) == part_count
        for o, g, r in zip(optimized, generic, reference):
            np.testing.assert_array_equal(to_numpy(o), to_numpy(g))
            np.testing.assert_array_equal(to_numpy(o), r)

    def test_cost_is_shape_only_and_positive(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((6,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = tuple(_rect_dense(numpy_ctx, X, Y, np.zeros((4, 6))) for _ in range(3))
        x = numpy_ctx.asarray(np.zeros(6))
        cost = stacked_apply_cost(parts, x)
        assert cost is not None
        # mats 3*(4*6) + one shared vec 6 + out 3*4 = 72 + 6 + 12 = 90 elems * 8 bytes.
        assert cost.peak_bytes == 90 * 8
        assert cost.flops > 0

    def test_not_applicable_for_nonuniform_codomain(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        Y4 = sc.DenseCoordinateSpace((4,), numpy_ctx)
        Y8 = sc.DenseCoordinateSpace((8,), numpy_ctx)
        parts = (
            _rect_dense(numpy_ctx, X, Y4, np.zeros((4, 4))),
            _rect_dense(numpy_ctx, X, Y8, np.zeros((8, 4))),
        )
        x = numpy_ctx.asarray(np.ones(4))
        assert not stacked_apply_applicable(parts, x)
        assert stacked_apply_cost(parts, x) is None


# ===========================================================================
# sum-to-single-uniform-dense-batched-rapply  (ADR-016 dispatch spec)
# ===========================================================================
class TestSumToSingleUniformRapply:
    @pytest.mark.parametrize("part_count,m,n", [(2, 4, 4), (4, 6, 8), (8, 5, 3)])
    def test_matches_generic(self, numpy_ctx, part_count, m, n):
        """Broadcast rapply (M_k^H @ y) == per-component generic == NumPy ground truth."""
        rng = np.random.default_rng(seed=part_count * 100 + m * 10 + n + 5)
        X = sc.DenseCoordinateSpace((n,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((m,), numpy_ctx)
        mats = [rng.standard_normal((m, n)) for _ in range(part_count)]
        parts = tuple(_rect_dense(numpy_ctx, X, Y, mat) for mat in mats)
        y = numpy_ctx.asarray(rng.standard_normal(m))

        assert sum_to_single_rapply_applicable(parts, y)
        optimized = sum_to_single_rapply_optimized(parts, y)
        generic = sum_to_single_rapply_generic(parts, y)
        reference = tuple(to_numpy(mat).conj().T @ to_numpy(y) for mat in mats)

        assert len(optimized) == len(generic) == part_count
        for o, g, r in zip(optimized, generic, reference):
            np.testing.assert_array_equal(to_numpy(o), to_numpy(g))
            np.testing.assert_array_equal(to_numpy(o), r)

    def test_not_applicable_for_nonuniform_domain(self, numpy_ctx):
        X4 = sc.DenseCoordinateSpace((4,), numpy_ctx)
        X8 = sc.DenseCoordinateSpace((8,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((4,), numpy_ctx)
        parts = (
            _rect_dense(numpy_ctx, X4, Y, np.zeros((4, 4))),
            _rect_dense(numpy_ctx, X8, Y, np.zeros((4, 8))),
        )
        y = numpy_ctx.asarray(np.ones(4))
        assert not sum_to_single_rapply_applicable(parts, y)
        assert sum_to_single_rapply_cost(parts, y) is None

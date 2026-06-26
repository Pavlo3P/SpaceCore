"""Tests for ADR-021 Tier-2 ``fuse()`` operator multiplication.

``fuse()`` collapses maximal subtrees of densely-fusible operators into a single
materialized ``DenseLinOp`` (multiplying the matrices), leaving matrix-free and
other non-materializable leaves intact. Because fusing reassociates the
arithmetic (matrix product then apply vs. apply in sequence), correctness is
checked with ``allclose``, not ``array_equal`` (per ADR-021, Tier-2 is not held
to bit-exactness).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy


def _dense(ctx, dom, cod, array):
    return sc.DenseLinOp(ctx.asarray(array), dom, cod, ctx)


class TestFuseComposed:
    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_dense_composition_fuses_to_single_dense(self, numpy_ctx, n):
        """A @ B of dense operators fuses to one DenseLinOp with the matrix product."""
        rng = np.random.default_rng(n)
        X = sc.DenseCoordinateSpace((n,), numpy_ctx)
        A_mat = rng.standard_normal((n, n))
        B_mat = rng.standard_normal((n, n))
        A = _dense(numpy_ctx, X, X, A_mat)
        B = _dense(numpy_ctx, X, X, B_mat)
        lazy = A @ B
        fused = lazy.fuse()

        assert isinstance(fused, sc.DenseLinOp)
        x = numpy_ctx.asarray(rng.standard_normal(n))
        np.testing.assert_allclose(to_numpy(fused.apply(x)), to_numpy(lazy.apply(x)), rtol=1e-10, atol=1e-12)
        # The fused matrix is the matrix product.
        np.testing.assert_allclose(to_numpy(fused.to_matrix()), A_mat @ B_mat, rtol=1e-10, atol=1e-12)

    def test_dense_chain_fuses_to_single_dense(self, numpy_ctx):
        """A @ B @ C of dense operators collapses to one DenseLinOp."""
        rng = np.random.default_rng(7)
        n = 12
        X = sc.DenseCoordinateSpace((n,), numpy_ctx)
        mats = [rng.standard_normal((n, n)) for _ in range(3)]
        A, B, C = (_dense(numpy_ctx, X, X, m) for m in mats)
        lazy = A @ B @ C
        fused = lazy.fuse()

        assert isinstance(fused, sc.DenseLinOp)
        x = numpy_ctx.asarray(rng.standard_normal(n))
        np.testing.assert_allclose(to_numpy(fused.apply(x)), to_numpy(lazy.apply(x)), rtol=1e-10, atol=1e-12)

    def test_rectangular_composition_preserves_domain_codomain(self, numpy_ctx):
        """B: X->Y, A: Y->Z fuses to a single DenseLinOp X->Z with the right shapes."""
        rng = np.random.default_rng(3)
        X = sc.DenseCoordinateSpace((5,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((4,), numpy_ctx)
        Z = sc.DenseCoordinateSpace((3,), numpy_ctx)
        B = _dense(numpy_ctx, X, Y, rng.standard_normal((4, 5)))
        A = _dense(numpy_ctx, Y, Z, rng.standard_normal((3, 4)))
        lazy = A @ B
        fused = lazy.fuse()

        assert isinstance(fused, sc.DenseLinOp)
        assert fused.domain == X and fused.codomain == Z
        x = numpy_ctx.asarray(rng.standard_normal(5))
        np.testing.assert_allclose(to_numpy(fused.apply(x)), to_numpy(lazy.apply(x)), rtol=1e-10, atol=1e-12)

    def test_leaf_fuse_returns_self(self, numpy_ctx):
        """A dense leaf is already fused; fuse() returns it unchanged."""
        X = sc.DenseCoordinateSpace((4,), numpy_ctx)
        A = _dense(numpy_ctx, X, X, np.eye(4))
        assert A.fuse() is A


class TestFuseMatrixFreeRail:
    def test_matrix_free_operand_is_not_densified(self, numpy_ctx):
        """A @ M with M matrix-free stays lazy; the matrix-free leaf is preserved."""
        rng = np.random.default_rng(11)
        n = 6
        X = sc.DenseCoordinateSpace((n,), numpy_ctx)
        A_mat = rng.standard_normal((n, n))
        M_mat = rng.standard_normal((n, n))
        A = _dense(numpy_ctx, X, X, A_mat)
        M = sc.MatrixFreeLinOp(
            lambda x: numpy_ctx.asarray(M_mat) @ x,
            lambda y: numpy_ctx.asarray(M_mat).T @ y,
            X, X, numpy_ctx,
        )
        lazy = A @ M
        fused = lazy.fuse()

        # The matrix-free operand must NOT be materialized into a dense operator.
        assert not isinstance(fused, sc.DenseLinOp)
        x = numpy_ctx.asarray(rng.standard_normal(n))
        np.testing.assert_allclose(to_numpy(fused.apply(x)), to_numpy(lazy.apply(x)), rtol=1e-10, atol=1e-12)

    def test_dense_pair_around_matrix_free_fuses_only_the_dense_neighbours(self, numpy_ctx):
        """A @ B with both dense fuses even when an unrelated matrix-free op exists."""
        rng = np.random.default_rng(5)
        n = 5
        X = sc.DenseCoordinateSpace((n,), numpy_ctx)
        A = _dense(numpy_ctx, X, X, rng.standard_normal((n, n)))
        B = _dense(numpy_ctx, X, X, rng.standard_normal((n, n)))
        M = sc.MatrixFreeLinOp(lambda x: x, lambda y: y, X, X, numpy_ctx)
        lazy = (A @ B) @ M
        fused = lazy.fuse()
        # Outer composition keeps M lazy, but A @ B inside is collapsed.
        assert not isinstance(fused, sc.DenseLinOp)
        x = numpy_ctx.asarray(rng.standard_normal(n))
        np.testing.assert_allclose(to_numpy(fused.apply(x)), to_numpy(lazy.apply(x)), rtol=1e-10, atol=1e-12)


class TestFuseMetricAdjoint:
    def test_weighted_composition_adjoint_is_consistent(self, numpy_ctx):
        """On a non-Euclidean (weighted) space, the fused metric adjoint matches.

        The shared middle-space Riesz maps cancel, so the fused DenseLinOp's
        metric adjoint equals B* @ A* up to rounding — the load-bearing
        correctness claim for fusing across a non-Euclidean geometry.
        """
        rng = np.random.default_rng(21)
        n = 4
        weights = numpy_ctx.asarray(np.array([2.0, 5.0, 11.0, 3.0]))
        W = sc.DenseCoordinateSpace((n,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights))
        A = _dense(numpy_ctx, W, W, rng.standard_normal((n, n)))
        B = _dense(numpy_ctx, W, W, rng.standard_normal((n, n)))
        lazy = A @ B
        fused = lazy.fuse()
        assert isinstance(fused, sc.DenseLinOp)

        x = numpy_ctx.asarray(rng.standard_normal(n))
        y = numpy_ctx.asarray(rng.standard_normal(n))
        np.testing.assert_allclose(to_numpy(fused.apply(x)), to_numpy(lazy.apply(x)), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(to_numpy(fused.rapply(y)), to_numpy(lazy.rapply(y)), rtol=1e-10, atol=1e-12)

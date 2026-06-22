"""Tests for :class:`spacecore.DiagonalLinOp` — coordinatewise diagonal operator.

Checklist section 6:

* ``apply`` / ``rapply`` — flat and tensor-shaped diagonals scale coordinatewise.
* Complex diagonals — adjoint uses the conjugate diagonal; ``is_hermitian``
  predicate is ``True`` for real-valued, ``False`` for genuinely complex.
* ``vapply`` / ``rvapply`` — batched application matches the per-row loop.
* ``to_matrix`` / ``to_dense`` match ``numpy.diag`` of the flattened diagonal.
* ``A`` — dense representation is cached and equals ``numpy.diag``.
* Weighted / metric adjoint — the dot-test identity holds in Euclidean,
  weighted, and complex spaces; non-Euclidean spaces without Riesz maps are
  rejected.
* Product / TreeSpace domains — apply/rapply use flatten/unflatten paths.
* Pytree round-trip and ``_convert`` dtype change.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, to_numpy


def _weighted_space(weights, ctx):
    weights = ctx.asarray(weights)
    return sc.DenseCoordinateSpace(
        tuple(weights.shape), ctx, geometry=sc.WeightedInnerProduct(weights)
    )


# ===========================================================================
# apply / rapply — flat and tensor-shaped diagonals
# ===========================================================================
class TestApplyRapply:
    def test_apply_and_rapply_flat_diagonal(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([1.0, 2.0, 3.0]), space, numpy_ctx)
        x = numpy_ctx.asarray([4.0, -1.0, 0.5])

        np.testing.assert_allclose(to_numpy(op.apply(x)), [4.0, -2.0, 1.5])
        np.testing.assert_allclose(to_numpy(op.rapply(x)), [4.0, -2.0, 1.5])

    def test_apply_and_rapply_tensor_shaped_diagonal(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), space, numpy_ctx)
        x = numpy_ctx.asarray([[2.0, -1.0], [0.5, 3.0]])

        np.testing.assert_allclose(to_numpy(op.apply(x)), [[2.0, -2.0], [1.5, 12.0]])
        np.testing.assert_allclose(to_numpy(op.rapply(x)), [[2.0, -2.0], [1.5, 12.0]])


# ===========================================================================
# Complex diagonal — conjugate adjoint and hermitian predicate
# ===========================================================================
class TestComplexHermitian:
    def test_complex_adjoint_identity_and_hermitian_predicate(self, numpy_complex_ctx):
        ctx = numpy_complex_ctx
        space = sc.DenseCoordinateSpace((2,), ctx)
        hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 0.0j, 2.0 + 0.0j]), space, ctx)
        non_hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 2.0j, 3.0 - 1.0j]), space, ctx)
        u = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
        v = ctx.asarray([1.5 + 0.5j, -2.0j])

        lhs = space.inner(non_hermitian.apply(u), v)
        rhs = space.inner(u, non_hermitian.rapply(v))

        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))
        assert hermitian.is_hermitian() is True
        assert non_hermitian.is_hermitian() is False

    def test_complex_rapply_conjugates_diagonal(self, numpy_complex_ctx):
        ctx = numpy_complex_ctx
        space = sc.DenseCoordinateSpace((2,), ctx)
        non_hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 2.0j, 3.0 - 1.0j]), space, ctx)
        v = ctx.asarray([1.5 + 0.5j, -2.0j])

        np.testing.assert_allclose(
            to_numpy(non_hermitian.rapply(v)),
            np.conj(to_numpy(non_hermitian.diagonal)) * to_numpy(v),
        )


# ===========================================================================
# vapply / rvapply — batched application matches per-row loop
# ===========================================================================
class TestBatched:
    def test_vapply_and_rvapply_with_leading_batch(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([1.0, 2.0, 3.0]), space, numpy_ctx)
        xs = numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])

        expected = np.array([[1.0, 4.0, 9.0], [4.0, -2.0, 1.5]])
        np.testing.assert_allclose(to_numpy(op.vapply(xs)), expected)
        np.testing.assert_allclose(to_numpy(op.rvapply(xs)), expected)

    def test_batched_apply_matches_loop(self, numpy_ctx):
        # Folded from tests/linops/test_batched_apply.py
        # (test_diagonal_linop_batched_apply_matches_loop).
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -1.0, 0.5]), space, numpy_ctx)
        xs = numpy_ctx.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])

        np.testing.assert_allclose(
            to_numpy(op.vapply(xs)),
            np.stack([to_numpy(op.apply(x)) for x in xs], axis=0),
        )
        np.testing.assert_allclose(
            to_numpy(op.rvapply(xs)),
            np.stack([to_numpy(op.rapply(y)) for y in xs], axis=0),
        )


# ===========================================================================
# to_dense / to_matrix / A — dense representation
# ===========================================================================
class TestToDense:
    def test_to_dense_matches_numpy_diagonal_for_tensor_space(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        diagonal = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
        op = sc.DiagonalLinOp(diagonal, space, numpy_ctx)

        flat = to_numpy(diagonal).reshape((4,))
        np.testing.assert_allclose(to_numpy(op.to_matrix()), np.diag(flat))
        np.testing.assert_allclose(
            to_numpy(op.to_dense()), np.diag(flat).reshape((2, 2, 2, 2))
        )

    def test_A_is_cached_and_equals_numpy_diag(self, numpy_ctx):
        # Folded from tests/linops/test_to_dense.py
        # (test_diagonal_linop_A_is_cached).
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([1.0, 2.0, 3.0]), space, numpy_ctx)

        A = op.A

        assert op.A is A
        np.testing.assert_allclose(to_numpy(A), np.diag([1.0, 2.0, 3.0]))


# ===========================================================================
# Adjoint dot-test identity — Euclidean / weighted / complex
# ===========================================================================
class TestAdjointIdentity:
    @staticmethod
    def _assert_adjoint_identity(op, x, y):
        lhs = op.codomain.inner(op.apply(x), y)
        rhs = op.domain.inner(x, op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    def test_euclidean_real_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_euclidean_real_adjoint_identity_for_matrix_backed_ops).
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        diagonal = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -1.0, 0.5]), space, numpy_ctx)
        z = numpy_ctx.asarray([1.0, -2.0, 0.75])
        w = numpy_ctx.asarray([-0.5, 3.0, 1.25])

        self._assert_adjoint_identity(diagonal, z, w)

    def test_euclidean_complex_adjoint_identity(self, numpy_complex_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_euclidean_complex_adjoint_identity_for_matrix_backed_ops).
        ctx = numpy_complex_ctx
        space = sc.DenseCoordinateSpace((3,), ctx)
        diagonal = sc.DiagonalLinOp(
            ctx.asarray([2.0 + 1.0j, -1.0j, 0.5 - 0.25j]), space, ctx
        )
        z = ctx.asarray([1.0 + 1.0j, -2.0, 0.75 - 0.5j])
        w = ctx.asarray([-0.5j, 3.0 + 0.25j, 1.25])

        self._assert_adjoint_identity(diagonal, z, w)

    def test_weighted_metric_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_weighted_vector_metric_adjoint_for_dense_sparse_and_diagonal).
        diagonal_space = _weighted_space([2.0, 3.0], numpy_ctx)
        diagonal = sc.DiagonalLinOp(numpy_ctx.asarray([2.0, -0.5]), diagonal_space, numpy_ctx)
        self._assert_adjoint_identity(
            diagonal, numpy_ctx.asarray([1.0, -2.0]), numpy_ctx.asarray([3.0, 0.25])
        )

    def test_weighted_metric_adjoint_identity_and_batches(self, numpy_ctx):
        space = _weighted_space([2.0, 5.0, 7.0], numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([1.0, -2.0, 0.5]), space, numpy_ctx)
        x = numpy_ctx.asarray([0.25, -1.5, 2.0])
        y = numpy_ctx.asarray([2.0, -0.5, 1.25])
        xs = numpy_ctx.asarray([[0.25, -1.5, 2.0], [3.0, 0.5, -1.0]])
        ys = numpy_ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 4.0, 0.75]])

        np.testing.assert_allclose(
            to_numpy(space.inner(op.apply(x), y)),
            to_numpy(space.inner(x, op.rapply(y))),
        )
        np.testing.assert_allclose(
            to_numpy(op.vapply(xs)), np.stack([to_numpy(op.apply(xi)) for xi in xs])
        )
        np.testing.assert_allclose(
            to_numpy(op.rvapply(ys)), np.stack([to_numpy(op.rapply(yi)) for yi in ys])
        )

    def test_non_euclidean_without_riesz_is_rejected(self, numpy_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_non_euclidean_space_without_riesz_maps_is_rejected).
        class BrokenInnerProduct(sc.InnerProduct):
            def inner(self, ops, x, y):
                return ops.vdot(x, 2.0 * y)

        class BrokenSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx):
                super().__init__(shape, ctx)
                self.geometry = BrokenInnerProduct()

        space = BrokenSpace((2,), numpy_ctx)

        with pytest.raises(TypeError, match="riesz/riesz_inverse"):
            sc.DiagonalLinOp(numpy_ctx.asarray([1.0, 2.0]), space, numpy_ctx)


# ===========================================================================
# Product / TreeSpace domains — flatten/unflatten paths
# ===========================================================================
class TestProductSpace:
    def test_product_space_diagonal_uses_flatten_unflatten_paths(self, numpy_ctx):
        ctx = numpy_ctx
        x1 = sc.DenseCoordinateSpace((2,), ctx)
        x2 = sc.DenseCoordinateSpace((1,), ctx)
        space = sc.TreeSpace.from_leaf_spaces((x1, x2), ctx)
        op = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0, 0.5]), space, ctx)
        x = (ctx.asarray([1.0, 3.0]), ctx.asarray([-2.0]))
        xs = (ctx.asarray([[1.0, 3.0], [-1.0, 4.0]]), ctx.asarray([[-2.0], [0.5]]))
        ys = (ctx.asarray([[2.0, -1.0], [0.25, 3.0]]), ctx.asarray([[4.0], [-2.0]]))

        y = op.apply(x)
        np.testing.assert_allclose(to_numpy(y[0]), [2.0, -3.0])
        np.testing.assert_allclose(to_numpy(y[1]), [-1.0])
        np.testing.assert_allclose(to_numpy(op.rapply(x)[0]), [2.0, -3.0])
        np.testing.assert_allclose(to_numpy(op.rapply(x)[1]), [-1.0])

        actual_v = op.vapply(xs)
        expected_v_rows = tuple(
            op.apply((xs[0][i], xs[1][i])) for i in range(xs[0].shape[0])
        )
        np.testing.assert_allclose(
            to_numpy(actual_v[0]), np.stack([to_numpy(row[0]) for row in expected_v_rows])
        )
        np.testing.assert_allclose(
            to_numpy(actual_v[1]), np.stack([to_numpy(row[1]) for row in expected_v_rows])
        )

        actual_rv = op.rvapply(ys)
        expected_rv_rows = tuple(
            op.rapply((ys[0][i], ys[1][i])) for i in range(ys[0].shape[0])
        )
        np.testing.assert_allclose(
            to_numpy(actual_rv[0]), np.stack([to_numpy(row[0]) for row in expected_rv_rows])
        )
        np.testing.assert_allclose(
            to_numpy(actual_rv[1]), np.stack([to_numpy(row[1]) for row in expected_rv_rows])
        )

    def test_euclidean_product_space_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_matrix_backed_ops_accept_euclidean_product_space).
        ctx = numpy_ctx
        space = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
        )
        diagonal_op = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0, 0.5]), space, ctx)
        x = (ctx.asarray([1.0, -2.0]), ctx.asarray([0.5]))
        y = (ctx.asarray([3.0, 0.25]), ctx.asarray([-1.5]))

        lhs = space.inner(diagonal_op.apply(x), y)
        rhs = space.inner(x, diagonal_op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    def test_product_space_missing_riesz_is_rejected(self, numpy_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_product_space_with_component_missing_riesz_maps_is_rejected).
        ctx = numpy_ctx

        class BrokenInnerProduct(sc.InnerProduct):
            def inner(self, ops, x, y):
                return ops.vdot(x, 2.0 * y)

        broken = sc.DenseCoordinateSpace((2,), ctx, geometry=BrokenInnerProduct())
        euclidean = sc.DenseCoordinateSpace((1,), ctx)
        product = sc.TreeSpace.from_leaf_spaces((broken, euclidean), ctx)

        with pytest.raises(TypeError, match="Riesz maps"):
            sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), product, ctx)


# ===========================================================================
# Pytree round-trip and _convert
# ===========================================================================
class TestPytreeAndConvert:
    @pytest.mark.skipif(not has_jax(), reason="jax is not installed")
    def test_pytree_flatten_unflatten_round_trip(self, numpy_ctx):
        import jax

        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([1.0, 2.0, 3.0]), space, numpy_ctx)

        leaves, treedef = jax.tree_util.tree_flatten(op)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)

        assert restored == op
        np.testing.assert_allclose(
            to_numpy(restored.apply(numpy_ctx.asarray([1.0, 1.0, 1.0]))), [1.0, 2.0, 3.0]
        )

    def test_convert_changes_context_dtype(self, numpy_ctx, numpy_f32_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DiagonalLinOp(numpy_ctx.asarray([1.0, 2.0, 3.0]), space, numpy_ctx)

        converted = op._convert(numpy_f32_ctx)

        assert converted.ctx == numpy_f32_ctx
        assert converted.domain.ctx == numpy_f32_ctx
        assert converted.diagonal.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(
            to_numpy(converted.apply(numpy_f32_ctx.asarray([1.0, 1.0, 1.0]))),
            [1.0, 2.0, 3.0],
        )

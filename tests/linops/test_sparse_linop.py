"""Tests for :class:`spacecore.SparseLinOp` — sparse coordinate matrix operator.

Checklist item 12:

* Construction + Euclidean forward/adjoint action (``apply`` / ``rapply``),
  with ``to_dense`` / ``to_matrix`` materialization.
* Rectangular and tensor-shaped Euclidean behavior with batched
  ``vapply`` / ``rvapply`` matching per-row application.
* Complex spaces use the conjugate transpose on ``rapply``.
* Subclasses of ``VectorSpace`` are accepted; coordinate product spaces
  (``TreeSpace``) are accepted on both domain and codomain.
* The stored sparse matrix ``A`` is the exact object supplied at
  construction (no copy / conversion); ``to_sparse`` returns it; the
  transpose is cached so repeated ``rapply`` does not retranspose.
* Weighted and general (non-Euclidean) metric adjoint identity:
  ``<A x, y>_cod == <x, A* y>_dom``.
* Product spaces missing Riesz maps are rejected with a ``Riesz maps``
  message.
* ``convert`` preserves the action while converting sparse storage dtype
  and keeping the space types.
* Batched lifting fast paths (with checks disabled) match the raw sparse
  matrix products.
* JAX jit of ``apply`` / ``rapply`` when jax is available.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy


class _TransposeCountingCSR(sps.csr_matrix):
    """A CSR matrix that counts how often ``.T`` is accessed."""

    def __init__(self, *args, counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter

    @property
    def T(self):
        if self.counter is not None:
            self.counter["calls"] += 1
        return super().T


def _assert_adjoint_identity(op, x, y):
    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))


# ===========================================================================
# Construction + Euclidean forward / adjoint action
# ===========================================================================
class TestEuclideanAction:
    def test_construct_apply_rapply_and_materialize(self, numpy_ctx):
        ctx = numpy_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)

        x = ctx.asarray([7.0, 8.0])
        y = ctx.asarray([1.0, -1.0, 2.0])

        np.testing.assert_allclose(to_numpy(op.apply(x)), dense @ np.asarray(x))
        np.testing.assert_allclose(to_numpy(op.rapply(y)), dense.T @ np.asarray(y))
        np.testing.assert_allclose(to_numpy(op.to_dense()), dense)
        np.testing.assert_allclose(to_numpy(op.to_matrix()), dense)

    def test_rectangular_batched_apply_and_rapply(self, numpy_ctx):
        ctx = numpy_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        dense = np.array([[1.0, -2.0], [3.0, 0.5], [0.25, 4.0]])
        op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)

        xs = ctx.asarray([[1.0, 2.0], [-3.0, 4.0], [0.5, -1.5]])
        ys = ctx.asarray([[2.0, -1.0, 0.5], [1.5, 3.0, -2.0]])

        assert op.to_dense().shape == (3, 2)
        assert op.to_matrix().shape == (3, 2)
        np.testing.assert_allclose(to_numpy(op.apply(xs[0])), dense @ np.asarray(xs[0]))
        np.testing.assert_allclose(to_numpy(op.rapply(ys[0])), dense.T @ np.asarray(ys[0]))
        for i in range(xs.shape[0]):
            np.testing.assert_allclose(to_numpy(op.vapply(xs)[i]), to_numpy(op.apply(xs[i])))
        for i in range(ys.shape[0]):
            np.testing.assert_allclose(to_numpy(op.rvapply(ys)[i]), to_numpy(op.rapply(ys[i])))

    def test_tensor_shaped_euclidean_behavior(self, numpy_ctx):
        ctx = numpy_ctx
        dom = sc.DenseCoordinateSpace((2, 2), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        dense = np.array(
            [
                [1.0, -2.0, 0.5, 3.0],
                [0.25, 4.0, -1.0, 2.0],
                [3.0, 0.5, 2.0, -0.75],
            ]
        )
        op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)
        x = ctx.asarray([[1.0, 2.0], [-3.0, 0.5]])
        y = ctx.asarray([2.0, -1.0, 0.25])
        xs = ctx.asarray([[[1.0, 0.0], [2.0, -1.0]], [[0.5, 3.0], [-2.0, 1.5]]])
        ys = ctx.asarray([[2.0, -1.0, 0.25], [0.5, 3.0, -2.0]])

        np.testing.assert_allclose(to_numpy(op.apply(x)), dense @ np.asarray(x).reshape(-1))
        np.testing.assert_allclose(
            to_numpy(op.rapply(y)), (dense.T @ np.asarray(y)).reshape((2, 2))
        )
        for i in range(xs.shape[0]):
            np.testing.assert_allclose(to_numpy(op.vapply(xs)[i]), to_numpy(op.apply(xs[i])))
        for i in range(ys.shape[0]):
            np.testing.assert_allclose(to_numpy(op.rvapply(ys)[i]), to_numpy(op.rapply(ys[i])))


# ===========================================================================
# Complex conjugate-transpose adjoint
# ===========================================================================
class TestComplexAdjoint:
    def test_complex_rapply_uses_conjugate_transpose(self, numpy_complex_ctx):
        ctx = numpy_complex_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((2,), ctx)
        dense = np.array(
            [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0j, 4.0 + 0.5j]],
            dtype=np.complex128,
        )
        op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)
        y = ctx.asarray([1.0 - 1.0j, 2.0 + 0.25j])

        np.testing.assert_allclose(to_numpy(op.rapply(y)), dense.conj().T @ np.asarray(y))


# ===========================================================================
# Space acceptance: subclasses and coordinate product spaces
# ===========================================================================
class TestSpaceAcceptance:
    def test_accepts_euclidean_vector_space_subclass(self, numpy_ctx):
        ctx = numpy_ctx

        class WeightedVectorSpace(sc.DenseCoordinateSpace):
            pass

        space = WeightedVectorSpace((2,), ctx)
        matrix = ctx.assparse([[1.0, 0.0], [0.0, 1.0]])
        op = sc.SparseLinOp(matrix, space, space, ctx)
        x = ctx.asarray([2.0, -1.0])

        assert type(op.domain) is WeightedVectorSpace
        np.testing.assert_allclose(to_numpy(op.apply(x)), to_numpy(x))
        np.testing.assert_allclose(to_numpy(op.rapply(x)), to_numpy(x))

    def test_accepts_coordinate_product_spaces(self, numpy_ctx):
        ctx = numpy_ctx
        vector = sc.DenseCoordinateSpace((2,), ctx)
        product = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
        )
        matrix = ctx.assparse(np.eye(2))
        x_product = (ctx.asarray([1.0]), ctx.asarray([-2.0]))
        x_vector = ctx.asarray([0.5, 3.0])

        to_vector = sc.SparseLinOp(matrix, product, vector, ctx)
        to_product = sc.SparseLinOp(matrix, vector, product, ctx)

        np.testing.assert_allclose(to_numpy(to_vector.apply(x_product)), [1.0, -2.0])
        y = to_product.apply(x_vector)
        np.testing.assert_allclose(to_numpy(y[0]), [0.5])
        np.testing.assert_allclose(to_numpy(y[1]), [3.0])

    def test_product_space_missing_riesz_maps_is_rejected(self, numpy_ctx):
        ctx = numpy_ctx

        class BrokenInnerProduct(sc.InnerProduct):
            def inner(self, ops, x, y):
                return ops.vdot(x, 2.0 * y)

        broken = sc.DenseCoordinateSpace((2,), ctx, geometry=BrokenInnerProduct())
        euclidean = sc.DenseCoordinateSpace((1,), ctx)
        product = sc.TreeSpace.from_leaf_spaces((broken, euclidean), ctx)

        with pytest.raises(TypeError, match="Riesz maps"):
            sc.SparseLinOp(ctx.assparse(np.eye(3)), product, product, ctx)


# ===========================================================================
# Stored sparse matrix: identity, no-copy, cached transpose
# ===========================================================================
class TestStoredMatrix:
    def test_A_and_to_sparse_return_stored_object(self, numpy_ctx):
        ctx = numpy_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        matrix = sps.csr_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        op = sc.SparseLinOp(matrix, dom, cod, ctx)

        assert op.A is matrix
        assert op.to_sparse() is matrix

    def test_to_dense_matches_apply(self, numpy_ctx):
        ctx = numpy_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        op = sc.SparseLinOp(sps.csr_matrix(dense), dom, cod, ctx)

        np.testing.assert_allclose(to_numpy(op.to_dense()), dense)
        x = ctx.asarray([7.0, 8.0])
        matrix = to_numpy(op.to_dense()).reshape(
            (int(np.prod(op.codomain.shape)), int(np.prod(op.domain.shape)))
        )
        y_from_dense = matrix @ to_numpy(op.domain.flatten(x))
        y_from_apply = to_numpy(op.codomain.flatten(op.apply(x)))
        np.testing.assert_allclose(y_from_dense, y_from_apply)

    def test_reuses_cached_transpose(self, numpy_ctx):
        ctx = numpy_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        counter = {"calls": 0}
        A = _TransposeCountingCSR([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], counter=counter)

        op = sc.SparseLinOp(A, dom, cod, ctx)
        transpose_calls = counter["calls"]

        op.rapply(ctx.asarray([1.0, -1.0, 2.0]))
        op.rapply(ctx.asarray([3.0, -2.0, 1.0]))

        assert transpose_calls == 1
        assert counter["calls"] == transpose_calls


# ===========================================================================
# Metric adjoint identity (weighted + general)
# ===========================================================================
class TestMetricAdjoint:
    def test_weighted_metric_adjoint_identity(self, numpy_ctx):
        ctx = numpy_ctx
        domain = sc.DenseCoordinateSpace(
            (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0]))
        )
        codomain = sc.DenseCoordinateSpace(
            (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
        )
        dense = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        op = sc.SparseLinOp(ctx.assparse(dense), domain, codomain, ctx)
        x = ctx.asarray([0.25, -1.5])
        y = ctx.asarray([2.0, -0.5, 1.25])

        _assert_adjoint_identity(op, x, y)

    def test_general_metric_adjoint_identity(self, numpy_ctx):
        ctx = numpy_ctx

        class ScalingInnerProduct(sc.InnerProduct):
            def __init__(self, weights):
                self.weights = weights

            def inner(self, ops, x, y):
                return ops.vdot(x, self.weights * y)

            def riesz(self, ops, x):
                return self.weights * x

            def riesz_inverse(self, ops, x):
                return x / self.weights

            @property
            def is_euclidean(self):
                return False

        domain = sc.DenseCoordinateSpace(
            (2,), ctx, geometry=ScalingInnerProduct(ctx.asarray([2.0, 5.0]))
        )
        codomain = sc.DenseCoordinateSpace(
            (3,), ctx, geometry=ScalingInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
        )
        dense = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        op = sc.SparseLinOp(ctx.assparse(dense), domain, codomain, ctx)
        x = ctx.asarray([0.25, -1.5])
        y = ctx.asarray([2.0, -0.5, 1.25])

        _assert_adjoint_identity(op, x, y)


# ===========================================================================
# Conversion
# ===========================================================================
class TestConvert:
    def test_convert_preserves_action_and_converts_sparse_storage(
        self, numpy_f32_ctx, numpy_ctx
    ):
        src = numpy_f32_ctx
        dst = numpy_ctx
        dom = sc.DenseCoordinateSpace((2,), src)
        cod = sc.DenseCoordinateSpace((3,), src)
        dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        op = sc.SparseLinOp(src.assparse(dense), dom, cod, src)

        op2 = op.convert(dst)
        x = op2.ctx.asarray([7.0, 8.0])

        assert op2 is not op
        assert type(op2.dom) is sc.DenseCoordinateSpace
        assert type(op2.cod) is sc.DenseCoordinateSpace
        assert op2.ops.get_dtype(op2.A) == dst.dtype
        np.testing.assert_allclose(
            to_numpy(op2.apply(x)), dense.astype(np.float64) @ np.asarray(x)
        )


# ===========================================================================
# Batched lifting fast paths (checks disabled)
# ===========================================================================
class TestBatchedLifting:
    def test_fast_paths_without_checks(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        sparse = ctx.assparse(sps.csr_matrix([[1.0, 0.0], [0.0, 4.0], [5.0, 6.0]]))
        op = sc.SparseLinOp(sparse, dom, cod, ctx)
        xs = ctx.asarray([[7.0, 8.0], [1.0, -1.0], [0.5, 2.0]])
        ys = ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        np.testing.assert_allclose(to_numpy(op.vapply(xs)), (sparse @ np.asarray(xs).T).T)
        np.testing.assert_allclose(to_numpy(op.rvapply(ys)), (sparse.T @ np.asarray(ys).T).T)


# ===========================================================================
# JAX jit
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJit:
    def test_jit_apply_and_rapply(self):
        import jax

        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        op = sc.SparseLinOp(
            ctx.assparse(
                np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jax_real_dtype())
            ),
            sc.DenseCoordinateSpace((2,), ctx),
            sc.DenseCoordinateSpace((3,), ctx),
            ctx,
        )
        x = ctx.asarray([7.0, 8.0])

        apply_jit = jax.jit(lambda Aop, z: Aop.apply(z))
        rapply_jit = jax.jit(lambda Aop, z: Aop.rapply(z))

        np.testing.assert_allclose(to_numpy(apply_jit(op, x)), [23.0, 53.0, 83.0])
        np.testing.assert_allclose(
            to_numpy(rapply_jit(op, ctx.asarray([1.0, -1.0, 2.0]))), [8.0, 10.0]
        )

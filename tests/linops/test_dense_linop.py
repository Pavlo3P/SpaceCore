"""Tests for :class:`spacecore.DenseLinOp` — dense tensor-backed operator.

Checklist item 11:

* ``apply`` / ``rapply`` perform the raw coordinate matrix action and its
  (conjugate-)transpose adjoint on Euclidean spaces.
* Complex operators use the conjugate transpose in ``rapply``.
* ``is_hermitian`` uses the Euclidean matrix: ``True`` / ``False`` on square
  ops, ``False`` on rectangular ops; weighted spaces use the metric adjoint.
* The constructor accepts Euclidean ``DenseCoordinateSpace`` subclasses and
  ``HermitianSpace``; rejects bad ``A`` shapes and non-Euclidean spaces that
  lack Riesz maps.
* The flattened matrix is computed once at construction and reused across
  ``apply`` / ``rapply`` (no per-call reshape).
* Adjoint identity ``<A x, y>_cod == <x, A† y>_dom`` holds on Euclidean,
  weighted-fused, and Hermitian spaces (real and complex).
* Weighted-fused mode (``_mode.name == 'WEIGHTED_FUSED'``) precomputes
  ``_weighted_A2H`` and is preserved/recomputed after ``convert``.
* ``to_dense`` / ``to_matrix`` / ``A`` return the stored tensor / flattened
  matrix.
* ``convert`` preserves the action, returns ``self`` for a same-context
  convert, rejects complex->real narrowing, and crosses numpy->jax.
* ``tree_flatten`` / ``tree_unflatten`` round-trip.
* Fast-path batched ``vapply`` / ``rvapply`` without checks match the raw
  matrix products.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy


_MATRIX = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class ReshapeCountingArray(np.ndarray):
    """ndarray subclass that counts ``reshape`` calls on the original array."""

    def __new__(cls, data, counter):
        obj = np.asarray(data).view(cls)
        obj.counter = counter
        obj._track_reshape = True
        return obj

    def __array_finalize__(self, obj):
        self.counter = getattr(obj, "counter", None)
        self._track_reshape = False

    def reshape(self, *shape, **kwargs):
        if self.counter is not None and self._track_reshape:
            self.counter["calls"] += 1
        return super().reshape(*shape, **kwargs)


def _assert_adjoint_identity(op, x, y, rtol=1e-6, atol=1e-6):
    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=rtol, atol=atol)


# ===========================================================================
# apply / rapply
# ===========================================================================
class TestApplyRapply:
    def test_apply_matches_matrix_product(self, numpy_ctx):
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(numpy_ctx.asarray(_MATRIX), dom, cod, numpy_ctx)
        x = numpy_ctx.asarray([7.0, 8.0])
        np.testing.assert_allclose(to_numpy(op.apply(x)), _MATRIX @ np.array([7.0, 8.0]))

    def test_rapply_matches_transpose_product(self, numpy_ctx):
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(numpy_ctx.asarray(_MATRIX), dom, cod, numpy_ctx)
        y = numpy_ctx.asarray([1.0, -1.0, 2.0])
        np.testing.assert_allclose(to_numpy(op.rapply(y)), _MATRIX.T @ np.array([1.0, -1.0, 2.0]))

    def test_rectangular_apply_and_rapply(self, numpy_ctx):
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        matrix = np.array([[1.0, -2.0], [3.0, 0.5], [0.25, 4.0]])
        op = sc.DenseLinOp(numpy_ctx.asarray(matrix), dom, cod, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0])
        y = numpy_ctx.asarray([2.0, -1.0, 0.5])
        np.testing.assert_allclose(to_numpy(op.apply(x)), matrix @ np.array([1.0, 2.0]))
        np.testing.assert_allclose(to_numpy(op.rapply(y)), matrix.T @ np.array([2.0, -1.0, 0.5]))

    def test_accepts_euclidean_vector_space_subclass(self, numpy_ctx):
        # Source: legacy test_dense_linop.py
        class WeightedVectorSpace(sc.DenseCoordinateSpace):
            pass

        space = WeightedVectorSpace((2,), numpy_ctx)
        matrix = numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]])
        op = sc.DenseLinOp(matrix, space, space, numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0])

        assert type(op.domain) is WeightedVectorSpace
        np.testing.assert_allclose(to_numpy(op.apply(x)), to_numpy(x))
        np.testing.assert_allclose(to_numpy(op.rapply(x)), to_numpy(x))


# ===========================================================================
# Complex adjoint uses conjugate transpose
# ===========================================================================
class TestComplexAdjoint:
    def test_complex_rapply_uses_conjugate_transpose(self, numpy_complex_ctx):
        # Source: legacy test_dense_linop.py
        ctx = numpy_complex_ctx
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((2,), ctx)
        matrix = np.array(
            [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0j, 4.0 + 0.5j]], dtype=np.complex128
        )
        op = sc.DenseLinOp(ctx.asarray(matrix), dom, cod, ctx)
        y = ctx.asarray([1.0 - 1.0j, 2.0 + 0.25j])

        np.testing.assert_allclose(
            to_numpy(op.rapply(y)), matrix.conj().T @ np.array([1.0 - 1.0j, 2.0 + 0.25j])
        )


# ===========================================================================
# is_hermitian on Euclidean matrix
# ===========================================================================
class TestIsHermitian:
    def test_hermitian_matrix_is_true(self, numpy_complex_ctx):
        # Source: legacy test_dense_linop.py
        ctx = numpy_complex_ctx
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(
            ctx.asarray([[2.0, 1.0 + 2.0j], [1.0 - 2.0j, 5.0]]), space, space, ctx
        )
        assert op.is_hermitian() is True

    def test_non_hermitian_matrix_is_false(self, numpy_complex_ctx):
        # Source: legacy test_dense_linop.py
        ctx = numpy_complex_ctx
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), space, space, ctx)
        assert op.is_hermitian() is False

    def test_rectangular_is_false(self, numpy_complex_ctx):
        # Source: legacy test_dense_linop.py
        ctx = numpy_complex_ctx
        space = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        op = sc.DenseLinOp(ctx.asarray(np.ones((3, 2))), space, cod, ctx)
        assert op.is_hermitian() is False

    def test_weighted_metric_hermitian_detection(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        weights = numpy_ctx.asarray([2.0, 3.0])
        space = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights)
        )
        weighted_self_adjoint = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 3.0], [2.0, 4.0]]), space, space, numpy_ctx
        )
        coordinate_symmetric_only = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 1.0], [1.0, 4.0]]), space, space, numpy_ctx
        )
        assert weighted_self_adjoint.is_hermitian() is True
        assert coordinate_symmetric_only.is_hermitian() is False


# ===========================================================================
# Adjoint identity (euclidean / weighted / complex / hermitian)
# ===========================================================================
class TestAdjointIdentity:
    def test_euclidean_real_adjoint_identity(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        domain = sc.DenseCoordinateSpace((2,), numpy_ctx)
        codomain = sc.DenseCoordinateSpace((3,), numpy_ctx)
        matrix = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        op = sc.DenseLinOp(numpy_ctx.asarray(matrix), domain, codomain, numpy_ctx)
        _assert_adjoint_identity(
            op, numpy_ctx.asarray([0.25, -1.5]), numpy_ctx.asarray([2.0, -0.5, 1.25])
        )

    def test_euclidean_complex_adjoint_identity(self, numpy_complex_ctx):
        # Source: legacy test_adjoint_identity.py
        ctx = numpy_complex_ctx
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.array([[1.0 + 0.5j, -2.0j], [0.5 - 1.0j, 3.0], [4.0, -1.0 + 2.0j]])
        op = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
        _assert_adjoint_identity(
            op,
            ctx.asarray([0.25 + 1.0j, -1.5 + 0.5j]),
            ctx.asarray([2.0 - 0.25j, -0.5 + 1.0j, 1.25j]),
        )

    def test_weighted_vector_metric_adjoint_identity(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        domain = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 5.0]))
        )
        codomain = sc.DenseCoordinateSpace(
            (3,),
            numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([3.0, 7.0, 11.0])),
        )
        matrix = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        op = sc.DenseLinOp(numpy_ctx.asarray(matrix), domain, codomain, numpy_ctx)
        _assert_adjoint_identity(
            op, numpy_ctx.asarray([0.25, -1.5]), numpy_ctx.asarray([2.0, -0.5, 1.25])
        )

    def test_weighted_scalar_metric_adjoint_counterexample(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        domain = sc.DenseCoordinateSpace(
            (1,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0]))
        )
        codomain = sc.DenseCoordinateSpace(
            (1,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([3.0]))
        )
        op = sc.DenseLinOp(numpy_ctx.asarray([[5.0]]), domain, codomain, numpy_ctx)
        y = numpy_ctx.asarray([4.0])
        # A† y = R_X^{-1} A^T R_Y y = (1/2) * 5 * 3 * 4 = 30.
        np.testing.assert_allclose(to_numpy(op.rapply(y)), [30.0], rtol=1e-6, atol=1e-6)
        _assert_adjoint_identity(op, numpy_ctx.asarray([1.25]), y)

    def test_accepts_hermitian_space_and_satisfies_adjoint_identity(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        identity_tensor = numpy_ctx.asarray(np.eye(4).reshape((2, 2, 2, 2)))
        op = sc.DenseLinOp(identity_tensor, space, space, numpy_ctx)
        x = numpy_ctx.asarray([[1.0, 2.0], [2.0, 3.0]])
        y = numpy_ctx.asarray([[4.0, -1.0], [-1.0, 2.0]])
        _assert_adjoint_identity(op, x, y)


# ===========================================================================
# Weighted-fused mode
# ===========================================================================
class TestWeightedFusedMode:
    def test_fused_mode_selected_and_matches_generic_metric(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        from spacecore.linop._metric import metric_rapply

        domain = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 5.0]))
        )
        codomain = sc.DenseCoordinateSpace(
            (3,),
            numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([3.0, 7.0, 11.0])),
        )
        matrix = numpy_ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        op = sc.DenseLinOp(matrix, domain, codomain, numpy_ctx)
        y = numpy_ctx.asarray([2.0, -0.5, 1.25])

        assert op._mode.name == "WEIGHTED_FUSED"
        np.testing.assert_allclose(to_numpy(op.rapply(y)), to_numpy(op._weighted_A2H @ y))
        np.testing.assert_allclose(
            to_numpy(op.rapply(y)),
            to_numpy(metric_rapply(op.domain, op.codomain, op._euclidean_rapply_core, y)),
        )

    def test_fused_mode_recomputed_after_convert(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="none")
        domain = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 5.0]))
        )
        codomain = sc.DenseCoordinateSpace(
            (3,),
            numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([3.0, 7.0, 11.0])),
        )
        op = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]]),
            domain,
            codomain,
            numpy_ctx,
        )

        converted = op.convert(new_ctx)

        assert op._mode.name == "WEIGHTED_FUSED"
        assert converted._mode.name == "WEIGHTED_FUSED"
        assert hasattr(converted, "_weighted_A2H")
        _assert_adjoint_identity(
            converted,
            new_ctx.asarray([0.25, -1.5]),
            new_ctx.asarray([2.0, -0.5, 1.25]),
            rtol=1e-5,
            atol=1e-5,
        )


# ===========================================================================
# Non-Euclidean without Riesz maps is rejected
# ===========================================================================
class TestValidation:
    def test_bad_shape_raises(self, numpy_ctx):
        # Source: legacy test_dense_linop.py
        with pytest.raises(TypeError, match="Expected A.shape"):
            sc.DenseLinOp(
                numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
                sc.DenseCoordinateSpace((2,), numpy_ctx),
                sc.DenseCoordinateSpace((3,), numpy_ctx),
                numpy_ctx,
            )

    def test_non_euclidean_without_riesz_is_rejected(self, numpy_ctx):
        # Source: legacy test_adjoint_identity.py
        class BrokenInnerProduct(sc.InnerProduct):
            def inner(self, ops, x, y):
                return ops.vdot(x, 2.0 * y)

        class BrokenSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx):
                super().__init__(shape, ctx)
                self.geometry = BrokenInnerProduct()

        space = BrokenSpace((2,), numpy_ctx)

        with pytest.raises(TypeError, match="MatrixFreeLinOp"):
            sc.DenseLinOp(
                numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), space, space, numpy_ctx
            )


# ===========================================================================
# to_dense / to_matrix / A
# ===========================================================================
class TestToDenseToMatrix:
    def test_to_dense_returns_stored_array(self, numpy_ctx):
        # Source: legacy test_to_dense.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = numpy_ctx.asarray(_MATRIX)
        op = sc.DenseLinOp(A, dom, cod, numpy_ctx)
        assert op.to_dense() is A

    def test_a_property_returns_stored_array(self, numpy_ctx):
        # Source: legacy test_to_dense.py / test_dense_linop.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = numpy_ctx.asarray(_MATRIX)
        op = sc.DenseLinOp(A, dom, cod, numpy_ctx)
        assert op.A is A

    def test_to_dense_matches_apply(self, numpy_ctx):
        # Source: legacy test_to_dense.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(numpy_ctx.asarray(_MATRIX), dom, cod, numpy_ctx)
        x = numpy_ctx.asarray([7.0, 8.0])
        dense = op.to_dense()
        matrix = np.asarray(to_numpy(dense)).reshape(
            (int(np.prod(op.codomain.shape)), int(np.prod(op.domain.shape)))
        )
        y_from_dense = matrix @ to_numpy(op.domain.flatten(x))
        y_from_apply = to_numpy(op.codomain.flatten(op.apply(x)))
        np.testing.assert_allclose(y_from_dense, y_from_apply)

    def test_to_matrix_shape(self, numpy_ctx):
        # Source: legacy test_dense_linop.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(numpy_ctx.asarray(_MATRIX), dom, cod, numpy_ctx)
        assert op.to_dense().shape == (3, 2)
        assert op.to_matrix().shape == (3, 2)

    def test_to_matrix_flattens_tensor_operator(self, numpy_ctx):
        dom = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = numpy_ctx.asarray(np.arange(12.0).reshape(3, 2, 2))
        op = sc.DenseLinOp(A, dom, cod, numpy_ctx)
        assert op.to_matrix().shape == (3, 4)


# ===========================================================================
# Cached matrix reshape (computed once)
# ===========================================================================
class TestCachedReshape:
    def test_reuses_cached_matrix_reshape(self, numpy_ctx):
        # Source: legacy test_dense_linop.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        counter = {"calls": 0}
        A = ReshapeCountingArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], counter)

        op = sc.DenseLinOp(A, dom, cod, numpy_ctx)
        matrix_reshape_calls = counter["calls"]

        op.apply(numpy_ctx.asarray([7.0, 8.0]))
        op.rapply(numpy_ctx.asarray([1.0, -1.0, 2.0]))
        op.apply(numpy_ctx.asarray([9.0, 10.0]))
        op.rapply(numpy_ctx.asarray([3.0, -2.0, 1.0]))

        assert matrix_reshape_calls == 1
        assert counter["calls"] == matrix_reshape_calls


# ===========================================================================
# convert
# ===========================================================================
class TestConvert:
    def test_convert_preserves_action(self, numpy_f32_ctx):
        # Source: legacy test_dense_linop.py / test_conversion_linops.py
        src = numpy_f32_ctx
        dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
        op = sc.DenseLinOp(
            src.asarray(_MATRIX),
            sc.DenseCoordinateSpace((2,), src),
            sc.DenseCoordinateSpace((3,), src),
            src,
        )
        op2 = op.convert(dst)
        x = op2.ctx.asarray([7.0, 8.0])
        assert type(op2.dom) is sc.DenseCoordinateSpace
        assert type(op2.cod) is sc.DenseCoordinateSpace
        assert op2.ops.get_dtype(op2.A) == dst.dtype
        np.testing.assert_allclose(to_numpy(op2.apply(x)), [23.0, 53.0, 83.0])

    def test_convert_to_same_context_returns_self(self, numpy_f32_ctx):
        # Source: legacy test_conversion_linops.py
        ctx = numpy_f32_ctx
        X = sc.DenseCoordinateSpace((2,), ctx)
        Y = sc.DenseCoordinateSpace((3,), ctx)
        op = sc.DenseLinOp(ctx.asarray(_MATRIX), X, Y, ctx)
        assert op.convert(ctx) is op

    def test_convert_rejects_complex_to_real_narrowing(self):
        # Source: legacy test_conversion_linops.py
        src = sc.Context(sc.NumpyOps(), dtype=np.complex64)
        dst = sc.Context(sc.NumpyOps(), dtype=np.float32)
        space = sc.DenseCoordinateSpace((2,), src)
        op = sc.DenseLinOp(src.asarray([[1.0, 0.0j], [0.0j, 1.0]]), space, space, src)
        with pytest.raises(TypeError, match="rejected complex-valued input.*x.real"):
            op.convert(dst)

    @pytest.mark.skipif(not has_jax(), reason="jax is not installed")
    def test_convert_numpy_to_jax(self):
        # Source: legacy test_dense_linop.py / test_conversion_linops.py
        dt = jax_real_dtype()
        src = sc.Context(sc.NumpyOps(), dtype=dt)
        dst = sc.Context(sc.JaxOps(), dtype=dt)
        op = sc.DenseLinOp(
            src.asarray(_MATRIX),
            sc.DenseCoordinateSpace((2,), src),
            sc.DenseCoordinateSpace((3,), src),
            src,
        )
        op2 = op.convert(dst)
        assert op2.ctx.ops.family == "jax"


# ===========================================================================
# pytree
# ===========================================================================
class TestPytree:
    def test_tree_flatten_unflatten_round_trip(self, numpy_ctx):
        # Source: legacy test_dense_linop.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        matrix = numpy_ctx.asarray(_MATRIX)
        op = sc.DenseLinOp(matrix, dom, cod, numpy_ctx)

        children, aux = op.tree_flatten()
        restored = sc.DenseLinOp.tree_unflatten(aux, children)

        assert type(restored.dom) is sc.DenseCoordinateSpace
        assert type(restored.cod) is sc.DenseCoordinateSpace
        np.testing.assert_allclose(to_numpy(restored.to_dense()), to_numpy(matrix))
        np.testing.assert_allclose(
            to_numpy(restored.apply(numpy_ctx.asarray([7.0, 8.0]))),
            to_numpy(op.apply(numpy_ctx.asarray([7.0, 8.0]))),
        )


# ===========================================================================
# Batched fast paths (without checks)
# ===========================================================================
class TestBatched:
    def test_fast_path_vapply_rvapply_without_checks(self):
        # Source: legacy test_batched_lifting.py
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        dom = sc.DenseCoordinateSpace((2,), ctx)
        cod = sc.DenseCoordinateSpace((3,), ctx)
        matrix = ctx.asarray(_MATRIX)
        op = sc.DenseLinOp(matrix, dom, cod, ctx)
        xs = ctx.asarray([[7.0, 8.0], [1.0, -1.0], [0.5, 2.0]])
        ys = ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        np.testing.assert_allclose(
            to_numpy(op.vapply(xs)), np.asarray(to_numpy(xs)) @ np.asarray(to_numpy(matrix)).T
        )
        np.testing.assert_allclose(
            to_numpy(op.rvapply(ys)), np.asarray(to_numpy(ys)) @ np.asarray(to_numpy(matrix))
        )

    def test_vapply_rvapply_match_stacked_apply(self, numpy_ctx):
        # Source: legacy test_batched_lifting.py
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(numpy_ctx.asarray(_MATRIX), dom, cod, numpy_ctx)
        xs = numpy_ctx.asarray([[7.0, 8.0], [1.0, -1.0], [0.5, 2.0]])
        ys = numpy_ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        expected_v = np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
        expected_rv = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
        np.testing.assert_allclose(to_numpy(op.vapply(xs)), expected_v)
        np.testing.assert_allclose(to_numpy(op.rvapply(ys)), expected_rv)

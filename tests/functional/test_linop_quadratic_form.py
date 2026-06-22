"""Tests for :class:`spacecore.LinOpQuadraticForm`.

Checklist section 7, ``LinOpQuadraticForm``:

* Construction guards: ``Q`` must be a square ``LinOp``, ``linear`` a
  ``LinearFunctional`` on ``Q.domain``, ``a`` scalar, ``Q`` Hermitian.
* ``value(x) = 1/2 <x, Qx> + linear(x) + a``.
* ``grad(x) = Q x + linear.representer`` (Hermitian assumption).
* ``hess_apply(x) = Q x``.
* Matrix-free ``Q`` is *not* validated for the Hermitian assumption.
* ``vvalue`` / ``vgrad`` match element-wise loops.
* ``__eq__``, ``tree_flatten`` / ``tree_unflatten`` round-trip, ``_convert``.

Analytic value/gradient references (incl. weighted geometry) live in
:mod:`tests.functional.test_generated_functionals` and
:mod:`tests.functional.test_metric_gradient`.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


def _quadratic_problem(ctx):
    """``q(x) = 1/2<x, diag(2,4) x> + <[1,-1], x> + 3``."""
    dom = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 4.0]]), dom, dom, ctx)
    linear = sc.InnerProductFunctional(ctx.asarray([1.0, -1.0]), dom, ctx)
    return sc.LinOpQuadraticForm(Q, linear, 3.0, ctx)


# ===========================================================================
# Construction guards
# ===========================================================================
class TestConstruction:
    def test_rejects_non_linop_operator(self, numpy_ctx):
        with pytest.raises(TypeError, match="Q must be a LinOp"):
            sc.LinOpQuadraticForm(numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), ctx=numpy_ctx)

    def test_rejects_non_linear_functional_linear_term(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Q = sc.IdentityLinOp(space, numpy_ctx)
        with pytest.raises(TypeError, match="linear must be a LinearFunctional"):
            sc.LinOpQuadraticForm(Q, linear="nope", ctx=numpy_ctx)

    def test_requires_square_operator(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        Q = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), X, Y, numpy_ctx
        )
        with pytest.raises(ValueError, match="Q.domain == Q.codomain"):
            sc.LinOpQuadraticForm(Q, ctx=numpy_ctx)

    def test_requires_linear_domain_matches_operator(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        other = sc.DenseCoordinateSpace((3,), numpy_ctx)
        Q = sc.IdentityLinOp(space, numpy_ctx)
        linear = sc.InnerProductFunctional(numpy_ctx.asarray([1.0, 2.0, 3.0]), other, numpy_ctx)
        with pytest.raises(ValueError, match="linear.domain must match Q.domain"):
            sc.LinOpQuadraticForm(Q, linear, ctx=numpy_ctx)

    def test_rejects_non_hermitian_dense_operator(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Q = sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0], [0.0, 3.0]]), space, space, numpy_ctx)
        with pytest.raises(ValueError, match="Hermitian"):
            sc.LinOpQuadraticForm(Q, ctx=numpy_ctx)

    def test_rejects_nonscalar_constant(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        space = sc.DenseCoordinateSpace((2,), ctx)
        Q = sc.IdentityLinOp(space, ctx)
        with pytest.raises(ValueError, match="scalar batch"):
            sc.LinOpQuadraticForm(Q, a=ctx.asarray([0.0, 0.0]), ctx=ctx)

    def test_explicit_context_overrides_inferred(self, numpy_f32_ctx, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_f32_ctx)
        Q = sc.DenseLinOp(
            numpy_f32_ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), space, space, numpy_f32_ctx
        )
        linear = sc.InnerProductFunctional(numpy_f32_ctx.asarray([1.0, 2.0]), space)
        q = sc.LinOpQuadraticForm(Q, linear, 0.0, numpy_ctx)
        assert q.ctx == numpy_ctx
        assert q.Q.ctx == numpy_ctx
        assert q.linear.ctx == numpy_ctx


# ===========================================================================
# value / grad / hess_apply
# ===========================================================================
class TestValueGradHess:
    def test_value_grad_hess_match_hand_computation(self, numpy_ctx):
        q = _quadratic_problem(numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0])
        np.testing.assert_allclose(to_numpy(q.value(x)), 12.0)
        np.testing.assert_allclose(to_numpy(q.grad(x)), [5.0, -5.0])
        np.testing.assert_allclose(to_numpy(q.hess_apply(x)), [4.0, -4.0])

    def test_call_alias_matches_value(self, numpy_ctx):
        q = _quadratic_problem(numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0])
        np.testing.assert_allclose(to_numpy(q(x)), to_numpy(q.value(x)))

    def test_gradient_without_linear_is_operator_action(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Q = sc.DenseLinOp(numpy_ctx.asarray([[2.0, 1.0], [1.0, 4.0]]), space, space, numpy_ctx)
        q = sc.LinOpQuadraticForm(Q, ctx=numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0])
        np.testing.assert_allclose(to_numpy(q.grad(x)), to_numpy(Q.apply(x)))
        np.testing.assert_allclose(to_numpy(q.hess_apply(x)), to_numpy(Q.apply(x)))


# ===========================================================================
# Matrix-free Hermitian assumption is not validated
# ===========================================================================
class TestMatrixFreeHermitian:
    def test_matrix_free_operator_is_not_validated(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)

        def apply(x):
            return numpy_ctx.asarray([x[0] + 2.0 * x[1], 3.0 * x[1]])

        def rapply(y):
            return numpy_ctx.asarray([y[0], 2.0 * y[0] + 3.0 * y[1]])

        Q = sc.MatrixFreeLinOp(apply, rapply, space, space, numpy_ctx)
        q = sc.LinOpQuadraticForm(Q, ctx=numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0])
        np.testing.assert_allclose(to_numpy(q.grad(x)), to_numpy(Q.apply(x)))


# ===========================================================================
# Batched value / gradient
# ===========================================================================
class TestBatched:
    def test_vvalue_and_vgrad_match_elementwise(self, numpy_ctx):
        q = _quadratic_problem(numpy_ctx)
        xs = numpy_ctx.asarray([[2.0, -1.0], [0.0, 3.0], [1.5, 2.0]])
        expected_values = q.ops.stack(tuple(q.value(x) for x in xs), axis=0)
        expected_grads = q.ops.stack(tuple(q.grad(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(q.vvalue(xs)), to_numpy(expected_values))
        np.testing.assert_allclose(to_numpy(q.vgrad(xs)), to_numpy(expected_grads))

    def test_bad_batch_shape_raises_under_checks(self, numpy_ctx):
        q = _quadratic_problem(numpy_ctx)
        with pytest.raises(Exception):
            q.vvalue(numpy_ctx.asarray([[1.0, 2.0, 3.0]]))


# ===========================================================================
# __eq__
# ===========================================================================
class TestEquality:
    def test_equal_when_terms_match(self, numpy_ctx):
        assert _quadratic_problem(numpy_ctx) == _quadratic_problem(numpy_ctx)

    def test_not_equal_when_offset_differs(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Q = sc.IdentityLinOp(space, numpy_ctx)
        a = sc.LinOpQuadraticForm(Q, a=1.0, ctx=numpy_ctx)
        b = sc.LinOpQuadraticForm(Q, a=2.0, ctx=numpy_ctx)
        assert a != b

    def test_not_equal_to_other_type(self, numpy_ctx):
        assert (_quadratic_problem(numpy_ctx) == 42) is False


# ===========================================================================
# Pytree round-trip + convert
# ===========================================================================
class TestPytreeAndConvert:
    def test_tree_flatten_unflatten_round_trip(self, numpy_ctx):
        q = _quadratic_problem(numpy_ctx)
        children, aux = q.tree_flatten()
        restored = sc.LinOpQuadraticForm.tree_unflatten(aux, children)
        x = numpy_ctx.asarray([2.0, -1.0])
        assert restored == q
        np.testing.assert_allclose(to_numpy(restored.value(x)), to_numpy(q.value(x)))

    def test_convert_preserves_value_across_dtype(self, numpy_f32_ctx, numpy_ctx):
        q = _quadratic_problem(numpy_f32_ctx)
        g = q.convert(numpy_ctx)
        assert g.ctx == numpy_ctx
        assert g.Q.ctx == numpy_ctx
        x = numpy_ctx.asarray([2.0, -1.0])
        np.testing.assert_allclose(to_numpy(g.value(x)), 12.0)

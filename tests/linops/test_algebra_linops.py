"""Tests for the algebra LinOps in :mod:`spacecore.linop._algebra`.

Checklist items 4–10, one class per LinOp:

* :class:`spacecore.IdentityLinOp` — apply / rapply identity, ``H is self``-ish,
  ``to_dense = I``, ``is_hermitian = True``.
* :class:`spacecore.ZeroLinOp` — apply returns codomain zero, ``H`` is
  ``ZeroLinOp(cod, dom)``, ``is_hermitian = True`` iff square.
* :class:`spacecore.ScaledLinOp` — apply = α·op.apply, rapply = conj(α)·op.rapply,
  batched, ``_convert``.
* :class:`spacecore.SumLinOp` — apply sums parts, ``parts`` property, batched.
* :class:`spacecore.ComposedLinOp` — apply = left∘right, rapply = right.H∘left.H.
* :class:`spacecore.MatrixFreeLinOp` — supplied callables used verbatim,
  ``rapply`` NOT wrapped by Riesz on non-Euclidean (ADR-009), pytree round-trip
  preserves callables.
* ``_AdjointViewLinOp`` (the ``op.H`` return) — apply == op.rapply,
  rapply == op.apply, ``.H is op`` (idempotent double-adjoint).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import (
    has_jax,
    has_torch,
    jax_complex_dtype,
    to_numpy,
    torch_complex_dtype,
)


# ===========================================================================
# IdentityLinOp
# ===========================================================================
class TestIdentityLinOp:
    def test_apply_returns_input(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(op.apply(x), x)

    def test_apply_unchecked_returns_literal_input(self, numpy_ctx):
        """With checks disabled, ``apply(x)`` and ``rapply(x)`` return the literal ``x``."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        X = sc.DenseCoordinateSpace((3,), ctx)
        op = sc.IdentityLinOp(X, ctx)
        x = ctx.asarray([1.0, 2.0, 3.0])
        assert op.apply(x) is x
        # rapply is a distinct code branch; confirm it is also a literal pass-through.
        assert op.rapply(x) is x

    def test_rapply_returns_input(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(op.rapply(x), x)

    def test_to_dense_is_identity_matrix(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        np.testing.assert_allclose(to_numpy(op.to_dense()), np.eye(3))

    def test_is_hermitian_true(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        assert sc.IdentityLinOp(X, numpy_ctx).is_hermitian() is True

    def test_equality(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((4,), numpy_ctx)
        assert sc.IdentityLinOp(X, numpy_ctx) == sc.IdentityLinOp(X, numpy_ctx)
        assert sc.IdentityLinOp(X, numpy_ctx) != sc.IdentityLinOp(Y, numpy_ctx)

    def test_pytree_round_trip(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        children, aux = op.tree_flatten()
        rebuilt = sc.IdentityLinOp.tree_unflatten(aux, children)
        assert rebuilt == op


# ===========================================================================
# ZeroLinOp
# ===========================================================================
class TestZeroLinOp:
    def test_apply_returns_codomain_zero(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.ZeroLinOp(X, Y, numpy_ctx)
        out = op.apply(numpy_ctx.asarray([1.0, 2.0]))
        np.testing.assert_allclose(out, np.zeros(3))

    def test_rapply_returns_domain_zero(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.ZeroLinOp(X, Y, numpy_ctx)
        out = op.rapply(numpy_ctx.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, np.zeros(2))

    def test_H_is_zero_with_swapped_spaces(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.ZeroLinOp(X, Y, numpy_ctx)
        # H.apply(y) returns op.rapply(y), which is domain.zeros() of shape (2,).
        out = op.H.apply(numpy_ctx.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, np.zeros(2))

    def test_is_hermitian_true_when_square(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        assert sc.ZeroLinOp(X, X, numpy_ctx).is_hermitian() is True

    def test_is_hermitian_false_when_rectangular(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        assert sc.ZeroLinOp(X, Y, numpy_ctx).is_hermitian() is False

    def test_to_dense_is_zero_matrix(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.ZeroLinOp(X, Y, numpy_ctx)
        np.testing.assert_allclose(to_numpy(op.to_dense()), np.zeros((3, 2)))

    def test_pytree_round_trip(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.ZeroLinOp(X, Y, numpy_ctx)
        children, aux = op.tree_flatten()
        rebuilt = sc.ZeroLinOp.tree_unflatten(aux, children)
        assert rebuilt == op


# ===========================================================================
# ScaledLinOp
# ===========================================================================
class TestScaledLinOp:
    def test_apply_scales_underlying_apply(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        identity = sc.IdentityLinOp(X, numpy_ctx)
        op = sc.ScaledLinOp(3.0, identity)
        np.testing.assert_allclose(op.apply(numpy_ctx.asarray([1.0, 2.0, 3.0])),
                                   [3.0, 6.0, 9.0])

    def test_rapply_conjugates_scalar_on_complex(self, numpy_complex_ctx):
        """rapply = conj(alpha) * op.rapply(y) on complex."""
        X = sc.DenseCoordinateSpace((2,), numpy_complex_ctx)
        identity = sc.IdentityLinOp(X, numpy_complex_ctx)
        alpha = 2.0 + 3.0j
        op = sc.ScaledLinOp(alpha, identity)
        y = numpy_complex_ctx.asarray([1.0 + 0.0j, 0.0 + 1.0j])
        np.testing.assert_allclose(op.rapply(y),
                                   np.conj(alpha) * to_numpy(y))

    def test_rapply_real_scalar_is_unchanged(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        identity = sc.IdentityLinOp(X, numpy_ctx)
        op = sc.ScaledLinOp(2.5, identity)
        y = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(op.rapply(y), [2.5, 5.0, 7.5])

    def test_scalar_property(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.ScaledLinOp(7.0, sc.IdentityLinOp(X, numpy_ctx))
        assert op.scalar == 7.0


# ===========================================================================
# SumLinOp
# ===========================================================================
class TestSumLinOp:
    def test_apply_sums_part_actions(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        a = sc.IdentityLinOp(X, numpy_ctx)
        b = sc.IdentityLinOp(X, numpy_ctx)
        op = sc.SumLinOp((a, b))
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(op.apply(x), 2.0 * to_numpy(x))

    def test_rapply_sums_part_adjoints(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        a = sc.IdentityLinOp(X, numpy_ctx)
        b = sc.IdentityLinOp(X, numpy_ctx)
        op = sc.SumLinOp((a, b))
        y = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(op.rapply(y), 2.0 * to_numpy(y))

    def test_parts_property(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        a = sc.IdentityLinOp(X, numpy_ctx)
        b = sc.IdentityLinOp(X, numpy_ctx)
        op = sc.SumLinOp((a, b))
        assert len(op.parts) == 2


# ===========================================================================
# ComposedLinOp
# ===========================================================================
class TestComposedLinOp:
    def test_apply_is_left_of_right(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DenseLinOp(numpy_ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, numpy_ctx)
        B = sc.DenseLinOp(numpy_ctx.asarray([[5.0, 0.0], [0.0, 7.0]]), X, X, numpy_ctx)
        op = sc.ComposedLinOp(A, B)  # A @ B
        x = numpy_ctx.asarray([1.0, 1.0])
        np.testing.assert_allclose(op.apply(x), [10.0, 21.0])

    def test_rapply_is_right_H_of_left_H(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DenseLinOp(numpy_ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, numpy_ctx)
        B = sc.DenseLinOp(numpy_ctx.asarray([[5.0, 0.0], [0.0, 7.0]]), X, X, numpy_ctx)
        op = sc.ComposedLinOp(A, B)
        # (A @ B).rapply(y) = B.rapply(A.rapply(y)) = B^T A^T y
        y = numpy_ctx.asarray([1.0, 1.0])
        np.testing.assert_allclose(op.rapply(y), [10.0, 21.0])


# ===========================================================================
# MatrixFreeLinOp — supplied callables used verbatim
# ===========================================================================
class TestMatrixFreeLinOp:
    def test_apply_uses_supplied_forward(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)
        called = []

        def fwd(x):
            called.append("fwd")
            return x * 2.0

        op = sc.MatrixFreeLinOp(fwd, lambda y: y, X, X, numpy_ctx)
        out = op.apply(numpy_ctx.asarray([1.0, 2.0, 3.0]))
        assert called == ["fwd"]
        np.testing.assert_allclose(out, [2.0, 4.0, 6.0])

    def test_rapply_uses_supplied_reverse_verbatim(self, numpy_ctx):
        """ADR-009: ``rapply`` calls the user callable directly, no Riesz wrapping."""
        weights = numpy_ctx.asarray([2.0, 3.0])
        X = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        called = []

        def rev(y):
            called.append(y)
            return y * 5.0  # user's chosen adjoint, untouched by Riesz

        op = sc.MatrixFreeLinOp(lambda x: x, rev, X, X, numpy_ctx)
        out = op.rapply(numpy_ctx.asarray([1.0, 2.0]))
        assert len(called) == 1
        np.testing.assert_allclose(out, [5.0, 10.0])

    def test_pytree_round_trip_preserves_callables(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((3,), numpy_ctx)

        def fwd(x):
            return x * 2.0

        def rev(y):
            return y * 3.0

        op = sc.MatrixFreeLinOp(fwd, rev, X, X, numpy_ctx)
        children, aux = op.tree_flatten()
        rebuilt = sc.MatrixFreeLinOp.tree_unflatten(aux, children)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(rebuilt.apply(x), op.apply(x))
        np.testing.assert_allclose(rebuilt.rapply(x), op.rapply(x))


# ===========================================================================
# _AdjointViewLinOp — A.H
# ===========================================================================
class TestAdjointViewLinOp:
    def test_view_apply_equals_op_rapply(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, numpy_ctx,
        )
        view = A.H
        y = numpy_ctx.asarray([1.0, -1.0, 2.0])
        np.testing.assert_allclose(view.apply(y), A.rapply(y))

    def test_view_rapply_equals_op_apply(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, numpy_ctx,
        )
        view = A.H
        x = numpy_ctx.asarray([1.0, 2.0])
        np.testing.assert_allclose(view.rapply(x), A.apply(x))

    def test_view_H_returns_original(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        # The double-adjoint view returns the original operator.
        assert op.H.H is op

    def test_view_swaps_domain_and_codomain(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, numpy_ctx,
        )
        assert A.H.domain == A.codomain
        assert A.H.codomain == A.domain


# ===========================================================================
# Shared helpers for folded coverage
# ===========================================================================
def _assert_adjoint_identity(op, x, y, rtol=1e-6, atol=1e-6):
    """Check ``<A x, y>_cod == <x, A* y>_dom`` using the space inner products."""
    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=rtol, atol=atol)


def _weighted_space_class():
    """A ``DenseCoordinateSpace`` subclass carrying explicit weights (for ADR-009)."""

    class WeightedVectorSpace(sc.DenseCoordinateSpace):
        def __init__(self, weights, ctx):
            self.weights = ctx.asarray(weights)
            super().__init__(
                tuple(self.weights.shape), ctx,
                geometry=sc.WeightedInnerProduct(self.weights),
            )

        def _convert(self, new_ctx):
            return WeightedVectorSpace(new_ctx.asarray(self.weights), new_ctx)

        def __eq__(self, other):
            return (
                type(other) is type(self)
                and super().__eq__(other)
                and np.allclose(to_numpy(self.weights), to_numpy(other.weights))
            )

        def __hash__(self):
            return super().__hash__()

    return WeightedVectorSpace


def _non_euclidean_matrix_free_fixture(ctx):
    """Return a non-Euclidean matrix-free operator plus probe data (ADR-009)."""
    WeightedVectorSpace = _weighted_space_class()
    gx = np.array([2.0, 5.0])
    gy = np.array([3.0, 7.0, 11.0])
    matrix_np = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    metric_adjoint_np = np.diag(1.0 / gx) @ matrix_np.T @ np.diag(gy)

    domain = WeightedVectorSpace(gx, ctx)
    codomain = WeightedVectorSpace(gy, ctx)
    matrix = ctx.asarray(matrix_np)
    metric_adjoint = ctx.asarray(metric_adjoint_np)

    op = sc.MatrixFreeLinOp(
        lambda z: matrix @ z, lambda w: metric_adjoint @ w, domain, codomain, ctx,
    )
    return {
        "op": op,
        "domain": domain,
        "codomain": codomain,
        "matrix_np": matrix_np,
        "metric_adjoint_np": metric_adjoint_np,
        "gx": gx,
        "gy": gy,
        "x": ctx.asarray([0.25, -1.5]),
        "y": ctx.asarray([2.0, -0.5, 1.25]),
        "ys": ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]]),
    }


def _complex_matrix():
    return np.array(
        [
            [1.0 + 2.0j, 3.0 - 1.0j],
            [-2.0 + 0.5j, 0.25 + 4.0j],
            [1.5 - 3.0j, -0.75 + 2.0j],
        ]
    )


def _algebra_cases(ctx):
    """The 7 algebra-class cases ``(op, x, y)`` over a complex context.

    Order matches the legacy parametrization in test_algebra.py: Scaled, Sum,
    Composed, Zero, Identity, MatrixFree, AdjointView.
    """
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    matrix = ctx.asarray(_complex_matrix())
    A = sc.DenseLinOp(matrix, dom, cod, ctx)
    B = sc.DenseLinOp(ctx.asarray((0.5 - 0.25j) * _complex_matrix()), dom, cod, ctx)
    C = sc.DenseLinOp(
        ctx.asarray([[2.0 - 1.0j, -0.5 + 0.25j], [1.25 + 2.0j, -3.0 + 0.5j]]), dom, dom, ctx,
    )
    x = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
    y = ctx.asarray([1.0 + 0.5j, -2.0j, 0.75 - 1.25j])
    z = ctx.asarray([-1.0 + 0.5j, 2.0 - 0.25j])
    matrix_free = sc.MatrixFreeLinOp(
        lambda v: matrix @ v,
        lambda w: ctx.ops.conj(ctx.ops.transpose(matrix)) @ w,
        dom,
        cod,
        ctx,
    )
    return [
        ((2.0 + 3.0j) * A, x, y),
        (A + B, x, y),
        (A @ C, z, y),
        (sc.ZeroLinOp(dom, cod, ctx), x, y),
        (sc.IdentityLinOp(dom, ctx), x, x),
        (matrix_free, x, y),
        (A.H, y, x),
    ]


# ===========================================================================
# MatrixFreeLinOp.from_coordinate_adjoint — ADR-009 metric wrapping
# (folded from test_adjoint_identity.py)
# ===========================================================================
class TestMatrixFreeFromCoordinateAdjoint:
    def test_rejects_non_euclidean_space_without_riesz_maps(self, numpy_ctx):
        """from_coordinate_adjoint refuses a non-Euclidean space lacking Riesz maps.

        Source: legacy test_adjoint_identity.py
        ::test_non_euclidean_space_without_riesz_maps_is_rejected (the
        from_coordinate_adjoint sub-assert; the DenseLinOp/DiagonalLinOp
        sub-asserts live in test_dense_linop.py / test_diagonal_linop.py).
        """
        class BrokenInnerProduct(sc.InnerProduct):
            def inner(self, ops, x, y):
                return ops.vdot(x, 2.0 * y)

        class BrokenSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx):
                super().__init__(shape, ctx)
                self.geometry = BrokenInnerProduct()

        space = BrokenSpace((2,), numpy_ctx)
        with pytest.raises(ValueError, match="MatrixFreeLinOp.from_coordinate_adjoint"):
            sc.MatrixFreeLinOp.from_coordinate_adjoint(
                lambda x: x, lambda y: y, space, space, numpy_ctx
            )

    def test_wraps_coordinate_adjoint_to_metric_adjoint(self, numpy_ctx):
        """The supplied Euclidean coordinate adjoint is Riesz-wrapped to the metric adjoint."""
        fixture = _non_euclidean_matrix_free_fixture(numpy_ctx)
        domain = fixture["domain"]
        codomain = fixture["codomain"]
        matrix = numpy_ctx.asarray(fixture["matrix_np"])
        metric_adjoint_np = fixture["metric_adjoint_np"]
        x = fixture["x"]
        y = fixture["y"]

        def apply(z):
            return matrix @ z

        def coordinate_rapply(w):
            return matrix.T @ w

        op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
            apply, coordinate_rapply, domain, codomain, numpy_ctx,
        )
        np.testing.assert_allclose(to_numpy(op.apply(x)), to_numpy(apply(x)))
        np.testing.assert_allclose(to_numpy(op.rapply(y)), metric_adjoint_np @ to_numpy(y))
        assert not np.allclose(to_numpy(op.rapply(y)), to_numpy(coordinate_rapply(y)))
        _assert_adjoint_identity(op, x, y)

    def test_distinct_from_direct_constructor(self, numpy_ctx):
        """``from_coordinate_adjoint`` wraps the callable; the direct constructor stores it verbatim."""
        fixture = _non_euclidean_matrix_free_fixture(numpy_ctx)
        domain = fixture["domain"]
        codomain = fixture["codomain"]
        matrix = numpy_ctx.asarray(fixture["matrix_np"])
        metric_adjoint = numpy_ctx.asarray(fixture["metric_adjoint_np"])
        y = fixture["y"]

        def apply(z):
            return matrix @ z

        def metric_rapply(w):
            return metric_adjoint @ w

        def coordinate_rapply(w):
            return matrix.T @ w

        direct = sc.MatrixFreeLinOp(apply, metric_rapply, domain, codomain, numpy_ctx)
        wrapped = sc.MatrixFreeLinOp.from_coordinate_adjoint(
            apply, coordinate_rapply, domain, codomain, numpy_ctx,
        )
        assert direct.rapply_fn is metric_rapply
        assert wrapped.rapply_fn is not coordinate_rapply
        np.testing.assert_allclose(to_numpy(direct.rapply(y)), to_numpy(metric_rapply(y)))
        np.testing.assert_allclose(to_numpy(wrapped.rapply(y)), to_numpy(metric_rapply(y)))
        assert not np.allclose(to_numpy(wrapped.rapply(y)), to_numpy(coordinate_rapply(y)))

    def test_direct_constructor_rapply_not_riesz_wrapped_on_non_euclidean(self, numpy_ctx):
        """ADR-009: a directly supplied ``rapply`` is used verbatim, never Riesz-corrected."""
        fixture = _non_euclidean_matrix_free_fixture(numpy_ctx)
        op = fixture["op"]
        x = fixture["x"]
        y = fixture["y"]
        matrix_np = fixture["matrix_np"]
        metric_adjoint_np = fixture["metric_adjoint_np"]
        gx = fixture["gx"]
        gy = fixture["gy"]

        expected_reverse = metric_adjoint_np @ to_numpy(y)
        double_corrected = (
            np.diag(1.0 / gx) @ metric_adjoint_np @ np.diag(gy) @ to_numpy(y)
        )
        np.testing.assert_allclose(to_numpy(op.apply(x)), matrix_np @ to_numpy(x))
        np.testing.assert_allclose(to_numpy(op.rapply(y)), expected_reverse)
        assert not np.allclose(to_numpy(op.rapply(y)), matrix_np.T @ to_numpy(y))
        assert not np.allclose(to_numpy(op.rapply(y)), double_corrected)
        _assert_adjoint_identity(op, x, y)

    def test_method_based_riesz_space_is_accepted(self, numpy_ctx):
        """Spaces exposing ``riesz``/``riesz_inverse`` methods are usable for wrapping."""

        class MethodInnerProduct(sc.InnerProduct):
            def inner(self, ops, x, y):
                return ops.vdot(x, numpy_ctx.asarray([2.0, 5.0]) * y)

        class MethodRieszSpace(sc.DenseCoordinateSpace):
            def __init__(self, ctx):
                super().__init__((2,), ctx, geometry=MethodInnerProduct())
                self.weights = ctx.asarray([2.0, 5.0])

            def riesz(self, x):
                return self.weights * x

            def riesz_inverse(self, x):
                return x / self.weights

        space = MethodRieszSpace(numpy_ctx)
        op = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), space, space, numpy_ctx,
        )
        _assert_adjoint_identity(
            op, numpy_ctx.asarray([0.25, -1.5]), numpy_ctx.asarray([2.0, -0.5]),
        )

    def test_convert_preserves_direct_reverse_without_riesz(self, numpy_ctx):
        """Converting a direct matrix-free op keeps the user's reverse callable as-is."""
        new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        WeightedVectorSpace = _weighted_space_class()
        domain = WeightedVectorSpace([2.0, 5.0], numpy_ctx)
        codomain = WeightedVectorSpace([3.0, 7.0, 11.0], numpy_ctx)
        matrix = numpy_ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        metric_adjoint = numpy_ctx.asarray(
            np.diag([1.0 / 2.0, 1.0 / 5.0])
            @ to_numpy(matrix.T)
            @ np.diag([3.0, 7.0, 11.0])
        )

        def apply(z):
            return matrix @ z

        def rapply(w):
            return metric_adjoint @ w

        op = sc.MatrixFreeLinOp(apply, rapply, domain, codomain, numpy_ctx)
        converted = op.convert(new_ctx)
        x = new_ctx.asarray([0.25, -1.5])
        y = new_ctx.asarray([2.0, -0.5, 1.25])

        assert converted.rapply_fn is rapply
        np.testing.assert_allclose(to_numpy(converted.rapply(y)), to_numpy(rapply(y)))
        _assert_adjoint_identity(converted, x, y)

    def test_convert_preserves_batched_reverse(self, numpy_ctx):
        """``from_coordinate_adjoint`` with a batched coordinate adjoint survives ``convert``."""
        new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        WeightedVectorSpace = _weighted_space_class()
        domain = WeightedVectorSpace([2.0, 5.0], numpy_ctx)
        codomain = WeightedVectorSpace([3.0, 7.0, 11.0], numpy_ctx)
        matrix = numpy_ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        calls = {"rvapply": 0}

        def rvapply(ws):
            calls["rvapply"] += 1
            return ws @ matrix

        op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
            lambda z: matrix @ z,
            lambda w: matrix.T @ w,
            domain,
            codomain,
            numpy_ctx,
            coordinate_rvapply=rvapply,
        )
        converted = op.convert(new_ctx)
        ys = new_ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])
        expected = np.stack([to_numpy(converted.rapply(y)) for y in ys], axis=0)

        np.testing.assert_allclose(to_numpy(converted.rvapply(ys)), expected)
        assert calls["rvapply"] == 1

    def test_convert_without_rvapply_uses_fallback(self, numpy_ctx):
        """When no batched coordinate adjoint is given, ``rvapply_fn`` stays ``None``."""
        new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        WeightedVectorSpace = _weighted_space_class()
        domain = WeightedVectorSpace([2.0, 5.0], numpy_ctx)
        codomain = WeightedVectorSpace([3.0, 7.0, 11.0], numpy_ctx)
        matrix = numpy_ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
        op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
            lambda z: matrix @ z, lambda w: matrix.T @ w, domain, codomain, numpy_ctx,
        )
        converted = op.convert(new_ctx)
        ys = new_ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])
        expected = np.stack([to_numpy(converted.rapply(y)) for y in ys], axis=0)

        assert op.rvapply_fn is None
        assert converted.rvapply_fn is None
        np.testing.assert_allclose(to_numpy(converted.rvapply(ys)), expected)


# ===========================================================================
# Algebra class hierarchy and check-policy independence
# (folded from test_algebra_linop.py)
# ===========================================================================
class TestAlgebraClassHierarchy:
    def _op(self, ctx):
        X = sc.DenseCoordinateSpace((2,), ctx)
        return sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X, X, ctx)

    def test_no_adjoint_linop_symbol_exported(self):
        """There is no public ``AdjointLinOp``; ``A.H`` returns the private view."""
        assert not hasattr(sc, "AdjointLinOp")

    def test_instances_are_linops(self, numpy_ctx):
        A = self._op(numpy_ctx)
        assert isinstance(2.0 * A, sc.LinOp)
        assert isinstance(A + A, sc.LinOp)
        assert isinstance(A @ A, sc.LinOp)
        assert isinstance(A.H, sc.LinOp)
        assert isinstance(sc.ZeroLinOp(A.domain, A.codomain, A.ctx), sc.LinOp)
        assert isinstance(sc.IdentityLinOp(A.domain, A.ctx), sc.LinOp)
        assert isinstance(
            sc.MatrixFreeLinOp(A.apply, A.rapply, A.domain, A.codomain, A.ctx), sc.LinOp,
        )

    def test_classes_subclass_linop(self):
        assert issubclass(sc.ScaledLinOp, sc.LinOp)
        assert issubclass(sc.SumLinOp, sc.LinOp)
        assert issubclass(sc.ComposedLinOp, sc.LinOp)
        assert issubclass(sc.ZeroLinOp, sc.LinOp)
        assert issubclass(sc.IdentityLinOp, sc.LinOp)
        assert issubclass(sc.MatrixFreeLinOp, sc.LinOp)

    def test_check_policy_mismatch_does_not_block_algebra(self):
        """Operands with differing ``check_level`` still combine (dtype matches)."""
        checked = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
        unchecked = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        A = self._op(checked)
        B = self._op(unchecked)
        assert isinstance(A + B, sc.SumLinOp)
        assert isinstance(A @ B, sc.ComposedLinOp)


# ===========================================================================
# ScaledLinOp / SumLinOp batched dispatch (folded from test_algebra.py)
# ===========================================================================
class TestScaledBatchedDispatch:
    def test_scaled_paths_use_space_scale_and_scale_batch(self, numpy_ctx):
        """``apply``/``rapply``/``vapply``/``rvapply`` route through the space scale helpers."""

        class CountingVectorSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx, counter):
                self.counter = counter
                super().__init__(shape, ctx)

            def scale(self, a, x):
                self.counter["scale"] += 1
                return super().scale(a, x)

            def scale_batch(self, a, x):
                self.counter["scale_batch"] += 1
                return super().scale_batch(a, x)

            def _convert(self, new_ctx):
                return CountingVectorSpace(self.shape, new_ctx, self.counter)

        domain_counter = {"scale": 0, "scale_batch": 0}
        codomain_counter = {"scale": 0, "scale_batch": 0}
        X = CountingVectorSpace((2,), numpy_ctx, domain_counter)
        Y = CountingVectorSpace((2,), numpy_ctx, codomain_counter)
        A = sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), X, Y, numpy_ctx)
        op = 3.0 * A

        op.apply(numpy_ctx.asarray([1.0, -2.0]))
        assert codomain_counter["scale"] == 1
        assert domain_counter["scale"] == 0

        op.rapply(numpy_ctx.asarray([0.5, 4.0]))
        assert domain_counter["scale"] == 1

        op.vapply(numpy_ctx.asarray([[1.0, -2.0], [0.5, 4.0]]))
        assert codomain_counter["scale_batch"] == 1

        op.rvapply(numpy_ctx.asarray([[0.5, 4.0], [3.0, -1.0]]))
        assert domain_counter["scale_batch"] == 1


class TestSumBatchedDispatch:
    def test_batched_accumulation_uses_space_add_batch(self, numpy_ctx):
        """``vapply``/``rvapply`` accumulate via the space ``add_batch`` helper."""

        class CountingVectorSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx, counter):
                self.counter = counter
                super().__init__(shape, ctx)

            def add_batch(self, x, y):
                self.counter["calls"] += 1
                return super().add_batch(x, y)

            def _convert(self, new_ctx):
                return CountingVectorSpace(self.shape, new_ctx, self.counter)

        domain_counter = {"calls": 0}
        codomain_counter = {"calls": 0}
        X = CountingVectorSpace((2,), numpy_ctx, domain_counter)
        Y = CountingVectorSpace((2,), numpy_ctx, codomain_counter)
        A = sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), X, Y, numpy_ctx)
        B = sc.DenseLinOp(numpy_ctx.asarray([[0.5, -4.0], [2.0, 5.0]]), X, Y, numpy_ctx)
        op = A + B

        op.vapply(numpy_ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]))
        assert codomain_counter["calls"] == 1
        assert domain_counter["calls"] == 0

        op.rvapply(numpy_ctx.asarray([[5.0, -2.0], [0.5, -1.0]]))
        assert domain_counter["calls"] == 1


# ===========================================================================
# Python ``sum()`` builtin over LinOps (folded from test_algebra.py)
# ===========================================================================
class TestPythonSumBuiltin:
    def test_sum_starts_from_zero_and_accumulates(self, numpy_ctx):
        """``sum([...])`` works because ``0 + A`` returns ``A`` (``__radd__``)."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X, X, numpy_ctx)
        B = sc.DenseLinOp(numpy_ctx.asarray([[0.5, 1.0], [1.5, 2.0]]), X, X, numpy_ctx)
        C = sc.DenseLinOp(numpy_ctx.asarray([[-2.0, -4.0], [-6.0, -8.0]]), X, X, numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0])

        op = sum([A, B, C])
        expected = to_numpy(A.apply(x)) + to_numpy(B.apply(x)) + to_numpy(C.apply(x))
        np.testing.assert_allclose(to_numpy(op.apply(x)), expected)

    def test_zero_plus_op_is_op(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        A = sc.DenseLinOp(numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), X, X, numpy_ctx)
        assert 0 + A is A


# ===========================================================================
# MatrixFreeLinOp vapply callback vs fallback (folded from test_batched_lifting.py)
# ===========================================================================
class TestMatrixFreeBatchedCallbacks:
    def test_vapply_rvapply_use_callbacks_when_supplied(self, numpy_ctx):
        """Supplied batched callbacks are used; counters confirm exactly one call each."""
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        calls = {"vapply": 0, "rvapply": 0}

        def apply(x):
            return numpy_ctx.asarray(matrix @ np.asarray(x))

        def rapply(y):
            return numpy_ctx.asarray(matrix.T @ np.asarray(y))

        def vapply(xs):
            calls["vapply"] += 1
            return numpy_ctx.asarray(np.asarray(xs) @ matrix.T)

        def rvapply(ys):
            calls["rvapply"] += 1
            return numpy_ctx.asarray(np.asarray(ys) @ matrix)

        op = sc.MatrixFreeLinOp(apply, rapply, dom, cod, numpy_ctx, vapply, rvapply)
        xs = numpy_ctx.asarray([[7.0, 8.0], [1.0, -1.0]])
        ys = numpy_ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        expected_v = np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
        expected_rv = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
        np.testing.assert_allclose(to_numpy(op.vapply(xs)), expected_v)
        np.testing.assert_allclose(to_numpy(op.rvapply(ys)), expected_rv)
        assert calls == {"vapply": 1, "rvapply": 1}

    def test_vapply_rvapply_fall_back_when_callbacks_absent(self, numpy_ctx):
        """Without batched callbacks, fallback batching matches a scalar loop."""
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        op = sc.MatrixFreeLinOp(
            lambda x: numpy_ctx.asarray(matrix @ np.asarray(x)),
            lambda y: numpy_ctx.asarray(matrix.T @ np.asarray(y)),
            dom,
            cod,
            numpy_ctx,
        )
        xs = numpy_ctx.asarray([[7.0, 8.0], [1.0, -1.0]])
        ys = numpy_ctx.asarray([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        expected_v = np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
        expected_rv = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
        np.testing.assert_allclose(to_numpy(op.vapply(xs)), expected_v)
        np.testing.assert_allclose(to_numpy(op.rvapply(ys)), expected_rv)


# ===========================================================================
# Cross-backend complex adjoint identity for all 7 algebra cases
# (folded from test_algebra.py)
# ===========================================================================
_ALGEBRA_CASE_IDS = ["scaled", "sum", "composed", "zero", "identity", "matrix_free", "adjoint"]


def _backend_complex_params():
    params = [pytest.param("numpy", np.complex128, id="numpy")]
    if has_jax():
        params.append(pytest.param("jax", jax_complex_dtype(), id="jax"))
    if has_torch():
        params.append(pytest.param("torch", torch_complex_dtype(), id="torch"))
    return params


def _ops_for_backend(name):
    if name == "numpy":
        return sc.NumpyOps()
    if name == "jax":
        return sc.JaxOps()
    if name == "torch":
        return sc.TorchOps()
    raise ValueError(f"Unknown backend {name!r}.")


class TestCrossBackendAlgebraAdjointIdentity:
    @pytest.mark.parametrize("backend_name, dtype", _backend_complex_params())
    @pytest.mark.parametrize("case_index", range(7), ids=_ALGEBRA_CASE_IDS)
    def test_complex_adjoint_identity(self, backend_name, dtype, case_index):
        ctx = sc.Context(_ops_for_backend(backend_name), dtype=dtype)
        op, x, y = _algebra_cases(ctx)[case_index]
        lhs = ctx.ops.vdot(op.apply(x), y)
        rhs = ctx.ops.vdot(x, op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)


# ===========================================================================
# jax pytree round-trip and jit for algebra expressions
# (folded from test_algebra.py)
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJaxAlgebra:
    @pytest.mark.parametrize("case_index", range(7), ids=_ALGEBRA_CASE_IDS)
    def test_pytree_roundtrip_for_algebra_classes(self, case_index, numpy_complex_ctx):
        import jax

        op, _, _ = _algebra_cases(numpy_complex_ctx)[case_index]
        leaves, treedef = jax.tree.flatten(op)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert rebuilt == op

    def test_jit_algebra_expression_matches_eager(self):
        import jax

        from tests._helpers import jax_real_dtype

        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        X = sc.DenseCoordinateSpace((2,), ctx)
        Y = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DenseLinOp(
            ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, ctx,
        )
        B = sc.DenseLinOp(
            ctx.asarray([[0.5, -1.0], [2.0, 1.0], [-0.5, 3.0]]), X, Y, ctx,
        )
        C = sc.DenseLinOp(ctx.asarray([[2.0, -1.0], [0.25, 1.5]]), X, X, ctx)
        expr = (2 * A + B) @ C
        x = ctx.asarray([1.0, -2.0])

        apply_jit = jax.jit(lambda op, z: op.apply(z))
        np.testing.assert_allclose(
            to_numpy(apply_jit(expr, x)), to_numpy(expr.apply(x)),
        )

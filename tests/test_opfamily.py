"""Point-indexed operator families and the functional-weighted map ``F · A``.

``m(x) = F(x) A x`` is not linear, so it is *not* a ``LinOp``. These tests pin
the three things that follow from that:

* the map is what it claims and is genuinely non-linear;
* freezing the point (``at``) recovers an ordinary ``LinOp`` that participates
  in the operator algebra;
* the derivative (``linearize_at``) is a *different* operator from the frozen
  member, and its adjoint is the **metric** adjoint.

The metric and complex cases are the load-bearing ones: a Euclidean-only suite
would pass with a coordinate adjoint and wrong conjugation.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


# ``spacecore.opfamily`` is a top-level module spanning linop and functional, so
# this suite lives at the tests root and declares the fixture the per-object
# suites (tests/functional, tests/linops) provide via their own conftest.
@pytest.fixture
def numpy_ctx():
    """Float64 NumPy context (default ``standard`` check level)."""
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _matrix():
    return np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]])


def _euclidean(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray(_matrix()), X, X, ctx)
    return X, A, sc.SquaredL2NormFunctional(X)


def _weighted(ctx):
    weights = ctx.asarray(np.array([2.0, 5.0, 11.0]))
    X = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(weights))
    A = sc.DenseLinOp(ctx.asarray(_matrix()), X, X, ctx)
    return X, A, sc.SquaredL2NormFunctional(X)


def _complex_weighted():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    weights = ctx.asarray(np.array([2.0, 5.0, 11.0]))
    X = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(weights))
    A = sc.DenseLinOp(
        ctx.asarray(np.array([[1 + 1j, 2, 0], [0, 1, 3 - 2j], [2, 0, 1j]])), X, X, ctx
    )
    # A complex-VALUED weight, so the conjugations in the adjoint are exercised.
    F = sc.InnerProductFunctional(ctx.asarray([1 + 1j, 2 - 0.5j, -1 + 0.3j]), X, ctx)
    return ctx, X, A, F


# ===========================================================================
# The map
# ===========================================================================
class TestFunctionalWeightedMap:
    def test_value_is_the_weighted_image(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        np.testing.assert_allclose(
            to_numpy((F * A).apply(x)), float(F.value(x)) * to_numpy(A.apply(x))
        )

    def test_is_not_a_linop(self, numpy_ctx):
        """The whole point: it must not enter code that assumes linearity."""
        X, A, F = _euclidean(numpy_ctx)
        m = F * A
        assert isinstance(m, sc.OperatorFamily)
        assert not isinstance(m, sc.LinOp)

    def test_is_genuinely_nonlinear(self, numpy_ctx):
        """``m(2x) != 2 m(x)`` — this is why it cannot be a ``LinOp``."""
        X, A, F = _euclidean(numpy_ctx)
        m = F * A
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        assert not np.allclose(to_numpy(m.apply(X.scale(2.0, x))), 2.0 * to_numpy(m.apply(x)))

    def test_both_operand_orders_work(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        np.testing.assert_allclose(to_numpy((F * A).apply(x)), to_numpy((A * F).apply(x)))

    def test_call_is_apply(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        np.testing.assert_allclose(to_numpy((F * A)(x)), to_numpy((F * A).apply(x)))


# ===========================================================================
# Freezing the point recovers a LinOp
# ===========================================================================
class TestFrozenMember:
    def test_at_returns_a_scaled_linop(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        member = (F * A).at(x)
        assert isinstance(member, sc.LinOp)
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        np.testing.assert_allclose(
            to_numpy(member.apply(h)), float(F.value(x)) * to_numpy(A.apply(h))
        )

    def test_at_x_applied_to_x_is_the_map(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        m = F * A
        np.testing.assert_allclose(to_numpy(m.at(x).apply(x)), to_numpy(m.apply(x)))

    def test_member_participates_in_the_operator_algebra(self, numpy_ctx):
        """Freezing buys the whole algebra back: ``@``, ``+``, ``.H``."""
        X, A, F = _euclidean(numpy_ctx)
        member = (F * A).at(numpy_ctx.asarray([1.0, 2.0, -1.0]))
        assert isinstance(member @ A, sc.LinOp)
        assert isinstance(member + A, sc.LinOp)
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        np.testing.assert_allclose(
            to_numpy(member.H.apply(h)),
            float(F.value(numpy_ctx.asarray([1.0, 2.0, -1.0]))) * to_numpy(A.H.apply(h)),
        )


# ===========================================================================
# The derivative is a different operator from the frozen member
# ===========================================================================
class TestLinearization:
    @pytest.mark.parametrize("build", [_euclidean, _weighted], ids=["euclidean", "weighted"])
    def test_matches_finite_differences(self, numpy_ctx, build):
        X, A, F = build(numpy_ctx)
        m = F * A
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        eps = 1e-6
        finite_difference = (
            to_numpy(m.apply(X.axpy(eps, h, x))) - to_numpy(m.apply(X.axpy(-eps, h, x)))
        ) / (2.0 * eps)
        np.testing.assert_allclose(
            to_numpy(m.linearize_at(x).apply(h)), finite_difference, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize("build", [_euclidean, _weighted], ids=["euclidean", "weighted"])
    def test_adjoint_identity_holds_in_the_space_geometry(self, numpy_ctx, build):
        """``<D h, w>_Y == <h, D^# w>_X`` — the *metric* adjoint (ADR-009)."""
        X, A, F = build(numpy_ctx)
        D = (F * A).linearize_at(numpy_ctx.asarray([1.0, 2.0, -1.0]))
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        w = numpy_ctx.asarray([1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            to_numpy(X.inner(D.apply(h), w)), to_numpy(X.inner(h, D.rapply(w)))
        )

    def test_complex_weight_conjugations_are_right(self):
        """A complex-valued weight on a weighted metric: both conjugations bite."""
        ctx, X, A, F = _complex_weighted()
        m = F * A
        x = ctx.asarray([1 + 0j, 2 - 1j, -1 + 2j])
        h = ctx.asarray([0.5 + 1j, -1 + 0j, 2 + 0j])
        w = ctx.asarray([1 + 1j, 1 + 0j, 1 - 1j])
        D = m.linearize_at(x)

        eps = 1e-6
        finite_difference = (
            to_numpy(m.apply(X.axpy(eps, h, x))) - to_numpy(m.apply(X.axpy(-eps, h, x)))
        ) / (2.0 * eps)
        np.testing.assert_allclose(
            to_numpy(D.apply(h)), finite_difference, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            to_numpy(X.inner(D.apply(h), w)), to_numpy(X.inner(h, D.rapply(w)))
        )

    def test_derivative_differs_from_the_frozen_member(self, numpy_ctx):
        """They are two different operators; conflating them is the easy mistake."""
        X, A, F = _euclidean(numpy_ctx)
        m = F * A
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        assert not np.allclose(
            to_numpy(m.linearize_at(x).apply(h)), to_numpy(m.at(x).apply(h))
        )

    def test_derivative_equals_frozen_member_for_a_constant_weight(self, numpy_ctx):
        """A constant family has zero rank-one correction, so they coincide."""
        X, A, _ = _euclidean(numpy_ctx)
        C = sc.ConstantFunctional(X, 3.0, numpy_ctx)
        m = sc.FunctionalScaledOperator(C, A)   # built directly: the factory would collapse it
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        np.testing.assert_allclose(
            to_numpy(m.linearize_at(x).apply(h)), to_numpy(m.at(x).apply(h))
        )

    def test_base_class_has_no_linearization(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)

        class Bare(sc.OperatorFamily):
            """A family that only implements ``at`` — the minimal subclass."""

            def at(self, x):
                return A

            def tree_flatten(self):
                return (), (self.domain, self.codomain, self.ctx)

            @classmethod
            def tree_unflatten(cls, aux, children):
                return cls(*aux)

        with pytest.raises(NotImplementedError, match="linearize_at"):
            Bare(X, X, numpy_ctx).linearize_at(numpy_ctx.asarray([1.0, 2.0, -1.0]))


# ===========================================================================
# Factory: the linear case is not forced through the non-linear type
# ===========================================================================
class TestFactory:
    def test_constant_weight_collapses_to_a_scaled_linop(self, numpy_ctx):
        X, A, _ = _euclidean(numpy_ctx)
        result = sc.ConstantFunctional(X, 3.0, numpy_ctx) * A
        assert isinstance(result, sc.LinOp)
        h = numpy_ctx.asarray([0.5, -1.0, 2.0])
        np.testing.assert_allclose(to_numpy(result.apply(h)), 3.0 * to_numpy(A.apply(h)))

    def test_zero_weight_collapses_to_zero(self, numpy_ctx):
        X, A, _ = _euclidean(numpy_ctx)
        assert isinstance(sc.ZeroFunctional(X, numpy_ctx) * A, sc.ZeroLinOp)

    def test_collapse_is_structural_not_value_based(self, numpy_ctx):
        """A functional that merely *happens* to be constant is not recognized."""
        X, A, _ = _euclidean(numpy_ctx)
        constant_valued = sc.InnerProductFunctional(
            numpy_ctx.asarray([0.0, 0.0, 0.0]), X, numpy_ctx
        )
        assert isinstance(constant_valued * A, sc.FunctionalScaledOperator)

    def test_domain_mismatch_raises(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        other = sc.SquaredL2NormFunctional(sc.DenseCoordinateSpace((2,), numpy_ctx))
        with pytest.raises(ValueError, match="same domain|domain =="):
            other * A

    def test_type_guards(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        with pytest.raises(TypeError, match="Functional"):
            sc.FunctionalScaledOperator("nope", A)
        with pytest.raises(TypeError, match="LinOp"):
            sc.FunctionalScaledOperator(F, "nope")


# ===========================================================================
# Container protocol
# ===========================================================================
class TestContainerProtocol:
    def test_equality(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        assert sc.FunctionalScaledOperator(F, A) == sc.FunctionalScaledOperator(F, A)
        other = sc.DenseLinOp(numpy_ctx.asarray(np.eye(3)), X, X, numpy_ctx)
        assert sc.FunctionalScaledOperator(F, A) != sc.FunctionalScaledOperator(F, other)

    def test_pytree_round_trip(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        m = sc.FunctionalScaledOperator(F, A)
        children, aux = m.tree_flatten()
        assert sc.FunctionalScaledOperator.tree_unflatten(aux, children) == m

    def test_convert_moves_both_operands(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        target = sc.Context(sc.NumpyOps(), dtype=np.float32)
        moved = sc.FunctionalScaledOperator(F, A).convert(target)
        assert moved.ctx == target
        assert moved.domain.ctx == target and moved.op.ctx == target

    def test_domain_and_codomain(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        m = F * A
        assert m.domain == A.domain and m.codomain == A.codomain

    def test_repr_marks_it_as_nonlinear(self, numpy_ctx):
        X, A, F = _euclidean(numpy_ctx)
        assert "⇝" in repr(F * A)   # not "→", which denotes a linear arrow

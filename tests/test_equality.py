"""Tests for the tiered ``__eq__`` rule: backend compatibility first, then algebra.

Contract under test (see docs/dev/0.4.0-eq-implementation-spec.md):

* Tier 1 — backend compatibility: same concrete type AND same backend
  (ops family + dtype), ignoring ``check_level``. Failure -> ``NotImplemented``
  (so foreign-type comparisons stay symmetric and fall back to identity).
* Tier 2 — cheap structural checks (shapes, spaces, counts, treedef) before any
  numerical comparison.
* Tier 3 — ``allclose`` on values, with ``equal_nan=True`` for reflexivity.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


@pytest.fixture
def ctx32():
    return sc.Context(sc.NumpyOps(), dtype=np.float32)


@pytest.fixture
def cctx():
    return sc.Context(sc.NumpyOps(), dtype=np.complex128)


def _maybe_ctx(family, dtype):
    from spacecore.contextual._state import normalize_context

    try:
        return normalize_context(family, dtype=dtype)
    except Exception:
        return None


# ===========================================================================
# Tier 1 — backend compatibility (shared gate)
# ===========================================================================
class TestBackendGate:
    def test_check_level_ignored_for_spaces(self, ctx):
        # check_level is policy, not backend: must not affect identity.
        assert sc.DenseVectorSpace((3,), ctx) == sc.DenseVectorSpace(
            (3,), ctx, check_level="strict"
        )

    def test_check_level_ignored_for_linops(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        vs = sc.DenseVectorSpace((3,), ctx, check_level="strict")
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        As = sc.DenseLinOp(ctx.asarray(np.eye(3)), vs, vs, ctx, check_level="strict")
        assert A == As

    def test_dtype_is_part_of_identity(self, ctx, ctx32):
        # float32 vs float64: different representation -> not equal.
        assert sc.DenseVectorSpace((3,), ctx) != sc.DenseVectorSpace((3,), ctx32)

    def test_cross_backend_not_equal(self, ctx):
        other = _maybe_ctx("jax", np.float64) or _maybe_ctx("torch", np.float64)
        if other is None:
            pytest.skip("no second backend available")
        assert sc.DenseVectorSpace((3,), ctx) != sc.DenseVectorSpace((3,), other)

    def test_foreign_type_symmetric(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert (v == 5) is False and (5 == v) is False
        assert (A == 5) is False and (5 == A) is False

    def test_gate_returns_notimplemented_on_foreign_type(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        assert v.__eq__(5) is NotImplemented
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert A.__eq__(object()) is NotImplemented


# ===========================================================================
# Spaces — field, shape, geometry, decisions
# ===========================================================================
class TestSpaceEquality:
    def test_field_distinguishes(self, ctx, cctx):
        assert sc.DenseVectorSpace((3,), ctx) != sc.DenseVectorSpace((3,), cctx)

    def test_shape(self, ctx):
        assert sc.DenseVectorSpace((3,), ctx) == sc.DenseVectorSpace((3,), ctx)
        assert sc.DenseVectorSpace((3,), ctx) != sc.DenseVectorSpace((4,), ctx)

    def test_geometry_kind_and_weights(self, ctx):
        w = sc.WeightedInnerProduct(ctx.asarray(np.arange(1, 4, dtype=float)))
        w2 = sc.WeightedInnerProduct(ctx.asarray(np.arange(1, 4, dtype=float)))
        wd = sc.WeightedInnerProduct(ctx.asarray(np.arange(2, 5, dtype=float)))
        assert sc.DenseVectorSpace((3,), ctx, geometry=w) == sc.DenseVectorSpace((3,), ctx, geometry=w2)
        assert sc.DenseVectorSpace((3,), ctx, geometry=w) != sc.DenseVectorSpace((3,), ctx, geometry=wd)
        assert sc.DenseVectorSpace((3,), ctx, geometry=w) != sc.DenseVectorSpace((3,), ctx)  # euclidean

    def test_weighted_geometry_shape_guard(self, ctx):
        # Regression: np.allclose used to broadcast [2.] vs [2.,2.,2.] to True.
        w1 = sc.WeightedInnerProduct(ctx.asarray(np.array([2.0])))
        w3 = sc.WeightedInnerProduct(ctx.asarray(np.array([2.0, 2.0, 2.0])))
        assert w1 != w3

    def test_hermitian_tolerances_excluded(self, ctx):
        # Decision: atol/rtol/enforce_herm are membership policy, not identity.
        assert sc.HermitianSpace(3, atol=0.0, ctx=ctx) == sc.HermitianSpace(3, atol=1e-6, ctx=ctx)
        assert sc.HermitianSpace(3, enforce_herm=True, ctx=ctx) == sc.HermitianSpace(
            3, enforce_herm=False, ctx=ctx
        )
        assert sc.HermitianSpace(3, ctx=ctx) != sc.HermitianSpace(4, ctx=ctx)

    def test_stacked_base_is_load_bearing(self, ctx):
        a = sc.DenseVectorSpace((3,), ctx).stacked(4)
        b = sc.DenseVectorSpace((3,), ctx).stacked(4)
        c = sc.DenseVectorSpace((5,), ctx).stacked(4)
        assert a == b and a != c

    def test_tree_treedef_and_leaves(self, ctx):
        leaves = (sc.DenseVectorSpace((3,), ctx), sc.DenseVectorSpace((2,), ctx))
        t1 = sc.TreeSpace((0, 0), leaves, ctx=ctx)
        t2 = sc.TreeSpace((0, 0), leaves, ctx=ctx)
        t3 = sc.TreeSpace((0, 0), (sc.DenseVectorSpace((3,), ctx), sc.DenseVectorSpace((9,), ctx)), ctx=ctx)
        assert t1 == t2 and t1 != t3

    def test_spaces_unhashable(self, ctx):
        with pytest.raises(TypeError):
            hash(sc.DenseVectorSpace((3,), ctx))


# ===========================================================================
# Linear operators
# ===========================================================================
class TestLinOpEquality:
    def _v(self, ctx, n=3):
        return sc.DenseVectorSpace((n,), ctx)

    def test_dense_values(self, ctx):
        v = self._v(ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert A == sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert A != sc.DenseLinOp(ctx.asarray(2 * np.eye(3)), v, v, ctx)

    def test_dense_nan_reflexive(self, ctx):
        v = self._v(ctx)
        m = np.eye(3)
        m[0, 0] = np.nan
        A = sc.DenseLinOp(ctx.asarray(m), v, v, ctx)
        assert A == A  # equal_nan=True

    def test_dense_geometry_in_domain_matters(self, ctx):
        # Same matrix, domains differ only in geometry -> different operator.
        v = self._v(ctx)
        w = sc.WeightedInnerProduct(ctx.asarray(np.arange(1, 4, dtype=float)))
        vw = sc.DenseVectorSpace((3,), ctx, geometry=w)
        assert sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx) != sc.DenseLinOp(
            ctx.asarray(np.eye(3)), vw, vw, ctx
        )

    def test_sparse_nan_not_equal_to_finite(self, ctx):
        # Regression: allclose_sparse was NaN-blind, so a NaN entry compared
        # "close" to a finite one and two different operators returned True.
        import scipy.sparse as sps

        v = self._v(ctx)

        def mk(corner):
            m = sps.eye(3, format="csr").tolil()
            m[0, 0] = corner
            return sc.SparseLinOp(ctx.assparse(m.tocsr()), v, v, ctx)

        assert (mk(np.nan) == mk(5.0)) is False
        assert (mk(5.0) == mk(np.nan)) is False
        assert (mk(np.nan) == mk(np.nan)) is False   # sparse NaN is non-reflexive (documented)
        assert (mk(2.0) == mk(2.0)) is True

    def test_cross_type_not_equal(self, ctx):
        # Mathematically identity, structurally different types -> not equal.
        v = self._v(ctx)
        ident = sc.IdentityLinOp(v, ctx)
        diag = sc.DiagonalLinOp(ctx.asarray(np.ones(3)), v, ctx)
        dense = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert ident != diag and ident != dense and diag != dense

    def test_scaled_returns_python_bool(self, ctx):
        # Regression: 0-d array scalar comparison used to leak np.bool_.
        v = self._v(ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        for scal in (2.0, np.float64(2.0)):
            res = (scal * A) == (2.0 * A)
            assert type(res) is bool and res is True
        assert ((2.0 * A) == (3.0 * A)) is False

    def test_scaled_leak_through_composition(self, ctx):
        v = self._v(ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        res = ((2.0 * A) @ A) == ((3.0 * A) @ A)
        assert type(res) is bool and res is False

    def test_scaled_nan_scalar_reflexive(self, ctx):
        # Regression: bool(nan == nan) is False, so a NaN-scaled op must use the
        # NaN-reflexive scalar comparison to stay equal to itself.
        v = self._v(ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        for scal in (float("nan"), np.float64("nan")):
            s = scal * A
            assert (s == s) is True
            assert ((s @ A) == (s @ A)) is True   # through composition
            assert (s.H == s.H) is True           # through adjoint
        # distinct scalars (one NaN) still unequal
        assert ((float("nan") * A) == (2.0 * A)) is False

    def test_sum_is_ordered(self, ctx):
        v = self._v(ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        ident = sc.IdentityLinOp(v, ctx)
        # Distinct operands in different order must not be equal.
        assert (ident + A) != (A + ident)

    def test_matrixfree_callable_identity(self, ctx):
        v = self._v(ctx)
        def f(x):
            return x

        a = sc.MatrixFreeLinOp(f, f, v, v, ctx)
        assert a == sc.MatrixFreeLinOp(f, f, v, v, ctx)
        assert a != sc.MatrixFreeLinOp(lambda x: x, lambda x: x, v, v, ctx)

    def test_adjoint(self, ctx):
        v = self._v(ctx)
        A = sc.DenseLinOp(ctx.asarray(np.arange(9.0).reshape(3, 3)), v, v, ctx)
        B = sc.DenseLinOp(ctx.asarray(np.arange(9.0).reshape(3, 3)), v, v, ctx)
        assert A.H == B.H

    def test_tree_linop_parts_and_structure(self, ctx):
        v2 = sc.DenseVectorSpace((2,), ctx)
        v3 = sc.DenseVectorSpace((3,), ctx)
        def blk():
            return sc.BlockDiagonalLinOp(
                [
                    sc.DenseLinOp(ctx.asarray(np.eye(3)), v3, v3, ctx),
                    sc.DenseLinOp(ctx.asarray(np.eye(2)), v2, v2, ctx),
                ]
            )
        assert blk() == blk()
        other = sc.BlockDiagonalLinOp(
            [
                sc.DenseLinOp(ctx.asarray(2 * np.eye(3)), v3, v3, ctx),
                sc.DenseLinOp(ctx.asarray(np.eye(2)), v2, v2, ctx),
            ]
        )
        assert blk() != other


# ===========================================================================
# Functionals
# ===========================================================================
class TestFunctionalEquality:
    def _v(self, ctx, n=3):
        return sc.DenseVectorSpace((n,), ctx)

    def test_base_returns_notimplemented(self, ctx):
        f = sc.InnerProductFunctional(ctx.asarray(np.ones(3)), self._v(ctx), ctx)
        assert sc.Functional.__eq__(f, object()) is NotImplemented

    def test_inner_product(self, ctx):
        v = self._v(ctx)
        a = sc.InnerProductFunctional(ctx.asarray(np.ones(3)), v, ctx)
        assert a == sc.InnerProductFunctional(ctx.asarray(np.ones(3)), v, ctx)
        assert a != sc.InnerProductFunctional(ctx.asarray(np.arange(3.0)), v, ctx)

    def test_quadratic_linear_none_safe(self, ctx):
        v = self._v(ctx)
        Q = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        lin = sc.InnerProductFunctional(ctx.asarray(np.ones(3)), v, ctx)
        q_none = sc.LinOpQuadraticForm(Q, ctx=ctx)
        q_lin = sc.LinOpQuadraticForm(Q, linear=lin, ctx=ctx)
        assert q_none == sc.LinOpQuadraticForm(Q, ctx=ctx)
        assert q_none != q_lin           # None vs functional, no crash
        assert q_lin == sc.LinOpQuadraticForm(Q, linear=lin, ctx=ctx)

    def test_composed(self, ctx):
        v, w = self._v(ctx), sc.DenseVectorSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.ones((2, 3))), v, w, ctx)
        def fn(x):
            return ctx.ops.vdot(ctx.asarray(np.ones(2)), x)

        F = sc.MatrixFreeLinearFunctional(fn, w, ctx)
        comp = F.compose(A)
        assert type(comp).__name__ == "ComposedFunctional"
        assert comp == F.compose(A)


# ===========================================================================
# Cross-backend gate (parametrized)
# ===========================================================================
@pytest.mark.parametrize("family", ["jax", "torch"])
def test_same_backend_equal_cross_backend_not(family):
    np_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    other = _maybe_ctx(family, np.float64)
    if other is None:
        pytest.skip(f"{family} unavailable")
    a = sc.DenseVectorSpace((3,), other)
    b = sc.DenseVectorSpace((3,), other)
    assert a == b                                   # same backend
    assert a != sc.DenseVectorSpace((3,), np_ctx)   # cross-backend

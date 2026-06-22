"""Tests for the compact ``__repr__`` system (``spacecore._repr``).

Covers the shared helpers and the per-object representations for spaces,
linear operators, functionals, and inner-product geometries. The contract is:

* algebra leads, a terse ``backend=..., dtype=...`` tag follows;
* array contents are never dumped into the repr;
* nesting is bounded (operands collapse to an arrow form);
* ``repr`` never raises for any constructible object.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore import _repr


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


@pytest.fixture
def cctx():
    return sc.Context(sc.NumpyOps(), dtype=np.complex128)


# ===========================================================================
# Shared helpers
# ===========================================================================
class TestHelpers:
    def test_field_symbol(self):
        assert _repr.field_symbol("real") == "ℝ"
        assert _repr.field_symbol("complex") == "ℂ"
        assert _repr.field_symbol(None) == "?"
        assert _repr.field_symbol("weird") == "?"

    def test_format_dtype_short_name(self):
        assert _repr.format_dtype(np.dtype("float64")) == "float64"
        assert _repr.format_dtype(np.dtype("complex128")) == "complex128"
        assert _repr.format_dtype(None) == "None"

    def test_format_dtype_strips_backend_prefix(self):
        # Simulate a torch-style qualified name without importing torch.
        class FakeDtype:
            def __str__(self):
                return "torch.float32"

        assert _repr.format_dtype(FakeDtype()) == "float32"

    def test_shape_descriptor(self):
        assert _repr.shape_descriptor("real", ()) == "ℝ"
        assert _repr.shape_descriptor("real", (5,)) == "ℝ^5"
        assert _repr.shape_descriptor("complex", (2, 3)) == "ℂ^(2, 3)"

    def test_summarize_value_array(self):
        out = _repr.summarize_value(np.zeros((2, 3)))
        assert out == "<array shape=(2, 3), dtype=float64>"

    def test_summarize_value_scalar(self):
        assert _repr.summarize_value(np.float64(3.5)) == "3.5"

    def test_summarize_value_tuple_recurses(self):
        out = _repr.summarize_value((1, np.zeros((4,))))
        assert out == "(1, <array shape=(4,), dtype=float64>)"

    def test_summarize_value_singleton_tuple_keeps_comma(self):
        assert _repr.summarize_value((None,)) == "(None,)"

    def test_summarize_value_complex_0d_keeps_imaginary(self):
        # Regression: float() used to silently drop the imaginary part.
        assert _repr.summarize_value(np.complex128(2 + 3j)) == "2+3j"

    def test_summarize_value_python_complex(self):
        assert _repr.summarize_value(2 + 3j) == "2+3j"

    def test_scalar_format_python_float_matches_0d_array(self):
        assert _repr.summarize_value(2.0) == _repr.summarize_value(np.float64(2.0))
        assert _repr.summarize_value(2.0) == "2"

    def test_summarize_value_int_is_exact_not_collapsed(self):
        # ints (shapes, counts) must not go through .6g (would give 1e+06).
        assert _repr.summarize_value(1000000) == "1000000"

    def test_format_dtype_accepts_bare_type(self):
        assert _repr.format_dtype(np.float64) == "float64"

    def test_summarize_value_defensive_on_pathological_shape(self):
        class Bad:
            @property
            def shape(self):
                raise RuntimeError("boom")

        class NonIterableShape:
            shape = 5

        # Must not raise.
        assert isinstance(_repr.summarize_value(Bad()), str)
        assert isinstance(_repr.summarize_value(NonIterableShape()), str)

    def test_truncated_join(self):
        assert _repr.truncated_join(["a", "b"], " + ") == "a + b"
        out = _repr.truncated_join([str(i) for i in range(10)], " + ", limit=3)
        assert out == "0 + 1 + 2 + …(+7 more)"

    def test_summarize_value_defers_to_short_repr(self, ctx):
        space = sc.DenseVectorSpace((3,), ctx)
        # A space has .shape/.dtype but must NOT be summarized as an array.
        assert _repr.summarize_value(space) == space._short_repr()
        assert "array" not in _repr.summarize_value(space)


# ===========================================================================
# Spaces
# ===========================================================================
class TestSpaceRepr:
    def test_dense_coordinate(self, ctx):
        s = sc.DenseCoordinateSpace((2, 3), ctx)
        assert repr(s) == "DenseCoordinateSpace(ℝ^(2, 3), backend='numpy', dtype=float64)"

    def test_dense_vector_1d_drops_tuple(self, ctx):
        s = sc.DenseVectorSpace((5,), ctx)
        assert repr(s) == "DenseVectorSpace(ℝ^5, backend='numpy', dtype=float64)"

    def test_complex_field_symbol(self, cctx):
        s = sc.DenseVectorSpace((4,), cctx)
        assert "ℂ^4" in repr(s)
        assert "dtype=complex128" in repr(s)

    def test_scalar_space(self, ctx):
        s = sc.DenseCoordinateSpace((), ctx)
        assert repr(s) == "DenseCoordinateSpace(ℝ, backend='numpy', dtype=float64)"

    def test_hermitian(self, ctx):
        s = sc.HermitianSpace(4, ctx=ctx)
        assert "Herm(4)" in repr(s)

    def test_weighted_geometry_flagged(self, ctx):
        w = sc.WeightedInnerProduct(ctx.asarray(np.arange(1, 6, dtype=float)))
        s = sc.DenseVectorSpace((5,), ctx, geometry=w)
        assert "weighted" in repr(s)

    def test_euclidean_not_flagged_weighted(self, ctx):
        s = sc.DenseVectorSpace((5,), ctx)
        assert "weighted" not in repr(s)

    def test_stacked(self, ctx):
        s = sc.DenseVectorSpace((5,), ctx).stacked(8)
        assert "8×ℝ^5" in repr(s)

    def test_stacked_shows_public_class_name(self, ctx):
        # Regression: repr used to leak the private dispatch subclass name.
        s = sc.DenseVectorSpace((5,), ctx).stacked(8)
        assert repr(s).startswith("StackedSpace(")
        assert not type(s).__name__.startswith("StackedSpace")  # actual type is private

    def test_tree(self, ctx):
        t = sc.TreeSpace.from_template(
            [0, 0], [sc.DenseVectorSpace((3,), ctx), sc.DenseVectorSpace((2,), ctx)], ctx=ctx
        )
        assert "Tree(ℝ^3, ℝ^2)" in repr(t)

    def test_tree_shows_public_class_name(self, ctx):
        t = sc.TreeSpace.from_template(
            [0, 0], [sc.DenseVectorSpace((3,), ctx), sc.DenseVectorSpace((2,), ctx)], ctx=ctx
        )
        assert repr(t).startswith("TreeSpace(")

    def test_tree_wide_abbreviates(self, ctx):
        leaves = [sc.DenseVectorSpace((1,), ctx) for _ in range(7)]
        t = sc.TreeSpace.from_template(list(range(7)), leaves, ctx=ctx)
        assert "…(+3)" in repr(t)

    def test_weighted_marker_survives_nesting(self, ctx):
        # Regression: the 'weighted' flag used to disappear when nested.
        w = sc.WeightedInnerProduct(ctx.asarray(np.arange(1, 4, dtype=float)))
        vw = sc.DenseVectorSpace((3,), ctx, geometry=w)
        assert "[weighted]" in repr(vw)
        assert "[weighted]" in repr(vw.stacked(4))


# ===========================================================================
# Linear operators
# ===========================================================================
class TestLinOpRepr:
    def test_dense_arrow(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(5)), v, v, ctx)
        assert repr(A) == "DenseLinOp(ℝ^5 → ℝ^5, backend='numpy', dtype=float64)"

    def test_dense_does_not_dump_values(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        # Distinctive values that would be visible if the matrix were printed.
        A = sc.DenseLinOp(ctx.asarray(np.full((3, 3), 7.0)), v, v, ctx)
        assert "7" not in repr(A)

    def test_diagonal(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        D = sc.DiagonalLinOp(ctx.asarray(np.arange(1, 6, dtype=float)), v, ctx)
        assert repr(D) == "DiagonalLinOp(ℝ^5 → ℝ^5, backend='numpy', dtype=float64)"

    def test_sparse_shows_nnz(self, ctx):
        import scipy.sparse as sps

        v = sc.DenseVectorSpace((5,), ctx)
        Sp = sc.SparseLinOp(ctx.assparse(sps.eye(5)), v, v, ctx)
        assert "nnz=5" in repr(Sp)

    def test_identity_single_space(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        assert repr(sc.IdentityLinOp(v, ctx)) == "IdentityLinOp(ℝ^5, backend='numpy', dtype=float64)"

    def test_scaled(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(5)), v, v, ctx)
        assert "2 · DenseLinOp(ℝ^5 → ℝ^5)" in repr(2.0 * A)

    def test_scaled_complex_scalar_not_dumped(self, cctx):
        v = sc.DenseVectorSpace((3,), cctx)
        A = sc.DenseLinOp(cctx.asarray(np.eye(3)), v, v, cctx)
        r = repr((2 + 3j) * A)
        assert "2+3j ·" in r
        assert "array(" not in r

    def test_sum_wide_is_abbreviated(self, ctx):
        v = sc.DenseVectorSpace((2,), ctx)
        ops = [sc.DiagonalLinOp(ctx.asarray(np.ones(2)), v, ctx) for _ in range(10)]
        total = ops[0]
        for op in ops[1:]:
            total = total + op
        assert "more)" in repr(total)

    def test_sum(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(5)), v, v, ctx)
        D = sc.DiagonalLinOp(ctx.asarray(np.ones(5)), v, ctx)
        assert " + " in repr(A + D)

    def test_composed(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(5)), v, v, ctx)
        D = sc.DiagonalLinOp(ctx.asarray(np.ones(5)), v, ctx)
        assert " ∘ " in repr(A @ D)

    def test_adjoint(self, ctx):
        v = sc.DenseVectorSpace((5,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(5)), v, v, ctx)
        assert repr(A.H).endswith(", backend='numpy', dtype=float64)")
        assert ".H" in repr(A.H)

    def test_adjoint_shows_public_class_name(self, ctx):
        # Regression: repr used to leak the private _AdjointViewLinOp name.
        v = sc.DenseVectorSpace((5,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(5)), v, v, ctx)
        assert repr(A.H).startswith("AdjointLinOp(")

    def test_nesting_is_bounded(self, ctx):
        # Deeply nested algebra must not recurse past one operand level.
        v = sc.DenseVectorSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        expr = (2.0 * A) @ (A + A)
        r = repr(expr)
        # Operands appear in their bounded arrow form, not with their own operands.
        assert "ScaledLinOp(ℝ^3 → ℝ^3)" in r
        assert "SumLinOp(ℝ^3 → ℝ^3)" in r
        # The inner scalar/operands of those operands are NOT expanded.
        assert "2.0 ·" not in r

    def test_short_repr_has_no_backend_tag(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert A._short_repr() == "DenseLinOp(ℝ^3 → ℝ^3)"
        assert "backend" not in A._short_repr()


# ===========================================================================
# Functionals and geometries
# ===========================================================================
class TestFunctionalRepr:
    def test_inner_product_functional_arrow(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        f = sc.InnerProductFunctional(ctx.asarray(np.ones(3)), v, ctx)
        assert repr(f) == "InnerProductFunctional(ℝ^3 → ℝ, backend='numpy', dtype=float64)"

    def test_inner_product_functional_hides_representer(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        f = sc.InnerProductFunctional(ctx.asarray(np.full(3, 9.0)), v, ctx)
        assert "9" not in repr(f)

    def test_quadratic_form_arrow(self, ctx):
        v = sc.DenseVectorSpace((3,), ctx)
        Q = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        qf = sc.LinOpQuadraticForm(Q, ctx=ctx)
        assert repr(qf) == "LinOpQuadraticForm(ℝ^3 → ℝ, backend='numpy', dtype=float64)"

    def test_composed_functional_shows_operands(self, ctx):
        # A matrix-free linear functional ∘ a non-square operator does not
        # simplify, so this exercises ComposedFunctional._repr_body directly.
        v = sc.DenseVectorSpace((3,), ctx)
        w = sc.DenseVectorSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.ones((2, 3))), v, w, ctx)
        F = sc.MatrixFreeLinearFunctional(
            lambda x: ctx.ops.vdot(ctx.asarray(np.ones(2)), x), w, ctx
        )
        comp = F.compose(A)
        assert type(comp).__name__ == "ComposedFunctional"
        r = repr(comp)
        assert "MatrixFreeLinearFunctional(ℝ^2 → ℝ) ∘ DenseLinOp(ℝ^3 → ℝ^2)" in r

    def test_euclidean_geometry_repr(self):
        assert repr(sc.EuclideanInnerProduct()) == "EuclideanInnerProduct()"

    def test_weighted_geometry_repr(self, ctx):
        w = sc.WeightedInnerProduct(ctx.asarray(np.arange(1, 4, dtype=float)))
        assert repr(w) == "WeightedInnerProduct(weights=<array shape=(3,), dtype=float64>)"


# ===========================================================================
# Dataclasses that are not ContextBound (must summarize, not dump, arrays)
# ===========================================================================
class TestDataclassRepr:
    def _tree(self, ctx):
        return sc.TreeSpace(
            (0, 0), (sc.DenseVectorSpace((3,), ctx), sc.DenseVectorSpace((2,), ctx)), ctx=ctx
        )

    def test_tree_element_summarizes_leaves(self, ctx):
        # Regression: dataclass auto-repr dumped full leaf arrays.
        t = self._tree(ctx)
        el = sc.TreeElement(t, (ctx.asarray(np.arange(3.0)), ctx.asarray(np.arange(2.0))))
        r = repr(el)
        assert r.startswith("TreeElement(")
        assert "<array shape=(3,), dtype=float64>" in r
        assert "array(" not in r  # no raw numpy dump
        assert "0., 1., 2." not in r

    def test_tree_element_large_leaves_stay_short(self, ctx):
        t = sc.TreeSpace((0,), (sc.DenseVectorSpace((1000,), ctx),), ctx=ctx)
        el = sc.TreeElement(t, (ctx.asarray(np.arange(1000.0)),))
        assert len(repr(el)) < 200

    def test_tree_spectral_decomposition_summarizes(self, ctx):
        from spacecore.space.concrete._tree_space import TreeSpectralDecomposition

        sd = TreeSpectralDecomposition((ctx.asarray(np.ones(3)),), (None,))
        r = repr(sd)
        assert r.startswith("TreeSpectralDecomposition(")
        assert "<array shape=(3,), dtype=float64>" in r
        assert "array(" not in r


# ===========================================================================
# Cross-cutting invariants
# ===========================================================================
class TestInvariants:
    def _all_objects(self, ctx):
        import scipy.sparse as sps

        v = sc.DenseVectorSpace((4,), ctx)
        w = sc.DenseVectorSpace((4,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(4)), v, w, ctx)
        D = sc.DiagonalLinOp(ctx.asarray(np.ones(4)), v, ctx)
        Sp = sc.SparseLinOp(ctx.assparse(sps.eye(4)), v, v, ctx)
        f = sc.InnerProductFunctional(ctx.asarray(np.ones(4)), v, ctx)
        return [
            v,
            sc.DenseCoordinateSpace((2, 2), ctx),
            sc.HermitianSpace(3, ctx=ctx),
            v.stacked(2),
            A,
            D,
            Sp,
            2.0 * A,
            A + D,
            A @ D,
            A.H,
            sc.IdentityLinOp(v, ctx),
            f,
            f.compose(A),
        ]

    def test_repr_never_raises(self, ctx):
        for obj in self._all_objects(ctx):
            assert isinstance(repr(obj), str)

    def test_repr_starts_with_repr_class_name(self, ctx):
        for obj in self._all_objects(ctx):
            assert repr(obj).startswith(obj._repr_class_name() + "(")

    def test_repr_carries_backend_tag(self, ctx):
        for obj in self._all_objects(ctx):
            assert "backend='numpy'" in repr(obj)
            assert "dtype=float64" in repr(obj)

    def test_no_check_level_leak(self, ctx):
        # check_level is policy, not identity: it must never appear in a repr.
        for obj in self._all_objects(ctx):
            assert "check_level" not in repr(obj)


# ===========================================================================
# Cross-backend consistency (Goal #5)
# ===========================================================================
def _maybe_ctx(family, dtype):
    from spacecore._contextual._state import normalize_context

    try:
        return normalize_context(family, dtype=dtype)
    except Exception:
        return None


@pytest.mark.parametrize("family", ["jax", "torch"])
class TestCrossBackend:
    def test_space_and_linop_repr(self, family):
        ctx = _maybe_ctx(family, np.float32)
        if ctx is None:
            pytest.skip(f"{family} backend unavailable")
        v = sc.DenseVectorSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray(np.eye(3)), v, v, ctx)
        assert repr(v) == f"DenseVectorSpace(ℝ^3, backend='{family}', dtype=float32)"
        assert repr(A) == f"DenseLinOp(ℝ^3 → ℝ^3, backend='{family}', dtype=float32)"

    def test_dtype_is_unqualified(self, family):
        ctx = _maybe_ctx(family, np.float32)
        if ctx is None:
            pytest.skip(f"{family} backend unavailable")
        v = sc.DenseVectorSpace((3,), ctx)
        # No backend-qualified dtype like 'torch.float32' in the tag.
        assert "dtype=float32" in repr(v)
        assert "torch." not in repr(v)

"""Tests for :class:`spacecore.StackedLinOp`.

Checklist item 17:

* ``dom = single`` (shared); ``cod = TreeSpace`` (per-component).
* ``apply(x)`` returns a tree element whose leaf ``i`` is ``parts[i].apply(x)``.
* ``rapply(y)`` sums component adjoints back into the shared domain.
* Batched ``vapply`` / ``rvapply`` preserve the same shape semantics.
* ``from_operators`` constructs the tuple-style cod TreeSpace.
* The legacy 4-arg ``(dom, cod, parts, ctx)`` constructor accepts a structured
  (NamedTuple-templated) codomain and round-trips it through the adjoint.
* Weighted/Euclidean adjoint dot-test ``<Ax, y>_cod == <x, A^H y>_dom``.
* ``rvapply`` accumulates via the shared-domain ``add_batch`` helper.
* Wrong-tuple-layout batched inputs are rejected.
* Constructor rejects empty parts (TreeLinOp base contract).
* JAX pytree round-trip preserves the layout; jit-compiled apply/rapply.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy


class _State(NamedTuple):
    a: object
    b: object


def _dense(ctx, matrix, dom, cod):
    return sc.DenseLinOp(ctx.asarray(matrix), dom, cod, ctx)


def _weighted_space(weights, ctx):
    return sc.DenseCoordinateSpace(
        tuple(np.asarray(weights).shape),
        ctx,
        geometry=sc.WeightedInnerProduct(ctx.asarray(weights)),
    )


def _stacked_parts(ctx):
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y1 = sc.DenseCoordinateSpace((2,), ctx)
    Y2 = sc.DenseCoordinateSpace((1,), ctx)
    A1 = _dense(ctx, [[1.0, 2.0], [3.0, 4.0]], X, Y1)
    A2 = _dense(ctx, [[5.0, 6.0]], X, Y2)
    return A1, A2


# ===========================================================================
# Construction validation
# ===========================================================================
class TestConstruction:
    def test_from_empty_operators_raises(self):
        with pytest.raises(Exception):
            sc.StackedLinOp.from_operators(())

    def test_from_operators_builds_tree_codomain(self, numpy_ctx):
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        assert op.domain == A1.domain
        assert isinstance(op.codomain, sc.TreeSpace)
        assert op.codomain.arity == 2


# ===========================================================================
# apply: replicates component actions on the shared input
# ===========================================================================
class TestApply:
    def test_apply_returns_tree_of_component_outputs(self, numpy_ctx):
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        y = op.apply(numpy_ctx.asarray([10.0, 20.0]))
        np.testing.assert_allclose(to_numpy(y[0]), [50.0, 110.0])
        np.testing.assert_allclose(to_numpy(y[1]), [170.0])


# ===========================================================================
# rapply: sums component adjoints into the shared domain
# ===========================================================================
class TestRapply:
    def test_rapply_sums_component_adjoints(self, numpy_ctx):
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        x = op.rapply((numpy_ctx.asarray([2.0, -1.0]), numpy_ctx.asarray([3.0])))
        # A1^T [2, -1] + A2^T [3] = [2-3, 4-4] + [15, 18] = [-1, 0] + [15, 18] = [14, 18]
        np.testing.assert_allclose(to_numpy(x), [14.0, 18.0])


# ===========================================================================
# Batched vapply / rvapply
# ===========================================================================
class TestBatched:
    def test_vapply_matches_per_element_loop(self, numpy_ctx):
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        xs = numpy_ctx.asarray(np.asarray([[10.0, 20.0], [1.0, -1.0]]))
        ys = op.vapply(xs)
        # Compare per-row against op.apply.
        for i in range(2):
            ref = op.apply(xs[i])
            np.testing.assert_allclose(to_numpy(ys[0])[i], to_numpy(ref[0]))
            np.testing.assert_allclose(to_numpy(ys[1])[i], to_numpy(ref[1]))

    def test_rvapply_uses_shared_domain_add_batch(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py
        # (test_tree_linops_use_space_add_batch_for_accumulation).
        class CountingVectorSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx, counter):
                self.counter = counter
                super().__init__(shape, ctx)

            def add_batch(self, x, y):
                self.counter["calls"] += 1
                return super().add_batch(x, y)

            def _convert(self, new_ctx):
                return CountingVectorSpace(self.shape, new_ctx, self.counter)

        counter = {"calls": 0}
        shared = CountingVectorSpace((2,), numpy_ctx, counter)
        cod1 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        cod2 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        A1 = _dense(numpy_ctx, [[1.0, 2.0]], shared, cod1)
        A2 = _dense(numpy_ctx, [[3.0, -1.0]], shared, cod2)
        op = sc.StackedLinOp.from_operators((A1, A2))
        ys = (numpy_ctx.asarray([[1.0], [2.0]]), numpy_ctx.asarray([[3.0], [4.0]]))

        op.rvapply(ys)
        assert counter["calls"] == 1

    def test_wrong_tuple_layout_is_rejected(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py
        # (test_tree_linop_batch_checks_reject_wrong_tuple_layout), adapted to
        # the stacked codomain product structure.
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        with pytest.raises(ValueError, match="structure"):
            op.rvapply((numpy_ctx.asarray([[2.0, -1.0], [1.0, 0.5]]),))
        with pytest.raises(ValueError, match="trailing shape"):
            op.rvapply(
                (
                    numpy_ctx.asarray([[2.0, -1.0], [1.0, 0.5]]),
                    numpy_ctx.asarray([[3.0, 4.0], [5.0, 6.0]]),
                )
            )


# ===========================================================================
# Adjoint identity (weighted / Euclidean dot-test)
# ===========================================================================
class TestAdjointIdentity:
    def test_euclidean_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py adjoint flow.
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        x = numpy_ctx.asarray([0.25, -1.5])
        y = (numpy_ctx.asarray([2.0, -0.5]), numpy_ctx.asarray([1.25]))
        lhs = op.codomain.inner(op.apply(x), y)
        rhs = op.domain.inner(x, op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    def test_weighted_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_adjoint_identity.py
        # (test_weighted_stacked_linop_adjoint_identity) and
        # tests/linops/test_tree_linop_batching.py
        # (test_stacked_adjoint_identity_and_batched_rvapply_with_weighted_space).
        domain = _weighted_space([2.0, 5.0], numpy_ctx)
        cod0 = _weighted_space([3.0, 7.0, 11.0], numpy_ctx)
        cod1 = _weighted_space([13.0], numpy_ctx)
        A0 = _dense(numpy_ctx, [[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]], domain, cod0)
        A1 = _dense(numpy_ctx, [[2.0, -0.25]], domain, cod1)
        op = sc.StackedLinOp.from_operators((A0, A1))
        x = numpy_ctx.asarray([0.25, -1.5])
        y = (numpy_ctx.asarray([2.0, -0.5, 1.25]), numpy_ctx.asarray([-0.75]))
        lhs = op.codomain.inner(op.apply(x), y)
        rhs = op.domain.inner(x, op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    def test_weighted_batched_rvapply_matches_loop(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py
        # (test_stacked_adjoint_identity_and_batched_rvapply_with_weighted_space).
        domain = _weighted_space([2.0, 5.0], numpy_ctx)
        cod1 = _weighted_space([3.0, 7.0], numpy_ctx)
        cod2 = _weighted_space([11.0], numpy_ctx)
        A1 = _dense(numpy_ctx, [[1.0, 2.0], [3.0, -1.0]], domain, cod1)
        A2 = _dense(numpy_ctx, [[0.5, 4.0]], domain, cod2)
        op = sc.StackedLinOp.from_operators((A1, A2))
        ys = (
            numpy_ctx.asarray([[0.5, 3.0], [1.0, -2.0]]),
            numpy_ctx.asarray([[-1.0], [4.0]]),
        )
        rows = tuple(
            op.rapply((ys[0][i], ys[1][i])) for i in range(2)
        )
        expected = np.stack([to_numpy(r) for r in rows], axis=0)
        np.testing.assert_allclose(to_numpy(op.rvapply(ys)), expected)


# ===========================================================================
# Structured (NamedTuple-templated) tree codomain via the 4-arg constructor
# ===========================================================================
class TestStructuredTree:
    def test_structured_codomain_round_trips_through_adjoint(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py
        # (test_stacked_linop_returns_structured_codomain_and_accepts_it_for_adjoint).
        A1, A2 = _stacked_parts(numpy_ctx)
        cod = sc.TreeSpace.from_template(
            _State(numpy_ctx.asarray([0.0, 0.0]), numpy_ctx.asarray([0.0])),
            (A1.codomain, A2.codomain),
            ctx=numpy_ctx,
        )
        op = sc.StackedLinOp(A1.domain, cod, (A1, A2), numpy_ctx)

        y = op.apply(numpy_ctx.asarray([10.0, 20.0]))
        assert isinstance(y, _State)
        np.testing.assert_allclose(to_numpy(y.a), [50.0, 110.0])
        np.testing.assert_allclose(to_numpy(y.b), [170.0])

        x = op.rapply(_State(numpy_ctx.asarray([2.0, -1.0]), numpy_ctx.asarray([3.0])))
        np.testing.assert_allclose(to_numpy(x), [14.0, 18.0])
        np.testing.assert_allclose(to_numpy(op.H.apply(y)), [1230.0, 1560.0])

    def test_structured_batched_paths_preserve_namedtuple(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py
        # (test_tree_linop_batch_paths_use_tree_of_batched_leaves, stacked part).
        A1, A2 = _stacked_parts(numpy_ctx)
        cod = sc.TreeSpace.from_template(
            _State(numpy_ctx.asarray([0.0, 0.0]), numpy_ctx.asarray([0.0])),
            (A1.codomain, A2.codomain),
            ctx=numpy_ctx,
        )
        op = sc.StackedLinOp(A1.domain, cod, (A1, A2), numpy_ctx)

        ys = op.vapply(numpy_ctx.asarray([[10.0, 20.0], [1.0, 2.0]]))
        assert isinstance(ys, _State)
        np.testing.assert_allclose(to_numpy(ys.a), [[50.0, 110.0], [5.0, 11.0]])
        np.testing.assert_allclose(to_numpy(ys.b), [[170.0], [17.0]])

        xr = op.rvapply(
            _State(
                numpy_ctx.asarray([[2.0, -1.0], [1.0, 2.0]]),
                numpy_ctx.asarray([[3.0], [-1.0]]),
            )
        )
        np.testing.assert_allclose(to_numpy(xr), [[14.0, 18.0], [2.0, 4.0]])


# ===========================================================================
# JAX pytree round-trip + jit
# ===========================================================================
class TestPytree:
    def test_round_trip(self, numpy_ctx):
        A1, A2 = _stacked_parts(numpy_ctx)
        op = sc.StackedLinOp.from_operators((A1, A2))
        children, aux = op.tree_flatten()
        rebuilt = sc.StackedLinOp.tree_unflatten(aux, children)
        assert rebuilt == op


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJit:
    def test_jit_apply_and_rapply(self):
        # Folded from tests/linops/test_linop_jit.py (test_product_linops_jit_compile).
        import jax

        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        X = sc.DenseCoordinateSpace((2,), ctx)
        Y1 = sc.DenseCoordinateSpace((2,), ctx)
        Y2 = sc.DenseCoordinateSpace((1,), ctx)
        A1 = _dense(ctx, [[1.0, 2.0], [3.0, 4.0]], X, Y1)
        A2 = _dense(ctx, [[5.0, 6.0]], X, Y2)
        op = sc.StackedLinOp.from_operators((A1, A2))
        x = ctx.asarray([7.0, 8.0])

        apply_jit = jax.jit(lambda Aop, z: Aop.apply(z))
        rapply_jit = jax.jit(lambda Aop, a, b: Aop.rapply((a, b)))
        y = apply_jit(op, x)
        xr = rapply_jit(op, ctx.asarray([1.0, -1.0]), ctx.asarray([2.0]))

        np.testing.assert_allclose(to_numpy(y[0]), [23.0, 53.0])
        np.testing.assert_allclose(to_numpy(y[1]), [83.0])
        np.testing.assert_allclose(to_numpy(xr), [8.0, 10.0])

"""Tests for :class:`spacecore.SumToSingleLinOp`.

Checklist item 18:

* ``dom = TreeSpace`` (per-component); ``cod = single`` (shared).
* ``apply(x)`` sums component actions: ``Σ_i parts[i].apply(x[i])``.
* ``rapply(y)`` splits adjoint across the tree leaves.
* Batched ``vapply`` / ``rvapply`` preserve the same shape semantics.
* The legacy 4-arg ``(dom, cod, parts, ctx)`` constructor accepts a structured
  (NamedTuple-templated) domain and round-trips it through the adjoint.
* Weighted/Euclidean adjoint dot-test ``<Ax, y>_cod == <x, A^H y>_dom``.
* ``vapply`` accumulates via the shared-codomain ``add_batch`` helper.
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


def _sum_parts(ctx):
    X1 = sc.DenseCoordinateSpace((2,), ctx)
    X2 = sc.DenseCoordinateSpace((3,), ctx)
    Y = sc.DenseCoordinateSpace((2,), ctx)
    A1 = _dense(ctx, [[1.0, 2.0], [3.0, 4.0]], X1, Y)
    A2 = _dense(ctx, [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], X2, Y)
    return A1, A2


# ===========================================================================
# Construction validation
# ===========================================================================
class TestConstruction:
    def test_from_empty_operators_raises(self):
        with pytest.raises(Exception):
            sc.SumToSingleLinOp.from_operators(())

    def test_from_operators_builds_tree_domain(self, numpy_ctx):
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        assert isinstance(op.domain, sc.TreeSpace)
        assert op.domain.arity == 2
        assert op.codomain == A1.codomain


# ===========================================================================
# apply: sums per-component actions
# ===========================================================================
class TestApply:
    def test_apply_sums_components(self, numpy_ctx):
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        y = op.apply(
            (numpy_ctx.asarray([10.0, 20.0]), numpy_ctx.asarray([1.0, 2.0, 3.0]))
        )
        # A1 [10, 20] + A2 [1, 2, 3] = [50, 110] + [38, 56] = [88, 166]
        np.testing.assert_allclose(to_numpy(y), [88.0, 166.0])


# ===========================================================================
# rapply: splits adjoint across the tree
# ===========================================================================
class TestRapply:
    def test_rapply_splits_to_tree_components(self, numpy_ctx):
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        x = op.rapply(numpy_ctx.asarray([2.0, -1.0]))
        # A1^T [2, -1] = [2-3, 4-4] = [-1, 0]; A2^T [2, -1] = [10-8, 12-9, 14-10] = [2, 3, 4]
        np.testing.assert_allclose(to_numpy(x[0]), [-1.0, 0.0])
        np.testing.assert_allclose(to_numpy(x[1]), [2.0, 3.0, 4.0])


# ===========================================================================
# Batched vapply / rvapply
# ===========================================================================
class TestBatched:
    def test_vapply_matches_per_element_loop(self, numpy_ctx):
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        xs = (
            numpy_ctx.asarray([[10.0, 20.0], [1.0, -1.0]]),
            numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        ys = op.vapply(xs)
        for i in range(2):
            ref = op.apply((xs[0][i], xs[1][i]))
            np.testing.assert_allclose(to_numpy(ys)[i], to_numpy(ref))

    def test_vapply_uses_shared_codomain_add_batch(self, numpy_ctx):
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
        dom1 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        dom2 = sc.DenseCoordinateSpace((1,), numpy_ctx)
        B1 = _dense(numpy_ctx, [[1.0], [2.0]], dom1, shared)
        B2 = _dense(numpy_ctx, [[3.0], [-1.0]], dom2, shared)
        op = sc.SumToSingleLinOp.from_operators((B1, B2))
        xs = (numpy_ctx.asarray([[1.0], [2.0]]), numpy_ctx.asarray([[3.0], [4.0]]))

        op.vapply(xs)
        assert counter["calls"] == 1

    def test_wrong_tuple_layout_is_rejected(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py
        # (test_tree_linop_batch_checks_reject_wrong_tuple_layout), adapted to
        # the sum-to-single domain product structure.
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        with pytest.raises(ValueError, match="structure"):
            op.vapply((numpy_ctx.asarray([[10.0, 20.0], [1.0, -1.0]]),))
        with pytest.raises(ValueError, match="trailing shape"):
            op.vapply(
                (
                    numpy_ctx.asarray([[10.0, 20.0], [1.0, -1.0]]),
                    numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
                )
            )


# ===========================================================================
# Adjoint identity (weighted / Euclidean dot-test)
# ===========================================================================
class TestAdjointIdentity:
    def test_euclidean_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py adjoint flow.
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        x = (numpy_ctx.asarray([0.25, -1.5]), numpy_ctx.asarray([2.0, -0.5, 1.25]))
        y = numpy_ctx.asarray([1.0, -2.0])
        lhs = op.codomain.inner(op.apply(x), y)
        rhs = op.domain.inner(x, op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    def test_weighted_adjoint_identity(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py
        # (test_sum_to_single_adjoint_identity_and_batched_vapply_with_weighted_space).
        dom1 = _weighted_space([2.0, 5.0], numpy_ctx)
        dom2 = _weighted_space([3.0], numpy_ctx)
        cod = _weighted_space([7.0, 11.0], numpy_ctx)
        A1 = _dense(numpy_ctx, [[1.0, 2.0], [3.0, -1.0]], dom1, cod)
        A2 = _dense(numpy_ctx, [[0.5], [4.0]], dom2, cod)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        x = (numpy_ctx.asarray([1.0, -2.0]), numpy_ctx.asarray([3.0]))
        y = numpy_ctx.asarray([0.5, 3.0])
        lhs = op.codomain.inner(op.apply(x), y)
        rhs = op.domain.inner(x, op.rapply(y))
        np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    def test_weighted_batched_vapply_matches_loop(self, numpy_ctx):
        # Folded from tests/linops/test_tree_linop_batching.py
        # (test_sum_to_single_adjoint_identity_and_batched_vapply_with_weighted_space).
        dom1 = _weighted_space([2.0, 5.0], numpy_ctx)
        dom2 = _weighted_space([3.0], numpy_ctx)
        cod = _weighted_space([7.0, 11.0], numpy_ctx)
        A1 = _dense(numpy_ctx, [[1.0, 2.0], [3.0, -1.0]], dom1, cod)
        A2 = _dense(numpy_ctx, [[0.5], [4.0]], dom2, cod)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        xs = (
            numpy_ctx.asarray([[1.0, -2.0], [0.5, 4.0]]),
            numpy_ctx.asarray([[3.0], [-1.0]]),
        )
        rows = tuple(op.apply((xs[0][i], xs[1][i])) for i in range(2))
        expected = np.stack([to_numpy(r) for r in rows], axis=0)
        np.testing.assert_allclose(to_numpy(op.vapply(xs)), expected)


# ===========================================================================
# Structured (NamedTuple-templated) tree domain via the 4-arg constructor
# ===========================================================================
class TestStructuredTree:
    def test_structured_domain_round_trips_through_adjoint(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py
        # (test_sum_to_single_accepts_structured_domain_and_returns_it_for_adjoint).
        A1, A2 = _sum_parts(numpy_ctx)
        dom = sc.TreeSpace.from_template(
            _State(numpy_ctx.asarray([0.0, 0.0]), numpy_ctx.asarray([0.0, 0.0, 0.0])),
            (A1.domain, A2.domain),
            ctx=numpy_ctx,
        )
        op = sc.SumToSingleLinOp(dom, A1.codomain, (A1, A2), numpy_ctx)

        x = _State(numpy_ctx.asarray([10.0, 20.0]), numpy_ctx.asarray([1.0, 2.0, 3.0]))
        y = op.apply(x)
        np.testing.assert_allclose(to_numpy(y), [88.0, 166.0])

        xr = op.rapply(numpy_ctx.asarray([2.0, -1.0]))
        assert isinstance(xr, _State)
        np.testing.assert_allclose(to_numpy(xr.a), [-1.0, 0.0])
        np.testing.assert_allclose(to_numpy(xr.b), [2.0, 3.0, 4.0])

        adj = op.H.apply(y)
        assert isinstance(adj, _State)
        np.testing.assert_allclose(to_numpy(adj.a), [586.0, 840.0])
        np.testing.assert_allclose(to_numpy(adj.b), [1768.0, 2022.0, 2276.0])

    def test_structured_batched_paths_preserve_namedtuple(self, numpy_ctx):
        # Folded from tests/linops/test_tree_structure.py
        # (test_tree_linop_batch_paths_use_tree_of_batched_leaves, sum part).
        A1, A2 = _sum_parts(numpy_ctx)
        dom = sc.TreeSpace.from_template(
            _State(numpy_ctx.asarray([0.0, 0.0]), numpy_ctx.asarray([0.0, 0.0, 0.0])),
            (A1.domain, A2.domain),
            ctx=numpy_ctx,
        )
        op = sc.SumToSingleLinOp(dom, A1.codomain, (A1, A2), numpy_ctx)

        xb = _State(
            numpy_ctx.asarray([[10.0, 20.0], [1.0, 2.0]]),
            numpy_ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        np.testing.assert_allclose(
            to_numpy(op.vapply(xb)), [[88.0, 166.0], [97.0, 148.0]]
        )

        xr = op.rvapply(numpy_ctx.asarray([[2.0, -1.0], [1.0, 2.0]]))
        assert isinstance(xr, _State)
        np.testing.assert_allclose(to_numpy(xr.a), [[-1.0, 0.0], [7.0, 10.0]])
        np.testing.assert_allclose(to_numpy(xr.b), [[2.0, 3.0, 4.0], [21.0, 24.0, 27.0]])


# ===========================================================================
# JAX pytree round-trip + jit
# ===========================================================================
class TestPytree:
    def test_round_trip(self, numpy_ctx):
        A1, A2 = _sum_parts(numpy_ctx)
        op = sc.SumToSingleLinOp.from_operators((A1, A2))
        children, aux = op.tree_flatten()
        rebuilt = sc.SumToSingleLinOp.tree_unflatten(aux, children)
        assert rebuilt == op


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJit:
    def test_jit_apply_and_rapply(self):
        # Folded from tests/linops/test_linop_jit.py (test_product_linops_jit_compile).
        import jax

        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        X = sc.DenseCoordinateSpace((2,), ctx)
        Y = sc.DenseCoordinateSpace((2,), ctx)
        A1 = _dense(ctx, [[1.0, 2.0], [3.0, 4.0]], X, Y)
        op = sc.SumToSingleLinOp.from_operators((A1, A1))
        x = ctx.asarray([7.0, 8.0])

        sum_apply = jax.jit(lambda Aop, a, b: Aop.apply((a, b)))
        sum_rapply = jax.jit(lambda Aop, z: Aop.rapply(z))

        np.testing.assert_allclose(to_numpy(sum_apply(op, x, x)), [46.0, 106.0])
        yr = sum_rapply(op, ctx.asarray([1.0, -1.0]))
        np.testing.assert_allclose(to_numpy(yr[0]), [-2.0, -2.0])
        np.testing.assert_allclose(to_numpy(yr[1]), [-2.0, -2.0])

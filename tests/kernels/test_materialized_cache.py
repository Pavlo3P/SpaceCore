"""ADR-022: build-time materialized-form cache for the batched folds.

Pins the contract from ``docs/dev/adr/022_caching.md``:

* the cached value is the *input-independent* stacked block-matrix array
  ``ops.stack([matrix(p) for p in parts])`` — one slot per matrix accessor
  (``_A2`` apply, ``_A2H`` rapply, ``_A2T`` vapply, ``_A2H.T`` rvapply);
* it is built once on first optimized use and reused across applies;
* caching never changes results (cached == generic), in ``on`` and ``verify``;
* it is excluded from operator identity (``__eq__``) and is dropped/rebuilt on a
  pytree round-trip;
* a fold that contains a matrix-free block is never cache-materialized;
* with dispatch ``off`` (the default) the cache is never populated.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import has_jax, to_numpy

from spacecore.kernels import dispatch_mode
from spacecore.kernels.specs._batched import CachedStackParts, stacked_block_matrices
from spacecore.kernels.specs.block_batched import (
    _A2,
    _A2H,
    _A2H_T,
    _A2T,
    block_batched_optimized,
    block_batched_rapply_optimized,
    block_batched_rvapply_optimized,
    block_batched_vapply_optimized,
)
from spacecore.kernels.specs.stacked_batched import _A2 as STACKED_A2
from spacecore.kernels.specs.stacked_batched import _A2H as SUM_TO_SINGLE_A2H
from spacecore.linop.tree._block import BlockDiagonalLinOp
from spacecore.linop.tree._from_single import StackedLinOp
from spacecore.linop.tree._to_single import SumToSingleLinOp


def _blocks(ctx, n, k, seed=0):
    """Return ``(space, mats, blocks)`` for ``k`` uniform ``n x n`` dense blocks."""
    rng = np.random.default_rng(seed)
    space = sc.DenseCoordinateSpace((n,), ctx)
    mats = [rng.standard_normal((n, n)) for _ in range(k)]
    blocks = tuple(sc.DenseLinOp(ctx.asarray(m), space, space, ctx) for m in mats)
    return space, mats, blocks


def _vec_components(ctx, n, k, seed=1):
    rng = np.random.default_rng(seed)
    return tuple(ctx.asarray(rng.standard_normal(n)) for _ in range(k))


# ---------------------------------------------------------------------------
# Wiring: the fold operators carry a (lazy, empty) cache.
# ---------------------------------------------------------------------------
def test_fold_operators_wrap_parts_with_empty_cache(numpy_ctx):
    _, _, blocks = _blocks(numpy_ctx, 4, 3)
    for op in (
        BlockDiagonalLinOp.from_operators(blocks),
        StackedLinOp.from_operators(blocks),
        SumToSingleLinOp.from_operators(blocks),
    ):
        assert isinstance(op.parts, CachedStackParts)
        assert op.parts._stack_cache == {}  # nothing materialized until first use


def test_dispatch_off_leaves_cache_empty(numpy_ctx):
    """The default (off) path never touches the cache."""
    _, _, blocks = _blocks(numpy_ctx, 4, 3)
    op = BlockDiagonalLinOp.from_operators(blocks)
    x = op.domain._from_components(_vec_components(numpy_ctx, 4, 3))
    op.apply(x)  # default mode == "off"
    assert op.parts._stack_cache == {}


# ---------------------------------------------------------------------------
# What is cached: the stacked block-matrix array, one slot per accessor.
# ---------------------------------------------------------------------------
def test_one_slot_per_accessor_equals_fresh_stack(numpy_ctx):
    _, _, blocks = _blocks(numpy_ctx, 4, 3)
    parts = CachedStackParts(blocks)
    x_parts = _vec_components(numpy_ctx, 4, 3)
    xb_parts = tuple(
        numpy_ctx.asarray(np.random.default_rng(s).standard_normal((3, 4)))
        for s in range(3)
    )

    block_batched_optimized(parts, x_parts)  # _A2 (apply)
    block_batched_rapply_optimized(parts, x_parts)  # _A2H (rapply)
    block_batched_vapply_optimized(parts, xb_parts)  # _A2T (vapply)
    block_batched_rvapply_optimized(parts, xb_parts)  # _A2H.T (rvapply)

    assert set(parts._stack_cache) == {_A2, _A2H, _A2T, _A2H_T}
    for accessor in (_A2, _A2H, _A2T, _A2H_T):
        fresh = numpy_ctx.ops.stack([accessor(b) for b in blocks])
        np.testing.assert_array_equal(
            to_numpy(parts._stack_cache[accessor]), to_numpy(fresh)
        )


def test_stack_reused_across_applies_but_not_for_plain_tuple(numpy_ctx):
    _, _, blocks = _blocks(numpy_ctx, 4, 3)

    cached = CachedStackParts(blocks)
    first = stacked_block_matrices(cached, _A2)
    second = stacked_block_matrices(cached, _A2)
    assert first is second  # memoized: identical object reused

    plain = tuple(blocks)  # no cache -> rebuilt every call (status quo)
    assert stacked_block_matrices(plain, _A2) is not stacked_block_matrices(plain, _A2)


# ---------------------------------------------------------------------------
# Caching never changes results.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["on", "verify"])
def test_cache_preserves_results_across_repeated_applies(numpy_ctx, mode):
    _, mats, blocks = _blocks(numpy_ctx, 4, 3)
    op = BlockDiagonalLinOp.from_operators(blocks)
    comps = _vec_components(numpy_ctx, 4, 3)
    x = op.domain._from_components(comps)
    reference = [m @ to_numpy(c) for m, c in zip(mats, comps)]

    with dispatch_mode(mode):
        first = op.codomain._components(op.apply(x))
        second = op.codomain._components(op.apply(x))  # second apply hits the cache

    assert len(op.parts._stack_cache) == 1  # _A2 only
    for a, b, r in zip(first, second, reference):
        np.testing.assert_array_equal(to_numpy(a), to_numpy(b))
        np.testing.assert_allclose(to_numpy(a), r, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# The cache is excluded from the operator's mathematical identity.
# ---------------------------------------------------------------------------
def test_cache_excluded_from_equality(numpy_ctx):
    _, _, blocks = _blocks(numpy_ctx, 4, 3)
    populated = BlockDiagonalLinOp.from_operators(blocks)
    empty = BlockDiagonalLinOp.from_operators(blocks)

    x = populated.domain._from_components(_vec_components(numpy_ctx, 4, 3))
    with dispatch_mode("on"):
        populated.apply(x)

    assert len(populated.parts._stack_cache) == 1
    assert empty.parts._stack_cache == {}
    assert populated == empty  # a populated cache must not change identity
    assert empty == populated


# ---------------------------------------------------------------------------
# Pytree round-trip drops the cache and rebuilds it lazily; still correct.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not has_jax(), reason="pytree round-trip needs jax")
def test_pytree_roundtrip_rebuilds_empty_cache(numpy_ctx):
    import jax

    _, mats, blocks = _blocks(numpy_ctx, 4, 3)
    op = BlockDiagonalLinOp.from_operators(blocks)
    comps = _vec_components(numpy_ctx, 4, 3)
    x = op.domain._from_components(comps)

    with dispatch_mode("on"):
        op.apply(x)
    assert len(op.parts._stack_cache) == 1

    leaves, treedef = jax.tree_util.tree_flatten(op)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt.parts, CachedStackParts)
    assert rebuilt.parts._stack_cache == {}  # dropped on flatten, rebuilt empty

    with dispatch_mode("on"):
        out = rebuilt.codomain._components(rebuilt.apply(x))
    for a, m, c in zip(out, mats, comps):
        np.testing.assert_allclose(to_numpy(a), m @ to_numpy(c), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# A matrix-free operand is never cache-materialized (ADR-008 rail).
# ---------------------------------------------------------------------------
def test_matrix_free_block_is_never_cache_materialized(numpy_ctx):
    space, mats, dense_blocks = _blocks(numpy_ctx, 4, 2)
    free = sc.IdentityLinOp(space)  # _core_kernel_set != "dense", no _A2
    op = BlockDiagonalLinOp.from_operators(dense_blocks + (free,))

    comps = _vec_components(numpy_ctx, 4, 3)
    x = op.domain._from_components(comps)
    with dispatch_mode("on"):
        out = op.codomain._components(op.apply(x))

    assert op.parts._stack_cache == {}  # fold inapplicable -> nothing materialized
    expected = [
        mats[0] @ to_numpy(comps[0]),
        mats[1] @ to_numpy(comps[1]),
        to_numpy(comps[2]),  # identity block
    ]
    for a, r in zip(out, expected):
        np.testing.assert_allclose(to_numpy(a), r, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# The broadcast folds (stacked.apply / sum_to_single.rapply) cache too.
# ---------------------------------------------------------------------------
def test_stacked_apply_caches_and_reuses(numpy_ctx):
    _, mats, blocks = _blocks(numpy_ctx, 4, 3)
    op = StackedLinOp.from_operators(blocks)
    x = _vec_components(numpy_ctx, 4, 1)[0]

    with dispatch_mode("on"):
        first = op.codomain._components(op.apply(x))
        op.apply(x)

    assert set(op.parts._stack_cache) == {STACKED_A2}
    for a, m in zip(first, mats):
        np.testing.assert_allclose(to_numpy(a), m @ to_numpy(x), rtol=1e-12, atol=1e-12)


def test_sum_to_single_rapply_caches_and_reuses(numpy_ctx):
    _, mats, blocks = _blocks(numpy_ctx, 4, 3)
    op = SumToSingleLinOp.from_operators(blocks)
    y = _vec_components(numpy_ctx, 4, 2)[0]

    with dispatch_mode("on"):
        first = op.domain._components(op.rapply(y))
        op.rapply(y)

    assert set(op.parts._stack_cache) == {SUM_TO_SINGLE_A2H}
    for a, m in zip(first, mats):
        np.testing.assert_allclose(
            to_numpy(a), m.conj().T @ to_numpy(y), rtol=1e-12, atol=1e-12
        )

"""Every optimized kernel must match its generic reference numerically.

The correctness_ref field on each :class:`KernelSpec` points back at one
of the test functions below. Adding a new kernel means adding both the
spec and a function here named ``test_<spec name with underscores>``.

Each test takes the same inputs the bench case uses and asserts the
optimized result equals the generic result within the spec's tolerance
*and* equals the NumPy ground truth that the bench case carries.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from bench._operations import kernel_probe_cases
from spacecore.kernels import registry as kernel_registry
from spacecore.kernels.block_diagonal import (
    block_diagonal_dense_apply_applicable,
    block_diagonal_dense_apply_generic,
    block_diagonal_dense_apply_optimized,
)
from spacecore.kernels.composed import (
    composed_chain_apply_applicable,
    composed_chain_apply_generic,
    composed_chain_apply_optimized,
)


# ---------------------------------------------------------------------------
# Per-kernel parity tests (correctness references)


@pytest.mark.parametrize(
    "block_count,block_size",
    [(1, 4), (2, 8), (4, 32), (8, 16), (16, 64)],
)
def test_block_diagonal_dense_apply_matches_generic(block_count, block_size):
    """``block-diagonal-dense-apply`` parity across block-count / size grid."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    ops = ctx.ops
    rng = np.random.default_rng(seed=block_count * 1000 + block_size)
    matrices = tuple(
        ctx.asarray(rng.standard_normal((block_size, block_size)))
        for _ in range(block_count)
    )
    leaves = tuple(
        ctx.asarray(rng.standard_normal(block_size))
        for _ in range(block_count)
    )
    assert block_diagonal_dense_apply_applicable(matrices, leaves, ops)
    generic = block_diagonal_dense_apply_generic(matrices, leaves, ops)
    optimized = block_diagonal_dense_apply_optimized(matrices, leaves, ops)
    assert len(generic) == len(optimized) == block_count
    for g, o in zip(generic, optimized):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(o))


def test_block_diagonal_dense_apply_rejects_mismatched_lengths():
    """Applicability returns False on shape-mismatched inputs."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    matrix = ctx.asarray(np.eye(4))
    leaf = ctx.asarray(np.zeros(4))
    assert not block_diagonal_dense_apply_applicable(
        (matrix, matrix), (leaf,), ctx.ops
    )
    assert not block_diagonal_dense_apply_applicable((), (), ctx.ops)


@pytest.mark.parametrize("chain_length", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [16, 64, 128])
def test_composed_chain_apply_matches_generic(chain_length, n):
    """``composed-chain-apply`` parity across chain-length / size grid."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    rng = np.random.default_rng(seed=chain_length * 100 + n)
    space = sc.DenseCoordinateSpace((n,), ctx)
    linops = tuple(
        sc.DenseLinOp(ctx.asarray(rng.standard_normal((n, n))), space, space, ctx)
        for _ in range(chain_length)
    )
    x = ctx.asarray(rng.standard_normal(n))
    assert composed_chain_apply_applicable(linops, x)
    generic = composed_chain_apply_generic(linops, x)
    optimized = composed_chain_apply_optimized(linops, x)
    np.testing.assert_array_equal(np.asarray(generic), np.asarray(optimized))


def test_composed_chain_apply_empty_returns_input_identity():
    """An empty chain returns the input unchanged (matches generic)."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x = ctx.asarray(np.asarray([1.0, 2.0, 3.0]))
    assert composed_chain_apply_optimized((), x) is x
    assert composed_chain_apply_generic((), x) is x


# ---------------------------------------------------------------------------
# Bench-case-driven parity (every registered case must pass tolerance)


@pytest.mark.parametrize(
    "case_id,kernel_name,case",
    kernel_probe_cases(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_bench_probe_generic_optimized_match_reference(case_id, kernel_name, case):
    """Every bench probe's three callables (generic/optimized/reference) agree.

    This is the integration layer: it confirms that the bench probes are
    correctly wired and that registering a new kernel automatically gets
    the same numeric check the per-kernel parametrized tests give.
    """
    spec = kernel_registry.get(kernel_name)
    generic = case.sc()
    optimized = case.optimized() if case.optimized else generic
    reference = case.reference() if case.reference else case.bare()

    def _to_tuple(value):
        return value if isinstance(value, tuple) else (value,)

    g, o, r = _to_tuple(generic), _to_tuple(optimized), _to_tuple(reference)
    assert len(g) == len(o) == len(r)
    for gi, oi, ri in zip(g, o, r):
        np.testing.assert_allclose(
            np.asarray(gi), np.asarray(ri),
            rtol=max(spec.rtol, 1e-12), atol=max(spec.atol, 1e-12),
            err_msg=f"{case_id}: generic differs from reference",
        )
        np.testing.assert_allclose(
            np.asarray(oi), np.asarray(gi),
            rtol=max(spec.rtol, 1e-12), atol=max(spec.atol, 1e-12),
            err_msg=f"{case_id}: optimized differs from generic",
        )

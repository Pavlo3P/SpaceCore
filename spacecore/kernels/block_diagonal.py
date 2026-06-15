"""Block-diagonal dense-leaf apply kernel.

The generic path for a ``BlockDiagonalLinOp`` over a tree of
``DenseLinOp`` blocks goes through:

1. ``BlockDiagonalLinOp.apply``'s ``@checked_method`` wrapper
2. ``TreeSpace._components`` to unpack the input
3. one ``DenseLinOp.apply`` per leaf, each with its own
   ``@checked_method`` wrapper
4. ``TreeSpace._from_components`` to repack the output

When every block is a ``DenseLinOp`` over a Euclidean dense coordinate
space, the inner work is just ``matrix @ leaf``. The optimized kernel
skips the per-leaf ``checked_method`` and operator-allocation overhead
and calls ``ops.matmul`` (or the cached underlying array operation)
directly on the matrices and leaves.

Applicability is intentionally narrow: every block must be a
``DenseLinOp`` and the kernel takes a flat sequence of matrices and a
matching flat sequence of input leaves. Callers that have a
``BlockDiagonalLinOp`` and a tree element can flatten through the
domain's ``_components``.
"""
from __future__ import annotations

from typing import Any, Sequence

from ._policy import KernelSpec
from ._registry import registry


def block_diagonal_dense_apply_generic(
    matrices: Sequence[Any], leaves: Sequence[Any], ops: Any
) -> tuple[Any, ...]:
    """Reference implementation: per-leaf ``ops.matmul``.

    ``ops`` is a backend ``BackendOps`` instance and is used for
    ``matmul``. This is the contract the optimized kernel must match.
    """
    return tuple(ops.matmul(m, x) for m, x in zip(matrices, leaves))


def block_diagonal_dense_apply_optimized(
    matrices: Sequence[Any], leaves: Sequence[Any], ops: Any
) -> tuple[Any, ...]:
    """Optimized implementation: cached method lookup, tight loop.

    Hoists the bound method out of the loop so the dispatch lookup runs
    once per call rather than once per leaf. On NumpyOps this is a small
    constant-factor improvement that shows up at high block counts.
    """
    matmul = ops.matmul
    out: list[Any] = []
    for m, x in zip(matrices, leaves):
        out.append(matmul(m, x))
    return tuple(out)


def block_diagonal_dense_apply_applicable(
    matrices: Sequence[Any], leaves: Sequence[Any], ops: Any
) -> bool:
    """Applicable when both sequences have the same nonzero length.

    Any further per-leaf shape/dtype validation is left to the backend
    ``matmul`` — the generic implementation does the same.
    """
    if len(matrices) != len(leaves):
        return False
    return len(matrices) > 0


SPEC = registry.register(
    KernelSpec(
        name="block-diagonal-dense-apply",
        generic=block_diagonal_dense_apply_generic,
        optimized=block_diagonal_dense_apply_optimized,
        applicable=block_diagonal_dense_apply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::test_block_diagonal_dense_apply_matches_generic"
        ),
        benchmark_id="kernels.block_diagonal_dense_apply",
        rtol=0.0,
        atol=0.0,
        notes=(
            "Skips per-leaf checked_method validation by hoisting "
            "ops.matmul. Caller is responsible for compatibility of "
            "matrices and leaves; use only when blocks are known dense."
        ),
    )
)

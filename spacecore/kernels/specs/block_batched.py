"""Batched dense matmul kernel for uniform block-diagonal operators.

Dispatch-eligible :class:`KernelSpec` routed at the
``"linop.block_diagonal.apply"`` call site (``BlockDiagonalLinOp._apply_unchecked``).
When every block is a dense operator over a Euclidean *flat* coordinate space,
all blocks share one ``(m, n)`` matrix shape and dtype, and every input
component shares that dtype (stacking heterogeneous dtypes would promote and
compute a block in the wrong precision), the per-block loop

    ``(A_0 @ x_0, A_1 @ x_1, ..., A_{K-1} @ x_{K-1})``

is replaced by a single batched matmul over the stacked operands ``(K, m, n)``
and ``(K, n)`` — one backend call instead of ``K``. The per-slice result of a
stacked ``matmul`` is bit-identical to the individual ``A_k @ x_k`` on NumPy
(verified; ``einsum`` is *not* bit-identical and is deliberately avoided), so the
spec ships ``rtol == atol == 0`` and restricts ``applicable`` to the NumPy
backend until the same equivalence is verified for the others.

This is a **materializing** fast path: stacking the blocks allocates ``O(K·m·n)``
extra bytes beyond the generic loop's working set, so the spec carries a
shape-only :class:`KernelCost`. The dispatcher checks that estimate against the
context's memory budget before selecting it — a wide stack on a tight budget
falls through to the generic loop.

Block kind and mode are read from the duck-typed ``_core_kernel_set`` /
``_mode`` / ``_A2`` attributes the dense core stamps on the operator, so this
module imports nothing from :mod:`spacecore.linop`.
"""
from __future__ import annotations

from typing import Any, Sequence

from ._policy import KernelCost, KernelSpec
from ._registry import registry

_BLOCK_DIAGONAL_APPLY_KEY = "linop.block_diagonal.apply"

# Batching wins only past a couple of blocks; below this the stack overhead
# dominates, so the spec reports itself inapplicable (compute profitability
# lives in ``applicable``, per ADR-016).
_MIN_BLOCKS = 2


def block_diagonal_apply_generic(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Reference: each block's core on its matching component.

    Byte-identical to the ``"linop.block_diagonal.apply"`` call site's inline
    path; re-exposed so the correctness test can call it directly.
    """
    return tuple(p._apply_core(xi) for p, xi in zip(parts, x_parts))


def _uniform_flat_dense(parts: Sequence[Any]) -> "tuple[tuple[int, int], Any] | None":
    """Return the shared ``((m, n), dtype)`` of uniform flat-dense blocks.

    Returns ``None`` unless every block is a flat-dense (``EUCLIDEAN_FLAT``)
    operator sharing one matrix shape *and* one dtype. Stacking promotes a mixed
    set to a common dtype, which would compute a lower-precision block in higher
    precision and break bit-exactness versus the per-block loop. (A
    ``BlockDiagonalLinOp`` already enforces one context, hence one dtype, across
    its blocks; this guard keeps the spec exact for any direct caller too.)
    Reads only operator metadata (``_core_kernel_set``, ``_mode``, ``_A2.shape``,
    ``_A2.dtype``) — never operand data.
    """
    from ..core.dense import _DenseMode

    if len(parts) < _MIN_BLOCKS:
        return None
    shape: "tuple[int, int] | None" = None
    dtype: Any = None
    for p in parts:
        if getattr(p, "_core_kernel_set", None) != "dense":
            return None
        if getattr(p, "_mode", None) is not _DenseMode.EUCLIDEAN_FLAT:
            return None
        p_shape = tuple(p._A2.shape)
        if len(p_shape) != 2:
            return None
        p_dtype = p._A2.dtype
        if shape is None:
            shape = p_shape  # type: ignore[assignment]
            dtype = p_dtype
        elif p_shape != shape or p_dtype != dtype:
            return None
    return shape, dtype  # type: ignore[return-value]


def block_batched_applicable(parts: Sequence[Any], x_parts: Sequence[Any]) -> bool:
    """Applicable to a uniform flat-dense block tuple on the NumPy backend.

    Requires the input components to share the blocks' dtype as well: at
    ``check_level="none"`` a tree element may carry heterogeneous-dtype leaves,
    and ``stack`` would promote them — computing a block in the wrong precision.
    Such inputs fall through to the exact per-block generic loop.
    """
    if len(parts) != len(x_parts):
        return False
    info = _uniform_flat_dense(parts)
    if info is None:
        return False
    _, dtype = info
    if any(getattr(x, "dtype", None) != dtype for x in x_parts):
        return False
    # Bit-exactness of stacked vs per-block matmul is verified for NumPy only.
    return parts[0].ops.family == "numpy"


def block_batched_optimized(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Single batched ``matmul`` over the stacked blocks and components."""
    ops = parts[0].ops
    stacked = ops.stack([p._A2 for p in parts])      # (K, m, n)
    x_batched = ops.stack(list(x_parts))             # (K, n)
    y_batched = ops.matmul(stacked, x_batched[..., None])[..., 0]  # (K, m)
    return tuple(y_batched[k] for k in range(len(parts)))


def block_batched_cost(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the stacking fast path."""
    info = _uniform_flat_dense(parts)
    if info is None:
        return None
    (m, n), _ = info
    k = len(parts)
    itemsize = int(parts[0]._A2.itemsize)
    # Extra allocations: stacked matrices (K·m·n), stacked input (K·n), and the
    # batched output (K·m). The generic loop allocates only O(K·m) for outputs.
    peak_bytes = (k * m * n + k * n + k * m) * itemsize
    return KernelCost(peak_bytes=peak_bytes, flops=2 * k * m * n)


SPEC = registry.register(
    KernelSpec(
        name="block-diagonal-uniform-dense-batched",
        generic=block_diagonal_apply_generic,
        optimized=block_batched_optimized,
        applicable=block_batched_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestBlockDiagonalUniformBatched::test_matches_generic"
        ),
        benchmark_id="kernels.block_diagonal_uniform_batched",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_BLOCK_DIAGONAL_APPLY_KEY,
        priority=10,
        cost=block_batched_cost,
        notes=(
            "Uniform flat-dense blocks -> one batched matmul. Materializing "
            "(stacks O(K*m*n)); the shape-only cost gates it on the memory "
            "budget. NumPy-only until cross-backend bit-exactness is verified."
        ),
    )
)

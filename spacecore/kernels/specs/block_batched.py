"""Batched dense matmul kernels for uniform block-diagonal operators.

Dispatch-eligible :class:`KernelSpec` objects routed at the block-diagonal call
sites (``BlockDiagonalLinOp._apply_unchecked`` / ``_rapply_unchecked``). When
every block is a dense operator over a Euclidean *flat* coordinate space, all
blocks share one matrix shape and dtype, and every input component shares that
dtype (stacking heterogeneous dtypes would promote and compute a block in the
wrong precision), the per-block loop is replaced by a single batched ``matmul``
over the stacked operands — one backend call instead of ``K``:

* **apply** (``linop.block_diagonal.apply``) stacks ``_A2`` and computes
  ``(A_0 @ x_0, ..., A_{K-1} @ x_{K-1})``.
* **rapply** (``linop.block_diagonal.rapply``) stacks the Euclidean adjoint
  ``_A2H`` and computes ``(A_0^H @ y_0, ..., A_{K-1}^H @ y_{K-1})``. The
  ``EUCLIDEAN_FLAT`` guard is load-bearing: a non-Euclidean adjoint goes through
  a metric/Riesz path, so a plain ``_A2H`` stack would be wrong.

Both share the one batched-matmul helper in :mod:`._batched`; each spec is a thin
wrapper naming a matrix accessor. The per-slice result of a stacked ``matmul`` is
bit-identical to the individual block on NumPy (verified; ``einsum`` is *not*
bit-identical and is deliberately avoided), so the specs ship ``rtol == atol ==
0`` and restrict ``applicable`` to the NumPy backend until the same equivalence
is verified for the others.

These are **materializing** fast paths: stacking the blocks allocates extra
bytes beyond the generic loop's working set, so each spec carries a shape-only
:class:`KernelCost`; the dispatcher checks it against the context's memory budget
before selecting it.

Block kind and mode are read from the duck-typed ``_core_kernel_set`` / ``_mode``
/ ``_A2`` / ``_A2H`` attributes the dense core stamps on the operator, so this
module imports nothing from :mod:`spacecore.linop`.
"""
from __future__ import annotations

from math import prod
from typing import Any, Sequence

from . import _batched
from ._policy import KernelCost, KernelSpec
from ._registry import registry

_BLOCK_DIAGONAL_APPLY_KEY = "linop.block_diagonal.apply"
_BLOCK_DIAGONAL_RAPPLY_KEY = "linop.block_diagonal.rapply"
_BLOCK_DIAGONAL_VAPPLY_KEY = "linop.block_diagonal.vapply"
_BLOCK_DIAGONAL_RVAPPLY_KEY = "linop.block_diagonal.rvapply"


def _A2(p: Any) -> Any:
    return p._A2


def _A2H(p: Any) -> Any:
    return p._A2H


def _A2T(p: Any) -> Any:
    return p._A2T


def _A2H_T(p: Any) -> Any:
    return p._A2H.T


def _leading(batch: Any, cols: int) -> int:
    """Flattened batch size ``M`` of an operand reshaped to ``(-1, cols)``."""
    return prod(tuple(batch.shape)) // cols


# ---------------------------------------------------------------------------
# Forward apply
# ---------------------------------------------------------------------------
def block_diagonal_apply_generic(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Apply each block's core to its matching component (reference).

    Byte-identical to the ``"linop.block_diagonal.apply"`` call site's inline
    path; re-exposed so the correctness test can call it directly.
    """
    return tuple(p._apply_core(xi) for p, xi in zip(parts, x_parts))


def block_batched_applicable(parts: Sequence[Any], x_parts: Sequence[Any]) -> bool:
    """Applicable to a uniform flat-dense block tuple on the NumPy backend."""
    if len(parts) != len(x_parts):
        return False
    info = _batched.uniform_flat_dense(parts, _A2)
    if info is None:
        return False
    _, dtype = info
    if not _batched.operands_share_dtype(x_parts, dtype):
        return False
    return _batched.is_numpy(parts)


def block_batched_optimized(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Single batched ``matmul`` over the stacked blocks and components."""
    return _batched.batched_matvec(parts, x_parts, _A2)


def block_batched_cost(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the stacking fast path."""
    info = _batched.uniform_flat_dense(parts, _A2)
    if info is None:
        return None
    return _batched.matvec_cost(info[0], len(parts), int(parts[0]._A2.itemsize))


# ---------------------------------------------------------------------------
# Adjoint rapply (Euclidean-flat only)
# ---------------------------------------------------------------------------
def block_diagonal_rapply_generic(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Apply each block's adjoint core to its matching component (reference).

    Byte-identical to the ``"linop.block_diagonal.rapply"`` call site's inline
    path (``parts[i]._rapply_core`` is exactly ``self._rapply_parts[i]``).
    """
    return tuple(p._rapply_core(yi) for p, yi in zip(parts, y_parts))


def block_batched_rapply_applicable(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> bool:
    """Applicable to a uniform flat-dense adjoint block tuple on NumPy.

    The shared ``EUCLIDEAN_FLAT`` mode (checked by ``uniform_flat_dense``) is
    what makes ``_A2H @ y`` the exact adjoint: a non-Euclidean adjoint would use
    ``metric_rapply`` / ``_weighted_A2H`` instead, so those operators are
    excluded and fall through to the generic per-block loop.
    """
    if len(parts) != len(y_parts):
        return False
    info = _batched.uniform_flat_dense(parts, _A2H)
    if info is None:
        return False
    _, dtype = info
    if not _batched.operands_share_dtype(y_parts, dtype):
        return False
    return _batched.is_numpy(parts)


def block_batched_rapply_optimized(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Single batched ``matmul`` over the stacked adjoint blocks and components."""
    return _batched.batched_matvec(parts, y_parts, _A2H)


def block_batched_rapply_cost(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the adjoint stacking fast path."""
    info = _batched.uniform_flat_dense(parts, _A2H)
    if info is None:
        return None
    return _batched.matvec_cost(info[0], len(parts), int(parts[0]._A2H.itemsize))


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


RAPPLY_SPEC = registry.register(
    KernelSpec(
        name="block-diagonal-uniform-dense-batched-rapply",
        generic=block_diagonal_rapply_generic,
        optimized=block_batched_rapply_optimized,
        applicable=block_batched_rapply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestBlockDiagonalUniformRapply::test_matches_generic"
        ),
        benchmark_id="kernels.block_diagonal_uniform_batched_rapply",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_BLOCK_DIAGONAL_RAPPLY_KEY,
        priority=10,
        cost=block_batched_rapply_cost,
        notes=(
            "Euclidean-flat adjoint dual of the apply fold: stack _A2H -> one "
            "batched matmul. Materializing; NumPy-only until cross-backend "
            "bit-exactness is verified."
        ),
    )
)


# ---------------------------------------------------------------------------
# Batched vapply / rvapply (Euclidean-flat only, transpose-on-right)
# ---------------------------------------------------------------------------
def block_diagonal_vapply_generic(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Apply each block's batched core to its matching component (reference).

    Byte-identical to the ``"linop.block_diagonal.vapply"`` call site's inline
    path; pins the check-free ``_vapply_core`` (the dense core's transpose-on-
    right ``xs.reshape((-1, n)) @ _A2T`` orientation).
    """
    return tuple(p._vapply_core(xi) for p, xi in zip(parts, x_parts))


def block_batched_vapply_applicable(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> bool:
    """Applicable to a uniform flat-dense batched tuple on the NumPy backend."""
    if len(parts) != len(x_parts):
        return False
    info = _batched.uniform_flat_dense(parts, _A2T)
    if info is None:
        return False
    _, dtype = info
    if not _batched.operands_share_dtype(x_parts, dtype):
        return False
    return _batched.is_numpy(parts)


def block_batched_vapply_optimized(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """One batched ``matmul`` ``X_k @ A2T_k`` over the stacked operands."""
    return _batched.batched_right_matmul(parts, x_parts, _A2T, parts[0]._dom_size)


def block_batched_vapply_cost(
    parts: Sequence[Any], x_parts: Sequence[Any]
) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the batched vapply fast path."""
    info = _batched.uniform_flat_dense(parts, _A2T)
    if info is None:
        return None
    cols = parts[0]._dom_size
    lead = _leading(x_parts[0], cols)
    return _batched.right_matmul_cost(info[0], len(parts), lead, int(parts[0]._A2T.itemsize))


def block_diagonal_rvapply_generic(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """Apply each block's batched adjoint core to its component (reference).

    Byte-identical to the ``"linop.block_diagonal.rvapply"`` call site's inline
    path; pins the check-free ``_rvapply_core`` (``ys.reshape((-1, m)) @
    _A2H.T``). Euclidean-flat only: a non-Euclidean adjoint uses a metric path.
    """
    return tuple(p._rvapply_core(yi) for p, yi in zip(parts, y_parts))


def block_batched_rvapply_applicable(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> bool:
    """Applicable to a uniform flat-dense batched adjoint tuple on NumPy."""
    if len(parts) != len(y_parts):
        return False
    info = _batched.uniform_flat_dense(parts, _A2H_T)
    if info is None:
        return False
    _, dtype = info
    if not _batched.operands_share_dtype(y_parts, dtype):
        return False
    return _batched.is_numpy(parts)


def block_batched_rvapply_optimized(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> tuple[Any, ...]:
    """One batched ``matmul`` ``Y_k @ A2H_k.T`` over the stacked operands."""
    return _batched.batched_right_matmul(parts, y_parts, _A2H_T, parts[0]._cod_size)


def block_batched_rvapply_cost(
    parts: Sequence[Any], y_parts: Sequence[Any]
) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the batched rvapply fast path."""
    info = _batched.uniform_flat_dense(parts, _A2H_T)
    if info is None:
        return None
    cols = parts[0]._cod_size
    lead = _leading(y_parts[0], cols)
    return _batched.right_matmul_cost(info[0], len(parts), lead, int(parts[0]._A2H.itemsize))


VAPPLY_SPEC = registry.register(
    KernelSpec(
        name="block-diagonal-uniform-dense-batched-vapply",
        generic=block_diagonal_vapply_generic,
        optimized=block_batched_vapply_optimized,
        applicable=block_batched_vapply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestBlockDiagonalUniformVapply::test_matches_generic"
        ),
        benchmark_id="kernels.block_diagonal_uniform_batched_vapply",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_BLOCK_DIAGONAL_VAPPLY_KEY,
        priority=10,
        cost=block_batched_vapply_cost,
        notes=(
            "Batched twin of the apply fold (transpose-on-right X @ A2T). "
            "Materializing; NumPy-only until cross-backend bit-exactness is "
            "verified."
        ),
    )
)


RVAPPLY_SPEC = registry.register(
    KernelSpec(
        name="block-diagonal-uniform-dense-batched-rvapply",
        generic=block_diagonal_rvapply_generic,
        optimized=block_batched_rvapply_optimized,
        applicable=block_batched_rvapply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestBlockDiagonalUniformRvapply::test_matches_generic"
        ),
        benchmark_id="kernels.block_diagonal_uniform_batched_rvapply",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_BLOCK_DIAGONAL_RVAPPLY_KEY,
        priority=10,
        cost=block_batched_rvapply_cost,
        notes=(
            "Euclidean-flat batched adjoint dual (transpose-on-right Y @ "
            "A2H.T). Materializing; NumPy-only until cross-backend "
            "bit-exactness is verified."
        ),
    )
)

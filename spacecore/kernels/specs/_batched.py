"""Shared batched-matmul fast path for uniform flat-dense operator tuples.

Several ADR-016 dispatch specs (block-diagonal apply/rapply/vapply/rvapply, the
stacked / sum-to-single broadcast folds) are the *same* optimization: when every
leaf is a flat-dense (``EUCLIDEAN_FLAT``) operator sharing one ``(rows, cols)``
matrix shape and dtype, a per-leaf loop over ``K`` backend ``matmul`` calls is
replaced by a single batched ``matmul`` over the stacked operands. They differ
only in *which* cached matrix each leaf contributes (``_A2`` for the forward
action, ``_A2H`` for the Euclidean adjoint, ``_A2.T`` / ``_A2H.T`` for the
batched transposed-right orientation) and in which call site is wired.

This module centralizes the uniformity guard and the two batched-``matmul``
primitives so each spec is a thin wrapper that names a matrix accessor and an
orientation — never a re-implementation. That is the single guard against the
per-shape spec sprawl ADR-016 warns about.

A stacked ``matmul`` is bit-identical to the per-leaf ``matmul`` on NumPy
(verified); ``einsum`` is **not** bit-identical and is deliberately avoided. The
wrapping specs therefore ship ``rtol == atol == 0`` and restrict ``applicable``
to the NumPy backend until the same equivalence is verified for the others.

Everything here reads operator/operand *metadata* (``_core_kernel_set``,
``_mode``, ``.shape``, ``.dtype``, ``.itemsize``) only — never operand data, and
never the result being estimated.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from ._policy import KernelCost

# Batching wins only past a couple of leaves; below this the stack overhead
# dominates, so a wrapping spec reports itself inapplicable (ADR-016 puts
# compute profitability in ``applicable``, not the memory gate).
MIN_BLOCKS = 2

Matrix = Callable[[Any], Any]


def uniform_flat_dense(
    parts: Sequence[Any], matrix: Matrix
) -> "tuple[tuple[int, int], Any] | None":
    """Return the shared ``((rows, cols), dtype)`` of uniform flat-dense parts.

    ``matrix`` reads the cached 2-D matrix contributed by one part (e.g.
    ``lambda p: p._A2`` for the forward action, ``lambda p: p._A2H`` for the
    Euclidean adjoint). Returns ``None`` unless there are at least
    :data:`MIN_BLOCKS` parts and every part is a flat-dense
    (``_DenseMode.EUCLIDEAN_FLAT``) operator whose matrix shares one shape *and*
    one dtype. ``EUCLIDEAN_FLAT`` is load-bearing: a non-Euclidean adjoint uses
    a metric/Riesz path, so a plain matrix stack would be numerically wrong.
    """
    from ..core.dense import _DenseMode

    if len(parts) < MIN_BLOCKS:
        return None
    shape: "tuple[int, int] | None" = None
    dtype: Any = None
    for p in parts:
        if getattr(p, "_core_kernel_set", None) != "dense":
            return None
        if getattr(p, "_mode", None) is not _DenseMode.EUCLIDEAN_FLAT:
            return None
        mat = matrix(p)
        m_shape = tuple(mat.shape)
        if len(m_shape) != 2:
            return None
        if shape is None:
            shape = m_shape  # type: ignore[assignment]
            dtype = mat.dtype
        elif m_shape != shape or mat.dtype != dtype:
            return None
    return shape, dtype  # type: ignore[return-value]


def is_numpy(parts: Sequence[Any]) -> bool:
    """Return whether the parts run on the NumPy backend (bit-exactness gate)."""
    return parts[0].ops.family == "numpy"


def operands_share_dtype(operands: Sequence[Any], dtype: Any) -> bool:
    """Return whether every operand already carries ``dtype``.

    At ``check_level="none"`` a tree element may carry heterogeneous-dtype
    leaves; ``stack`` would promote them and compute a block in the wrong
    precision, breaking bit-exactness versus the per-block loop. Such inputs
    fall through to the exact generic path.
    """
    return all(getattr(o, "dtype", None) == dtype for o in operands)


def batched_matvec(
    parts: Sequence[Any], vecs: Sequence[Any], matrix: Matrix
) -> tuple[Any, ...]:
    """Return ``tuple(M_k @ v_k)`` via one batched ``matmul``.

    ``M_k = matrix(parts[k])`` has shape ``(r, c)``; ``vecs[k]`` has shape
    ``(c,)``; result component ``k`` has shape ``(r,)``. Per-slice identical to
    the individual ``M_k @ v_k`` on NumPy.
    """
    ops = parts[0].ops
    mats = ops.stack([matrix(p) for p in parts])     # (K, r, c)
    stacked = ops.stack(list(vecs))                  # (K, c)
    out = ops.matmul(mats, stacked[..., None])[..., 0]  # (K, r)
    return tuple(out[k] for k in range(len(parts)))


def batched_right_matmul(
    parts: Sequence[Any], batches: Sequence[Any], matrix: Matrix, cols: int
) -> tuple[Any, ...]:
    """Return ``tuple(X_k @ R_k)`` via one batched ``matmul`` (transpose-on-right).

    Mirrors the dense ``vapply``/``rvapply`` core orientation
    ``xs.reshape((-1, cols)) @ R``. ``R_k = matrix(parts[k])`` has shape
    ``(cols, r)``; ``batches[k]`` flattens to ``(-1, cols)``; result component
    ``k`` has shape ``(-1, r)``. Per-slice identical to the individual
    ``X_k @ R_k`` on NumPy.
    """
    ops = parts[0].ops
    rights = ops.stack([matrix(p) for p in parts])             # (K, cols, r)
    stacked = ops.stack([b.reshape((-1, cols)) for b in batches])  # (K, M, cols)
    out = ops.matmul(stacked, rights)                          # (K, M, r)
    return tuple(out[k] for k in range(len(parts)))


def matvec_cost(shape: tuple[int, int], k: int, itemsize: int) -> KernelCost:
    """Peak extra bytes for :func:`batched_matvec`: stacked mats + vecs + out."""
    r, c = shape
    peak_bytes = (k * r * c + k * c + k * r) * itemsize
    return KernelCost(peak_bytes=peak_bytes, flops=2 * k * r * c)


def right_matmul_cost(
    shape: tuple[int, int], k: int, lead: int, itemsize: int
) -> KernelCost:
    """Peak extra bytes for :func:`batched_right_matmul`: stacked mats + batch + out.

    ``shape`` is ``(cols, r)`` (the right matrix), ``lead`` is the flattened
    batch size ``M`` shared by every component.
    """
    cols, r = shape
    peak_bytes = (k * cols * r + k * lead * cols + k * lead * r) * itemsize
    return KernelCost(peak_bytes=peak_bytes, flops=2 * k * lead * cols * r)

"""Batched dense matmul kernels for the stacked / sum-to-single broadcast folds.

Two ADR-016 dispatch specs for the *broadcast-no-sum* directions of the
single-domain / single-codomain tree operators, where one operand is applied to
every component and the per-component results are returned as a tuple (no
reduction, so the fold is exactly bit-equal to the per-component loop):

* **stacked.apply** (``linop.stacked.apply``) — ``StackedLinOp`` applies the one
  shared ``x`` through every component ``A_i`` to a product codomain; stack
  ``_A2`` and compute ``(A_0 @ x, ..., A_{K-1} @ x)``.
* **sum_to_single.rapply** (``linop.sum_to_single.rapply``) — ``SumToSingleLinOp``
  applies the one shared ``y`` through every component adjoint ``A_i^H`` to a
  product domain; stack ``_A2H`` and compute ``(A_0^H @ y, ..., A_{K-1}^H @ y)``.

The *reduction* directions of these operators (``stacked.rapply`` /
``sum_to_single.apply`` and their batched twins) already carry an exact inline
flat-dense fast path and a float-add order that must be preserved, so they are
deliberately **not** routed here.

Both share the broadcast batched-matmul helper in :mod:`._batched`; each is a
thin wrapper naming a matrix accessor. Components share a domain (stacked) or a
codomain (sum-to-single) but may differ on the other side, so the uniformity
guard additionally requires a uniform matrix *shape* — heterogeneous-shape
operators fall through to the generic loop. Guarded to ``EUCLIDEAN_FLAT`` and the
NumPy backend, with a shape-only memory cost, exactly as the block-diagonal
folds.
"""
from __future__ import annotations

from typing import Any, Sequence

from . import _batched
from ._policy import KernelCost, KernelSpec
from ._registry import registry

_STACKED_APPLY_KEY = "linop.stacked.apply"
_SUM_TO_SINGLE_RAPPLY_KEY = "linop.sum_to_single.rapply"


def _A2(p: Any) -> Any:
    return p._A2


def _A2H(p: Any) -> Any:
    return p._A2H


# ---------------------------------------------------------------------------
# stacked.apply (one shared x -> tuple of component outputs)
# ---------------------------------------------------------------------------
def stacked_apply_generic(parts: Sequence[Any], x: Any) -> tuple[Any, ...]:
    """Apply the shared input through each component core (reference).

    Byte-identical to the ``"linop.stacked.apply"`` call site's inline path
    (``self._apply_parts[i]`` is exactly ``parts[i]._apply_core``).
    """
    return tuple(p._apply_core(x) for p in parts)


def stacked_apply_applicable(parts: Sequence[Any], x: Any) -> bool:
    """Applicable to uniform flat-dense components and a matching-dtype ``x`` on NumPy."""
    info = _batched.uniform_flat_dense(parts, _A2)
    if info is None:
        return False
    _, dtype = info
    if not _batched.operands_share_dtype((x,), dtype):
        return False
    return _batched.is_numpy(parts)


def stacked_apply_optimized(parts: Sequence[Any], x: Any) -> tuple[Any, ...]:
    """One batched ``matmul`` of the stacked component matrices against shared ``x``."""
    return _batched.batched_matvec_shared(parts, x, _A2)


def stacked_apply_cost(parts: Sequence[Any], x: Any) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the stacked apply fast path."""
    info = _batched.uniform_flat_dense(parts, _A2)
    if info is None:
        return None
    return _batched.matvec_shared_cost(info[0], len(parts), int(parts[0]._A2.itemsize))


# ---------------------------------------------------------------------------
# sum_to_single.rapply (one shared y -> tuple of component adjoint outputs)
# ---------------------------------------------------------------------------
def sum_to_single_rapply_generic(parts: Sequence[Any], y: Any) -> tuple[Any, ...]:
    """Apply the shared input through each component adjoint core (reference).

    Byte-identical to the ``"linop.sum_to_single.rapply"`` call site's inline
    path. Euclidean-flat only: a non-Euclidean adjoint uses a metric/Riesz path.
    """
    return tuple(p._rapply_core(y) for p in parts)


def sum_to_single_rapply_applicable(parts: Sequence[Any], y: Any) -> bool:
    """Applicable to uniform flat-dense adjoints and a matching-dtype ``y`` on NumPy."""
    info = _batched.uniform_flat_dense(parts, _A2H)
    if info is None:
        return False
    _, dtype = info
    if not _batched.operands_share_dtype((y,), dtype):
        return False
    return _batched.is_numpy(parts)


def sum_to_single_rapply_optimized(parts: Sequence[Any], y: Any) -> tuple[Any, ...]:
    """One batched ``matmul`` of the stacked adjoint matrices against shared ``y``."""
    return _batched.batched_matvec_shared(parts, y, _A2H)


def sum_to_single_rapply_cost(parts: Sequence[Any], y: Any) -> "KernelCost | None":
    """Shape-only peak-extra-cost estimate of the sum-to-single rapply fast path."""
    info = _batched.uniform_flat_dense(parts, _A2H)
    if info is None:
        return None
    return _batched.matvec_shared_cost(info[0], len(parts), int(parts[0]._A2H.itemsize))


STACKED_APPLY_SPEC = registry.register(
    KernelSpec(
        name="stacked-uniform-dense-batched-apply",
        generic=stacked_apply_generic,
        optimized=stacked_apply_optimized,
        applicable=stacked_apply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestStackedUniformApply::test_matches_generic"
        ),
        benchmark_id="kernels.stacked_uniform_batched_apply",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_STACKED_APPLY_KEY,
        priority=10,
        cost=stacked_apply_cost,
        notes=(
            "Broadcast one shared x through uniform flat-dense components -> one "
            "batched matmul. Materializing; NumPy-only until cross-backend "
            "bit-exactness is verified."
        ),
    )
)


SUM_TO_SINGLE_RAPPLY_SPEC = registry.register(
    KernelSpec(
        name="sum-to-single-uniform-dense-batched-rapply",
        generic=sum_to_single_rapply_generic,
        optimized=sum_to_single_rapply_optimized,
        applicable=sum_to_single_rapply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestSumToSingleUniformRapply::test_matches_generic"
        ),
        benchmark_id="kernels.sum_to_single_uniform_batched_rapply",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_SUM_TO_SINGLE_RAPPLY_KEY,
        priority=10,
        cost=sum_to_single_rapply_cost,
        notes=(
            "Broadcast one shared y through uniform flat-dense adjoints -> one "
            "batched matmul. Euclidean-flat only; materializing; NumPy-only "
            "until cross-backend bit-exactness is verified."
        ),
    )
)

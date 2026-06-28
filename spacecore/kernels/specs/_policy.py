"""Policy types that every optimized kernel must satisfy.

A :class:`KernelSpec` is the contract that ties a fast path to (a) the
generic implementation it must match numerically, (b) the test that pins
that match across the generator-driven case set, and (c) the benchmark
that proves it actually wins. The two exceptions reject registration
attempts that skip either rail.

ADR-016 (dispatch policy, accepted) adds the structural-dispatch metadata
that lets the benchmarked-spec layer be *routed* through a single
dispatcher instead of named explicitly at each call site: a
``dispatch_key`` naming the operation family a call site requests, an
integer ``priority``, and an optional shape-only :class:`KernelCost`
estimator for fast paths that allocate more than ``O(1)`` extra memory.
``name`` stays the unique identity; many specs may share a ``dispatch_key``.

This module intentionally has no runtime behavior beyond field validation.
It describes the shape of a kernel registration and the rules the registry
and dispatcher enforce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


class GenericImpl(Protocol):
    """The reference implementation of a kernel.

    Called when a caller wants behavior identical to the un-optimized
    code path. The optimized implementation is tested against this.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class OptimizedImpl(Protocol):
    """The optimized implementation of a kernel.

    Must return values numerically equivalent to ``GenericImpl`` within
    the kernel's documented tolerance whenever ``Applicability`` returns
    ``True`` for the inputs.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class Applicability(Protocol):
    """Predicate that returns ``True`` when the optimized kernel is safe.

    Receives the same arguments as the kernel's call signature. May raise
    only on programmer errors: the dispatcher treats a raised exception as
    "not applicable, plus a bug to fix" — it skips the spec and emits a
    ``RuntimeWarning`` rather than letting the apply call crash.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> bool: ...


@dataclass(frozen=True, slots=True)
class KernelCost:
    """Predicted peak extra cost of an optimized kernel's fast path.

    Computed **from shapes and dtypes only** — never from operand data, and
    never by allocating the result being estimated. The dispatcher uses
    :attr:`peak_bytes` to decide whether a materializing fast path fits the
    context's memory budget before selecting it (ADR-016's memory gate).

    Attributes
    ----------
    peak_bytes
        Predicted peak *extra* bytes the optimized path allocates beyond the
        ``O(1)`` working set the generic path already needs (forming a Gram
        matrix, multiplying ``dense @ dense`` into one matrix, stacking blocks
        for a batched call, ...).
    flops
        Predicted floating-point operation count of the optimized path.
        Informational: *compute* profitability is the spec author's
        responsibility, encoded in ``applicable`` from shapes. The dispatcher
        gates on memory, not flops.
    """

    peak_bytes: int
    flops: int = 0

    def __post_init__(self) -> None:
        if self.peak_bytes < 0:
            raise ValueError("KernelCost.peak_bytes must be non-negative.")
        if self.flops < 0:
            raise ValueError("KernelCost.flops must be non-negative.")


class CostEstimator(Protocol):
    """Shape-only predictor of an optimized kernel's peak extra cost.

    Receives the same arguments as the kernel's call signature and returns a
    :class:`KernelCost`, or ``None`` when no estimate can be produced (unknown
    or symbolic shapes). The dispatcher treats ``None`` as *unaffordable* and
    skips the spec — **no estimate, no fuse**. The estimator must read shapes
    and dtypes only; it must never touch operand data or allocate the result.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> "KernelCost | None": ...


class MissingReferenceError(ValueError):
    """Raised when a kernel attempts to register without a correctness ref."""


class MissingBenchmarkError(ValueError):
    """Raised when a kernel attempts to register without a benchmark id."""


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """The full contract for one optimized kernel.

    Attributes
    ----------
    name
        Stable kebab-case identifier used by tests, benchmarks, and
        documentation. Must be unique across the registry.
    generic
        Reference implementation. Must agree numerically with
        ``optimized`` whenever ``applicable`` returns ``True``.
    optimized
        Fast-path implementation.
    applicable
        Predicate that returns ``True`` when ``optimized`` is safe to
        call. The default is ``lambda *a, **k: True``.
    correctness_ref
        Pytest node id of the test that pins ``optimized`` against
        ``generic`` on every applicable generated case. Required.
        Example: ``"tests/kernels/test_kernels_match_generic.py::``
        ``test_block_diagonal_dense_apply_matches_generic"``.
    benchmark_id
        Identifier of the bench case that measures the optimized vs
        generic ratio. Required.
    rtol, atol
        Tolerance used by the matching test. Defaults are tight; loosen
        only when an underlying backend op already disagrees with the
        same tolerance. A spec is **dispatch-eligible** (auto-routable by
        the dispatcher) only when ``rtol == atol == 0`` — exact
        equivalence. A spec with loosened tolerances may register and be
        called explicitly but is never auto-routed.
    dispatch_key
        Name of the operation *family* a call site requests, e.g.
        ``"linop.composed.apply"``. Empty (the default) means the spec is
        not dispatch-routable: it is an explicit-entry kernel only. Many
        specs may share a ``dispatch_key``; the dispatcher selects among
        them by descending :attr:`priority`.
    priority
        Selection order within a ``dispatch_key``. Higher wins. Two
        dispatch-eligible specs that share a ``dispatch_key`` **and** a
        ``priority`` are a registration-time error (ambiguous selection).
    cost
        Shape-only :class:`KernelCost` estimator, required for any
        dispatch-eligible spec whose ``optimized`` path allocates more than
        ``O(1)`` extra memory. The dispatcher checks the estimate against
        the context's memory budget before selecting the spec. ``None``
        (the default) declares the fast path non-materializing; a
        materializing spec with no ``cost`` must not auto-dispatch.
    notes
        One-line note about *why* this kernel exists or when *not* to
        use it.
    """

    name: str
    generic: GenericImpl
    optimized: OptimizedImpl
    applicable: Applicability
    correctness_ref: str
    benchmark_id: str
    rtol: float = 1e-12
    atol: float = 1e-12
    dispatch_key: str = ""
    priority: int = 0
    cost: Optional[CostEstimator] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.correctness_ref:
            raise MissingReferenceError(
                f"kernel {self.name!r}: correctness_ref is required"
            )
        if not self.benchmark_id:
            raise MissingBenchmarkError(
                f"kernel {self.name!r}: benchmark_id is required"
            )
        if not callable(self.generic):
            raise TypeError(f"kernel {self.name!r}: generic must be callable")
        if not callable(self.optimized):
            raise TypeError(f"kernel {self.name!r}: optimized must be callable")
        if not callable(self.applicable):
            raise TypeError(f"kernel {self.name!r}: applicable must be callable")
        if self.cost is not None and not callable(self.cost):
            raise TypeError(f"kernel {self.name!r}: cost must be callable or None")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError(f"kernel {self.name!r}: priority must be an int")

    @property
    def is_dispatch_eligible(self) -> bool:
        """Return whether the dispatcher may auto-route to this spec.

        A spec is eligible only when it names a ``dispatch_key`` *and* claims
        exact equivalence (``rtol == atol == 0``). Specs with loosened
        tolerances, or with no ``dispatch_key``, are explicit-entry only.
        """
        return bool(self.dispatch_key) and self.rtol == 0.0 and self.atol == 0.0

"""Policy types that every optimized kernel must satisfy.

A :class:`KernelSpec` is the contract that ties a fast path to (a) the
generic implementation it must match numerically, (b) the test that pins
that match across the generator-driven case set, and (c) the benchmark
that proves it actually wins. The two exceptions reject registration
attempts that skip either rail.

This module intentionally has no behavior. It only describes the shape of
a kernel registration and the rules the registry will enforce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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
    only on programmer errors (the registry treats raised exceptions as
    "not applicable, plus a bug to fix").
    """

    def __call__(self, *args: Any, **kwargs: Any) -> bool: ...


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
        generic ratio. Must be reachable through
        :func:`bench._operations.kernel_benchmark_ids`. Required.
    rtol, atol
        Tolerance used by the matching test. Defaults are tight; loosen
        only when an underlying backend op already disagrees with the
        same tolerance.
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

"""Benchmarked numerical kernels and their registration policy.

A :class:`KernelSpec` ties a hand-optimized fast path to a generic reference, an
applicability predicate, a correctness-reference test, and a benchmark id. These
kernels are opt-in: a clearly-scoped call site selects them explicitly; nothing
auto-dispatches. See ``docs/source/design/kernels_policy.rst`` for the contract
and :mod:`spacecore.kernels` for how this layer relates to the core-kernel layer.

Importing this subpackage registers every :class:`KernelSpec`.
"""
from __future__ import annotations

from ._policy import KernelSpec, MissingBenchmarkError, MissingReferenceError
from ._registry import KernelRegistry, registry

# Importing the kernel modules triggers their ``registry.register(...)`` calls.
from . import block_diagonal  # noqa: F401  (side-effect: registration)
from . import composed  # noqa: F401  (side-effect: registration)

__all__ = [
    "KernelSpec",
    "KernelRegistry",
    "MissingBenchmarkError",
    "MissingReferenceError",
    "registry",
]

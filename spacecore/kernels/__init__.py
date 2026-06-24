"""Optimized kernels for SpaceCore, organized into two layers.

This subpackage keeps optimization logic out of the operator class bodies and
groups it by kind:

**Core kernels** — :mod:`spacecore.kernels.core`. The check-free cores of an
operator's apply (or a functional's evaluation): the body that runs once the
public method has validated its boundary. Operators bind one with the
:func:`core_kernels` class decorator; binding is static, so consuming a core
costs nothing at call time. These cores *are* on the default apply/eval path.

**Benchmarked numerical kernels** — :mod:`spacecore.kernels.specs`. Heavier
fast paths described by :class:`KernelSpec`, each tied to a generic reference, an
applicability predicate, a correctness test, and a benchmark. Per ADR-016 they
are *routable* through the single :func:`dispatch` entry point by structural
match on a ``dispatch_key``, gated by :func:`dispatch_mode` (``off`` by default)
and a shape-only memory cost estimate. A spec with no ``dispatch_key`` stays
explicit-entry only.

The public names of both layers are re-exported here for convenience, so
``spacecore.kernels.core_kernels`` and ``spacecore.kernels.KernelSpec`` resolve
without reaching into the subpackages. Importing this package registers every
core-kernel set and every :class:`KernelSpec`.
"""
from __future__ import annotations

from .core import (
    CoreKernelSet,
    core_kernel_names,
    core_kernels,
    get_core_kernels,
    register_core_kernels,
)
from .specs import (
    DispatchAmbiguityError,
    DispatchMode,
    DispatchVerificationError,
    KernelCost,
    KernelRegistry,
    KernelSpec,
    MissingBenchmarkError,
    MissingReferenceError,
    dispatch,
    dispatch_mode,
    effective_mode,
    get_dispatch_mode,
    get_memory_budget_fraction,
    registry,
    set_dispatch_mode,
    set_memory_budget_fraction,
    should_consult_dispatch,
)

__all__ = [
    # Core-kernel layer
    "CoreKernelSet",
    "core_kernels",
    "core_kernel_names",
    "get_core_kernels",
    "register_core_kernels",
    # Benchmarked-spec layer
    "KernelSpec",
    "KernelCost",
    "KernelRegistry",
    "MissingBenchmarkError",
    "MissingReferenceError",
    # Dispatch (ADR-016)
    "DispatchAmbiguityError",
    "DispatchVerificationError",
    "DispatchMode",
    "dispatch",
    "dispatch_mode",
    "effective_mode",
    "get_dispatch_mode",
    "set_dispatch_mode",
    "should_consult_dispatch",
    "get_memory_budget_fraction",
    "set_memory_budget_fraction",
    "registry",
]

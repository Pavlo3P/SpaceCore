"""Benchmarked numerical kernels and their registration policy.

A :class:`KernelSpec` ties a hand-optimized fast path to a generic reference, an
applicability predicate, a correctness-reference test, and a benchmark id. Per
ADR-016 the layer is *routable*: a spec that names a ``dispatch_key`` and claims
exact equivalence (``rtol == atol == 0``) is **dispatch-eligible** and selected
by the :func:`dispatch` entry point through structural match. Dispatch is
``off`` by default, so the catalog stays inert until a key is turned on. Specs
with no ``dispatch_key`` (or loosened tolerances) remain explicit-entry only.
See ``docs/source/design/kernels_policy.rst`` for the contract and
:mod:`spacecore.kernels` for how this layer relates to the core-kernel layer.

Importing this subpackage registers every :class:`KernelSpec`.
"""
from __future__ import annotations

from ._dispatch import (
    DispatchMode,
    DispatchVerificationError,
    dispatch,
    dispatch_mode,
    effective_mode,
    get_dispatch_mode,
    get_memory_budget_fraction,
    set_dispatch_mode,
    set_memory_budget_fraction,
    should_consult_dispatch,
)
from ._batched import CachedStackParts
from ._policy import (
    KernelCost,
    KernelSpec,
    MissingBenchmarkError,
    MissingReferenceError,
)
from ._registry import DispatchAmbiguityError, KernelRegistry, registry

# Importing the kernel modules triggers their ``registry.register(...)`` calls.
from . import block_diagonal  # noqa: F401  (side-effect: registration)
from . import composed  # noqa: F401  (side-effect: registration)
from . import composed_simplify  # noqa: F401  (side-effect: dispatch specs)
from . import block_batched  # noqa: F401  (side-effect: dispatch specs)
from . import stacked_batched  # noqa: F401  (side-effect: dispatch specs)

__all__ = [
    "CachedStackParts",
    "KernelSpec",
    "KernelCost",
    "KernelRegistry",
    "MissingBenchmarkError",
    "MissingReferenceError",
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

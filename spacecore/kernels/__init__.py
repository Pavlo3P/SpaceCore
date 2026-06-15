"""Optimized kernels and their registration policy.

This subpackage hosts hand-optimized fast paths for selected operator and
space families. The submodule exists to make optimization opt-in,
auditable, and bounded.

Policy (see ``docs/source/design/kernels_policy.rst``):

* Every kernel must register a :class:`KernelSpec` that names a *generic*
  reference implementation, an *optimized* implementation, an
  applicability predicate, a correctness-reference test id, and a
  benchmark id.
* Tests verify that the optimized implementation matches the generic
  reference on every applicable generated case.
* Benchmarks confirm the optimization actually wins on its target cases.
* No dispatch or fusion rules are wired in this release. The 0.6.0
  design decision will authorize dispatch; until then, optimized kernels
  are called explicitly by user code or by clearly-scoped call sites
  inside SpaceCore.

The submodule is import-safe: importing :mod:`spacecore.kernels`
registers every kernel definition but does not change any default code
path. Existing :class:`LinOp` apply / rapply paths in
:mod:`spacecore.linop` are unchanged.
"""
from __future__ import annotations

from ._policy import KernelSpec, MissingBenchmarkError, MissingReferenceError
from ._registry import KernelRegistry, registry

# Importing the kernel modules triggers their ``registry.register(...)``
# calls. They sit at the bottom so the public types above are visible
# before circular imports could try to use them.
from . import block_diagonal  # noqa: F401  (side-effect: registration)
from . import composed  # noqa: F401  (side-effect: registration)

__all__ = [
    "KernelSpec",
    "KernelRegistry",
    "MissingBenchmarkError",
    "MissingReferenceError",
    "registry",
]

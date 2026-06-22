"""Core apply / evaluation kernels and their binding rules.

The check-free cores of LinOp ``apply``/``rapply``/``vapply``/``rvapply`` and
Functional ``value``/``grad``/``vvalue``/``vgrad`` live here as concrete
functions (``algebra``, ``dense``, ``diagonal``, ``sparse``, ``functional``),
together with the rules that bind them onto operator classes (``_rules``).

Importing this subpackage registers every core-kernel set. The binding is static
(class-definition time via the :func:`core_kernels` decorator), so consuming a
core costs nothing at call time. See :mod:`spacecore.kernels` for how this layer
relates to the benchmarked-spec layer.
"""
from __future__ import annotations

from ._rules import (
    CoreKernelSet,
    core_kernel_names,
    core_kernels,
    get_core_kernels,
    register_core_kernels,
)

# Importing the concrete kernel modules registers their core-kernel sets.
from . import algebra  # noqa: F401  (side-effect: registration)
from . import dense  # noqa: F401  (side-effect: registration)
from . import diagonal  # noqa: F401  (side-effect: registration)
from . import sparse  # noqa: F401  (side-effect: registration)
from . import functional  # noqa: F401  (side-effect: registration)

__all__ = [
    "CoreKernelSet",
    "core_kernels",
    "core_kernel_names",
    "get_core_kernels",
    "register_core_kernels",
]

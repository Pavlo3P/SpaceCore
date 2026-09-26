"""Check-free operator cores for SpaceCore.

:mod:`spacecore.kernels.core` holds the check-free cores of an operator's apply
(or a functional's evaluation): the body that runs once the public method has
validated its boundary. Operators bind one with the :func:`core_kernels` class
decorator; binding is static, so consuming a core costs nothing at call time.
These cores are on the default apply/eval path.

The benchmarked-spec dispatch layer of ADR-016 was removed in 0.5.0a: its fast
paths applied only to eager NumPy, where they measured mostly slower than the
generic loops, and never under ``jax.jit``.
"""
from __future__ import annotations

from .core import (
    CoreKernelSet,
    core_kernel_names,
    core_kernels,
    get_core_kernels,
    register_core_kernels,
)

__all__ = [
    "CoreKernelSet",
    "core_kernels",
    "core_kernel_names",
    "get_core_kernels",
    "register_core_kernels",
]

"""Benchmark and overhead-diagnostics tooling for SpaceCore.

Importing :mod:`bench` does *not* mutate any global state. Float64 mode
for the optional backends is enabled by :func:`bench.enable_jax_x64` and
:func:`bench.enable_torch_x64`, which the runner calls once per process
before constructing any probe. Tests and library code that import
``bench`` for its types remain unaffected.
"""
from __future__ import annotations


def enable_jax_x64() -> None:
    """Turn on JAX 64-bit mode for fair float64 comparisons.

    JAX defaults to float32 to match TPU hardware; running benchmarks
    in that mode would compare SpaceCore's float64 path against a
    float32 JAX kernel, which is unfair on dtype-promotion-sensitive
    operations. The bench runner calls this once per process before
    constructing any JAX probe.
    """
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass


def enable_torch_x64() -> None:
    """Set Torch's default dtype to float64 for fair comparisons.

    Torch defaults to float32; benchmarking SpaceCore's float64 path
    against a float32 Torch kernel is unfair on dtype-sensitive operations
    (the same argument as :func:`enable_jax_x64`). The runner calls this
    once per process before constructing any Torch probe.

    Apple **MPS is float32-only hardware**, so it cannot honor this; the
    device-aware probe builds its MPS case at float32 explicitly, and the
    correctness gate keeps a float32-width tolerance for MPS only.
    """
    try:
        import torch

        torch.set_default_dtype(torch.float64)
    except ImportError:
        pass

"""Benchmark and overhead-diagnostics tooling for SpaceCore.

Importing :mod:`bench` does *not* mutate any global state. JAX x64 mode
is enabled by :func:`bench.enable_jax_x64`, which is called from the
runner before any JAX probe is constructed. Tests and library code
that import ``bench`` for its types remain unaffected.
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

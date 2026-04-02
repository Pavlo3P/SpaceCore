from __future__ import annotations
import importlib.util
import numpy as np


def has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


def jax_real_dtype():
    if not has_jax():
        return np.float32
    import jax
    return np.float64 if bool(jax.config.read("jax_enable_x64")) else np.float32


def jax_complex_dtype():
    return np.complex128 if jax_real_dtype() == np.float64 else np.complex64


def to_numpy(x):
    if isinstance(x, tuple):
        return tuple(to_numpy(xi) for xi in x)
    return np.asarray(x)


def prod(shape):
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)

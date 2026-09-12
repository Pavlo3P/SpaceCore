"""Shared backend parametrization and context builders for ``tests/linalg``.

The linalg solvers are exercised across every installed backend, so the
per-object test files share one set of ``pytest.param`` lists and a single
context builder rather than redefining them in each module.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_cupy, has_jax, has_torch, jax_real_dtype, torch_real_dtype


def ops_for_backend(name: str):
    """Return a fresh ``BackendOps`` instance for ``name``."""
    if name == "numpy":
        return sc.NumpyOps()
    if name == "jax":
        return sc.JaxOps()
    if name == "torch":
        return sc.TorchOps()
    if name == "cupy":
        return sc.CuPyOps()
    raise ValueError(f"Unknown backend {name!r}.")


def make_ctx(backend_name: str = "numpy", dtype=np.float64, check_level: str = "none"):
    """Build a solver context and arm ``check_level`` for the objects built from it.

    ``check_level`` lives on the bound object rather than on the ``Context``, so the
    requested level is installed as the ambient default that seeds the spaces and
    operators the caller constructs next. Checks default to ``none`` (the solver hot
    path); the autouse fixture in ``tests/conftest.py`` restores the previous ambient
    level after each test.
    """
    sc.set_check_level(check_level)
    return sc.Context(ops_for_backend(backend_name), dtype=dtype)


def backend_params(*, cupy: bool = True):
    """Parametrize ``(backend_name, dtype)`` over every installed backend."""
    params = [
        pytest.param("numpy", np.float64, id="numpy"),
        pytest.param(
            "jax",
            jax_real_dtype(),
            marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
            id="jax",
        ),
        pytest.param(
            "torch",
            torch_real_dtype(),
            marks=pytest.mark.skipif(not has_torch(), reason="torch is not installed"),
            id="torch",
        ),
    ]
    if cupy:
        params.append(
            pytest.param(
                "cupy",
                np.float64,
                marks=pytest.mark.skipif(not has_cupy(), reason="cupy is not installed"),
                id="cupy",
            )
        )
    return params


def numpy_jax_params():
    """Parametrize ``(backend_name, dtype)`` over NumPy and JAX only."""
    return [
        pytest.param("numpy", np.float64, id="numpy"),
        pytest.param(
            "jax",
            jax_real_dtype(),
            marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
            id="jax",
        ),
    ]

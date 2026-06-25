"""Pytest fixtures for backend conformance tests.

These fixtures provide every available ``BackendOps`` instance to the
conformance suite. Tests written against the ``backend_ops`` fixture run
once per installed backend.

Real and complex dtype fixtures yield the standard 32- and 64-bit precision
pairs. A backend that cannot honor a given dtype end-to-end (JAX without
``jax_enable_x64``, Torch's default dtype, etc.) is filtered upstream via
:func:`tests.backend._conformance.backend_supports_dtype`.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests._helpers import has_cupy, has_jax, has_torch


def _all_families() -> list[str]:
    families = ["numpy"]
    if has_jax():
        families.append("jax")
    if has_torch():
        families.append("torch")
    if has_cupy():
        families.append("cupy")
    return families


@pytest.fixture(params=_all_families())
def backend_family(request) -> str:
    """The family name of the backend under test, e.g. ``"numpy"``."""
    return request.param


@pytest.fixture
def backend_ops(backend_family: str):
    """Instantiate ``BackendOps`` for the requested family.

    The returned object is a fresh ``NumpyOps`` / ``JaxOps`` / ``TorchOps`` /
    ``CuPyOps``. Optional families are skipped at fixture-collection time
    when their library is not importable.
    """
    sc = importlib.import_module("spacecore")
    if backend_family == "numpy":
        return sc.NumpyOps()
    if backend_family == "jax":
        if not hasattr(sc, "JaxOps"):
            pytest.skip("JaxOps is not exported (JAX not installed).")
        return sc.JaxOps()
    if backend_family == "torch":
        if not hasattr(sc, "TorchOps"):
            pytest.skip("TorchOps is not exported (Torch not installed).")
        return sc.TorchOps()
    if backend_family == "cupy":
        if not hasattr(sc, "CuPyOps"):
            pytest.skip("CuPyOps is not exported (CuPy not installed).")
        return sc.CuPyOps()
    raise AssertionError(f"unknown backend family {backend_family!r}")


@pytest.fixture(params=[np.float32, np.float64])
def real_dtype(request):
    """A real dtype; both 32- and 64-bit are exercised."""
    return request.param


@pytest.fixture(params=[np.complex64, np.complex128])
def complex_dtype(request):
    """A complex dtype; both 64- and 128-bit are exercised."""
    return request.param


@pytest.fixture(params=[np.float32, np.float64, np.complex64, np.complex128])
def conformance_dtype(request):
    """All four real/complex dtype combinations."""
    return request.param

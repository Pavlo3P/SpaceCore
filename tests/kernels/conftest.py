"""Shared fixtures for the per-object ``tests/kernels/`` suite."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def numpy_ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


@pytest.fixture
def numpy_complex_ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.complex128)


@pytest.fixture
def numpy_f32_ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float32)

"""Shared fixtures for the per-object ``tests/linalg/`` suite."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def numpy_ctx():
    """Float64 NumPy context (default ``standard`` check level)."""
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


@pytest.fixture
def numpy_complex_ctx():
    """Complex128 NumPy context."""
    return sc.Context(sc.NumpyOps(), dtype=np.complex128)

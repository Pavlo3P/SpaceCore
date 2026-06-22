"""Shared fixtures for the per-object ``tests/spaces/`` suite.

These fixtures are intentionally simple — most tests construct their own
spaces, but a few hot dtypes / contexts repeat often enough that a
fixture removes boilerplate.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def numpy_ctx():
    """Float64 NumPy context."""
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


@pytest.fixture
def numpy_complex_ctx():
    """Complex128 NumPy context."""
    return sc.Context(sc.NumpyOps(), dtype=np.complex128)


@pytest.fixture
def numpy_f32_ctx():
    """Float32 NumPy context, useful for testing convert across dtypes."""
    return sc.Context(sc.NumpyOps(), dtype=np.float32)


@pytest.fixture
def weighted_geometry_factory():
    """Build an SPD :class:`spacecore.WeightedInnerProduct` on a context."""

    def make(weights, ctx):
        return sc.WeightedInnerProduct(ctx.asarray(weights))

    return make

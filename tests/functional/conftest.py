"""Shared fixtures for the per-object ``tests/functional/`` suite.

Most tests build their own functionals, but a handful of hot contexts and
spaces repeat often enough that a fixture removes boilerplate. These mirror
the fixtures used by ``tests/spaces`` and ``tests/linops`` so the three
per-object suites read the same way.
"""
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


@pytest.fixture
def numpy_f32_ctx():
    """Float32 NumPy context, useful for testing ``convert`` across dtypes."""
    return sc.Context(sc.NumpyOps(), dtype=np.float32)


@pytest.fixture
def dense_space(numpy_ctx):
    """3-dimensional Euclidean dense coordinate space."""
    return sc.DenseCoordinateSpace((3,), numpy_ctx)


@pytest.fixture
def weighted_space(numpy_ctx):
    """3-dimensional weighted (non-Euclidean) dense coordinate space."""
    weights = numpy_ctx.asarray([2.0, 5.0, 11.0])
    return sc.DenseCoordinateSpace(
        (3,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights)
    )

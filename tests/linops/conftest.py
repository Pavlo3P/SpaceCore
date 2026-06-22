"""Shared fixtures for the per-object ``tests/linops/`` suite."""
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


@pytest.fixture
def dense_vector_space(numpy_ctx):
    return sc.DenseCoordinateSpace((3,), numpy_ctx)


@pytest.fixture
def weighted_vector_space(numpy_ctx):
    weights = numpy_ctx.asarray([2.0, 5.0, 11.0])
    return sc.DenseCoordinateSpace(
        (3,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
    )

"""CuPyOps-specific tests.

Generic operation conformance lives in :mod:`tests.backend.test_operations`;
this module covers behavior that is *only* meaningful for ``CuPyOps``:

* the ``cupy`` family identifier;
* default representation dtype (``float64``);
* GPU memory placement (``dense_array`` is ``cupy.ndarray``);
* the ``cupyx.scipy.sparse`` family used for sparse conversion;
* index-set / index-add round trips on GPU;
* ``__eq__`` / ``__hash__`` / ``__repr__`` of ``CuPyOps`` instances.

Skipped wholesale when CuPy is not importable or no usable CUDA device is
available.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import has_cupy, to_numpy

pytestmark = pytest.mark.skipif(
    not has_cupy(), reason="CuPy is not installed or no usable CUDA device is available"
)


@pytest.fixture
def ops():
    import spacecore as sc

    return sc.CuPyOps()


# ---------------------------------------------------------------------------
# Family and capability flags
# ---------------------------------------------------------------------------
def test_cupy_ops_family_string(ops):
    assert ops.family == "cupy"


def test_cupy_ops_allow_sparse_is_true(ops):
    assert ops.allow_sparse is True


def test_cupy_ops_has_native_vmap_is_false(ops):
    """CuPyOps does not advertise native vmap; uses the fallback loop."""
    assert ops.has_native_vmap is False


# ---------------------------------------------------------------------------
# Dtype defaulting
# ---------------------------------------------------------------------------
def test_cupy_ops_default_dtype_is_float64(ops):
    assert ops.sanitize_dtype(None) == np.float64


def test_cupy_ops_eps_default(ops):
    assert ops.eps(np.float64) == pytest.approx(float(np.finfo(np.float64).eps))


# ---------------------------------------------------------------------------
# Equality, hash, repr
# ---------------------------------------------------------------------------
def test_cupy_ops_equality_and_hash():
    import spacecore as sc

    a = sc.CuPyOps()
    b = sc.CuPyOps()
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1, b: 2} == {a: 2}


def test_cupy_ops_repr():
    import spacecore as sc

    assert "CuPyOps" in repr(sc.CuPyOps())
    assert "family='cupy'" in repr(sc.CuPyOps())


# ---------------------------------------------------------------------------
# GPU memory placement
# ---------------------------------------------------------------------------
def test_cupy_ops_dense_array_is_cupy_ndarray(ops):
    import cupy as cp

    assert ops.dense_array is cp.ndarray


def test_cupy_ops_zeros_returns_cupy_ndarray(ops):
    import cupy as cp

    x = ops.zeros((3, 3))
    assert isinstance(x, cp.ndarray)


# ---------------------------------------------------------------------------
# Sparse: cupyx.scipy.sparse
# ---------------------------------------------------------------------------
def test_cupy_ops_sparse_uses_cupyx_scipy_sparse(ops):
    import cupyx.scipy.sparse as css

    dense = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    sparse = ops.assparse(ops.asarray(dense))
    assert css.issparse(sparse)


def test_cupy_ops_sparse_matmul_round_trip(ops):
    dense = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sparse = ops.assparse(ops.asarray(dense))
    x = ops.asarray([7.0, 8.0])
    expected = dense @ np.asarray([7.0, 8.0])
    np.testing.assert_allclose(to_numpy(ops.sparse_matmul(sparse, x)), expected)


def test_cupy_ops_allclose_sparse_true(ops):
    dense = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a = ops.assparse(ops.asarray(dense))
    b = ops.assparse(ops.asarray(dense))
    assert ops.allclose_sparse(a, b) is True


# ---------------------------------------------------------------------------
# Index mutation on GPU
# ---------------------------------------------------------------------------
def test_cupy_ops_index_set_round_trip(ops):
    x = ops.asarray([1.0, 2.0, 3.0])
    y = ops.index_set(x, 1, ops.asarray(5.0), copy=True)
    np.testing.assert_allclose(to_numpy(y), [1.0, 5.0, 3.0])
    # Original unchanged.
    np.testing.assert_allclose(to_numpy(x), [1.0, 2.0, 3.0])


def test_cupy_ops_index_add_round_trip(ops):
    x = ops.asarray([1.0, 2.0, 3.0])
    z = ops.index_add(x, 0, ops.asarray(2.0), copy=True)
    np.testing.assert_allclose(to_numpy(z), [3.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Constants and astype(None)
# ---------------------------------------------------------------------------
def test_cupy_ops_constants_are_cached(ops):
    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e


def test_cupy_ops_astype_none_is_identity(ops):
    x = ops.asarray([1.0, 2.0])
    assert ops.astype(x, None) is x

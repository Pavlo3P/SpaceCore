"""NumpyOps-specific tests.

Generic operation conformance lives in :mod:`tests.backend.test_operations`;
this module covers behavior that is *only* meaningful for ``NumpyOps``:

* the ``numpy`` family identifier and the ``array_api_compat.numpy`` xp
  namespace;
* default representation dtype (``float64``);
* the sparse format SpaceCore returns from ``assparse`` (scipy CSR);
* ``__eq__`` / ``__hash__`` / ``__repr__`` of ``NumpyOps`` instances;
* the ``allow_sparse=True`` / ``has_native_vmap=False`` capability flags.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def ops():
    return sc.NumpyOps()


# ---------------------------------------------------------------------------
# Family and capability flags
# ---------------------------------------------------------------------------
def test_numpy_ops_family_string(ops):
    assert ops.family == "numpy"


def test_numpy_ops_allow_sparse_is_true(ops):
    assert ops.allow_sparse is True


def test_numpy_ops_has_native_vmap_is_false(ops):
    assert ops.has_native_vmap is False


def test_numpy_ops_xp_is_array_api_compat_numpy():
    assert sc.NumpyOps.xp.__name__ == "array_api_compat.numpy"


# ---------------------------------------------------------------------------
# Dtype defaulting
# ---------------------------------------------------------------------------
def test_numpy_ops_default_dtype_is_float64(ops):
    assert ops.sanitize_dtype(None) == np.float64


def test_numpy_ops_eps_distinguishes_precision(ops):
    assert ops.eps(np.float64) < ops.eps(np.float32)


# ---------------------------------------------------------------------------
# Equality, hash, repr
# ---------------------------------------------------------------------------
def test_numpy_ops_equality_and_hash():
    a = sc.NumpyOps()
    b = sc.NumpyOps()
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1, b: 2} == {a: 2}


def test_numpy_ops_repr():
    assert repr(sc.NumpyOps()) == "NumpyOps(family='numpy')"


def test_numpy_ops_not_equal_to_other_family():
    if hasattr(sc, "JaxOps"):
        assert sc.NumpyOps() != sc.JaxOps()
    assert sc.NumpyOps() != "numpy"


# ---------------------------------------------------------------------------
# Sparse format
# ---------------------------------------------------------------------------
def test_numpy_ops_sparse_format_is_scipy_csr(ops):
    import scipy.sparse as sps

    dense = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    sparse = ops.assparse(ops.asarray(dense))
    assert sps.issparse(sparse) and sparse.format == "csr"


def test_numpy_ops_sparse_array_types_present(ops):
    """``sparse_array`` exposes the SciPy sparse classes SpaceCore accepts."""
    types = ops.sparse_array
    assert types is not None
    assert all(isinstance(t, type) for t in types)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_numpy_ops_assparse_format_kwarg(ops, fmt):
    dense = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    out = ops.assparse(ops.asarray(dense), format=fmt)
    assert out.format == fmt


def test_numpy_ops_sparse_array_concrete_scipy_classes(ops):
    """backend-002: pin the CONCRETE scipy.sparse base classes, not just types."""
    import scipy.sparse as sps

    types = ops.sparse_array
    assert sps.spmatrix in types
    assert sps.sparray in types


# ---------------------------------------------------------------------------
# assparse on an EXISTING scipy sparse input (numpy/_ops.py ~134-143)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("out_fmt", ["csr", "csc", "coo"])
def test_numpy_ops_assparse_reformats_existing_sparse(ops, out_fmt):
    """gap-7: assparse on an existing sparse input re-emits the requested format.

    Start from a CSR sparse input and request each format; the returned
    object must carry the requested format and the same values.
    """
    import scipy.sparse as sps

    dense = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    existing = sps.csr_matrix(dense)
    out = ops.assparse(existing, format=out_fmt)
    assert out.format == out_fmt
    np.testing.assert_allclose(out.toarray(), dense)


def test_numpy_ops_assparse_existing_sparse_dtype_cast(ops):
    """gap-7: an astype/dtype change is applied to an existing sparse input."""
    import scipy.sparse as sps

    dense = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    existing = sps.csr_matrix(dense)
    assert existing.dtype == np.dtype(np.float32)
    out = ops.assparse(existing, format="csr", dtype=np.float64)
    assert out.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(out.toarray(), dense.astype(np.float64))


def test_numpy_ops_assparse_existing_sparse_unknown_format_raises(ops):
    """gap-7: an unknown format on an existing sparse input raises ValueError."""
    import scipy.sparse as sps

    existing = sps.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 2.0]]))
    with pytest.raises(ValueError, match="Unknown sparse format"):
        ops.assparse(existing, format="bogus")


# ---------------------------------------------------------------------------
# eigh rejects a scipy sparse matrix (spacecore/backend/_ops.py ~699-700)
# ---------------------------------------------------------------------------
def test_numpy_ops_eigh_rejects_scipy_sparse(ops):
    """gap-2: building a scipy sparse matrix and calling eigh raises TypeError."""
    import scipy.sparse as sps

    A = sps.csr_matrix(np.asarray([[2.0, 0.0], [0.0, 3.0]]))
    with pytest.raises(
        TypeError, match="eigh requires a dense array; sparse input is not supported."
    ):
        ops.eigh(A)


# ---------------------------------------------------------------------------
# scan xs=None with length=None raises the exact ValueError (numpy/_ops.py ~385)
# ---------------------------------------------------------------------------
def test_numpy_ops_scan_xs_none_requires_length(ops):
    """gap-5: scan(xs=None) without a length raises the documented ValueError."""
    init = ops.asarray(np.asarray(0.0))

    def body(carry, _x):
        return carry + 1.0, carry

    with pytest.raises(ValueError, match=r"scan\(xs=None\) requires an explicit `length`\."):
        ops.scan(body, init, None, length=None)


# ---------------------------------------------------------------------------
# Constants caching and astype(None)
# ---------------------------------------------------------------------------
def test_numpy_ops_constants_are_cached(ops):
    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e


def test_numpy_ops_astype_none_is_identity(ops):
    x = ops.asarray([1.0, 2.0])
    assert ops.astype(x, None) is x

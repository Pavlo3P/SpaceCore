"""Conversion-path conformance pins (Phase J11).

This module pins the cross-backend conversion paths exposed by
``spacecore.Context``:

* Round-trip through ``np.asarray(ctx.asarray(...))`` preserves values and
  dtype.
* ``DenseCoordinateSpace.convert(new_ctx)`` moves a space between backend
  families and round-trips back, along with one of its elements.
* Sparse conversion via ``ctx.assparse`` accepts a SciPy CSR input on every
  backend whose ``ops.allow_sparse`` is true, and otherwise raises.
* ``Context.asarray`` refuses an implicit complex-to-real cast (ADR-015
  Stage 1), but accepts the reverse real-to-complex broadening.
* ``ops.asarray(arr, dtype=...)`` returns the requested dtype unchanged.

Tests parametrize over the shared ``backend_ops`` fixture from
``tests/backend/conftest.py``. Backends not installed are skipped at
fixture collection time; dtypes a backend cannot honor end-to-end are
filtered via :func:`tests.backend._conformance.backend_supports_dtype`.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import scipy.sparse as sps

from tests._helpers import has_jax, to_numpy
from tests.backend._conformance import (
    assert_matches_reference,
    backend_supports_dtype,
)


@pytest.fixture
def numpy_ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


def _skip_if_unsupported(backend_ops, dtype) -> None:
    """Skip when the backend cannot honor ``dtype`` end-to-end."""
    if not backend_supports_dtype(backend_ops.family, dtype):
        pytest.skip(f"{backend_ops.family} does not natively support {np.dtype(dtype)}")


def _make_ctx(ops, dtype):
    sc = importlib.import_module("spacecore")
    return sc.Context(ops, dtype=dtype)


def _pick_real_dtype(family) -> np.dtype | None:
    """Return a real dtype the backend honors end-to-end, or ``None``."""
    for dt in (np.float64, np.float32):
        if backend_supports_dtype(family, dt):
            return dt
    return None


def _densify(out):
    """Materialize a backend sparse object as a NumPy ndarray."""
    if hasattr(out, "toarray"):
        return np.asarray(to_numpy(out.toarray()))
    if hasattr(out, "to_dense"):
        return np.asarray(to_numpy(out.to_dense()))
    if hasattr(out, "todense"):
        return np.asarray(out.todense())
    return None


# Round-trip: ctx.asarray -> np.asarray preserves values and dtype --------


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_asarray_roundtrip_preserves_values_and_dtype(backend_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    if np.dtype(dtype).kind == "c":
        src = np.asarray([1 + 2j, 3 - 4j, 5 + 0j], dtype=dtype)
    else:
        src = np.asarray([1.0, -2.0, 3.5], dtype=dtype)
    out = _make_ctx(backend_ops, dtype).asarray(src)
    back = to_numpy(out)
    assert np.dtype(back.dtype) == np.dtype(dtype)
    assert_matches_reference("asarray", out, src, dtype=dtype)


# Cross-backend convert: arrays and DenseCoordinateSpace elements ---------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cross_backend_array_convert_numpy_jax_roundtrip(numpy_ops, dtype):
    """NumPy -> JAX -> NumPy round-trip preserves values and dtype.

    Cross-backend transfer goes through a NumPy intermediate: each
    ``Context`` is responsible only for its own backend, so we materialize
    the source as a NumPy array (via ``to_numpy``) before handing it to
    the destination ``Context.asarray``. ``Context.convert`` is then
    exercised in its same-backend role for symmetry with the public API.
    """
    if not has_jax():
        pytest.skip("JAX not installed.")
    sc = importlib.import_module("spacecore")
    jax_ops = sc.JaxOps()
    if not backend_supports_dtype("jax", dtype):
        pytest.skip(f"JAX does not natively support {np.dtype(dtype)}")

    src = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    ctx_np = _make_ctx(numpy_ops, dtype)
    ctx_jx = _make_ctx(jax_ops, dtype)

    arr_np = ctx_np.asarray(src)
    arr_jx = ctx_jx.asarray(to_numpy(arr_np))
    assert jax_ops.is_dense(arr_jx)
    # Same-context convert: asarray on its own array (no-op-equivalent).
    assert jax_ops.is_dense(ctx_jx.convert(arr_jx))
    assert_matches_reference("asarray", arr_jx, src, dtype=dtype)

    arr_back = ctx_np.asarray(to_numpy(arr_jx))
    assert numpy_ops.is_dense(arr_back)
    assert np.dtype(arr_back.dtype) == np.dtype(dtype)
    assert_matches_reference("asarray", arr_back, src, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cross_backend_dense_coord_space_convert(numpy_ops, dtype):
    """``DenseCoordinateSpace.convert`` moves between NumPy and JAX losslessly."""
    if not has_jax():
        pytest.skip("JAX not installed.")
    sc = importlib.import_module("spacecore")
    jax_ops = sc.JaxOps()
    if not backend_supports_dtype("jax", dtype):
        pytest.skip(f"JAX does not natively support {np.dtype(dtype)}")

    ctx_np = _make_ctx(numpy_ops, dtype)
    ctx_jx = _make_ctx(jax_ops, dtype)

    space_np = sc.DenseCoordinateSpace((3,), ctx=ctx_np)
    space_jx = space_np.convert(ctx_jx)
    assert space_jx.ctx == ctx_jx
    assert space_jx.shape == space_np.shape
    assert space_jx.convert(ctx_np) == space_np

    # Move an element across the same path; staging through ``to_numpy`` so
    # each Context only sees its own backend's arrays.
    src = np.asarray([1.0, 2.0, 3.0], dtype=dtype)
    x_np = ctx_np.asarray(src)
    x_jx = ctx_jx.asarray(to_numpy(x_np))
    x_back = ctx_np.asarray(to_numpy(x_jx))
    assert_matches_reference("asarray", x_back, src, dtype=dtype)


# Sparse conversion (scipy.sparse.csr_matrix input) -----------------------


def test_assparse_from_scipy_csr_when_allowed(backend_ops):
    """``ctx.assparse(scipy_csr)`` succeeds on every sparse-capable backend.

    When a backend reports ``allow_sparse=False`` we expect the
    corresponding ``Context`` to refuse via ``Context.assert_sparse``.
    """
    dtype = _pick_real_dtype(backend_ops.family)
    if dtype is None:
        pytest.skip(f"{backend_ops.family} has no supported real dtype")
    dense = np.eye(3, dtype=dtype)
    csr = sps.csr_matrix(dense)
    ctx = _make_ctx(backend_ops, dtype)

    if not backend_ops.allow_sparse:
        with pytest.raises((TypeError, NotImplementedError)):
            ctx.assert_sparse(ctx.assparse(csr))
        return

    out = ctx.assparse(csr)
    assert backend_ops.is_sparse(out)
    dense_out = _densify(out)
    if dense_out is None:  # pragma: no cover - defensive guard
        pytest.skip(f"no known densification path for {type(out).__name__}")
    assert_matches_reference("asarray", dense_out, dense, dtype=dtype)


# Complex-to-real refusal (ADR-015 Stage 1) -------------------------------


@pytest.mark.parametrize(
    "via",
    ["ops", "ctx"],
    ids=["ops.asarray", "Context.asarray"],
)
def test_asarray_refuses_complex_to_real(backend_ops, via):
    """A complex source array cannot be cast to a real dtype implicitly."""
    if not backend_supports_dtype(backend_ops.family, np.float64):
        pytest.skip(f"{backend_ops.family} does not natively support float64")
    if not backend_supports_dtype(backend_ops.family, np.complex128):
        pytest.skip(f"{backend_ops.family} does not natively support complex128")

    carr = np.asarray([1 + 2j, 3 - 4j], dtype=np.complex128)
    with pytest.raises(TypeError, match="complex"):
        if via == "ops":
            backend_ops.asarray(carr, dtype=np.float64)
        else:
            _make_ctx(backend_ops, np.float64).asarray(carr)


# Real-to-complex broadening works everywhere -----------------------------


@pytest.mark.parametrize(
    "real_dt,complex_dt",
    [(np.float32, np.complex64), (np.float64, np.complex128)],
)
def test_asarray_real_to_complex_broadening(backend_ops, real_dt, complex_dt):
    _skip_if_unsupported(backend_ops, real_dt)
    _skip_if_unsupported(backend_ops, complex_dt)

    src = np.asarray([1.0, -2.0, 3.5], dtype=real_dt)
    out = backend_ops.asarray(src, dtype=complex_dt)
    back = to_numpy(out)
    assert np.dtype(back.dtype) == np.dtype(complex_dt)
    assert_matches_reference("asarray", out, src.astype(complex_dt), dtype=complex_dt)


# Dtype passthrough -------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "via",
    ["ops", "ctx"],
    ids=["ops.asarray(dtype=)", "Context.asarray"],
)
def test_asarray_dtype_passthrough(backend_ops, dtype, via):
    """The returned array carries the requested / configured dtype exactly."""
    _skip_if_unsupported(backend_ops, dtype)
    if np.dtype(dtype).kind == "c":
        src = [1 + 1j, 2 + 0j, 3 - 1j]
    else:
        src = [1.0, 2.0, 3.0]
    if via == "ops":
        out = backend_ops.asarray(src, dtype=dtype)
    else:
        out = _make_ctx(backend_ops, dtype).asarray(src)
    back = to_numpy(out)
    assert np.dtype(back.dtype) == np.dtype(dtype)

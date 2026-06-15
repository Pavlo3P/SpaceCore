"""Cross-backend conformance against the NumpyOps reference.

The ``backend_ops`` fixture (from ``tests/backend/conftest.py``) is
parametrized over every installed backend family. Each test computes the
operation on the backend under test and compares against ``NumpyOps`` as
the reference, applying the per-op tolerance from
``tests/backend/_conformance.py``.

Backends that are not installed are skipped at fixture-collection time.
Dtypes the backend cannot honor end-to-end are filtered via
:func:`tests.backend._conformance.backend_supports_dtype`; this avoids
shipping spurious failures when JAX is built without ``jax_enable_x64`` or
Torch's default dtype is ``float32``.

This module is the J3–J5 implementation for the
backend-conformance phase. J2's ``test_conformance_numpy.py`` pins NumPy
behavior against ``numpy`` / ``numpy.linalg``; this module pins JAX, Torch,
and CuPy behavior against ``NumpyOps``.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests._helpers import to_numpy
from tests.backend._conformance import (
    assert_eigh_identity,
    assert_matches_reference,
    backend_supports_dtype,
)


@pytest.fixture
def numpy_ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


def _skip_if_unsupported(backend_ops, dtype):
    if not backend_supports_dtype(backend_ops.family, dtype):
        pytest.skip(f"{backend_ops.family} does not natively support {np.dtype(dtype)}")


# ---------------------------------------------------------------------------
# Construction


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_asarray_matches_numpy(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = [1.0, 2.0, 3.0] if np.dtype(dtype).kind != "c" else [1 + 2j, 3 + 4j, 5 + 6j]
    ref = numpy_ops.asarray(src, dtype=dtype)
    out = backend_ops.asarray(src, dtype=dtype)
    assert_matches_reference("asarray", out, to_numpy(ref), dtype=dtype)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_zeros_ones_eye(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    assert_matches_reference(
        "zeros", backend_ops.zeros((2, 3), dtype=dtype), to_numpy(numpy_ops.zeros((2, 3), dtype=dtype)), dtype=dtype
    )
    assert_matches_reference(
        "ones", backend_ops.ones((2, 3), dtype=dtype), to_numpy(numpy_ops.ones((2, 3), dtype=dtype)), dtype=dtype
    )
    assert_matches_reference(
        "eye", backend_ops.eye(3, dtype=dtype), to_numpy(numpy_ops.eye(3, dtype=dtype)), dtype=dtype
    )


# ---------------------------------------------------------------------------
# Layout


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reshape_transpose_stack(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    arr = np.arange(6, dtype=dtype).reshape(2, 3)
    x_be = backend_ops.asarray(arr, dtype=dtype)
    x_np = numpy_ops.asarray(arr, dtype=dtype)
    assert_matches_reference(
        "reshape", backend_ops.reshape(x_be, (3, 2)), to_numpy(numpy_ops.reshape(x_np, (3, 2))), dtype=dtype
    )
    assert_matches_reference(
        "transpose", backend_ops.transpose(x_be), to_numpy(numpy_ops.transpose(x_np)), dtype=dtype
    )
    a = backend_ops.asarray([1.0, 2.0], dtype=dtype)
    b = backend_ops.asarray([3.0, 4.0], dtype=dtype)
    aN = numpy_ops.asarray([1.0, 2.0], dtype=dtype)
    bN = numpy_ops.asarray([3.0, 4.0], dtype=dtype)
    assert_matches_reference(
        "stack", backend_ops.stack([a, b], axis=0), to_numpy(numpy_ops.stack([aN, bN], axis=0)), dtype=dtype
    )


# ---------------------------------------------------------------------------
# Elementwise / reductions


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_elementwise_unary(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    y = numpy_ops.asarray(src, dtype=dtype)
    for op_name in ("abs", "sign", "sqrt"):
        if op_name == "sqrt":
            src_pos = np.abs(src)
            x = backend_ops.asarray(src_pos, dtype=dtype)
            y = numpy_ops.asarray(src_pos, dtype=dtype)
        assert_matches_reference(
            op_name, getattr(backend_ops, op_name)(x), to_numpy(getattr(numpy_ops, op_name)(y)), dtype=dtype
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_conj_real_imag_complex(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([1 + 2j, 3 - 1j, -2 + 5j], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    y = numpy_ops.asarray(src, dtype=dtype)
    for op_name in ("conj", "real", "imag"):
        out_dtype = backend_ops.real_dtype(dtype) if op_name in ("real", "imag") else dtype
        assert_matches_reference(
            op_name,
            getattr(backend_ops, op_name)(x),
            to_numpy(getattr(numpy_ops, op_name)(y)),
            dtype=out_dtype,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_reductions(backend_ops, numpy_ops, dtype, axis):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    y = numpy_ops.asarray(src, dtype=dtype)
    for op_name in ("sum", "mean", "min", "max"):
        assert_matches_reference(
            op_name,
            getattr(backend_ops, op_name)(x, axis=axis),
            to_numpy(getattr(numpy_ops, op_name)(y, axis=axis)),
            dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logsumexp(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    y = numpy_ops.asarray(src, dtype=dtype)
    for axis in (None, 0, 1):
        assert_matches_reference(
            "logsumexp",
            backend_ops.logsumexp(x, axis=axis),
            to_numpy(numpy_ops.logsumexp(y, axis=axis)),
            dtype=dtype,
        )


# ---------------------------------------------------------------------------
# Linear algebra


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_matmul(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    B = np.asarray([[5.0, 0.0], [1.0, 2.0]], dtype=dtype)
    xa = backend_ops.asarray(A, dtype=dtype)
    xb = backend_ops.asarray(B, dtype=dtype)
    ya = numpy_ops.asarray(A, dtype=dtype)
    yb = numpy_ops.asarray(B, dtype=dtype)
    assert_matches_reference(
        "matmul", backend_ops.matmul(xa, xb), to_numpy(numpy_ops.matmul(ya, yb)), dtype=dtype
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_vdot(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    if np.dtype(dtype).kind == "c":
        a = np.asarray([1 + 1j, 2 - 1j], dtype=dtype)
        b = np.asarray([3 + 0j, 1 - 2j], dtype=dtype)
    else:
        a = np.asarray([1.0, 2.0], dtype=dtype)
        b = np.asarray([3.0, 1.0], dtype=dtype)
    assert_matches_reference(
        "vdot",
        backend_ops.vdot(backend_ops.asarray(a, dtype=dtype), backend_ops.asarray(b, dtype=dtype)),
        to_numpy(numpy_ops.vdot(numpy_ops.asarray(a, dtype=dtype), numpy_ops.asarray(b, dtype=dtype))),
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eigh_identity(backend_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
    assert_eigh_identity(backend_ops, backend_ops.asarray(A, dtype=dtype), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solve_norm_cholesky(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
    b = np.asarray([1.0, 2.0], dtype=dtype)
    xA = backend_ops.asarray(A, dtype=dtype)
    xb = backend_ops.asarray(b, dtype=dtype)
    yA = numpy_ops.asarray(A, dtype=dtype)
    yb = numpy_ops.asarray(b, dtype=dtype)
    assert_matches_reference(
        "solve", backend_ops.solve(xA, xb), to_numpy(numpy_ops.solve(yA, yb)), dtype=dtype
    )
    assert_matches_reference(
        "norm", backend_ops.norm(xA), to_numpy(numpy_ops.norm(yA)), dtype=dtype
    )
    assert_matches_reference(
        "cholesky", backend_ops.cholesky(xA), to_numpy(numpy_ops.cholesky(yA)), dtype=dtype
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_svd(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    M = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=dtype)
    x = backend_ops.asarray(M, dtype=dtype)
    y = numpy_ops.asarray(M, dtype=dtype)
    u_be, s_be, vh_be = backend_ops.svd(x, full_matrices=False)
    u_np, s_np, vh_np = numpy_ops.svd(y, full_matrices=False)
    assert_matches_reference("svd", s_be, to_numpy(s_np), dtype=dtype)
    recon_be = backend_ops.matmul(backend_ops.matmul(u_be, backend_ops.diag(s_be)), vh_be)
    recon_np = numpy_ops.matmul(numpy_ops.matmul(u_np, numpy_ops.diag(s_np)), vh_np)
    assert_matches_reference("svd", recon_be, to_numpy(recon_np), dtype=dtype)


# ---------------------------------------------------------------------------
# Indexing / triangular


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_take_diag_tril_triu(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.arange(9, dtype=dtype).reshape(3, 3)
    x = backend_ops.asarray(src, dtype=dtype)
    y = numpy_ops.asarray(src, dtype=dtype)
    idx_be = backend_ops.asarray(np.asarray([0, 2], dtype=np.int64))
    idx_np = numpy_ops.asarray(np.asarray([0, 2], dtype=np.int64))
    assert_matches_reference(
        "take", backend_ops.take(x, idx_be, axis=1), to_numpy(numpy_ops.take(y, idx_np, axis=1)), dtype=dtype
    )
    assert_matches_reference("diag", backend_ops.diag(x), to_numpy(numpy_ops.diag(y)), dtype=dtype)
    assert_matches_reference("tril", backend_ops.tril(x), to_numpy(numpy_ops.tril(y)), dtype=dtype)
    assert_matches_reference("triu", backend_ops.triu(x), to_numpy(numpy_ops.triu(y)), dtype=dtype)

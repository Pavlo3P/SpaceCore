"""NumpyOps systematic comparison against numpy / numpy.linalg.

This module pins ``NumpyOps`` as the conformance reference. Every public
``BackendOps`` row in :doc:`docs/source/design/backend_conformance` that
lists ``test_conformance_numpy.py`` has a check here.

A few rows are exercised at the matrix-cell granularity rather than per
shape/dtype combination: predicates, dtype helpers, and constants. Where
``np.allclose`` would mask integer-typed mismatches,
``assert_matches_reference`` falls back to exact equality.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import scipy.special

from tests.backend._conformance import (
    assert_eigh_identity,
    assert_matches_reference,
    numpy_reference,
)


@pytest.fixture
def ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


# ---------------------------------------------------------------------------
# Metadata / predicates


def test_family_and_native_vmap(ops):
    assert ops.family == "numpy"
    assert ops.has_native_vmap is False
    assert ops.allow_sparse is True


def test_is_dense_is_sparse_is_array(ops):
    import scipy.sparse as sp

    dense = np.asarray([1.0, 2.0])
    sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    assert ops.is_dense(dense) is True
    assert ops.is_sparse(dense) is False
    assert ops.is_array(dense) is True
    assert ops.is_dense(sparse) is False
    assert ops.is_sparse(sparse) is True
    assert ops.is_array(sparse) is True
    assert ops.is_array(1.0) is False
    assert ops.is_array("nope") is False


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.complex64, np.complex128, np.int32, np.int64],
)
def test_get_dtype_shape_ndim_size(ops, dtype):
    x = ops.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    assert np.dtype(ops.get_dtype(x)) == np.dtype(dtype)
    assert ops.shape(x) == (2, 3)
    assert ops.ndim(x) == 2
    assert ops.size(x) == 6


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.float32, False),
        (np.float64, False),
        (np.complex64, True),
        (np.complex128, True),
        (np.int32, False),
    ],
)
def test_is_complex_dtype(ops, dtype, expected):
    assert ops.is_complex_dtype(dtype) is expected


@pytest.mark.parametrize(
    "complex_dtype,real_dtype",
    [(np.complex64, np.float32), (np.complex128, np.float64)],
)
def test_real_dtype_for_complex(ops, complex_dtype, real_dtype):
    assert np.dtype(ops.real_dtype(complex_dtype)) == np.dtype(real_dtype)


def test_real_dtype_for_real(ops):
    assert np.dtype(ops.real_dtype(np.float32)) == np.dtype(np.float32)
    assert np.dtype(ops.real_dtype(np.float64)) == np.dtype(np.float64)


# ---------------------------------------------------------------------------
# Construction and dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_asarray_round_trip(ops, dtype):
    src = np.asarray([1.0, 2.0, 3.0], dtype=dtype)
    out = ops.asarray(src, dtype=dtype)
    assert_matches_reference("asarray", out, src, dtype=dtype)


@pytest.mark.parametrize("src_dtype,dst_dtype", [
    (np.float32, np.float64),
    (np.float64, np.float32),
    (np.complex64, np.complex128),
    (np.float64, np.complex128),
])
def test_astype(ops, src_dtype, dst_dtype):
    x = ops.asarray([1.0, 2.0, 3.0], dtype=src_dtype)
    out = ops.astype(x, dst_dtype)
    assert np.dtype(ops.get_dtype(out)) == np.dtype(dst_dtype)
    assert_matches_reference(
        "astype", out, np.asarray([1.0, 2.0, 3.0], dtype=dst_dtype), dtype=dst_dtype
    )


def test_asarray_rejects_complex_to_real(ops):
    z = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128)
    with pytest.raises(Exception):
        ops.asarray(z, dtype=np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_zeros_ones_full(ops, dtype):
    assert_matches_reference("zeros", ops.zeros((2, 3), dtype=dtype), np.zeros((2, 3), dtype=dtype), dtype=dtype)
    assert_matches_reference("ones", ops.ones((2, 3), dtype=dtype), np.ones((2, 3), dtype=dtype), dtype=dtype)
    assert_matches_reference(
        "full", ops.full((2, 3), 7.0, dtype=dtype), np.full((2, 3), 7.0, dtype=dtype), dtype=dtype
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_zeros_like_ones_like_full_like(ops, dtype):
    x = ops.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    assert_matches_reference(
        "zeros_like", ops.zeros_like(x), np.zeros((2, 2), dtype=dtype), dtype=dtype
    )
    assert_matches_reference(
        "ones_like", ops.ones_like(x), np.ones((2, 2), dtype=dtype), dtype=dtype
    )
    assert_matches_reference(
        "full_like", ops.full_like(x, 9.0), np.full((2, 2), 9.0, dtype=dtype), dtype=dtype
    )


def test_arange_and_eye(ops):
    assert_matches_reference(
        "eye", ops.eye(4, 3, dtype=np.float64), np.eye(4, 3, dtype=np.float64), dtype=np.float64
    )
    out = ops.arange(0, 10, 2, dtype=np.float64)
    assert_matches_reference(
        "arange", out, np.arange(0, 10, 2, dtype=np.float64), dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Shape and layout


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ravel_reshape_transpose(ops, dtype):
    x = ops.asarray(np.arange(6, dtype=dtype).reshape(2, 3), dtype=dtype)
    ref = np.arange(6, dtype=dtype).reshape(2, 3)
    assert_matches_reference("ravel", ops.ravel(x), np.ravel(ref), dtype=dtype)
    assert_matches_reference("reshape", ops.reshape(x, (3, 2)), np.reshape(ref, (3, 2)), dtype=dtype)
    assert_matches_reference("transpose", ops.transpose(x), np.transpose(ref), dtype=dtype)


def test_stack_and_concatenate(ops):
    a = ops.asarray([1.0, 2.0], dtype=np.float64)
    b = ops.asarray([3.0, 4.0], dtype=np.float64)
    assert_matches_reference(
        "stack", ops.stack([a, b], axis=0), np.stack([np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])], axis=0), dtype=np.float64
    )
    assert_matches_reference(
        "concatenate",
        ops.concatenate([a, b], axis=0),
        np.concatenate([np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])], axis=0),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Elementwise


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_elementwise_unary(ops, dtype):
    src = np.asarray([1 + 2j, -3 + 4j], dtype=dtype if np.dtype(dtype).kind == "c" else np.complex128)
    if np.dtype(dtype).kind != "c":
        src = src.real.astype(dtype)
    x = ops.asarray(src, dtype=dtype)
    for op_name in ("conj", "real", "imag", "abs", "sign"):
        ref = numpy_reference(op_name)(src)
        assert_matches_reference(op_name, getattr(ops, op_name)(x), ref, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sqrt_exp_log(ops, dtype):
    src = np.asarray([0.25, 1.0, 4.0], dtype=dtype)
    x = ops.asarray(src, dtype=dtype)
    assert_matches_reference("sqrt", ops.sqrt(x), np.sqrt(src), dtype=dtype)
    assert_matches_reference("exp", ops.exp(x), np.exp(src), dtype=dtype)
    assert_matches_reference("log", ops.log(x), np.log(src), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_sqrt_complex(ops, dtype):
    src = np.asarray([-1.0 + 0j, 4.0 + 3.0j, 0.0 + 1.0j], dtype=dtype)
    x = ops.asarray(src, dtype=dtype)
    assert_matches_reference("sqrt", ops.sqrt(x), np.sqrt(src), dtype=dtype)


# ---------------------------------------------------------------------------
# Reductions


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
def test_sum_mean_min_max_prod(ops, dtype, axis):
    src = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
    x = ops.asarray(src, dtype=dtype)
    for op_name in ("sum", "mean", "min", "max", "prod"):
        ref = numpy_reference(op_name)(src, axis=axis)
        out = getattr(ops, op_name)(x, axis=axis)
        assert_matches_reference(op_name, out, ref, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_argmin_argmax_argsort_sort(ops, dtype):
    src = np.asarray([3.0, 1.0, 4.0, 1.0, 5.0], dtype=dtype)
    x = ops.asarray(src, dtype=dtype)
    assert_matches_reference("argmin", ops.argmin(x), np.argmin(src), dtype=dtype)
    assert_matches_reference("argmax", ops.argmax(x), np.argmax(src), dtype=dtype)
    assert_matches_reference("argsort", ops.argsort(x), np.argsort(src), dtype=dtype)
    assert_matches_reference("sort", ops.sort(x), np.sort(src), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logsumexp(ops, dtype):
    src = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=dtype)
    x = ops.asarray(src, dtype=dtype)
    for axis in (None, 0, 1):
        ref = scipy.special.logsumexp(src, axis=axis)
        assert_matches_reference("logsumexp", ops.logsumexp(x, axis=axis), ref, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isfinite_isnan(ops, dtype):
    src = np.asarray([1.0, np.nan, np.inf, -np.inf], dtype=dtype)
    x = ops.asarray(src, dtype=dtype)
    assert np.array_equal(np.asarray(ops.isfinite(x)), np.isfinite(src))
    assert np.array_equal(np.asarray(ops.isnan(x)), np.isnan(src))


# ---------------------------------------------------------------------------
# Linear algebra


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_vdot_matmul(ops, dtype):
    if np.dtype(dtype).kind == "c":
        a = np.asarray([1 + 1j, 2 - 1j], dtype=dtype)
        b = np.asarray([3 + 0j, 1 - 2j], dtype=dtype)
    else:
        a = np.asarray([1.0, 2.0], dtype=dtype)
        b = np.asarray([3.0, 1.0], dtype=dtype)
    x = ops.asarray(a, dtype=dtype)
    y = ops.asarray(b, dtype=dtype)
    assert_matches_reference("vdot", ops.vdot(x, y), np.vdot(a, b), dtype=dtype)
    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    B = np.asarray([[5.0, 0.0], [1.0, 2.0]], dtype=dtype)
    xA = ops.asarray(A, dtype=dtype)
    xB = ops.asarray(B, dtype=dtype)
    assert_matches_reference("matmul", ops.matmul(xA, xB), A @ B, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kron_einsum(ops, dtype):
    a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    b = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    xa = ops.asarray(a, dtype=dtype)
    xb = ops.asarray(b, dtype=dtype)
    assert_matches_reference("kron", ops.kron(xa, xb), np.kron(a, b), dtype=dtype)
    assert_matches_reference("einsum", ops.einsum("ij,jk->ik", xa, xb), np.einsum("ij,jk->ik", a, b), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eigh_identity(ops, dtype):
    if np.dtype(dtype).kind == "c":
        A = np.asarray([[4.0 + 0j, 1.0 - 1j], [1.0 + 1j, 3.0 + 0j]], dtype=dtype)
    else:
        A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
    assert_eigh_identity(ops, ops.asarray(A, dtype=dtype), dtype=dtype)


# ---------------------------------------------------------------------------
# Indexing


def test_take(ops):
    src = np.arange(12, dtype=np.float64).reshape(3, 4)
    x = ops.asarray(src, dtype=np.float64)
    idx = ops.asarray(np.asarray([0, 2]), dtype=np.int64)
    assert_matches_reference(
        "take", ops.take(x, idx, axis=1), np.take(src, np.asarray([0, 2]), axis=1), dtype=np.float64
    )


def test_diag_diagonal_tril_triu(ops):
    src = np.arange(9, dtype=np.float64).reshape(3, 3)
    x = ops.asarray(src, dtype=np.float64)
    assert_matches_reference("diag", ops.diag(x), np.diag(src), dtype=np.float64)
    assert_matches_reference("diagonal", ops.diagonal(x), np.diagonal(src), dtype=np.float64)
    assert_matches_reference("tril", ops.tril(x), np.tril(src), dtype=np.float64)
    assert_matches_reference("triu", ops.triu(x), np.triu(src), dtype=np.float64)


# ---------------------------------------------------------------------------
# Constants


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eps_is_finfo_eps(ops, dtype):
    assert ops.eps(dtype) == pytest.approx(float(np.finfo(dtype).eps))


def test_constants_are_scalars(ops):
    assert float(np.asarray(ops.pi)) == pytest.approx(float(np.pi))
    assert float(np.asarray(ops.e)) == pytest.approx(float(np.e))
    assert np.isinf(float(np.asarray(ops.inf)))
    assert np.isnan(float(np.asarray(ops.nan)))


# ---------------------------------------------------------------------------
# allclose


def test_allclose(ops):
    a = ops.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    b = ops.asarray([1.0, 2.0, 3.0 + 1e-10], dtype=np.float64)
    assert ops.allclose(a, b) is True or ops.allclose(a, b) == np.True_
    c = ops.asarray([1.0, 2.0, 3.5], dtype=np.float64)
    assert ops.allclose(a, c) is False or ops.allclose(a, c) == np.False_

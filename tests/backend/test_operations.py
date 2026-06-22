"""Uniform operations conformance: each backend vs its own native library.

Every test takes the parametrized ``backend_ops`` fixture and obtains its
truth via :func:`tests.backend._references.native_reference`. ``NumpyOps``
is checked against ``numpy``, ``JaxOps`` against ``jax.numpy``, ``TorchOps``
against ``torch``, ``CuPyOps`` against ``cupy``. This decouples backends:
a bug in ``NumpyOps.matmul`` will not silently propagate as truth for the
other backends.

Tests are grouped into one ``TestXxx`` class per BackendOps method category
so ``pytest -k TestSparse`` filters the run without hunting through files.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests._helpers import to_numpy
from tests.backend._conformance import (
    assert_matches_reference,
    backend_supports_dtype,
    tolerance_for,
)
from tests.backend._references import native_reference


# ---------------------------------------------------------------------------
# Small helpers used across categories
# ---------------------------------------------------------------------------
def _skip_if_dtype_unsupported(backend_ops, dtype) -> None:
    if not backend_supports_dtype(backend_ops.family, dtype):
        pytest.skip(f"{backend_ops.family} does not honor dtype {np.dtype(dtype)}")


def _default_real_dtype(backend_ops):
    """Backend-preferred real dtype: float64 when supported, else float32.

    Tests that do not parametrize over dtype use this so JAX without
    ``jax_enable_x64`` runs in float32 instead of erroring on float64.
    """
    if backend_supports_dtype(backend_ops.family, np.float64):
        return np.float64
    return np.float32


def _make_real(backend_ops, shape, dtype, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(dtype)
    return backend_ops.asarray(data, dtype=dtype)


def _make_complex(backend_ops, shape, dtype, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    data = (real + 1j * imag).astype(dtype)
    return backend_ops.asarray(data, dtype=dtype)


def _make_array(backend_ops, shape, dtype, seed: int = 0) -> Any:
    if np.dtype(dtype).kind == "c":
        return _make_complex(backend_ops, shape, dtype, seed)
    return _make_real(backend_ops, shape, dtype, seed)


def _make_hermitian(backend_ops, n: int, dtype, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (M + M.conj().T) * 0.5
    else:
        M = rng.standard_normal((n, n))
        H = (M + M.T) * 0.5
    H = H.astype(dtype)
    return backend_ops.asarray(H, dtype=dtype)


def _make_spd(backend_ops, n: int, dtype, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A = M.conj().T @ M + n * np.eye(n)
    else:
        M = rng.standard_normal((n, n))
        A = M.T @ M + n * np.eye(n)
    A = A.astype(dtype)
    return backend_ops.asarray(A, dtype=dtype)


def _matches(op_name: str, backend_result, reference, *, dtype, equal_nan: bool = False) -> None:
    assert_matches_reference(op_name, backend_result, reference, dtype=dtype, equal_nan=equal_nan)


# ===========================================================================
# Properties — pinned per family, not tolerance-based
# ===========================================================================
class TestProperties:
    def test_family_string(self, backend_ops):
        ref = native_reference(backend_ops)
        assert backend_ops.family == ref.expected_family

    def test_allow_sparse_matches_family(self, backend_ops):
        ref = native_reference(backend_ops)
        assert backend_ops.allow_sparse == ref.expected_allow_sparse

    def test_has_native_vmap_matches_family(self, backend_ops):
        ref = native_reference(backend_ops)
        assert backend_ops.has_native_vmap == ref.expected_has_native_vmap

    def test_dense_array_type(self, backend_ops):
        ref = native_reference(backend_ops)
        x = backend_ops.zeros((2, 2))
        assert isinstance(x, ref.dense_array_type)

    def test_sparse_array_type_present_when_supported(self, backend_ops):
        if backend_ops.sparse_array is None:
            return
        assert all(isinstance(t, type) for t in backend_ops.sparse_array)

    def test_sparse_array_concrete_classes_per_family(self, backend_ops):
        """backend-002: pin the CONCRETE sparse classes, not just ``isinstance(t, type)``.

        NumpyOps must expose the scipy.sparse base classes; TorchOps must expose
        ``torch.Tensor`` (the sparse-layout carrier). JAX/cupy are checked
        loosely since their sparse class set is environment-dependent.
        """
        types = backend_ops.sparse_array
        if backend_ops.family == "numpy":
            import scipy.sparse as sps

            assert sps.spmatrix in types
            assert sps.sparray in types
        elif backend_ops.family == "torch":
            import torch

            # Torch carries every sparse layout on the single Tensor class.
            assert types == (torch.Tensor,)
        else:
            assert types is None or all(isinstance(t, type) for t in types)


# ===========================================================================
# Predicates
# ===========================================================================
class TestPredicates:
    def test_is_dense_for_native_dense_array(self, backend_ops):
        x = backend_ops.zeros((2, 2))
        assert backend_ops.is_dense(x) is True

    def test_is_dense_false_for_python_list(self, backend_ops):
        assert backend_ops.is_dense([1.0, 2.0]) is False

    def test_is_array_for_native_dense(self, backend_ops):
        x = backend_ops.zeros((2, 2))
        assert backend_ops.is_array(x) is True

    def test_is_array_false_for_python_list(self, backend_ops):
        assert backend_ops.is_array([1.0, 2.0]) is False

    def test_is_sparse_false_for_dense_array(self, backend_ops):
        """gap-1: is_sparse(dense_array) is False on every backend."""
        x = backend_ops.zeros((2, 2))
        assert backend_ops.is_sparse(x) is False

    def test_is_sparse_false_when_sparse_array_is_none(self, backend_ops):
        """gap-1: a backend whose ``sparse_array`` is None reports is_sparse False.

        ``BackendOps.is_sparse`` short-circuits to False when ``sparse_array``
        is None. We probe the base-class logic directly against this backend's
        instance with ``sparse_array`` monkeypatched to None so the negative
        branch is exercised regardless of whether the installed backend
        natively returns None.
        """
        from spacecore.backend._ops import BackendOps

        x = backend_ops.zeros((2, 2))

        class _NoSparse:
            sparse_array = None
            is_sparse = BackendOps.is_sparse

        assert _NoSparse().is_sparse(x) is False
        # And a backend that genuinely has no sparse support also returns False.
        if backend_ops.sparse_array is None:
            assert backend_ops.is_sparse(x) is False


# ===========================================================================
# Dtype helpers
# ===========================================================================
class TestDtypeHelpers:
    def test_sanitize_none_returns_default_dtype(self, backend_ops):
        ref = native_reference(backend_ops)
        sanitized = backend_ops.sanitize_dtype(None)
        assert sanitized == ref.default_dtype

    @pytest.mark.parametrize("dtype, expected_complex", [
        (np.float32, False), (np.float64, False),
        (np.complex64, True), (np.complex128, True),
    ])
    def test_is_complex_dtype(self, backend_ops, dtype, expected_complex):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        assert backend_ops.is_complex_dtype(dtype) is expected_complex

    @pytest.mark.parametrize("complex_in, real_out", [
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ])
    def test_real_dtype_strips_complex(self, backend_ops, complex_in, real_out):
        _skip_if_dtype_unsupported(backend_ops, complex_in)
        out = backend_ops.real_dtype(complex_in)
        assert np.dtype(str(out).split(".")[-1]) == np.dtype(real_out)

    @pytest.mark.parametrize("real_in", [np.float32, np.float64])
    def test_real_dtype_is_identity_on_real(self, backend_ops, real_in):
        _skip_if_dtype_unsupported(backend_ops, real_in)
        out = backend_ops.real_dtype(real_in)
        assert backend_ops.sanitize_dtype(out) == backend_ops.sanitize_dtype(real_in)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_get_dtype_round_trip(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        x = backend_ops.zeros((2, 2), dtype=dtype)
        assert backend_ops.get_dtype(x) == backend_ops.sanitize_dtype(dtype)

    def test_get_dtype_rejects_non_array(self, backend_ops):
        with pytest.raises(TypeError):
            backend_ops.get_dtype([1.0, 2.0])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eps_matches_finfo(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        assert backend_ops.eps(dtype) == pytest.approx(float(np.finfo(dtype).eps))


# ===========================================================================
# Shape / size
# ===========================================================================
class TestShape:
    def test_shape_returns_tuple(self, backend_ops):
        x = backend_ops.zeros((3, 4, 5))
        assert backend_ops.shape(x) == (3, 4, 5)
        assert isinstance(backend_ops.shape(x), tuple)

    def test_ndim(self, backend_ops):
        x = backend_ops.zeros((3, 4, 5))
        assert backend_ops.ndim(x) == 3

    def test_size_product_of_shape(self, backend_ops):
        x = backend_ops.zeros((3, 4, 5))
        assert backend_ops.size(x) == 60


# ===========================================================================
# Constants
# ===========================================================================
class TestConstants:
    def test_inf_is_positive_infinity(self, backend_ops):
        assert float(to_numpy(backend_ops.inf)) == float("inf")

    def test_nan_is_nan(self, backend_ops):
        assert np.isnan(float(to_numpy(backend_ops.nan)))

    def test_pi(self, backend_ops):
        assert float(to_numpy(backend_ops.pi)) == pytest.approx(np.pi, rel=1e-7)

    def test_e(self, backend_ops):
        assert float(to_numpy(backend_ops.e)) == pytest.approx(np.e, rel=1e-7)

    def test_constants_are_cached(self, backend_ops):
        first = backend_ops.pi
        second = backend_ops.pi
        assert first is second


# ===========================================================================
# Construction
# ===========================================================================
class TestConstruction:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_zeros_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        _matches("zeros", backend_ops.zeros((3, 3), dtype=dtype),
                 ref.zeros((3, 3), dtype=dtype), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_ones_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        _matches("ones", backend_ops.ones((3, 3), dtype=dtype),
                 ref.ones((3, 3), dtype=dtype), dtype=dtype)

    def test_zeros_default_dtype_is_sanitized_default(self, backend_ops):
        x = backend_ops.zeros((2, 2))
        assert backend_ops.get_dtype(x) == backend_ops.sanitize_dtype(None)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_full_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        _matches("full", backend_ops.full((2, 3), 1.5, dtype=dtype),
                 ref.full((2, 3), 1.5, dtype=dtype), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eye_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        _matches("eye", backend_ops.eye(4, 5, dtype=dtype),
                 ref.eye(4, 5, dtype=dtype), dtype=dtype)
        _matches("eye", backend_ops.eye(3, dtype=dtype),
                 ref.eye(3, dtype=dtype), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_arange_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        _matches("arange", backend_ops.arange(5, dtype=dtype),
                 ref.arange(5, dtype=dtype), dtype=dtype)
        _matches("arange", backend_ops.arange(2, 10, dtype=dtype),
                 ref.arange(2, 10, dtype=dtype), dtype=dtype)
        _matches("arange", backend_ops.arange(0, 10, 2, dtype=dtype),
                 ref.arange(0, 10, 2, dtype=dtype), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_asarray_round_trip(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        data = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        _matches("asarray", backend_ops.asarray(data, dtype=dtype),
                 ref.asarray(data, dtype=dtype), dtype=dtype)

    def test_asarray_passthrough_for_native(self, backend_ops):
        x = backend_ops.zeros((3, 3))
        y = backend_ops.asarray(x)
        # Family-level equality of values
        assert np.array_equal(to_numpy(x), to_numpy(y))

    def test_asarray_refuses_complex_to_real(self, backend_ops):
        _skip_if_dtype_unsupported(backend_ops, np.complex64)
        z = backend_ops.asarray(np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64))
        with pytest.raises(TypeError):
            backend_ops.asarray(z, dtype=np.float32)

    @pytest.mark.parametrize("src_dtype, dst_dtype", [
        (np.float32, np.float64),
        (np.float64, np.float32),
        (np.float32, np.complex64),
    ])
    def test_astype_round_trip(self, backend_ops, src_dtype, dst_dtype):
        _skip_if_dtype_unsupported(backend_ops, src_dtype)
        _skip_if_dtype_unsupported(backend_ops, dst_dtype)
        x = _make_real(backend_ops, (3,), src_dtype)
        y = backend_ops.astype(x, dst_dtype)
        assert backend_ops.get_dtype(y) == backend_ops.sanitize_dtype(dst_dtype)

    def test_astype_none_is_identity(self, backend_ops):
        x = backend_ops.zeros((2, 2))
        assert backend_ops.astype(x, None) is x

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_zeros_like_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = backend_ops.ones((3, 4), dtype=dtype)
        x_ref = ref.ones((3, 4), dtype=dtype)
        _matches("zeros_like", backend_ops.zeros_like(x_be),
                 ref.zeros_like(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_ones_like_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = backend_ops.zeros((3, 4), dtype=dtype)
        x_ref = ref.zeros((3, 4), dtype=dtype)
        _matches("ones_like", backend_ops.ones_like(x_be),
                 ref.ones_like(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_full_like_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = backend_ops.zeros((3, 4), dtype=dtype)
        x_ref = ref.zeros((3, 4), dtype=dtype)
        _matches("full_like", backend_ops.full_like(x_be, 7.0),
                 ref.full_like(x_ref, 7.0), dtype=dtype)


# ===========================================================================
# Reshape / layout
# ===========================================================================
class TestReshape:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_ravel(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (2, 3), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("ravel", backend_ops.ravel(x_be), ref.ravel(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("new_shape", [(6,), (3, 2), (1, 6), (-1, 2)])
    def test_reshape(self, backend_ops, dtype, new_shape):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (2, 3), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("reshape", backend_ops.reshape(x_be, new_shape),
                 ref.reshape(x_ref, new_shape), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_transpose_default_reverses_axes(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (2, 3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("transpose", backend_ops.transpose(x_be), ref.transpose(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_transpose_with_axes(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (2, 3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("transpose", backend_ops.transpose(x_be, (1, 0, 2)),
                 ref.transpose(x_ref, (1, 0, 2)), dtype=dtype)

    def test_swapaxes(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (2, 3, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("swapaxes", backend_ops.swapaxes(x_be, 0, 2),
                 ref.swapaxes(x_ref, 0, 2), dtype=dt)

    def test_broadcast_to(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = backend_ops.asarray(np.asarray([1.0, 2.0, 3.0], dtype=dt), dtype=dt)
        x_ref = ref.asarray(np.asarray([1.0, 2.0, 3.0], dtype=dt), dtype=dt)
        _matches("broadcast_to", backend_ops.broadcast_to(x_be, (4, 3)),
                 ref.broadcast_to(x_ref, (4, 3)), dtype=dt)

    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_expand_dims_int(self, backend_ops, axis):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (2, 3), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("expand_dims", backend_ops.expand_dims(x_be, axis),
                 ref.expand_dims(x_ref, axis), dtype=dt)

    def test_expand_dims_tuple(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (2, 3), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("expand_dims", backend_ops.expand_dims(x_be, (0, 2)),
                 ref.expand_dims(x_ref, (0, 2)), dtype=dt)

    def test_squeeze_axis_none_drops_singletons(self, backend_ops):
        ref = native_reference(backend_ops)
        x_be = backend_ops.zeros((1, 3, 1, 4))
        x_ref = ref.zeros((1, 3, 1, 4))
        out_be = backend_ops.squeeze(x_be)
        out_ref = ref.squeeze(x_ref)
        assert backend_ops.shape(out_be) == tuple(out_ref.shape)

    @pytest.mark.parametrize("axis", [0, 2])
    def test_squeeze_explicit_axis(self, backend_ops, axis):
        ref = native_reference(backend_ops)
        x_be = backend_ops.zeros((1, 3, 1, 4))
        x_ref = ref.zeros((1, 3, 1, 4))
        out_be = backend_ops.squeeze(x_be, axis)
        out_ref = ref.squeeze(x_ref, axis)
        assert backend_ops.shape(out_be) == tuple(out_ref.shape)

    @pytest.mark.parametrize("source,destination", [(0, 2), (2, 0), (-1, 0), (0, -1)])
    def test_moveaxis(self, backend_ops, source, destination):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (2, 3, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("moveaxis", backend_ops.moveaxis(x_be, source, destination),
                 ref.moveaxis(x_ref, source, destination), dtype=dt)

    def test_stack(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        xs_be = [_make_real(backend_ops, (3,), dt, seed=i) for i in range(3)]
        xs_ref = [ref.asarray(to_numpy(x), dtype=dt) for x in xs_be]
        _matches("stack", backend_ops.stack(xs_be, axis=0),
                 ref.stack(xs_ref, axis=0), dtype=dt)

    @pytest.mark.parametrize("op_name", ["hstack", "vstack", "dstack", "column_stack"])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_stacking_helpers(self, backend_ops, op_name, shape):
        """hstack/vstack/dstack/column_stack match the native reference.

        Both the 1-D and 2-D input cases are exercised so the
        ``atleast_*d`` promotion paths each backend uses are covered.
        """
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        xs_be = [_make_real(backend_ops, shape, dt, seed=i) for i in range(2)]
        xs_ref = [ref.asarray(to_numpy(x), dtype=dt) for x in xs_be]
        be_fn = getattr(backend_ops, op_name)
        ref_fn = getattr(ref, op_name)
        _matches(op_name, be_fn(xs_be), ref_fn(xs_ref), dtype=dt)

    def test_concatenate(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        xs_be = [_make_real(backend_ops, (2, 3), dt, seed=i) for i in range(2)]
        xs_ref = [ref.asarray(to_numpy(x), dtype=dt) for x in xs_be]
        _matches("concatenate", backend_ops.concatenate(xs_be, axis=0),
                 ref.concatenate(xs_ref, axis=0), dtype=dt)

    def test_take(self, backend_ops):
        ref = native_reference(backend_ops)
        # take operates on integer indices; result dtype follows source.
        # Use int64 input so the comparison is dtype-stable.
        x_np = np.arange(12, dtype=np.int64).reshape(3, 4)
        x_be = backend_ops.asarray(x_np)
        x_ref = ref.asarray(x_np)
        idx_be = backend_ops.asarray(np.asarray([0, 2], dtype=np.int64))
        idx_ref = ref.asarray(np.asarray([0, 2], dtype=np.int64))
        _matches("take", backend_ops.take(x_be, idx_be, axis=1),
                 ref.take(x_ref, idx_ref, axis=1), dtype=np.int64)


# ===========================================================================
# Elementwise
# ===========================================================================
class TestElementwise:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_conj(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_array(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("conj", backend_ops.conj(x_be), ref.conj(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_real_imag_on_complex(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_complex(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        real_dt = np.float32 if dtype == np.complex64 else np.float64
        _skip_if_dtype_unsupported(backend_ops, real_dt)
        _matches("real", backend_ops.real(x_be), ref.real(x_ref), dtype=real_dt)
        _matches("imag", backend_ops.imag(x_be), ref.imag(x_ref), dtype=real_dt)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_real_on_real_is_identity(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("real", backend_ops.real(x_be), ref.real(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_imag_on_real(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        if backend_ops.family == "torch":
            pytest.skip(
                "torch.imag refuses real-dtype tensors; SpaceCore inherits "
                "this. Real-input semantics live in the backend's specifics file."
            )
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("imag", backend_ops.imag(x_be), ref.imag(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_abs(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_array(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        # abs of complex returns real; tolerance dtype is the real precision.
        if np.dtype(dtype).kind == "c":
            tol_dt = np.float32 if dtype == np.complex64 else np.float64
        else:
            tol_dt = dtype
        _matches("abs", backend_ops.abs(x_be), ref.abs(x_ref), dtype=tol_dt)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sign_on_real(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = backend_ops.asarray(np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=dtype), dtype=dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("sign", backend_ops.sign(x_be), ref.sign(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sqrt(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = backend_ops.asarray(np.asarray([0.25, 1.0, 4.0, 9.0], dtype=dtype), dtype=dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("sqrt", backend_ops.sqrt(x_be), ref.sqrt(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sqrt_complex(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_complex(backend_ops, (4,), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("sqrt", backend_ops.sqrt(x_be), ref.sqrt(x_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp_log(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = backend_ops.asarray(np.asarray([0.1, 1.0, 2.0, 5.0], dtype=dtype), dtype=dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("exp", backend_ops.exp(x_be), ref.exp(x_ref), dtype=dtype)
        _matches("log", backend_ops.log(x_be), ref.log(x_ref), dtype=dtype)

    def test_where(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        cond_np = np.asarray([True, False, True, False])
        x_np = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=dt)
        y_np = np.asarray([-1.0, -2.0, -3.0, -4.0], dtype=dt)
        cond_be = backend_ops.asarray(cond_np)
        cond_ref = ref.asarray(cond_np)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        y_be = backend_ops.asarray(y_np, dtype=dt)
        y_ref = ref.asarray(y_np, dtype=dt)
        _matches("where", backend_ops.where(cond_be, x_be, y_be),
                 ref.where(cond_ref, x_ref, y_ref), dtype=dt)

    def test_maximum_minimum(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.asarray([1.0, 5.0, 3.0], dtype=dt)
        y_np = np.asarray([4.0, 2.0, 6.0], dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        y_be = backend_ops.asarray(y_np, dtype=dt)
        y_ref = ref.asarray(y_np, dtype=dt)
        _matches("maximum", backend_ops.maximum(x_be, y_be),
                 ref.maximum(x_ref, y_ref), dtype=dt)
        _matches("minimum", backend_ops.minimum(x_be, y_be),
                 ref.minimum(x_ref, y_ref), dtype=dt)

    def test_clip(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        _matches("clip", backend_ops.clip(x_be, -1.0, 1.0),
                 ref.clip(x_ref, -1.0, 1.0), dtype=dt)

    def test_isfinite_isnan(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.asarray([1.0, np.inf, np.nan, -np.inf], dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        # bool outputs: compare via to_numpy + array_equal directly
        assert np.array_equal(to_numpy(backend_ops.isfinite(x_be)), to_numpy(ref.isfinite(x_ref)))
        assert np.array_equal(to_numpy(backend_ops.isnan(x_be)), to_numpy(ref.isnan(x_ref)))


# ===========================================================================
# Reductions — axis tuples, negative axes, keepdims
# ===========================================================================
class TestReductions:
    @pytest.mark.parametrize("op_name", ["sum", "mean", "min", "max", "prod"])
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduction(self, backend_ops, op_name, axis, keepdims):
        ref = native_reference(backend_ops)
        dtype = _default_real_dtype(backend_ops)
        # Use modest positive values so prod doesn't blow up.
        x_be = backend_ops.asarray(
            np.arange(1.0, 13.0, dtype=dtype).reshape(3, 4), dtype=dtype
        )
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        be_fn = getattr(backend_ops, op_name)
        ref_fn = getattr(ref, op_name)
        _matches(op_name, be_fn(x_be, axis=axis, keepdims=keepdims),
                 ref_fn(x_ref, axis=axis, keepdims=keepdims), dtype=dtype)

    def test_trace(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (5, 5), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("trace", backend_ops.trace(x_be), ref.trace(x_ref), dtype=dt)

    def test_sum_dtype_promotion(self, backend_ops):
        """``sum(dtype=float64)`` of float32 input lifts the result to float64."""
        _skip_if_dtype_unsupported(backend_ops, np.float64)
        _skip_if_dtype_unsupported(backend_ops, np.float32)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), np.float32)
        x_ref = ref.asarray(to_numpy(x_be), dtype=np.float32)
        be_out = backend_ops.sum(x_be, dtype=np.float64)
        ref_out = ref.sum(x_ref, dtype=np.float64)
        # Comparison via to_numpy is dtype-honest even when the backend
        # returns a 0-D scalar (NumPy) vs a 0-D tensor (Torch/JAX).
        be_np = np.asarray(to_numpy(be_out))
        assert be_np.dtype == np.dtype(np.float64)
        _matches("sum", be_out, ref_out, dtype=np.float64)


# ===========================================================================
# Index reductions
# ===========================================================================
class TestIndexReductions:
    def test_argsort_and_sort(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.asarray([3.0, 1.0, 4.0, 1.5, 9.0, 2.0], dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        # argsort returns ints; exact equality
        assert np.array_equal(to_numpy(backend_ops.argsort(x_be)), to_numpy(ref.argsort(x_ref)))
        _matches("sort", backend_ops.sort(x_be), ref.sort(x_ref), dtype=dt)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    def test_argmin_argmax(self, backend_ops, axis):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        assert np.array_equal(
            to_numpy(backend_ops.argmin(x_be, axis=axis)),
            to_numpy(ref.argmin(x_ref, axis=axis)),
        )
        assert np.array_equal(
            to_numpy(backend_ops.argmax(x_be, axis=axis)),
            to_numpy(ref.argmax(x_ref, axis=axis)),
        )


# ===========================================================================
# Linear algebra
# ===========================================================================
class TestLinalg:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_vdot(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_array(backend_ops, (5,), dtype)
        y_be = _make_array(backend_ops, (5,), dtype, seed=1)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        y_ref = ref.asarray(to_numpy(y_be), dtype=dtype)
        _matches("vdot", backend_ops.vdot(x_be, y_be), ref.vdot(x_ref, y_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matmul(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        A_be = _make_array(backend_ops, (3, 4), dtype)
        B_be = _make_array(backend_ops, (4, 2), dtype, seed=1)
        A_ref = ref.asarray(to_numpy(A_be), dtype=dtype)
        B_ref = ref.asarray(to_numpy(B_be), dtype=dtype)
        _matches("matmul", backend_ops.matmul(A_be, B_be),
                 ref.matmul(A_ref, B_ref), dtype=dtype)

    def test_kron(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        a_be = _make_real(backend_ops, (2, 2), dt)
        b_be = _make_real(backend_ops, (3, 2), dt, seed=1)
        a_ref = ref.asarray(to_numpy(a_be), dtype=dt)
        b_ref = ref.asarray(to_numpy(b_be), dtype=dt)
        _matches("kron", backend_ops.kron(a_be, b_be),
                 ref.kron(a_ref, b_ref), dtype=dt)

    def test_einsum(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        a_be = _make_real(backend_ops, (3, 4), dt)
        b_be = _make_real(backend_ops, (4, 5), dt, seed=1)
        a_ref = ref.asarray(to_numpy(a_be), dtype=dt)
        b_ref = ref.asarray(to_numpy(b_be), dtype=dt)
        _matches("einsum", backend_ops.einsum("ij,jk->ik", a_be, b_be),
                 ref.einsum("ij,jk->ik", a_ref, b_ref), dtype=dt)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eigh_satisfies_eigen_identity(self, backend_ops, dtype):
        """``A v_i = lambda_i v_i`` per column. Avoids eigvec-sign ambiguity."""
        _skip_if_dtype_unsupported(backend_ops, dtype)
        H = _make_hermitian(backend_ops, 4, dtype)
        eigvals, eigvecs = backend_ops.eigh(H)
        Av = backend_ops.matmul(H, eigvecs)
        lhs = to_numpy(Av)
        rhs = to_numpy(eigvecs) * to_numpy(eigvals)[None, :]
        tol = tolerance_for("eigh", dtype)
        assert np.allclose(lhs, rhs, rtol=tol.rtol, atol=tol.atol), (
            f"eigh identity failed for dtype={np.dtype(dtype)}"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eigvalsh_matches_native(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        H_be = _make_hermitian(backend_ops, 4, dtype)
        H_ref = ref.asarray(to_numpy(H_be), dtype=dtype)
        be_eigvals = np.sort(to_numpy(backend_ops.eigvalsh(H_be)))
        ref_eigvals = np.sort(to_numpy(ref.eigvalsh(H_ref)))
        _matches("eigvalsh", be_eigvals, ref_eigvals, dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_svd_singular_values_match(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        A_be = _make_real(backend_ops, (5, 3), dtype)
        A_ref = ref.asarray(to_numpy(A_be), dtype=dtype)
        be_s = np.sort(to_numpy(backend_ops.svd(A_be, full_matrices=False)[1]))[::-1]
        ref_s = np.sort(to_numpy(ref.svd(A_ref, full_matrices=False)[1]))[::-1]
        _matches("svd", be_s, ref_s, dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_solve(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        A_be = _make_spd(backend_ops, 4, dtype)
        b_be = _make_real(backend_ops, (4,), dtype, seed=2)
        A_ref = ref.asarray(to_numpy(A_be), dtype=dtype)
        b_ref = ref.asarray(to_numpy(b_be), dtype=dtype)
        _matches("solve", backend_ops.solve(A_be, b_be),
                 ref.solve(A_ref, b_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cholesky_lower_factor(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        A_be = _make_spd(backend_ops, 4, dtype)
        A_ref = ref.asarray(to_numpy(A_be), dtype=dtype)
        _matches("cholesky", backend_ops.cholesky(A_be),
                 ref.cholesky(A_ref), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_norm_default(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("norm", backend_ops.norm(x_be), ref.norm(x_ref), dtype=dtype)

    def test_norm_with_axis(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("norm", backend_ops.norm(x_be, axis=1),
                 ref.norm(x_ref, axis=1), dtype=dt)


# ===========================================================================
# Matrix helpers
# ===========================================================================
class TestMatrixHelpers:
    def test_diag_extract(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (4, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("diag", backend_ops.diag(x_be), ref.diag(x_ref), dtype=dt)

    def test_diag_construct_from_1d(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        v_np = np.asarray([1.0, 2.0, 3.0], dtype=dt)
        v_be = backend_ops.asarray(v_np, dtype=dt)
        v_ref = ref.asarray(v_np, dtype=dt)
        _matches("diag", backend_ops.diag(v_be), ref.diag(v_ref), dtype=dt)

    def test_diagonal(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (4, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("diagonal", backend_ops.diagonal(x_be),
                 ref.diagonal(x_ref), dtype=dt)

    def test_tril_triu(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_be = _make_real(backend_ops, (4, 4), dt)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dt)
        _matches("tril", backend_ops.tril(x_be), ref.tril(x_ref), dtype=dt)
        _matches("triu", backend_ops.triu(x_be), ref.triu(x_ref), dtype=dt)


# ===========================================================================
# allclose
# ===========================================================================
class TestAllclose:
    def test_allclose_true_on_equal(self, backend_ops):
        dt = _default_real_dtype(backend_ops)
        x = _make_real(backend_ops, (3, 4), dt)
        assert backend_ops.allclose(x, x) is True

    def test_allclose_false_on_perturbed(self, backend_ops):
        dt = _default_real_dtype(backend_ops)
        x = _make_real(backend_ops, (3, 4), dt)
        y = _make_real(backend_ops, (3, 4), dt, seed=1)
        assert backend_ops.allclose(x, y) is False

    def test_allclose_nan_handling(self, backend_ops):
        dt = _default_real_dtype(backend_ops)
        x_np = np.asarray([1.0, np.nan, 3.0], dtype=dt)
        x = backend_ops.asarray(x_np, dtype=dt)
        y = backend_ops.asarray(x_np, dtype=dt)
        assert backend_ops.allclose(x, y, equal_nan=False) is False
        assert backend_ops.allclose(x, y, equal_nan=True) is True


# ===========================================================================
# Special
# ===========================================================================
class TestSpecial:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_logsumexp_no_weights(self, backend_ops, dtype):
        _skip_if_dtype_unsupported(backend_ops, dtype)
        ref = native_reference(backend_ops)
        x_be = _make_real(backend_ops, (3, 4), dtype)
        x_ref = ref.asarray(to_numpy(x_be), dtype=dtype)
        _matches("logsumexp", backend_ops.logsumexp(x_be, axis=1),
                 ref.logsumexp(x_ref, axis=1), dtype=dtype)


# ===========================================================================
# Sparse — guarded; some backends do not support every code path
# ===========================================================================
class TestSparse:
    def _csr_skip(self, backend_ops) -> None:
        if not backend_ops.allow_sparse:
            pytest.skip(f"{backend_ops.family} does not support sparse arrays")

    def test_assparse_round_trips_through_dense(self, backend_ops):
        self._csr_skip(backend_ops)
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        dense_np = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=dt)
        dense_be = backend_ops.asarray(dense_np, dtype=dt)
        dense_ref = ref.asarray(dense_np, dtype=dt)
        be_sparse = backend_ops.assparse(dense_be)
        ref_sparse = ref.assparse_csr(dense_ref)
        be_dense_back = to_numpy(_densify_via_helper(backend_ops, be_sparse))
        ref_dense_back = to_numpy(ref.sparse_to_dense(ref_sparse))
        np.testing.assert_allclose(be_dense_back, dense_np.astype(np.float64))
        np.testing.assert_allclose(ref_dense_back, dense_np.astype(np.float64))

    def test_sparse_matmul_matches_dense_reference(self, backend_ops):
        self._csr_skip(backend_ops)
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        A_np = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=dt)
        b_np = np.asarray([1.0, 2.0, 3.0], dtype=dt)
        A_be_sparse = backend_ops.assparse(backend_ops.asarray(A_np, dtype=dt))
        b_be = backend_ops.asarray(b_np, dtype=dt)
        A_ref_sparse = ref.assparse_csr(ref.asarray(A_np, dtype=dt))
        b_ref = ref.asarray(b_np, dtype=dt)
        be_out = backend_ops.sparse_matmul(A_be_sparse, b_be)
        ref_out = ref.sparse_matmul(A_ref_sparse, b_ref)
        _matches("sparse_matmul", be_out, ref_out, dtype=dt)

    def test_allclose_sparse_true_on_equal(self, backend_ops):
        self._csr_skip(backend_ops)
        dt = _default_real_dtype(backend_ops)
        dense_np = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=dt)
        a = backend_ops.assparse(backend_ops.asarray(dense_np, dtype=dt))
        b = backend_ops.assparse(backend_ops.asarray(dense_np, dtype=dt))
        assert backend_ops.allclose_sparse(a, b) is True

    def test_allclose_sparse_false_on_perturbed(self, backend_ops):
        self._csr_skip(backend_ops)
        dt = _default_real_dtype(backend_ops)
        a = backend_ops.assparse(backend_ops.asarray(
            np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=dt), dtype=dt))
        b = backend_ops.assparse(backend_ops.asarray(
            np.asarray([[1.0, 0.0], [0.0, 2.5]], dtype=dt), dtype=dt))
        assert backend_ops.allclose_sparse(a, b) is False


def _densify_via_helper(backend_ops, sparse_obj) -> Any:
    """Round-trip a backend sparse object to dense for value comparison."""
    if backend_ops.family in ("numpy", "cupy"):
        return sparse_obj.toarray()
    if backend_ops.family == "torch":
        return sparse_obj.to_dense()
    if backend_ops.family == "jax":
        return sparse_obj.todense()
    raise AssertionError(backend_ops.family)


# ===========================================================================
# Index ops (index_set, index_add, ix_)
# ===========================================================================
class TestIndexOps:
    def test_index_set_writes_at_indices(self, backend_ops):
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.zeros(5, dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        idx = np.asarray([1, 3])
        vals_np = np.asarray([7.0, 9.0], dtype=dt)
        be_out = backend_ops.index_set(x_be, idx, backend_ops.asarray(vals_np, dtype=dt))
        ref_out = ref.index_set(x_ref, idx, ref.asarray(vals_np, dtype=dt))
        _matches("index_set", be_out, ref_out, dtype=dt)

    def test_index_add_distinct_indices(self, backend_ops):
        """All backends agree when indices are distinct (no accumulation)."""
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.zeros(5, dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        idx = np.asarray([0, 2, 4])
        vals_np = np.asarray([2.0, 3.0, 5.0], dtype=dt)
        be_out = backend_ops.index_add(x_be, idx, backend_ops.asarray(vals_np, dtype=dt))
        ref_out = ref.index_add(x_ref, idx, ref.asarray(vals_np, dtype=dt))
        expected = x_np.copy()
        np.add.at(expected, idx, vals_np)
        np.testing.assert_allclose(to_numpy(be_out), expected)
        np.testing.assert_allclose(to_numpy(ref_out), expected)

    def test_index_add_accumulates_at_repeated_indices(self, backend_ops):
        """np.add.at semantics for repeated indices — torch lacks this."""
        if backend_ops.family == "torch":
            pytest.skip(
                "TorchOps.index_add implements scatter-assign, not accumulate. "
                "Last-write-wins semantics tested in tests/backend/test_torch_ops.py."
            )
        ref = native_reference(backend_ops)
        dt = _default_real_dtype(backend_ops)
        x_np = np.zeros(5, dtype=dt)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        idx = np.asarray([1, 1, 3])
        vals_np = np.asarray([2.0, 3.0, 5.0], dtype=dt)
        be_out = backend_ops.index_add(x_be, idx, backend_ops.asarray(vals_np, dtype=dt))
        ref_out = ref.index_add(x_ref, idx, ref.asarray(vals_np, dtype=dt))
        expected = x_np.copy()
        np.add.at(expected, idx, vals_np)
        np.testing.assert_allclose(to_numpy(be_out), expected)
        np.testing.assert_allclose(to_numpy(ref_out), expected)

    def test_ix_open_mesh(self, backend_ops):
        ref = native_reference(backend_ops)
        # Compare the broadcast result on a small array.
        dt = _default_real_dtype(backend_ops)
        x_np = np.arange(20, dtype=dt).reshape(4, 5)
        x_be = backend_ops.asarray(x_np, dtype=dt)
        x_ref = ref.asarray(x_np, dtype=dt)
        be_mesh = backend_ops.ix_(np.asarray([0, 2]), np.asarray([1, 4]))
        ref_mesh = ref.ix_(np.asarray([0, 2]), np.asarray([1, 4]))
        # Apply both meshes to the same dense and compare the gathered result.
        gathered_be = to_numpy(x_be[be_mesh])
        gathered_ref = to_numpy(x_ref[ref_mesh] if backend_ops.family != "torch"
                                else x_ref[tuple(ref_mesh)])
        np.testing.assert_allclose(gathered_be, gathered_ref)


# ===========================================================================
# Control flow
# ===========================================================================
class TestControlFlow:
    def test_fori_loop_accumulates(self, backend_ops):
        ref = native_reference(backend_ops)
        init = backend_ops.asarray(np.asarray(0.0))
        be_out = backend_ops.fori_loop(0, 5, lambda i, acc: acc + (i + 1), init)
        ref_out = ref.fori_loop(0, 5, lambda i, acc: acc + (i + 1), 0.0)
        assert float(to_numpy(be_out)) == pytest.approx(float(ref_out))

    def test_while_loop_reaches_terminal(self, backend_ops):
        ref = native_reference(backend_ops)
        init = backend_ops.asarray(np.asarray(0.0))
        be_out = backend_ops.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, init)
        ref_out = ref.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, 0.0)
        assert float(to_numpy(be_out)) == pytest.approx(float(ref_out))

    def test_scan_accumulates_and_stacks(self, backend_ops):
        # JAX scan body must return matching pytree structures; keep simple.
        xs_be = backend_ops.asarray(np.arange(5.0))
        init_be = backend_ops.asarray(np.asarray(0.0))

        def body(carry, x):
            new_carry = carry + x
            return new_carry, x * 2.0

        final_carry, ys = backend_ops.scan(body, init_be, xs_be, length=5)
        assert float(to_numpy(final_carry)) == pytest.approx(0 + 1 + 2 + 3 + 4)
        np.testing.assert_allclose(to_numpy(ys), np.arange(5.0) * 2.0)

    def test_cond_selects_expected_branch(self, backend_ops):
        ref = native_reference(backend_ops)

        def true_fn(x):
            return x + 100.0

        def false_fn(x):
            return x - 100.0

        x_be = backend_ops.asarray(np.asarray(1.0))
        x_val = 1.0
        be_t = backend_ops.cond(True, true_fn, false_fn, x_be)
        be_f = backend_ops.cond(False, true_fn, false_fn, x_be)
        ref_t = ref.cond(True, true_fn, false_fn, x_val)
        ref_f = ref.cond(False, true_fn, false_fn, x_val)
        assert float(to_numpy(be_t)) == pytest.approx(ref_t)
        assert float(to_numpy(be_f)) == pytest.approx(ref_f)


# ===========================================================================
# vmap — fallback semantics; JAX-native path lives in test_jax_ops.py
# ===========================================================================
class TestVmap:
    def test_vmap_in_axes_0_matches_python_loop(self, backend_ops):
        xs = backend_ops.asarray(np.arange(12.0).reshape(4, 3))

        def fn(x):
            return x.sum() if backend_ops.family == "torch" else backend_ops.sum(x)

        result = backend_ops.vmap(fn, in_axes=0)(xs)
        result_np = to_numpy(result)
        expected = np.arange(12.0).reshape(4, 3).sum(axis=1)
        np.testing.assert_allclose(result_np, expected)

    def test_vmap_in_axes_none_broadcasts(self, backend_ops):
        scalar = backend_ops.asarray(np.asarray(7.0))
        xs = backend_ops.asarray(np.arange(4.0))

        def fn(a, b):
            return a + b

        result = backend_ops.vmap(fn, in_axes=(None, 0))(scalar, xs)
        np.testing.assert_allclose(to_numpy(result), np.arange(4.0) + 7.0)

    def test_vmap_multi_arg_in_axes(self, backend_ops):
        xs = backend_ops.asarray(np.arange(8.0).reshape(4, 2))
        ys = backend_ops.asarray(np.arange(8.0).reshape(4, 2))

        def fn(a, b):
            return a + b

        out = backend_ops.vmap(fn, in_axes=(0, 0))(xs, ys)
        np.testing.assert_allclose(to_numpy(out), to_numpy(xs) + to_numpy(ys))

    def test_vmap_out_axes_one_stacks_on_second_axis(self, backend_ops):
        """gap-4: a non-default ``out_axes=1`` stacks per-row outputs on axis 1.

        ``fn`` maps each length-3 row to a length-3 vector; mapping over the 4
        rows with ``out_axes=1`` yields a (3, 4) array — the transpose of the
        default ``out_axes=0`` (4, 3) stacking. Reference is an independent
        ``np.stack(..., axis=1)``.
        """
        data = np.arange(12.0).reshape(4, 3)
        xs = backend_ops.asarray(data)

        def fn(row):
            return row * 2.0

        out = backend_ops.vmap(fn, in_axes=0, out_axes=1)(xs)
        out_np = to_numpy(out)
        expected = np.stack([data[i] * 2.0 for i in range(4)], axis=1)
        assert out_np.shape == expected.shape
        np.testing.assert_allclose(out_np, expected)


# ===========================================================================
# vectorize — native path (NumPy/JAX) and Python-loop fallback (Torch/CuPy)
# ===========================================================================
class TestVectorize:
    def test_vectorize_elementwise_preserves_shape(self, backend_ops):
        """``vectorize(f)`` applies ``f`` to each element and keeps the shape."""
        data = np.arange(6.0).reshape(2, 3)
        x = backend_ops.asarray(data)

        def f(v):
            return v * v + 1.0

        out = backend_ops.vectorize(f)(x)
        assert backend_ops.shape(out) == (2, 3)
        np.testing.assert_allclose(to_numpy(out), data * data + 1.0)

    def test_vectorize_broadcasts_multiple_args(self, backend_ops):
        """Array arguments broadcast against one another, NumPy-style."""
        a = backend_ops.asarray(np.asarray([1.0, 2.0, 3.0]))
        b = backend_ops.asarray(np.asarray([10.0]))

        def g(x, y):
            return x + y

        out = backend_ops.vectorize(g)(a, b)
        np.testing.assert_allclose(to_numpy(out), np.asarray([11.0, 12.0, 13.0]))

    def test_vectorize_excluded_passes_arg_through(self, backend_ops):
        """An ``excluded`` positional argument is forwarded unvectorized."""
        a = backend_ops.asarray(np.asarray([1.0, 2.0, 3.0]))

        def scale(x, factor):
            return x * factor

        out = backend_ops.vectorize(scale, excluded=[1])(a, 2.0)
        np.testing.assert_allclose(to_numpy(out), np.asarray([2.0, 4.0, 6.0]))

    def test_vectorize_loop_fallback_matches_elementwise(self, backend_ops):
        """The Python-loop fallback reproduces elementwise semantics everywhere.

        Exercised directly (not only on backends whose ``xp`` lacks a native
        ``vectorize``) so the base-class branch is covered on every config.
        """
        data = np.arange(8.0).reshape(4, 2)
        x = backend_ops.asarray(data)

        def f(v):
            return v * 3.0

        out = backend_ops._vectorize_loop(f)(x)
        assert backend_ops.shape(out) == (4, 2)
        np.testing.assert_allclose(to_numpy(out), data * 3.0)

    def test_vectorize_loop_fallback_rejects_signature(self, backend_ops):
        """The fallback has no gufunc support and says so explicitly."""
        with pytest.raises(NotImplementedError):
            backend_ops._vectorize_loop(lambda v: v, signature="(n)->(n)")


# ===========================================================================
# eigh sparse rejection (gap-2)
# ===========================================================================
class TestEighSparseRejection:
    def test_eigh_rejects_sparse_input(self, backend_ops):
        """gap-2: ``eigh`` on a sparse array raises the exact base-class TypeError.

        Built where a sparse input is constructible (any backend with
        ``allow_sparse``); the message is pinned at
        ``spacecore/backend/_ops.py`` lines 699-700.
        """
        if not backend_ops.allow_sparse:
            pytest.skip(f"{backend_ops.family} does not support sparse arrays")
        dense = backend_ops.asarray(
            np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=_default_real_dtype(backend_ops))
        )
        sparse = backend_ops.assparse(dense)
        with pytest.raises(
            TypeError, match="eigh requires a dense array; sparse input is not supported."
        ):
            backend_ops.eigh(sparse)


# ===========================================================================
# concatenate dtype-cast branch (gap-3)
# ===========================================================================
class TestConcatenateDtype:
    def test_concatenate_with_explicit_dtype_casts_result(self, backend_ops):
        """gap-3: ``concatenate(..., dtype=)`` triggers the cast branch (_ops.py ~794).

        Inputs are float32; an explicit float64 dtype must promote the result.
        Values are checked against an independent ``np.concatenate``.
        """
        _skip_if_dtype_unsupported(backend_ops, np.float32)
        _skip_if_dtype_unsupported(backend_ops, np.float64)
        a_np = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.asarray([[5.0, 6.0]], dtype=np.float32)
        a_be = backend_ops.asarray(a_np, dtype=np.float32)
        b_be = backend_ops.asarray(b_np, dtype=np.float32)
        out = backend_ops.concatenate([a_be, b_be], axis=0, dtype=np.float64)
        assert backend_ops.get_dtype(out) == backend_ops.sanitize_dtype(np.float64)
        expected = np.concatenate([a_np, b_np], axis=0).astype(np.float64)
        np.testing.assert_allclose(to_numpy(out), expected)


# ===========================================================================
# scan — reverse and xs=None length-driven branches (gap-5)
# ===========================================================================
class TestScanBranches:
    def test_scan_reverse_accumulates_back_to_front(self, backend_ops):
        """gap-5: ``reverse=True`` accumulates from the last element backwards.

        With ``reverse=True`` the carry is the same total sum, but the stacked
        per-step outputs are computed back-to-front then re-ordered to input
        order. Reference is an independent reversed-accumulation in NumPy.
        """
        data = np.arange(5.0)
        xs = backend_ops.asarray(data)
        init = backend_ops.asarray(np.asarray(0.0))

        def body(carry, x):
            new_carry = carry + x
            return new_carry, carry  # emit running prefix BEFORE adding x

        final_carry, ys = backend_ops.scan(body, init, xs, reverse=True)

        # Independent reference: walk indices 4..0, emit carry then add.
        ref_carry = 0.0
        ref_ys = [0.0] * len(data)
        for i in range(len(data) - 1, -1, -1):
            ref_ys[i] = ref_carry
            ref_carry = ref_carry + data[i]
        assert float(to_numpy(final_carry)) == pytest.approx(ref_carry)
        np.testing.assert_allclose(to_numpy(ys), np.asarray(ref_ys))

    def test_scan_xs_none_length_driven(self, backend_ops):
        """gap-5: ``xs=None`` with an explicit ``length`` runs length steps.

        The body ignores ``x`` (which is None) and counts up; the stacked
        outputs are the running counter. Reference is an independent loop.
        """
        init = backend_ops.asarray(np.asarray(0.0))

        def body(carry, _x):
            new_carry = carry + 1.0
            return new_carry, new_carry

        final_carry, ys = backend_ops.scan(body, init, None, length=4)
        assert float(to_numpy(final_carry)) == pytest.approx(4.0)
        np.testing.assert_allclose(to_numpy(ys), np.asarray([1.0, 2.0, 3.0, 4.0]))


# ===========================================================================
# backend_kwargs passthrough (gap-6) — NumPy-only behavioral probes
# ===========================================================================
class TestBackendKwargsPassthrough:
    """Prove the ``backend_kwargs`` dict is forwarded to the backend call.

    Strategy: pass an unsupported kwarg and assert the backend raises a
    ``TypeError`` — the error only surfaces if the dict reaches the underlying
    numpy/scipy linalg routine. Restricted to NumPy where the kwarg surface is
    knowable; other backends are skipped (their linalg signatures differ).
    """

    def _numpy_only(self, backend_ops) -> None:
        if backend_ops.family != "numpy":
            pytest.skip("backend_kwargs probes are NumPy-specific")

    def test_matmul_forwards_unknown_kwarg(self, backend_ops):
        self._numpy_only(backend_ops)
        a = backend_ops.asarray(np.eye(2))
        b = backend_ops.asarray(np.eye(2))
        with pytest.raises(TypeError):
            backend_ops.matmul(a, b, backend_kwargs={"not_a_real_kwarg": 1})

    def test_solve_forwards_unknown_kwarg(self, backend_ops):
        self._numpy_only(backend_ops)
        A = backend_ops.asarray(np.eye(3))
        b = backend_ops.asarray(np.ones(3))
        with pytest.raises(TypeError):
            backend_ops.solve(A, b, backend_kwargs={"not_a_real_kwarg": 1})

    def test_svd_forwards_unknown_kwarg(self, backend_ops):
        self._numpy_only(backend_ops)
        A = backend_ops.asarray(np.eye(3))
        with pytest.raises(TypeError):
            backend_ops.svd(A, backend_kwargs={"not_a_real_kwarg": 1})

    def test_cholesky_forwards_unknown_kwarg(self, backend_ops):
        self._numpy_only(backend_ops)
        A = backend_ops.asarray(np.eye(3))
        with pytest.raises(TypeError):
            backend_ops.cholesky(A, backend_kwargs={"not_a_real_kwarg": 1})

    def test_eigvalsh_forwards_unknown_kwarg(self, backend_ops):
        self._numpy_only(backend_ops)
        A = backend_ops.asarray(np.eye(3))
        with pytest.raises(TypeError):
            backend_ops.eigvalsh(A, backend_kwargs={"not_a_real_kwarg": 1})

    def test_eigh_forwards_unknown_kwarg(self, backend_ops):
        self._numpy_only(backend_ops)
        A = backend_ops.asarray(np.eye(3))
        with pytest.raises(TypeError):
            backend_ops.eigh(A, backend_kwargs={"not_a_real_kwarg": 1})

    def test_matmul_empty_backend_kwargs_is_noop(self, backend_ops):
        """An empty dict forwards nothing and matches the default call."""
        self._numpy_only(backend_ops)
        a = backend_ops.asarray(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        b = backend_ops.asarray(np.asarray([[5.0, 6.0], [7.0, 8.0]]))
        out = backend_ops.matmul(a, b, backend_kwargs={})
        expected = np.asarray([[1.0, 2.0], [3.0, 4.0]]) @ np.asarray([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(to_numpy(out), expected)

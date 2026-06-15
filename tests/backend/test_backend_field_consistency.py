"""Cross-backend scalar-field (real vs complex) consistency pins.

Phase J8 of the backend-conformance suite. The matrix in
:doc:`docs/source/design/backend_conformance` flags a handful of field-
related rows that must agree across every installed backend:

- ``ops.is_complex_dtype`` classifies dtypes the same way ``np.dtype.kind``
  does — real dtypes are not complex, complex dtypes are.
- ``ops.real_dtype`` strips the imaginary half: ``complex64 → float32`` and
  ``complex128 → float64``. Real inputs round-trip unchanged.
- ``ops.conj`` is the identity on real arrays and flips the imaginary sign
  on complex arrays.
- ``ops.vdot`` conjugates the FIRST argument (NumPy convention). JAX, Torch
  and CuPy must match.
- ``ops.real`` / ``ops.imag`` collapse a complex dtype to the matching real
  dtype (``complex64 → float32``, ``complex128 → float64``).
- ``spacecore.Space.field`` follows the context dtype: a complex context
  yields ``"complex"``; a real context yields ``"real"``.

The vdot conjugation pin previously lived in
``tests/test_backend_ops_complex.py`` (see the marker comment in that file)
and is now expressed here via the parametrized ``backend_ops`` fixture so
each installed backend is exercised once.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests._helpers import to_numpy
from tests.backend._conformance import (
    _canonicalize_dtype,
    assert_matches_reference,
    backend_supports_dtype,
)


@pytest.fixture
def numpy_ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


def _skip_if_unsupported(backend_ops, dtype):
    """Skip if ``backend_ops.family`` cannot honor ``dtype`` end-to-end."""
    if not backend_supports_dtype(backend_ops.family, dtype):
        pytest.skip(f"{backend_ops.family} does not natively support {np.dtype(dtype)}")


# ---------------------------------------------------------------------------
# is_complex_dtype


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.float32, False),
        (np.float64, False),
        (np.complex64, True),
        (np.complex128, True),
    ],
)
def test_is_complex_dtype_matches_np_kind(backend_ops, dtype, expected):
    """``is_complex_dtype`` matches ``np.dtype(dtype).kind == 'c'``."""
    _skip_if_unsupported(backend_ops, dtype)
    sanitized = backend_ops.sanitize_dtype(dtype)
    assert backend_ops.is_complex_dtype(sanitized) is expected
    # And matches the NumPy-side classification we'd use as ground truth.
    assert (np.dtype(dtype).kind == "c") is expected


# ---------------------------------------------------------------------------
# real_dtype


@pytest.mark.parametrize(
    "complex_in,real_out",
    [
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ],
)
def test_real_dtype_strips_complex(backend_ops, complex_in, real_out):
    """``real_dtype`` returns the matching real precision."""
    _skip_if_unsupported(backend_ops, complex_in)
    out = backend_ops.real_dtype(complex_in)
    assert _canonicalize_dtype(out) == np.dtype(real_out)


@pytest.mark.parametrize("real_dtype_in", [np.float32, np.float64])
def test_real_dtype_is_identity_on_real(backend_ops, real_dtype_in):
    """A real dtype passed to ``real_dtype`` round-trips unchanged."""
    _skip_if_unsupported(backend_ops, real_dtype_in)
    out = backend_ops.real_dtype(real_dtype_in)
    assert _canonicalize_dtype(out) == np.dtype(real_dtype_in)


# ---------------------------------------------------------------------------
# conj


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_conj_is_identity_on_real(backend_ops, numpy_ops, dtype):
    """``conj`` on a real array equals the input (no imaginary part to flip)."""
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([1.0, -2.0, 3.5, -4.25], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    out = backend_ops.conj(x)
    # NumPy reference: conj on a real array is the same array.
    assert_matches_reference("conj", out, src, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_conj_flips_imag_on_complex(backend_ops, numpy_ops, dtype):
    """``conj`` on a complex array negates the imaginary part."""
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([1 + 2j, -3 + 4j, 5 - 6j, 0 + 1j], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    out = backend_ops.conj(x)
    ref = np.conj(src)
    assert_matches_reference("conj", out, ref, dtype=dtype)
    # Explicit imag-sign-flip identity: imag(conj(x)) == -imag(x).
    out_imag = to_numpy(backend_ops.imag(out))
    src_imag = src.imag
    assert_matches_reference(
        "imag", out_imag, -src_imag, dtype=backend_ops.real_dtype(dtype)
    )


# ---------------------------------------------------------------------------
# vdot (NumPy convention: conjugates the FIRST argument)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_vdot_conjugates_first_argument(backend_ops, numpy_ops, dtype):
    """``vdot(x, y)`` equals ``sum(conj(x) * y)`` for complex inputs.

    This is the NumPy convention; JAX, Torch and CuPy must match it so that
    ``DenseLinOp.rapply`` and other inner-product code paths agree across
    backends.
    """
    _skip_if_unsupported(backend_ops, dtype)
    # Hand-picked so that swapping the conjugation would flip a sign we
    # can read off — (1+2j, 3+4j) · (5+6j, 7+8j) with conj on x is
    # 70 - 8j; with conj on y it would be 70 + 8j.
    x_src = np.asarray([1 + 2j, 3 + 4j], dtype=dtype)
    y_src = np.asarray([5 + 6j, 7 + 8j], dtype=dtype)
    x = backend_ops.asarray(x_src, dtype=dtype)
    y = backend_ops.asarray(y_src, dtype=dtype)
    out = backend_ops.vdot(x, y)

    # Reference 1: NumPy's own vdot (the authoritative convention).
    ref = np.vdot(x_src, y_src)
    assert_matches_reference("vdot", out, ref, dtype=dtype)

    # Reference 2: spelled-out sum(conj(x) * y) — independent of np.vdot,
    # catches a backend that accidentally conjugates the second argument.
    spelled_out = np.sum(np.conj(x_src) * y_src)
    assert_matches_reference("vdot", out, spelled_out, dtype=dtype)
    # And the value itself: 70 - 8j (real backends could short-circuit).
    np.testing.assert_allclose(to_numpy(out), 70.0 - 8.0j, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vdot_real_agrees_with_numpy(backend_ops, numpy_ops, dtype):
    """For real dtypes ``vdot`` collapses to a plain inner product."""
    _skip_if_unsupported(backend_ops, dtype)
    x_src = np.asarray([1.0, 2.0, 3.0], dtype=dtype)
    y_src = np.asarray([4.0, -1.0, 0.5], dtype=dtype)
    x = backend_ops.asarray(x_src, dtype=dtype)
    y = backend_ops.asarray(y_src, dtype=dtype)
    assert_matches_reference(
        "vdot", backend_ops.vdot(x, y), np.vdot(x_src, y_src), dtype=dtype
    )


# ---------------------------------------------------------------------------
# real / imag dtype


@pytest.mark.parametrize(
    "complex_in,real_out",
    [
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ],
)
def test_real_imag_collapse_to_real_dtype(backend_ops, complex_in, real_out):
    """``real(x)`` and ``imag(x)`` of a complex array carry the matching real dtype."""
    _skip_if_unsupported(backend_ops, complex_in)
    src = np.asarray([1 + 2j, 3 - 4j, -5 + 0j], dtype=complex_in)
    x = backend_ops.asarray(src, dtype=complex_in)

    re = backend_ops.real(x)
    im = backend_ops.imag(x)

    # Dtype pin: the imaginary half drops, precision is preserved.
    assert _canonicalize_dtype(re.dtype) == np.dtype(real_out)
    assert _canonicalize_dtype(im.dtype) == np.dtype(real_out)

    # Value pin: matches NumPy's componentwise split.
    assert_matches_reference("real", re, src.real, dtype=real_out)
    assert_matches_reference("imag", im, src.imag, dtype=real_out)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_on_real_input_is_identity(backend_ops, dtype):
    """On a real array ``real`` returns the array unchanged."""
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([1.0, -2.0, 3.5], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    assert_matches_reference("real", backend_ops.real(x), src, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_imag_on_real_input(backend_ops, dtype):
    """``imag`` on a real array is zero where supported.

    Torch's ``imag`` raises ``RuntimeError`` for non-complex tensors — that
    is the documented deviation captured in ``backend_deviations.rst``. We
    pin both shapes of the contract: backends that implement ``imag`` for
    real inputs return zeros; backends that don't, raise.
    """
    _skip_if_unsupported(backend_ops, dtype)
    src = np.asarray([1.0, -2.0, 3.5], dtype=dtype)
    x = backend_ops.asarray(src, dtype=dtype)
    if backend_ops.family == "torch":
        with pytest.raises(RuntimeError):
            backend_ops.imag(x)
        return
    assert_matches_reference(
        "imag", backend_ops.imag(x), np.zeros_like(src), dtype=dtype
    )


# ---------------------------------------------------------------------------
# Cross-link: spacecore.Space.field follows the context dtype


def test_dense_vector_space_field_is_real_for_real_ctx():
    """A real context yields ``space.field == 'real'``."""
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseVectorSpace((4,), ctx)
    assert space.field == "real"


def test_dense_vector_space_field_is_complex_for_complex_ctx():
    """A complex context yields ``space.field == 'complex'``."""
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    space = sc.DenseVectorSpace((4,), ctx)
    assert space.field == "complex"


@pytest.mark.parametrize(
    "dtype,expected_field",
    [
        (np.float32, "real"),
        (np.float64, "real"),
        (np.complex64, "complex"),
        (np.complex128, "complex"),
    ],
)
def test_dense_vector_space_field_tracks_is_complex_dtype(
    backend_ops, dtype, expected_field
):
    """``Space.field`` and ``ops.is_complex_dtype`` agree across backends."""
    _skip_if_unsupported(backend_ops, dtype)
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(backend_ops, dtype=dtype)
    space = sc.DenseVectorSpace((3,), ctx)
    assert space.field == expected_field
    # And the underlying classifier agrees on the sanitized dtype.
    assert backend_ops.is_complex_dtype(ctx.dtype) is (expected_field == "complex")

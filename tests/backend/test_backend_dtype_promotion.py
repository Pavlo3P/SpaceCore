"""Phase J7: dtype-promotion pins per ADR-015 Stage 1.

These tests pin the *actual* dtype-promotion behavior of each backend
rather than forcing the backends to agree. ADR-015 Stage 1 intentionally
leaves promotion as backend-native: NumPy follows NEP 50, JAX follows its
own rules (and is 32-bit unless ``jax_enable_x64`` is set), and Torch
follows ``torch.result_type``. Each test therefore looks up the expected
dtype in a per-backend ``EXPECTED`` dict and asserts the actual result
matches.

Categories covered:

* ``asarray(list_of_floats, dtype=None)`` — backend-default real dtype.
* ``matmul(float32, float64)`` — mixed-precision promotion.
* ``matmul(real, complex)`` — universal contract: complex wins.
* Python scalar + array — host-scalar promotion.
* Bool + float — kind promotion.
* ``matmul(float32, complex128)`` — mixed kind + precision.

A backend that does not support a particular dtype end-to-end (JAX
without ``jax_enable_x64`` for ``float64``/``complex128``, Torch with a
non-float64 default) is skipped via :func:`backend_supports_dtype`.

The expected-dtype lookup is intentionally small and local to this file
so the per-backend pins can be inspected and updated in one place when
the ecosystem moves.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests._helpers import (
    jax_real_dtype,
    to_numpy,
    torch_real_dtype,
)
from tests.backend._conformance import (
    assert_matches_reference,
    backend_supports_dtype,
)


# ---------------------------------------------------------------------------
# Helpers


def _skip_if_unsupported(backend_ops, dtype) -> None:
    """Skip when the backend cannot honor ``dtype`` end-to-end."""
    if not backend_supports_dtype(backend_ops.family, dtype):
        pytest.skip(
            f"{backend_ops.family} does not natively support {np.dtype(dtype)}"
        )


def _actual_dtype(backend_ops, x) -> np.dtype:
    """Normalize a backend dtype to a ``numpy.dtype`` for comparison.

    Torch dtypes do not convert directly via ``np.dtype``; we route through
    ``to_numpy`` so the comparison is uniform across families.
    """
    return to_numpy(x).dtype


def _default_real_for(family: str) -> np.dtype:
    """Backend-default real dtype, mirroring ``_helpers.*_real_dtype``."""
    if family == "numpy":
        # NEP 50: a python list of floats becomes float64.
        return np.dtype(np.float64)
    if family == "jax":
        # JAX honours ``jax_enable_x64``; default is float32.
        return np.dtype(jax_real_dtype())
    if family == "torch":
        # Torch honours ``torch.get_default_dtype()``.
        td = torch_real_dtype()
        # ``torch_real_dtype`` returns either a numpy dtype (when torch is
        # absent) or a ``torch.dtype``; normalize via str().
        name = str(td).rsplit(".", 1)[-1]
        return np.dtype(name)
    if family == "cupy":
        return np.dtype(np.float64)
    raise AssertionError(f"unknown backend family {family!r}")


@pytest.fixture
def numpy_ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


# ---------------------------------------------------------------------------
# Per-test expected-dtype lookups
#
# Each dict maps backend family -> expected numpy dtype. We resolve the
# family at test time because torch's default dtype is runtime-configurable
# and jax's x64 flag is process-global; encoding the rule (not a frozen
# answer) keeps the pins truthful when the environment changes.

# matmul(float32, float64):
#   - NumPy NEP 50: promotes to float64.
#   - JAX (default x32): keeps float32; JAX (x64): promotes to float64.
#   - Torch: ``torch.result_type(float32, float64) == float64``.
MATMUL_F32_F64_EXPECTED = {
    "numpy": lambda: np.dtype(np.float64),
    "jax": lambda: np.dtype(jax_real_dtype()),  # f32 unless x64 enabled
    "torch": lambda: np.dtype(np.float64),
    "cupy": lambda: np.dtype(np.float64),
}

# matmul(float32, complex128):
#   - NumPy: complex128.
#   - JAX (default x32): keeps complex64 (paired with its real precision).
#   - JAX (x64): complex128.
#   - Torch: ``torch.result_type(float32, complex128) == complex128``.
MATMUL_F32_C128_EXPECTED = {
    "numpy": lambda: np.dtype(np.complex128),
    "jax": lambda: np.dtype(
        np.complex128 if jax_real_dtype() == np.float64 else np.complex64
    ),
    "torch": lambda: np.dtype(np.complex128),
    "cupy": lambda: np.dtype(np.complex128),
}

# array(float32) + python scalar 1.5:
#   - NumPy NEP 50: weak-typed scalar keeps array dtype -> float32.
#   - JAX: weak scalar keeps array dtype -> float32.
#   - Torch: ``result_type(float32, python_float)`` keeps float32.
ARRAY_F32_PLUS_PYFLOAT_EXPECTED = {
    "numpy": lambda: np.dtype(np.float32),
    "jax": lambda: np.dtype(np.float32),
    "torch": lambda: np.dtype(np.float32),
    "cupy": lambda: np.dtype(np.float32),
}

# bool array + float array (32-bit float):
#   - All backends: kind promotion -> float dtype of the float operand.
BOOL_PLUS_F32_EXPECTED = {
    "numpy": lambda: np.dtype(np.float32),
    "jax": lambda: np.dtype(np.float32),
    "torch": lambda: np.dtype(np.float32),
    "cupy": lambda: np.dtype(np.float32),
}


# ---------------------------------------------------------------------------
# asarray default dtype


def test_asarray_python_floats_default_dtype(backend_ops):
    """``ops.asarray([1.0, ...], dtype=None)`` returns the backend default.

    NumPy: float64. JAX: float32 in default mode, float64 with x64 enabled.
    Torch: ``torch.get_default_dtype()`` (float32 unless the user set
    float64 globally). CuPy: float64.
    """
    expected = _default_real_for(backend_ops.family)
    _skip_if_unsupported(backend_ops, expected)
    out = backend_ops.asarray([1.0, 2.0, 3.0])
    assert _actual_dtype(backend_ops, out) == expected, (
        f"{backend_ops.family}: asarray default dtype "
        f"{_actual_dtype(backend_ops, out)} != expected {expected}"
    )


# ---------------------------------------------------------------------------
# matmul promotion: float32 vs float64


def test_matmul_f32_f64_promotion(backend_ops, numpy_ops):
    """Record the backend's promotion for ``matmul(f32, f64)``.

    NumPy NEP 50 promotes to float64. JAX in 32-bit default keeps float32
    (and so the output dtype tracks ``jax_real_dtype()``). Torch follows
    ``torch.result_type``, which yields float64.
    """
    expected = MATMUL_F32_F64_EXPECTED[backend_ops.family]()
    _skip_if_unsupported(backend_ops, expected)

    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.asarray([[5.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    if not backend_supports_dtype(backend_ops.family, np.float64):
        # JAX without x64 cannot honour the float64 operand; degrade B to
        # match the backend so the test still pins the *promotion result*.
        B = B.astype(np.float32)

    xa = backend_ops.asarray(A, dtype=np.float32)
    xb = backend_ops.asarray(B, dtype=B.dtype)
    out = backend_ops.matmul(xa, xb)
    assert _actual_dtype(backend_ops, out) == expected, (
        f"{backend_ops.family}: matmul(f32, f64) dtype "
        f"{_actual_dtype(backend_ops, out)} != expected {expected}"
    )

    # Value sanity-check at the result's precision.
    ref = numpy_ops.matmul(
        numpy_ops.asarray(A, dtype=expected),
        numpy_ops.asarray(B.astype(expected), dtype=expected),
    )
    assert_matches_reference("matmul", out, to_numpy(ref), dtype=expected)


# ---------------------------------------------------------------------------
# matmul promotion: real vs complex (universal contract)


@pytest.mark.parametrize(
    "real_dtype,complex_dtype",
    [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
    ],
)
def test_matmul_real_complex_yields_complex(
    backend_ops, numpy_ops, real_dtype, complex_dtype
):
    """``matmul(real, complex)`` produces a complex array where supported.

    NumPy, JAX, and CuPy auto-promote real-times-complex to complex.
    Torch refuses mixed dtypes outright and raises ``RuntimeError``; that
    is a documented deviation (see :doc:`/design/backend_deviations`).
    Callers on Torch must pre-promote operands with ``ops.astype``.
    """
    _skip_if_unsupported(backend_ops, real_dtype)
    _skip_if_unsupported(backend_ops, complex_dtype)

    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=real_dtype)
    B = np.asarray([[1 + 1j, 0 + 0j], [0 - 1j, 2 + 0j]], dtype=complex_dtype)
    xa = backend_ops.asarray(A, dtype=real_dtype)
    xb = backend_ops.asarray(B, dtype=complex_dtype)
    if backend_ops.family == "torch":
        # Documented deviation: pin that the call raises.
        with pytest.raises(RuntimeError):
            backend_ops.matmul(xa, xb)
        return
    out = backend_ops.matmul(xa, xb)
    assert _actual_dtype(backend_ops, out).kind == "c", (
        f"{backend_ops.family}: matmul(real, complex) lost the imaginary "
        f"part — got dtype {_actual_dtype(backend_ops, out)}"
    )

    ref = numpy_ops.matmul(
        numpy_ops.asarray(A.astype(complex_dtype), dtype=complex_dtype),
        numpy_ops.asarray(B, dtype=complex_dtype),
    )
    assert_matches_reference("matmul", out, to_numpy(ref), dtype=complex_dtype)


# ---------------------------------------------------------------------------
# Python scalar + array


def test_array_plus_python_scalar_keeps_array_dtype(backend_ops):
    """``ops.asarray([...], dtype=f32) + 1.5`` keeps float32.

    NumPy NEP 50 treats untyped Python scalars as weakly typed and lets
    the array dtype win. JAX does the same. Torch follows the same rule
    via ``result_type`` for python-float-on-float-tensor.
    """
    expected = ARRAY_F32_PLUS_PYFLOAT_EXPECTED[backend_ops.family]()
    _skip_if_unsupported(backend_ops, expected)

    x = backend_ops.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    out = x + 1.5
    assert _actual_dtype(backend_ops, out) == expected, (
        f"{backend_ops.family}: f32_array + 1.5 dtype "
        f"{_actual_dtype(backend_ops, out)} != expected {expected}"
    )
    # Numeric check at f32 precision.
    np.testing.assert_allclose(
        to_numpy(out), np.asarray([2.5, 3.5, 4.5], dtype=expected), rtol=1e-6
    )


# ---------------------------------------------------------------------------
# Bool + float


def test_bool_plus_float_promotes_to_float(backend_ops):
    """``bool_array + float32_array`` promotes to ``float32`` on every backend.

    This is a kind-promotion pin: integer-kind (bool) loses to float-kind,
    and the float operand's precision is preserved.
    """
    expected = BOOL_PLUS_F32_EXPECTED[backend_ops.family]()
    _skip_if_unsupported(backend_ops, expected)

    b_src = np.asarray([True, False, True])
    f_src = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    b = backend_ops.asarray(b_src)
    f = backend_ops.asarray(f_src, dtype=np.float32)
    out = b + f

    actual = _actual_dtype(backend_ops, out)
    assert actual == expected, (
        f"{backend_ops.family}: bool + f32 dtype {actual} != expected {expected}"
    )
    # Boolean addition: True->1.0, False->0.0; compare exactly via NumPy.
    expected_vals = np.asarray([2.0, 2.0, 4.0], dtype=expected)
    assert np.allclose(to_numpy(out), expected_vals, rtol=1e-6)


# ---------------------------------------------------------------------------
# Mixed precision matmul: float32 @ complex128


def test_matmul_f32_complex128(backend_ops, numpy_ops):
    """Mixed kind+precision matmul: ``f32 @ c128`` pinned per backend.

    NumPy promotes to ``complex128`` (kind wins, then precision). JAX in
    default 32-bit keeps ``complex64`` (both inputs are demoted to the
    backend's working precision); JAX with x64 matches NumPy. Torch
    follows ``torch.result_type(float32, complex128) == complex128``.
    """
    expected = MATMUL_F32_C128_EXPECTED[backend_ops.family]()
    _skip_if_unsupported(backend_ops, expected)

    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.asarray(
        [[1 + 1j, 0 + 0j], [0 - 1j, 2 + 0j]], dtype=np.complex128
    )
    # When the backend doesn't carry complex128, demote B so the matmul
    # still exercises *promotion* (not unsupported-dtype rejection).
    b_dtype = np.complex128
    if not backend_supports_dtype(backend_ops.family, np.complex128):
        b_dtype = np.complex64
        B = B.astype(b_dtype)

    xa = backend_ops.asarray(A, dtype=np.float32)
    xb = backend_ops.asarray(B, dtype=b_dtype)
    out = backend_ops.matmul(xa, xb)

    actual = _actual_dtype(backend_ops, out)
    assert actual == expected, (
        f"{backend_ops.family}: matmul(f32, c128) dtype {actual} "
        f"!= expected {expected}"
    )

    # Numerical correctness against NumPy at the result's precision.
    ref = numpy_ops.matmul(
        numpy_ops.asarray(A.astype(expected), dtype=expected),
        numpy_ops.asarray(B.astype(expected), dtype=expected),
    )
    assert_matches_reference("matmul", out, to_numpy(ref), dtype=expected)

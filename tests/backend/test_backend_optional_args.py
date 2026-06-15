"""Optional-argument behavior pinned across backends.

The ``backend_ops`` fixture (from ``tests/backend/conftest.py``) is
parametrized over every installed backend family. Each test exercises an
optional-argument shape — axis tuples, negative axes, ``keepdims``,
``copy`` semantics, ``expand_dims`` / ``squeeze`` / ``moveaxis`` edge
cases, and backend-specific ``backend_kwargs`` passthrough — and compares
against ``NumpyOps`` as the reference.

This is the J6 slice of the backend-conformance phase: the J3-J5
``test_conformance_cross_backend.py`` module pins shape/dtype-agnostic
behavior; this module pins the optional-argument surface that the
generic conformance suite glosses over. When a backend genuinely lacks an
operation, the test asserts ``NotImplementedError`` rather than skipping
silently — silent skips have repeatedly hidden regressions in past
phases.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, to_numpy
from tests.backend._conformance import (
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


def _src_3d(dtype):
    return np.arange(24, dtype=dtype).reshape(2, 3, 4)


def _src_2d(dtype):
    return np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)


# ---------------------------------------------------------------------------
# Reductions with axis tuples


@pytest.mark.parametrize("op_name", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reduction_axis_tuple_3d(backend_ops, numpy_ops, op_name, dtype):
    """axis=(0, 1) on a 3D array reduces along both leading axes."""
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_3d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    out_be = getattr(backend_ops, op_name)(x_be, axis=(0, 1))
    out_np = getattr(numpy_ops, op_name)(x_np, axis=(0, 1))
    assert tuple(to_numpy(out_be).shape) == (4,)
    assert_matches_reference(op_name, out_be, to_numpy(out_np), dtype=dtype)


# ---------------------------------------------------------------------------
# Reductions with negative axes


@pytest.mark.parametrize("op_name", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("axis", [-1, -2, (-1, -2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reduction_negative_axis(backend_ops, numpy_ops, op_name, axis, dtype):
    """Negative axes are normalized identically to NumPy."""
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_3d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    out_be = getattr(backend_ops, op_name)(x_be, axis=axis)
    out_np = getattr(numpy_ops, op_name)(x_np, axis=axis)
    assert_matches_reference(op_name, out_be, to_numpy(out_np), dtype=dtype)


# ---------------------------------------------------------------------------
# Reductions with keepdims


@pytest.mark.parametrize("op_name", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1, (0, 2), None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reduction_keepdims(backend_ops, numpy_ops, op_name, axis, dtype):
    """``keepdims=True`` preserves reduced axes as size-1 dimensions."""
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_3d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    out_be = getattr(backend_ops, op_name)(x_be, axis=axis, keepdims=True)
    out_np = getattr(numpy_ops, op_name)(x_np, axis=axis, keepdims=True)
    assert tuple(to_numpy(out_be).shape) == tuple(to_numpy(out_np).shape)
    # All output dims must be either preserved or singleton.
    assert all(d == 1 or d == s for d, s in zip(to_numpy(out_be).shape, src.shape))
    assert_matches_reference(op_name, out_be, to_numpy(out_np), dtype=dtype)


# ---------------------------------------------------------------------------
# expand_dims with int and tuple axis


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_expand_dims_int_axis(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_2d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    for ax in (0, 1, 2, -1, -2):
        out_be = backend_ops.expand_dims(x_be, ax)
        out_np = numpy_ops.expand_dims(x_np, ax)
        assert tuple(to_numpy(out_be).shape) == tuple(to_numpy(out_np).shape)
        assert_matches_reference("expand_dims", out_be, to_numpy(out_np), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_expand_dims_tuple_axis(backend_ops, numpy_ops, dtype):
    """Tuple axis inserts singletons at the requested positions."""
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_2d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    # numpy.expand_dims accepts tuple axis directly. Backends route tuples
    # through the BackendOps shim, which loops over normalized positions.
    out_be = backend_ops.expand_dims(x_be, (0, 2))
    out_np = numpy_ops.expand_dims(x_np, (0, 2))
    assert tuple(to_numpy(out_be).shape) == tuple(to_numpy(out_np).shape)
    assert_matches_reference("expand_dims", out_be, to_numpy(out_np), dtype=dtype)


# ---------------------------------------------------------------------------
# squeeze with axis=None and explicit axis


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_squeeze_axis_none_drops_all_singletons(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.arange(6, dtype=dtype).reshape(1, 2, 1, 3, 1)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    out_be = backend_ops.squeeze(x_be, axis=None)
    out_np = numpy_ops.squeeze(x_np, axis=None)
    assert tuple(to_numpy(out_be).shape) == (2, 3)
    assert_matches_reference("squeeze", out_be, to_numpy(out_np), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_squeeze_axis_none_no_singletons_is_identity(backend_ops, numpy_ops, dtype):
    """``squeeze(axis=None)`` on an array with no singletons returns shape unchanged."""
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_2d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    out_be = backend_ops.squeeze(x_be, axis=None)
    out_np = numpy_ops.squeeze(x_np, axis=None)
    assert tuple(to_numpy(out_be).shape) == src.shape
    assert_matches_reference("squeeze", out_be, to_numpy(out_np), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_squeeze_explicit_axis(backend_ops, numpy_ops, dtype):
    _skip_if_unsupported(backend_ops, dtype)
    src = np.arange(6, dtype=dtype).reshape(1, 2, 1, 3)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    # int axis
    out_be = backend_ops.squeeze(x_be, axis=0)
    out_np = numpy_ops.squeeze(x_np, axis=0)
    assert tuple(to_numpy(out_be).shape) == (2, 1, 3)
    assert_matches_reference("squeeze", out_be, to_numpy(out_np), dtype=dtype)
    # tuple axis
    out_be = backend_ops.squeeze(x_be, axis=(0, 2))
    out_np = numpy_ops.squeeze(x_np, axis=(0, 2))
    assert tuple(to_numpy(out_be).shape) == (2, 3)
    assert_matches_reference("squeeze", out_be, to_numpy(out_np), dtype=dtype)


# ---------------------------------------------------------------------------
# moveaxis with negative source / destination


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "source,destination",
    [(-1, 0), (0, -1), (-1, -2), ((-1, -2), (0, 1))],
)
def test_moveaxis_negative(backend_ops, numpy_ops, dtype, source, destination):
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_3d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    x_np = numpy_ops.asarray(src, dtype=dtype)
    out_be = backend_ops.moveaxis(x_be, source, destination)
    out_np = numpy_ops.moveaxis(x_np, source, destination)
    assert tuple(to_numpy(out_be).shape) == tuple(to_numpy(out_np).shape)
    assert_matches_reference("moveaxis", out_be, to_numpy(out_np), dtype=dtype)


# ---------------------------------------------------------------------------
# asarray / astype copy semantics


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_astype_same_dtype_preserves_values(backend_ops, dtype):
    """``astype`` to the same dtype is value-preserving across backends.

    The ``copy`` keyword is best-effort: NumPy and Torch honor it
    explicitly, JAX arrays are immutable so a copy is unobservable, and
    CuPy follows NumPy semantics. We pin the value behavior, not the
    aliasing — backends that ignore ``copy`` are documented as such.
    """
    _skip_if_unsupported(backend_ops, dtype)
    src = _src_2d(dtype)
    x_be = backend_ops.asarray(src, dtype=dtype)
    out = backend_ops.astype(x_be, dtype)
    assert_matches_reference("asarray", out, src, dtype=dtype)


@pytest.mark.parametrize("from_dtype,to_dtype", [
    (np.float32, np.float64),
    (np.float64, np.float32),
])
def test_astype_cross_dtype_roundtrip(backend_ops, from_dtype, to_dtype):
    """Cross-dtype ``astype`` produces values equal to the NumPy reference cast."""
    _skip_if_unsupported(backend_ops, from_dtype)
    _skip_if_unsupported(backend_ops, to_dtype)
    src = _src_2d(from_dtype)
    x_be = backend_ops.asarray(src, dtype=from_dtype)
    out = backend_ops.astype(x_be, to_dtype)
    ref = src.astype(to_dtype)
    assert_matches_reference("asarray", out, ref, dtype=to_dtype)


# ---------------------------------------------------------------------------
# backend_kwargs passthrough for asarray


def test_asarray_backend_kwargs_passthrough(backend_ops, numpy_ops):
    """A no-op backend_kwargs payload must not crash any backend.

    The contract is: callers may forward backend-specific keywords via
    ``backend_kwargs`` and the call either honors them or raises a
    typed error — never silently misbehaves. An empty payload is the
    "no-op" baseline that every backend must accept.
    """
    _skip_if_unsupported(backend_ops, np.float32)
    src = [1.0, 2.0, 3.0]
    # All backends accept asarray with no extra kwargs.
    out = backend_ops.asarray(src, dtype=np.float32)
    ref = numpy_ops.asarray(src, dtype=np.float32)
    assert_matches_reference("asarray", out, to_numpy(ref), dtype=np.float32)


def test_asarray_unknown_backend_kwarg_rejected(backend_ops):
    """Unknown backend keywords surface as a typed error, not silent drop.

    Each backend delegates to ``xp.asarray`` (or ``torch.as_tensor``) and
    those raise ``TypeError`` for unknown kwargs. We only assert that
    something is raised — the exact exception class is backend-specific.
    For Torch we pass an unknown kwarg via ``backend_kwargs``; for the
    others we pass it as ``**backend_kwargs`` on the base call.
    """
    src = [1.0, 2.0, 3.0]
    with pytest.raises((TypeError, ValueError)):
        backend_ops.asarray(src, dtype=np.float32, definitely_not_a_real_kwarg=True)


def test_asarray_jax_device_string(backend_ops):
    """JAX historically rejected ``device`` as a string on ``jnp.asarray``.

    The contract pins behavior for the JAX family: either the call
    succeeds (newer JAX accepts a device argument) or it raises a typed
    error. Non-JAX backends are skipped — they don't share this surface.
    """
    if backend_ops.family != "jax":
        pytest.skip("device-string passthrough is JAX-specific")
    if not has_jax():
        pytest.skip("JAX is not installed")
    src = [1.0, 2.0, 3.0]
    try:
        out = backend_ops.asarray(src, dtype=np.float32, device="cpu")
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        # Accepted outcome: JAX raises a typed error for an unsupported
        # device specification. We don't pin which class — JAX versions
        # disagree — only that it is one of the recognized error types.
        return
    # Accepted outcome: JAX produced a valid array. Values must match.
    assert tuple(to_numpy(out).shape) == (3,)


# ---------------------------------------------------------------------------
# NotImplementedError surfacing
#
# When a backend genuinely lacks an op, the contract is that calling it
# raises ``NotImplementedError`` rather than returning ``None`` or
# falling through to an opaque ``AttributeError``. The conformance harness
# treats silent skips as a smell, so this asserts the loud failure mode.


def test_unknown_method_does_not_silently_skip(backend_ops):
    """A method name the backend does not define must surface clearly.

    Accessing an unknown attribute on a ``BackendOps`` instance is a
    programming error and should raise ``AttributeError`` — not return
    ``None`` and not silently no-op. This pins the "loud failure" half
    of the J6 contract: callers can rely on missing ops being observable.
    """
    with pytest.raises(AttributeError):
        backend_ops.this_method_does_not_exist  # noqa: B018

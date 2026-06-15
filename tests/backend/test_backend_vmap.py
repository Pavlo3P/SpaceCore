"""Cross-backend conformance for ``ops.vmap``.

Phase J9 of the backend-conformance suite. The Python-loop fallback
provided by :class:`spacecore.backend._ops.BackendOps` and the native
implementations (``jax.vmap``, ``torch.vmap``) must agree on the
batching contract: shape, structure, and error semantics. This module
pins that contract against a Python loop built directly with
``NumpyOps`` as the reference, so a future native implementation cannot
silently drift from the fallback.

Coverage:

* ``ops.vmap`` exists on every backend.
* ``in_axes=0`` matches a Python loop for ``f(x) = ops.sum(x * x)``.
* ``in_axes=None`` is a no-op for that argument.
* Multi-arg ``in_axes=(0, None)`` batches only the first argument for
  ``f(x, y) = ops.matmul(x, y)``.
* Returning a tuple from the function is preserved by ``vmap``.
* Mismatched leaf batch sizes raise (any exception).
* For JAX/Torch, the native vmap path is exercised when
  ``ops.has_native_vmap`` is ``True``; for NumPy/CuPy, the fallback
  loop path is exercised.

The dtype is fixed at ``float64`` (with a soft skip on backends that do
not natively support 64-bit) to keep the focus on the batching contract
rather than dtype interactions, which Phase J3-J5 already cover.
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


_DTYPE = np.float64


@pytest.fixture
def numpy_ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


def _skip_if_unsupported(backend_ops, dtype):
    if not backend_supports_dtype(backend_ops.family, dtype):
        pytest.skip(f"{backend_ops.family} does not natively support {np.dtype(dtype)}")


# ---------------------------------------------------------------------------
# Existence


def test_vmap_exists_on_every_backend(backend_ops):
    """Every backend exposes ``ops.vmap`` and ``ops.has_native_vmap``."""
    assert callable(backend_ops.vmap)
    # has_native_vmap is a boolean property defined on the base class.
    assert isinstance(backend_ops.has_native_vmap, bool)


def test_native_vmap_flag_matches_family(backend_ops):
    """JAX and Torch advertise native vmap; NumPy and CuPy use the fallback."""
    if backend_ops.family in ("jax", "torch"):
        assert backend_ops.has_native_vmap is True
    elif backend_ops.family in ("numpy", "cupy"):
        assert backend_ops.has_native_vmap is False


# ---------------------------------------------------------------------------
# Single-argument batching: f(x) = sum(x * x)


def test_vmap_in_axes_0_matches_python_loop(backend_ops, numpy_ops):
    """``vmap`` over leading axis matches a Python loop element-by-element."""
    _skip_if_unsupported(backend_ops, _DTYPE)
    src = np.arange(12, dtype=_DTYPE).reshape(3, 4)
    x_be = backend_ops.asarray(src, dtype=_DTYPE)

    def f_be(x):
        return backend_ops.sum(x * x)

    out = backend_ops.vmap(f_be, in_axes=0)(x_be)

    # Reference: Python loop using the NumPy backend.
    x_ref = numpy_ops.asarray(src, dtype=_DTYPE)
    ref_rows = [numpy_ops.sum(x_ref[i] * x_ref[i]) for i in range(src.shape[0])]
    ref = numpy_ops.stack(ref_rows, axis=0)

    assert_matches_reference("sum", out, to_numpy(ref), dtype=_DTYPE)
    assert to_numpy(out).shape == (src.shape[0],)


def test_vmap_in_axes_none_is_noop(backend_ops, numpy_ops):
    """``in_axes=None`` does not batch the argument."""
    _skip_if_unsupported(backend_ops, _DTYPE)
    src = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=_DTYPE)
    x_be = backend_ops.asarray(src, dtype=_DTYPE)

    def f_be(x):
        return backend_ops.sum(x * x)

    out = backend_ops.vmap(f_be, in_axes=None)(x_be)
    expected = numpy_ops.sum(numpy_ops.asarray(src, dtype=_DTYPE) ** 2)

    assert_matches_reference("sum", out, to_numpy(expected), dtype=_DTYPE)
    # No new leading axis was introduced.
    assert to_numpy(out).shape == ()


# ---------------------------------------------------------------------------
# Multi-argument batching: f(x, y) = matmul(x, y) with in_axes=(0, None)


def test_vmap_multi_arg_in_axes_batches_only_first(backend_ops, numpy_ops):
    """``vmap(f, in_axes=(0, None))`` batches only ``x`` in ``matmul(x, y)``."""
    _skip_if_unsupported(backend_ops, _DTYPE)
    batch = 3
    src_x = np.arange(batch * 2 * 4, dtype=_DTYPE).reshape(batch, 2, 4)
    src_y = np.arange(4 * 5, dtype=_DTYPE).reshape(4, 5)

    x_be = backend_ops.asarray(src_x, dtype=_DTYPE)
    y_be = backend_ops.asarray(src_y, dtype=_DTYPE)

    def f_be(x, y):
        return backend_ops.matmul(x, y)

    out = backend_ops.vmap(f_be, in_axes=(0, None))(x_be, y_be)

    # Reference: Python loop using the NumPy backend.
    x_ref = numpy_ops.asarray(src_x, dtype=_DTYPE)
    y_ref = numpy_ops.asarray(src_y, dtype=_DTYPE)
    ref_rows = [numpy_ops.matmul(x_ref[i], y_ref) for i in range(batch)]
    ref = numpy_ops.stack(ref_rows, axis=0)

    assert_matches_reference("matmul", out, to_numpy(ref), dtype=_DTYPE)
    assert to_numpy(out).shape == (batch, 2, 5)


# ---------------------------------------------------------------------------
# Structured outputs


def test_vmap_preserves_tuple_output(backend_ops, numpy_ops):
    """A function returning a tuple is mapped leaf-by-leaf, structure preserved."""
    _skip_if_unsupported(backend_ops, _DTYPE)
    src = np.arange(12, dtype=_DTYPE).reshape(3, 4)
    x_be = backend_ops.asarray(src, dtype=_DTYPE)

    def f_be(x):
        return backend_ops.sum(x), backend_ops.sum(x * x)

    out = backend_ops.vmap(f_be, in_axes=0)(x_be)
    assert isinstance(out, tuple)
    assert len(out) == 2

    x_ref = numpy_ops.asarray(src, dtype=_DTYPE)
    ref0 = numpy_ops.stack([numpy_ops.sum(x_ref[i]) for i in range(src.shape[0])], axis=0)
    ref1 = numpy_ops.stack(
        [numpy_ops.sum(x_ref[i] * x_ref[i]) for i in range(src.shape[0])], axis=0
    )
    assert_matches_reference("sum", out[0], to_numpy(ref0), dtype=_DTYPE)
    assert_matches_reference("sum", out[1], to_numpy(ref1), dtype=_DTYPE)


# ---------------------------------------------------------------------------
# Error semantics


def test_vmap_mismatched_batch_sizes_raises(backend_ops):
    """Vmap over inputs whose mapped axes disagree on size must raise.

    The Python-loop fallback sizes the loop from the first mapped argument
    and detects the mismatch only when a later argument runs out of slices
    along the mapped axis (raising ``IndexError``). Native ``jax.vmap``
    raises eagerly at call time. Either path is acceptable — the contract
    is "any exception".
    """
    _skip_if_unsupported(backend_ops, _DTYPE)
    # x has 5 batches, y has 3. The fallback loops 5 times and y[3] raises.
    src_x = np.arange(20, dtype=_DTYPE).reshape(5, 4)
    src_y = np.arange(12, dtype=_DTYPE).reshape(3, 4)
    x_be = backend_ops.asarray(src_x, dtype=_DTYPE)
    y_be = backend_ops.asarray(src_y, dtype=_DTYPE)

    def f_be(x, y):
        return backend_ops.sum(x * y)

    mapped = backend_ops.vmap(f_be, in_axes=(0, 0))
    with pytest.raises(Exception):
        result = mapped(x_be, y_be)
        # Some lazy backends materialize only on conversion; force it.
        to_numpy(result)


# ---------------------------------------------------------------------------
# Native vs fallback path coverage


def test_jax_exercises_native_vmap_path(backend_ops):
    """For JAX, ``vmap`` is the native ``jax.vmap`` and returns ``jax.Array``."""
    if backend_ops.family != "jax":
        pytest.skip("JAX-specific path test")
    if not has_jax():
        pytest.skip("JAX is not installed")
    assert backend_ops.has_native_vmap is True

    import jax  # noqa: F401  (imported inside the test, guarded)
    import jax.numpy as jnp

    _skip_if_unsupported(backend_ops, _DTYPE)
    src = np.arange(6, dtype=_DTYPE).reshape(2, 3)
    x_be = backend_ops.asarray(src, dtype=_DTYPE)

    def f_be(x):
        return backend_ops.sum(x * x)

    mapped = backend_ops.vmap(f_be, in_axes=0)
    out = mapped(x_be)
    # The native path returns a JAX array; the fallback would return a
    # stacked backend array too, but it would not be produced by jax.vmap.
    assert isinstance(out, jnp.ndarray)


def test_fallback_loop_path_for_numpy_and_cupy(backend_ops, numpy_ops):
    """NumPy and CuPy must not advertise native vmap and must still agree."""
    if backend_ops.family not in ("numpy", "cupy"):
        pytest.skip("Fallback-loop path test")
    assert backend_ops.has_native_vmap is False

    _skip_if_unsupported(backend_ops, _DTYPE)
    src = np.arange(8, dtype=_DTYPE).reshape(4, 2)
    x_be = backend_ops.asarray(src, dtype=_DTYPE)

    def f_be(x):
        return backend_ops.sum(x * x)

    out = backend_ops.vmap(f_be, in_axes=0)(x_be)

    x_ref = numpy_ops.asarray(src, dtype=_DTYPE)
    ref = numpy_ops.stack(
        [numpy_ops.sum(x_ref[i] * x_ref[i]) for i in range(src.shape[0])], axis=0
    )
    assert_matches_reference("sum", out, to_numpy(ref), dtype=_DTYPE)

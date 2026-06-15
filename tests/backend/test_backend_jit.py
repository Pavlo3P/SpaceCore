"""JIT-compatibility pins for the JAX backend (Phase J10).

This module declares the subset of ``BackendOps`` that is safe to call
inside ``jax.jit``. The harness in
:mod:`tests.backend._conformance` defines the per-op tolerance table; we
reuse it here so jit-compiled results land within the same envelope as
their eager counterparts.

What this module pins:

* ``NumpyOps`` is **not** traceable. Calling ``ops.asarray`` on a JAX
  tracer must raise — we accept any of ``TypeError``,
  ``NotImplementedError``, or JAX's ``TracerArrayConversionError`` /
  ``ConcretizationTypeError`` to keep the harness robust across JAX
  versions.
* ``JaxOps.{asarray, matmul, sum, eigh, solve}`` work end-to-end under
  ``jax.jit`` and match eager evaluation within the per-op tolerance.
* The static-shape introspection helpers
  ``JaxOps.{is_array, shape, ndim, get_dtype}`` are safe to call inside
  ``jit`` because they read trace-time metadata, not concrete values.
* The control-flow primitives
  ``JaxOps.{fori_loop, cond, while_loop}`` are jit-compatible.

Negative cases use substring matching on JAX's tracer-conversion error
classes; we never pin an exact message, because JAX rewords these
between minor versions.

The whole module is skipped when JAX is not installed.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, jax_real_dtype, to_numpy
from tests.backend._conformance import (
    assert_matches_reference,
    backend_supports_dtype,
)

pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


# ---------------------------------------------------------------------------
# Fixtures local to this module: we intentionally do not use the
# parametrized ``backend_ops`` fixture from conftest.py — every test here
# targets JAX (or NumPy, as a negative case) explicitly.


@pytest.fixture
def jax_ops():
    sc = importlib.import_module("spacecore")
    if not hasattr(sc, "JaxOps"):
        pytest.skip("JaxOps is not exported (JAX not installed).")
    return sc.JaxOps()


@pytest.fixture
def numpy_ops():
    sc = importlib.import_module("spacecore")
    return sc.NumpyOps()


@pytest.fixture
def jit_real_dtype():
    """Pick the largest real dtype JAX honors end-to-end in this session."""
    dt = jax_real_dtype()
    if not backend_supports_dtype("jax", dt):
        pytest.skip(f"jax does not natively support {np.dtype(dt)}")
    return dt


# ---------------------------------------------------------------------------
# NumpyOps is not traceable.


def test_numpy_ops_not_traceable_under_jit(numpy_ops):
    """NumpyOps inside ``jax.jit`` must raise — it has no tracing support.

    JAX reworded the relevant errors between releases (``TypeError`` →
    ``TracerArrayConversionError`` → ``ConcretizationTypeError``). We
    accept any of them; the pin is "it raises", not "it raises X".
    """
    import jax
    import jax.numpy as jnp

    expected = (
        TypeError,
        NotImplementedError,
        getattr(jax.errors, "TracerArrayConversionError", Exception),
        getattr(jax.errors, "ConcretizationTypeError", Exception),
    )

    @jax.jit
    def fn(x):
        # NumpyOps.asarray ultimately calls np.asarray, which cannot accept
        # a JAX tracer object.
        return numpy_ops.sum(numpy_ops.asarray(x))

    with pytest.raises(expected):
        fn(jnp.asarray([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# JaxOps numerics under jit: asarray / matmul / sum / eigh / solve.


def test_jit_asarray_and_sum(jax_ops, numpy_ops, jit_real_dtype):
    import jax

    src = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=jit_real_dtype)

    @jax.jit
    def fn(x):
        return jax_ops.sum(jax_ops.asarray(x))

    out_jit = fn(src)
    out_eager = jax_ops.sum(jax_ops.asarray(src))
    ref = numpy_ops.sum(numpy_ops.asarray(src, dtype=jit_real_dtype))

    assert_matches_reference("sum", out_jit, to_numpy(ref), dtype=jit_real_dtype)
    assert_matches_reference("sum", out_jit, to_numpy(out_eager), dtype=jit_real_dtype)


def test_jit_matmul(jax_ops, numpy_ops, jit_real_dtype):
    import jax

    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jit_real_dtype)
    B = np.asarray([[5.0, 0.0], [1.0, 2.0]], dtype=jit_real_dtype)

    @jax.jit
    def fn(a, b):
        return jax_ops.matmul(a, b)

    xa = jax_ops.asarray(A, dtype=jit_real_dtype)
    xb = jax_ops.asarray(B, dtype=jit_real_dtype)
    out_jit = fn(xa, xb)
    ref = numpy_ops.matmul(
        numpy_ops.asarray(A, dtype=jit_real_dtype),
        numpy_ops.asarray(B, dtype=jit_real_dtype),
    )

    assert_matches_reference("matmul", out_jit, to_numpy(ref), dtype=jit_real_dtype)


def test_jit_eigh(jax_ops, jit_real_dtype):
    """``eigh`` under jit must match its eager output.

    We compare jit vs eager (rather than vs NumPy) because eigenvector
    signs are ambiguous and not portable between LAPACK builds. The
    eager-vs-NumPy identity is pinned by the cross-backend conformance
    suite already.
    """
    import jax

    A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jit_real_dtype)

    @jax.jit
    def fn(x):
        return jax_ops.eigh(x)

    xA = jax_ops.asarray(A, dtype=jit_real_dtype)
    eig_jit, vec_jit = fn(xA)
    eig_eager, vec_eager = jax_ops.eigh(xA)

    assert_matches_reference("eigh", eig_jit, to_numpy(eig_eager), dtype=jit_real_dtype)
    # Reconstruction A = V diag(λ) V^T is sign-stable.
    recon_jit = jax_ops.matmul(
        jax_ops.matmul(vec_jit, jax_ops.diag(eig_jit)),
        jax_ops.transpose(vec_jit),
    )
    assert_matches_reference("eigh", recon_jit, A, dtype=jit_real_dtype)


def test_jit_solve(jax_ops, numpy_ops, jit_real_dtype):
    import jax

    A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jit_real_dtype)
    b = np.asarray([1.0, 2.0], dtype=jit_real_dtype)

    @jax.jit
    def fn(a, rhs):
        return jax_ops.solve(a, rhs)

    out_jit = fn(
        jax_ops.asarray(A, dtype=jit_real_dtype),
        jax_ops.asarray(b, dtype=jit_real_dtype),
    )
    ref = numpy_ops.solve(
        numpy_ops.asarray(A, dtype=jit_real_dtype),
        numpy_ops.asarray(b, dtype=jit_real_dtype),
    )
    assert_matches_reference("solve", out_jit, to_numpy(ref), dtype=jit_real_dtype)


# ---------------------------------------------------------------------------
# Static-shape introspection helpers are safe on traced inputs.


def test_jit_static_shape_helpers(jax_ops, jit_real_dtype):
    """``is_array``, ``shape``, ``ndim``, ``get_dtype`` work on tracers.

    These helpers read trace-time metadata only — they must not force
    concretization of values, so they are safe inside ``jax.jit``.
    """
    import jax

    src = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jit_real_dtype)
    captured: dict[str, object] = {}

    @jax.jit
    def fn(x):
        captured["is_array"] = jax_ops.is_array(x)
        captured["shape"] = jax_ops.shape(x)
        captured["ndim"] = jax_ops.ndim(x)
        captured["dtype"] = jax_ops.get_dtype(x)
        return jax_ops.sum(x)

    out = fn(jax_ops.asarray(src, dtype=jit_real_dtype))
    # The function returns; the helpers populated their results from the
    # tracer without raising.
    assert captured["is_array"] is True
    assert tuple(captured["shape"]) == (2, 3)
    assert int(captured["ndim"]) == 2
    assert np.dtype(captured["dtype"]) == np.dtype(jit_real_dtype)
    # Sanity check that the body actually executed.
    assert float(to_numpy(out)) == pytest.approx(float(src.sum()))


# ---------------------------------------------------------------------------
# Control flow under jit.


def test_jit_fori_loop(jax_ops, jit_real_dtype):
    import jax

    @jax.jit
    def fn(init):
        def body(i, carry):
            return carry + jax_ops.asarray(i, dtype=jit_real_dtype)

        return jax_ops.fori_loop(0, 5, body, init)

    out = fn(jax_ops.asarray(0.0, dtype=jit_real_dtype))
    assert_matches_reference(
        "sum",
        out,
        np.asarray(10.0, dtype=jit_real_dtype),
        dtype=jit_real_dtype,
    )


def test_jit_cond(jax_ops, jit_real_dtype):
    import jax

    @jax.jit
    def fn(pred, x):
        return jax_ops.cond(
            pred,
            lambda v: v * jax_ops.asarray(2.0, dtype=jit_real_dtype),
            lambda v: v * jax_ops.asarray(-1.0, dtype=jit_real_dtype),
            x,
        )

    x = jax_ops.asarray(3.0, dtype=jit_real_dtype)
    out_true = fn(True, x)
    out_false = fn(False, x)
    assert_matches_reference(
        "sum", out_true, np.asarray(6.0, dtype=jit_real_dtype), dtype=jit_real_dtype
    )
    assert_matches_reference(
        "sum", out_false, np.asarray(-3.0, dtype=jit_real_dtype), dtype=jit_real_dtype
    )


def test_jit_while_loop(jax_ops, jit_real_dtype):
    import jax

    limit = jax_ops.asarray(4.0, dtype=jit_real_dtype)
    one = jax_ops.asarray(1.0, dtype=jit_real_dtype)

    @jax.jit
    def fn(init):
        return jax_ops.while_loop(
            lambda c: c < limit,
            lambda c: c + one,
            init,
        )

    out = fn(jax_ops.asarray(0.0, dtype=jit_real_dtype))
    assert_matches_reference(
        "sum",
        out,
        np.asarray(4.0, dtype=jit_real_dtype),
        dtype=jit_real_dtype,
    )

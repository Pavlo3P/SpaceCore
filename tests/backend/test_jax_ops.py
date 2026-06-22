"""JaxOps-specific tests.

Generic operation conformance lives in :mod:`tests.backend.test_operations`;
this module covers behavior that is *only* meaningful for ``JaxOps``:

* the ``jax`` family identifier and the ``jax.numpy`` xp namespace;
* default representation dtype tracking ``jax_enable_x64``;
* native ``jax.vmap`` integration (not the Python-loop fallback);
* JIT round-trip correctness for representative ops, including control flow
  (``fori_loop``, ``while_loop``, ``cond``) and static-shape introspection
  inside ``jax.jit``;
* refusal of NumPy/Torch behavior under JIT (``NumpyOps`` is not traceable);
* ``__eq__`` / ``__hash__`` / ``__repr__`` of ``JaxOps`` instances.

Skipped wholesale when JAX is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import has_jax, jax_real_dtype
from tests.backend._conformance import assert_matches_reference, backend_supports_dtype

pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


@pytest.fixture
def ops():
    import spacecore as sc

    return sc.JaxOps()


@pytest.fixture
def jit_dtype():
    """Largest real dtype JAX honors end-to-end in this session."""
    dt = jax_real_dtype()
    if not backend_supports_dtype("jax", dt):
        pytest.skip(f"jax does not natively support {np.dtype(dt)}")
    return dt


# ---------------------------------------------------------------------------
# Family and capability flags
# ---------------------------------------------------------------------------
def test_jax_ops_family_string(ops):
    assert ops.family == "jax"


def test_jax_ops_allow_sparse_is_true(ops):
    """JaxOps reports allow_sparse=True; JAX exposes BCOO/BCSR experimentally."""
    assert ops.allow_sparse is True


def test_jax_ops_has_native_vmap_is_true(ops):
    assert ops.has_native_vmap is True


def test_jax_ops_xp_is_jax_numpy():
    import spacecore as sc

    assert sc.JaxOps.xp is sc.JaxOps.jnp


# ---------------------------------------------------------------------------
# Dtype defaulting
# ---------------------------------------------------------------------------
def test_jax_ops_default_dtype_tracks_x64_config(ops):
    """Default sanitize_dtype follows ``jax_enable_x64`` (x32 → float32)."""
    import jax

    expected = ops.jnp.float64 if bool(jax.config.read("jax_enable_x64")) else ops.jnp.float32
    sanitized = ops.sanitize_dtype(None)
    assert sanitized == expected


def test_jax_ops_eps_default(ops, jit_dtype):
    assert ops.eps(jit_dtype) == pytest.approx(float(np.finfo(jit_dtype).eps))


# ---------------------------------------------------------------------------
# Equality, hash, repr
# ---------------------------------------------------------------------------
def test_jax_ops_equality_and_hash():
    import spacecore as sc

    a = sc.JaxOps()
    b = sc.JaxOps()
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1, b: 2} == {a: 2}


def test_jax_ops_repr():
    import spacecore as sc

    assert "JaxOps" in repr(sc.JaxOps())
    assert "family='jax'" in repr(sc.JaxOps())


# ---------------------------------------------------------------------------
# Native vmap path (vs the Python-loop fallback used by NumPy/CuPy)
# ---------------------------------------------------------------------------
def test_jax_ops_native_vmap_returns_jax_array(ops, jit_dtype):
    """``ops.vmap(fn, in_axes=0)`` takes the ``jax.vmap`` path."""
    import jax.numpy as jnp

    src = np.arange(6, dtype=jit_dtype).reshape(2, 3)
    x_be = ops.asarray(src, dtype=jit_dtype)

    def fn(x):
        return ops.sum(x * x)

    out = ops.vmap(fn, in_axes=0)(x_be)
    assert isinstance(out, jnp.ndarray)
    expected = np.asarray([float((src[i] * src[i]).sum()) for i in range(src.shape[0])])
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# JIT round-trip — selected ops that BackendOps documents as jit-compatible
# ---------------------------------------------------------------------------
def test_jax_ops_jit_asarray_and_sum(ops, jit_dtype):
    import jax

    src = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=jit_dtype)

    @jax.jit
    def fn(x):
        return ops.sum(ops.asarray(x))

    out_jit = fn(src)
    out_eager = ops.sum(ops.asarray(src))
    assert_matches_reference("sum", out_jit, np.asarray(src).sum(), dtype=jit_dtype)
    assert_matches_reference("sum", out_jit, np.asarray(out_eager), dtype=jit_dtype)


def test_jax_ops_jit_matmul(ops, jit_dtype):
    import jax

    A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jit_dtype)
    B = np.asarray([[5.0, 0.0], [1.0, 2.0]], dtype=jit_dtype)

    @jax.jit
    def fn(a, b):
        return ops.matmul(a, b)

    out_jit = fn(ops.asarray(A, dtype=jit_dtype), ops.asarray(B, dtype=jit_dtype))
    assert_matches_reference("matmul", out_jit, A @ B, dtype=jit_dtype)


def test_jax_ops_jit_eigh(ops, jit_dtype):
    """Compare jit vs eager; eigenvector signs are not portable to NumPy."""
    import jax

    A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jit_dtype)
    xA = ops.asarray(A, dtype=jit_dtype)

    @jax.jit
    def fn(x):
        return ops.eigh(x)

    eig_jit, vec_jit = fn(xA)
    eig_eager, _ = ops.eigh(xA)
    assert_matches_reference("eigh", eig_jit, np.asarray(eig_eager), dtype=jit_dtype)
    # Reconstruction A = V diag(λ) V^T is sign-stable.
    recon = ops.matmul(ops.matmul(vec_jit, ops.diag(eig_jit)), ops.transpose(vec_jit))
    assert_matches_reference("eigh", recon, A, dtype=jit_dtype)


def test_jax_ops_jit_solve(ops, jit_dtype):
    import jax

    A = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jit_dtype)
    b = np.asarray([1.0, 2.0], dtype=jit_dtype)
    expected = np.linalg.solve(A.astype(np.float64), b.astype(np.float64)).astype(jit_dtype)

    @jax.jit
    def fn(a, rhs):
        return ops.solve(a, rhs)

    out_jit = fn(ops.asarray(A, dtype=jit_dtype), ops.asarray(b, dtype=jit_dtype))
    assert_matches_reference("solve", out_jit, expected, dtype=jit_dtype)


def test_jax_ops_jit_static_shape_helpers(ops, jit_dtype):
    """``is_array``, ``shape``, ``ndim``, ``get_dtype`` work on tracers."""
    import jax

    src = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jit_dtype)
    captured: dict[str, object] = {}

    @jax.jit
    def fn(x):
        captured["is_array"] = ops.is_array(x)
        captured["shape"] = ops.shape(x)
        captured["ndim"] = ops.ndim(x)
        captured["dtype"] = ops.get_dtype(x)
        return ops.sum(x)

    out = fn(ops.asarray(src, dtype=jit_dtype))
    assert captured["is_array"] is True
    assert tuple(captured["shape"]) == (2, 3)
    assert int(captured["ndim"]) == 2
    assert np.dtype(captured["dtype"]) == np.dtype(jit_dtype)
    assert float(np.asarray(out)) == pytest.approx(float(src.sum()))


# ---------------------------------------------------------------------------
# Control flow under JIT
# ---------------------------------------------------------------------------
def test_jax_ops_jit_fori_loop(ops, jit_dtype):
    import jax

    @jax.jit
    def fn(init):
        def body(i, carry):
            return carry + ops.asarray(i, dtype=jit_dtype)

        return ops.fori_loop(0, 5, body, init)

    out = fn(ops.asarray(0.0, dtype=jit_dtype))
    assert_matches_reference("sum", out, np.asarray(10.0, dtype=jit_dtype), dtype=jit_dtype)


def test_jax_ops_jit_cond(ops, jit_dtype):
    import jax

    @jax.jit
    def fn(pred, x):
        return ops.cond(
            pred,
            lambda v: v * ops.asarray(2.0, dtype=jit_dtype),
            lambda v: v * ops.asarray(-1.0, dtype=jit_dtype),
            x,
        )

    x = ops.asarray(3.0, dtype=jit_dtype)
    assert_matches_reference("sum", fn(True, x), np.asarray(6.0, dtype=jit_dtype), dtype=jit_dtype)
    assert_matches_reference("sum", fn(False, x), np.asarray(-3.0, dtype=jit_dtype), dtype=jit_dtype)


def test_jax_ops_jit_while_loop(ops, jit_dtype):
    import jax

    limit = ops.asarray(4.0, dtype=jit_dtype)
    one = ops.asarray(1.0, dtype=jit_dtype)

    @jax.jit
    def fn(init):
        return ops.while_loop(lambda c: c < limit, lambda c: c + one, init)

    out = fn(ops.asarray(0.0, dtype=jit_dtype))
    assert_matches_reference("sum", out, np.asarray(4.0, dtype=jit_dtype), dtype=jit_dtype)


# ---------------------------------------------------------------------------
# Negative: NumpyOps is not jit-traceable
# ---------------------------------------------------------------------------
def test_numpy_ops_is_not_traceable_under_jax_jit():
    """``NumpyOps`` inside ``jax.jit`` must raise.

    JAX has reworded the relevant errors between releases; the contract is
    "any of these", not "exactly that message".
    """
    import jax
    import jax.numpy as jnp
    import spacecore as sc

    numpy_ops = sc.NumpyOps()
    expected = (
        TypeError,
        NotImplementedError,
        getattr(jax.errors, "TracerArrayConversionError", Exception),
        getattr(jax.errors, "ConcretizationTypeError", Exception),
    )

    @jax.jit
    def fn(x):
        return numpy_ops.sum(numpy_ops.asarray(x))

    with pytest.raises(expected):
        fn(jnp.asarray([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Constants and astype(None)
# ---------------------------------------------------------------------------
def test_jax_ops_constants_are_cached(ops):
    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e


def test_jax_ops_astype_none_is_identity(ops, jit_dtype):
    x = ops.asarray([1.0, 2.0], dtype=jit_dtype)
    assert ops.astype(x, None) is x

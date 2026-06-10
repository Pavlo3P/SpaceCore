import importlib

import numpy as np
import pytest

from tests._helpers import (
    has_jax,
    has_torch,
    jax_complex_dtype,
    jax_real_dtype,
    to_numpy,
    torch_complex_dtype,
    torch_real_dtype,
)


DELEGATED_METHODS = (
    "reshape",
    "sum",
    "eigh",
    "trace",
    "concatenate",
    "transpose",
    "matmul",
)

TORCH_AAC_DELEGATED_METHODS = (
    "mean",
    "prod",
    "sort",
    "argsort",
    "argmin",
    "argmax",
    "clip",
    "take",
    "diagonal",
    "squeeze",
)


def test_numpy_ops_inherits_common_delegated_methods():
    sc = importlib.import_module("spacecore")

    assert sc.NumpyOps.xp.__name__ == "array_api_compat.numpy"
    for name in DELEGATED_METHODS:
        assert getattr(sc.NumpyOps, name) is getattr(sc.BackendOps, name)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_ops_inherits_common_delegated_methods():
    sc = importlib.import_module("spacecore")

    assert sc.JaxOps.xp is sc.JaxOps.jnp
    for name in DELEGATED_METHODS:
        assert getattr(sc.JaxOps, name) is getattr(sc.BackendOps, name)


def test_torch_ops_uses_aac_namespace_when_available():
    sc = importlib.import_module("spacecore")

    assert sc.TorchOps.xp.__name__ == "array_api_compat.torch"


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_ops_inherits_aac_delegated_methods():
    sc = importlib.import_module("spacecore")

    for name in TORCH_AAC_DELEGATED_METHODS:
        assert getattr(sc.TorchOps, name) is getattr(sc.BackendOps, name)


def _check_raw_delegated_ops(ops, dtype):
    x = ops.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
    y = ops.reshape(x, (3, 2))
    h = ops.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=dtype)
    cube = ops.reshape(ops.arange(24, dtype=dtype), (2, 3, 4))
    singleton = ops.reshape(ops.arange(6, dtype=dtype), (1, 2, 1, 3))

    np.testing.assert_allclose(
        to_numpy(ops.matmul(h, ops.asarray([1.0, 2.0], dtype=dtype))), [4.0, 7.0]
    )
    np.testing.assert_allclose(to_numpy(ops.reshape(x, (6,))), np.arange(1.0, 7.0))
    np.testing.assert_allclose(to_numpy(ops.sum(x, axis=0)), [5.0, 7.0, 9.0])
    np.testing.assert_allclose(
        to_numpy(ops.sum(cube, axis=(0, 2))), np.arange(24.0).reshape(2, 3, 4).sum(axis=(0, 2))
    )
    np.testing.assert_allclose(
        to_numpy(ops.prod(cube + 1, axis=(0, 2))),
        (np.arange(24.0).reshape(2, 3, 4) + 1).prod(axis=(0, 2)),
    )
    np.testing.assert_allclose(
        to_numpy(ops.mean(cube, axis=(0, 2))), np.arange(24.0).reshape(2, 3, 4).mean(axis=(0, 2))
    )
    assert ops.shape(ops.squeeze(singleton)) == (2, 3)
    np.testing.assert_allclose(to_numpy(ops.trace(h)), 5.0)
    np.testing.assert_allclose(
        to_numpy(ops.concatenate((x, x), axis=0)),
        np.concatenate((to_numpy(x), to_numpy(x)), axis=0),
    )
    np.testing.assert_allclose(to_numpy(ops.transpose(y)), to_numpy(y).T)

    evals, evecs = ops.eigh(h)
    np.testing.assert_allclose(
        to_numpy(evecs @ ops.diag(evals) @ ops.transpose(evecs)), to_numpy(h), atol=1e-6
    )


def test_numpy_raw_delegated_ops():
    sc = importlib.import_module("spacecore")

    _check_raw_delegated_ops(sc.NumpyOps(), np.float64)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_raw_delegated_ops():
    sc = importlib.import_module("spacecore")

    _check_raw_delegated_ops(sc.JaxOps(), jax_real_dtype())


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_raw_delegated_ops():
    sc = importlib.import_module("spacecore")

    _check_raw_delegated_ops(sc.TorchOps(), torch_real_dtype())


def test_numpy_eps_uses_default_dtype():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()

    np.testing.assert_allclose(
        ops.eps(ops.sanitize_dtype(None)), np.finfo(ops.sanitize_dtype(None)).eps
    )


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_eps_uses_default_dtype():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()

    np.testing.assert_allclose(ops.eps(jax_real_dtype()), np.finfo(jax_real_dtype()).eps)


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_eps_uses_default_dtype():
    sc = importlib.import_module("spacecore")
    import torch

    ops = sc.TorchOps()
    np.testing.assert_allclose(
        ops.eps(ops.sanitize_dtype(None)), torch.finfo(ops.sanitize_dtype(None)).eps
    )


def test_backend_ops_hash_and_repr():
    sc = importlib.import_module("spacecore")
    first = sc.NumpyOps()
    second = sc.NumpyOps()

    assert hash(first) == hash(second)
    assert {first: 1, second: 2} == {first: 2}
    assert repr(first) == "NumpyOps(family='numpy')"


def test_numpy_eps_distinguishes_dtype_precision():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()

    assert ops.eps(np.float64) < ops.eps(np.float32)


def test_numpy_constants_are_cached_and_astype_none_is_noop():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = ops.asarray([1.0, 2.0])

    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e
    assert ops.astype(x, None) is x


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_constants_are_cached_and_astype_none_is_noop():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    x = ops.asarray([1.0, 2.0], dtype=jax_real_dtype())

    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e
    assert ops.astype(x, None) is x


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_constants_are_cached_and_astype_none_is_noop():
    sc = importlib.import_module("spacecore")
    ops = sc.TorchOps()
    x = ops.asarray([1.0, 2.0], dtype=torch_real_dtype())

    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e
    assert ops.astype(x, None) is x


def _check_complex_adjoint(ops, dtype):
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(ops, dtype=dtype)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    A = ctx.asarray(
        [
            [1.0 + 2.0j, 3.0 - 1.0j],
            [-2.0 + 0.5j, 0.25 + 4.0j],
            [1.5 - 3.0j, -0.75 + 2.0j],
        ]
    )
    x = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
    y = ctx.asarray([1.0 + 0.5j, -2.0j, 0.75 - 1.25j])
    op = sc.DenseLinOp(A, dom, cod, ctx)

    lhs = ctx.ops.vdot(op.apply(x), y)
    rhs = ctx.ops.vdot(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)

    H = sc.HermitianSpace(2, ctx=ctx)
    raw = ctx.asarray([[1.0 + 0.0j, 2.0 + 3.0j], [-1.0 + 4.0j, -0.5 + 0.0j]])
    herm = H.symmetrize(raw)
    evals, evecs = H.spectral_decompose(herm)
    rebuilt = (evecs * evals) @ ctx.ops.conj(ctx.ops.transpose(evecs))
    np.testing.assert_allclose(to_numpy(rebuilt), to_numpy(herm), rtol=1e-6, atol=1e-6)


def test_numpy_complex_adjoint_and_hermitian_spectral_decompose():
    sc = importlib.import_module("spacecore")

    _check_complex_adjoint(sc.NumpyOps(), np.complex128)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_complex_adjoint_and_hermitian_spectral_decompose():
    sc = importlib.import_module("spacecore")

    _check_complex_adjoint(sc.JaxOps(), jax_complex_dtype())


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_complex_adjoint_and_hermitian_spectral_decompose():
    sc = importlib.import_module("spacecore")

    _check_complex_adjoint(sc.TorchOps(), torch_complex_dtype())

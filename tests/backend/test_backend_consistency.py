import importlib
import numpy as np
import pytest
from tests._helpers import has_jax, jax_real_dtype, to_numpy

pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


def test_numpy_and_jax_agree_on_values_for_supported_dtype():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ops = sc.NumpyOps()
    jx_ops = sc.JaxOps()
    x_np = np_ops.asarray([1.,2.,3.], dtype=dt)
    x_jx = jx_ops.asarray([1.,2.,3.], dtype=dt)
    assert np.allclose(to_numpy(x_np), to_numpy(x_jx))


def test_numpy_and_jax_dense_linop_agree_on_values():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ctx = sc.Context(sc.NumpyOps(), dtype=dt)
    jx_ctx = sc.Context(sc.JaxOps(), dtype=dt)
    Xn = sc.DenseCoordinateSpace((2,), np_ctx)
    Yn = sc.DenseCoordinateSpace((3,), np_ctx)
    Xj = sc.DenseCoordinateSpace((2,), jx_ctx)
    Yj = sc.DenseCoordinateSpace((3,), jx_ctx)
    data = [[1.,2.],[3.,4.],[5.,6.]]
    opn = sc.DenseLinOp(np_ctx.asarray(data), Xn, Yn, np_ctx)
    opj = sc.DenseLinOp(jx_ctx.asarray(data), Xj, Yj, jx_ctx)
    x_n = np_ctx.asarray([7.,8.])
    x_j = jx_ctx.asarray([7.,8.])
    assert np.allclose(to_numpy(opn.apply(x_n)), to_numpy(opj.apply(x_j)))


def test_numpy_and_jax_agree_on_new_metadata_and_shape_ops():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ops = sc.NumpyOps()
    jx_ops = sc.JaxOps()

    x_np = np_ops.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=dt)
    x_jx = jx_ops.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=dt)

    assert np_ops.shape(x_np) == jx_ops.shape(x_jx) == (2, 3)
    assert np_ops.ndim(x_np) == jx_ops.ndim(x_jx) == 2
    assert np_ops.size(x_np) == jx_ops.size(x_jx) == 6
    assert np.allclose(to_numpy(np_ops.astype(x_np, dt)), to_numpy(jx_ops.astype(x_jx, dt)))
    assert np.allclose(to_numpy(np_ops.zeros_like(x_np)), to_numpy(jx_ops.zeros_like(x_jx)))
    assert np.allclose(to_numpy(np_ops.ones_like(x_np)), to_numpy(jx_ops.ones_like(x_jx)))
    assert np.allclose(to_numpy(np_ops.full_like(x_np, 7.)), to_numpy(jx_ops.full_like(x_jx, 7.)))
    assert np.allclose(
        to_numpy(np_ops.broadcast_to(np_ops.asarray([1., 2., 3.], dtype=dt), (2, 3))),
        to_numpy(jx_ops.broadcast_to(jx_ops.asarray([1., 2., 3.], dtype=dt), (2, 3))),
    )
    assert np.allclose(to_numpy(np_ops.expand_dims(x_np, 0)), to_numpy(jx_ops.expand_dims(x_jx, 0)))
    assert np.allclose(
        to_numpy(np_ops.squeeze(np_ops.expand_dims(x_np, 0), axis=0)),
        to_numpy(jx_ops.squeeze(jx_ops.expand_dims(x_jx, 0), axis=0)),
    )
    assert np.allclose(to_numpy(np_ops.moveaxis(x_np, 0, 1)), to_numpy(jx_ops.moveaxis(x_jx, 0, 1)))
    assert np.allclose(to_numpy(np_ops.swapaxes(x_np, 0, 1)), to_numpy(jx_ops.swapaxes(x_jx, 0, 1)))


def test_numpy_and_jax_agree_on_new_reductions_and_elementwise_ops():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ops = sc.NumpyOps()
    jx_ops = sc.JaxOps()

    x_np = np_ops.asarray([[1., -2., np.nan], [4., 5., np.inf]], dtype=dt)
    x_jx = jx_ops.asarray([[1., -2., np.nan], [4., 5., np.inf]], dtype=dt)

    finite_np = np_ops.asarray([[1., -2., 3.], [4., 5., 6.]], dtype=dt)
    finite_jx = jx_ops.asarray([[1., -2., 3.], [4., 5., 6.]], dtype=dt)

    assert np.allclose(to_numpy(np_ops.mean(finite_np, axis=0)), to_numpy(jx_ops.mean(finite_jx, axis=0)))
    assert np.allclose(to_numpy(np_ops.min(finite_np, axis=1)), to_numpy(jx_ops.min(finite_jx, axis=1)))
    assert np.allclose(to_numpy(np_ops.max(finite_np, axis=1)), to_numpy(jx_ops.max(finite_jx, axis=1)))
    assert np.allclose(to_numpy(np_ops.minimum(finite_np, 2.)), to_numpy(jx_ops.minimum(finite_jx, 2.)))
    assert np.allclose(to_numpy(np_ops.clip(finite_np, -1., 4.)), to_numpy(jx_ops.clip(finite_jx, -1., 4.)))
    assert np.array_equal(to_numpy(np_ops.isfinite(x_np)), to_numpy(jx_ops.isfinite(x_jx)))
    assert np.array_equal(to_numpy(np_ops.isnan(x_np)), to_numpy(jx_ops.isnan(x_jx)))


def test_numpy_and_jax_agree_on_new_linalg_ops():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ops = sc.NumpyOps()
    jx_ops = sc.JaxOps()

    A_np = np_ops.asarray([[4., 1.], [1., 3.]], dtype=dt)
    A_jx = jx_ops.asarray([[4., 1.], [1., 3.]], dtype=dt)
    b_np = np_ops.asarray([1., 2.], dtype=dt)
    b_jx = jx_ops.asarray([1., 2.], dtype=dt)
    M_np = np_ops.asarray([[1., 2.], [3., 4.], [5., 7.]], dtype=dt)
    M_jx = jx_ops.asarray([[1., 2.], [3., 4.], [5., 7.]], dtype=dt)

    assert np.allclose(to_numpy(np_ops.norm(M_np)), to_numpy(jx_ops.norm(M_jx)))
    assert np.allclose(to_numpy(np_ops.solve(A_np, b_np)), to_numpy(jx_ops.solve(A_jx, b_jx)))
    assert np.allclose(to_numpy(np_ops.eigvalsh(A_np)), to_numpy(jx_ops.eigvalsh(A_jx)))
    assert np.allclose(to_numpy(np_ops.cholesky(A_np)), to_numpy(jx_ops.cholesky(A_jx)))

    u_np, s_np, vh_np = np_ops.svd(M_np, full_matrices=False)
    u_jx, s_jx, vh_jx = jx_ops.svd(M_jx, full_matrices=False)
    assert np.allclose(to_numpy(s_np), to_numpy(s_jx))
    assert np.allclose(to_numpy(u_np @ np_ops.diag(s_np) @ vh_np), to_numpy(u_jx @ jx_ops.diag(s_jx) @ vh_jx))


def test_numpy_and_jax_agree_on_new_indexing_and_triangular_ops():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ops = sc.NumpyOps()
    jx_ops = sc.JaxOps()

    x_np = np_ops.asarray([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=dt)
    x_jx = jx_ops.asarray([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=dt)
    idx_np = np_ops.asarray([2, 0])
    idx_jx = jx_ops.asarray([2, 0])

    assert np.allclose(to_numpy(np_ops.take(x_np, idx_np, axis=1)), to_numpy(jx_ops.take(x_jx, idx_jx, axis=1)))
    assert np.allclose(to_numpy(np_ops.diag(x_np)), to_numpy(jx_ops.diag(x_jx)))
    assert np.allclose(to_numpy(np_ops.diag(np_ops.diag(x_np))), to_numpy(jx_ops.diag(jx_ops.diag(x_jx))))
    assert np.allclose(to_numpy(np_ops.diagonal(x_np)), to_numpy(jx_ops.diagonal(x_jx)))
    assert np.allclose(to_numpy(np_ops.tril(x_np)), to_numpy(jx_ops.tril(x_jx)))
    assert np.allclose(to_numpy(np_ops.triu(x_np)), to_numpy(jx_ops.triu(x_jx)))

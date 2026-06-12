import importlib

import numpy as np
import pytest

from tests._helpers import cupy_complex_dtype, cupy_real_dtype, has_cupy, to_numpy


pytestmark = pytest.mark.skipif(
    not has_cupy(), reason="CuPy is not installed or no usable CUDA device is available"
)


def _ctx(dtype=np.float64):
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.CuPyOps(), dtype=dtype)


def test_cupy_ops_dense_creation_and_indexing():
    sc = importlib.import_module("spacecore")
    ops = sc.CuPyOps()
    x = ops.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    y = ops.index_set(x, 1, ops.asarray(5.0), copy=True)
    z = ops.index_add(y, 0, ops.asarray(2.0), copy=True)

    assert ops.family == "cupy"
    assert ops.is_dense(x)
    np.testing.assert_allclose(to_numpy(x), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(to_numpy(y), [1.0, 5.0, 3.0])
    np.testing.assert_allclose(to_numpy(z), [3.0, 5.0, 3.0])


def test_cupy_sparse_conversion_and_matmul():
    ctx = _ctx()
    dense = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sparse = ctx.assparse(dense)
    x = ctx.asarray([7.0, 8.0])

    assert ctx.ops.is_sparse(sparse)
    np.testing.assert_allclose(to_numpy(ctx.ops.sparse_matmul(sparse, x)), [23.0, 53.0, 83.0])
    assert ctx.ops.allclose_sparse(sparse, ctx.assparse(dense))


def test_cupy_dense_linop_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(ctx.asarray(dense), dom, cod, ctx)
    x = ctx.asarray([7.0, 8.0])
    y = ctx.asarray([1.0, -1.0, 2.0])

    np.testing.assert_allclose(to_numpy(op.apply(x)), dense @ np.asarray([7.0, 8.0]))
    np.testing.assert_allclose(to_numpy(op.rapply(y)), dense.T @ np.asarray([1.0, -1.0, 2.0]))


def test_cupy_sparse_linop_apply_and_to_dense():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)
    x = ctx.asarray([7.0, 8.0])

    np.testing.assert_allclose(to_numpy(op.apply(x)), dense @ np.asarray([7.0, 8.0]))
    np.testing.assert_allclose(to_numpy(op.to_dense()), dense)


def test_cupy_ops_reject_complex_to_real_casts():
    sc = importlib.import_module("spacecore")
    ops = sc.CuPyOps()
    x = ops.asarray([1.0 + 1.0j], dtype=cupy_complex_dtype())

    with pytest.raises(TypeError, match="rejected complex-valued input"):
        ops.asarray(x, dtype=cupy_real_dtype())
    with pytest.raises(TypeError, match="rejected complex-valued input"):
        ops.astype(x, cupy_real_dtype())

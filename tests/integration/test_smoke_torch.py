import importlib

import numpy as np
import pytest

from tests._helpers import has_torch, to_numpy, torch_real_dtype

pytestmark = pytest.mark.skipif(not has_torch(), reason="torch is not installed")


def test_torch_vector_hermitian_product_and_linop_smoke():
    sc = importlib.import_module("spacecore")
    dt = torch_real_dtype()
    ctx = sc.Context(sc.TorchOps(), dtype=dt, check_level="standard")

    X = sc.DenseCoordinateSpace((2,), ctx)
    x = ctx.asarray([1.0, 2.0])
    assert np.allclose(to_numpy(X.inner(x, x)), 5.0)

    H = sc.HermitianSpace(2, atol=1e-6, rtol=1e-6, ctx=ctx)
    h = ctx.asarray([[2.0, 1.0], [1.0, 2.0]])
    evals, evecs = H.spectral_decompose(h)
    assert np.allclose(to_numpy(evecs @ ctx.ops.diag(evals) @ evecs.T.conj()), to_numpy(h))

    P = sc.TreeSpace.from_leaf_spaces((X, sc.DenseCoordinateSpace((3,), ctx)), ctx)
    p = (x, ctx.asarray([3.0, 4.0, 5.0]))
    flat = P.flatten(p)
    assert flat.shape == (5,)
    assert all(ctx.ops.is_dense(part) for part in P.unflatten(flat))

    Y = sc.DenseCoordinateSpace((3,), ctx)
    A = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(A, X, Y, ctx)
    assert np.allclose(to_numpy(op.apply(x)), np.array([5.0, 11.0, 17.0]))


def test_torch_from_operators_joins_dtypes_through_backend():
    """Regression: inferring a context by joining leaf dtypes must use the
    operand backend's own ``result_type``, not NumPy's.

    ``BlockDiagonalLinOp.from_operators`` / ``StackedLinOp.from_operators`` infer
    a ``TreeSpace`` and join its leaves' dtypes. Routing that join through
    ``numpy.result_type`` raised ``TypeError: Cannot interpret 'torch.float64' as
    a data type`` on the torch backend.
    """
    sc = importlib.import_module("spacecore")
    dt = torch_real_dtype()
    ctx = sc.Context(sc.TorchOps(), dtype=dt)
    X = sc.DenseCoordinateSpace((3,), ctx)

    rng = np.random.default_rng(0)
    A = sc.DenseLinOp(ctx.asarray(rng.standard_normal((3, 3))), X, X, ctx)
    B = sc.DenseLinOp(ctx.asarray(rng.standard_normal((3, 3))), X, X, ctx)

    bd = sc.BlockDiagonalLinOp.from_operators((A, B))   # previously raised
    assert bd.ctx.dtype == dt
    sc.StackedLinOp.from_operators((A, B))               # same inference path
    sc.SumToSingleLinOp.from_operators((A, B))

    # ADR-021 fuse() goes through the same construction and must work on torch.
    fused = (A @ B).fuse()
    x = ctx.asarray(rng.standard_normal(3))
    assert np.allclose(to_numpy(fused.apply(x)), to_numpy((A @ B).apply(x)))

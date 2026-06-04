import importlib

import numpy as np
import pytest

from tests._helpers import has_torch, to_numpy, torch_real_dtype

pytestmark = pytest.mark.skipif(not has_torch(), reason="torch is not installed")


def test_torch_vector_hermitian_product_and_linop_smoke():
    sc = importlib.import_module("spacecore")
    dt = torch_real_dtype()
    ctx = sc.Context(sc.TorchOps(), dtype=dt, enable_checks=True)

    X = sc.VectorSpace((2,), ctx)
    x = ctx.asarray([1.0, 2.0])
    assert np.allclose(to_numpy(X.inner(x, x)), 5.0)

    H = sc.HermitianSpace(2, atol=1e-6, rtol=1e-6, ctx=ctx)
    h = ctx.asarray([[2.0, 1.0], [1.0, 2.0]])
    evals, evecs = H.spectral_decompose(h)
    assert np.allclose(to_numpy(evecs @ ctx.ops.diag(evals) @ evecs.T.conj()), to_numpy(h))

    P = sc.ProductSpace((X, sc.VectorSpace((3,), ctx)), ctx)
    p = (x, ctx.asarray([3.0, 4.0, 5.0]))
    flat = P.flatten(p)
    assert flat.shape == (5,)
    assert all(ctx.ops.is_dense(part) for part in P.unflatten(flat))

    Y = sc.VectorSpace((3,), ctx)
    A = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(A, X, Y, ctx)
    assert np.allclose(to_numpy(op.apply(x)), np.array([5.0, 11.0, 17.0]))

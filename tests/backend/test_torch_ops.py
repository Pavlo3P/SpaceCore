import importlib
import inspect

import numpy as np
import pytest

from tests._helpers import has_torch, to_numpy, torch_complex_dtype, torch_real_dtype

pytestmark = pytest.mark.skipif(not has_torch(), reason="torch is not installed")


def test_torch_ops_basic_array_creation_and_conversion():
    sc = importlib.import_module("spacecore")
    import torch

    ops = sc.TorchOps()
    x = ops.asarray([1, 2, 3], dtype=torch.float32)
    y = ops.asarray(np.array([1, 2, 3], dtype=np.float64))

    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float64
    assert ops.is_dense(x)
    assert not ops.is_sparse(x)


def test_torch_ops_signatures_extend_backendops_and_torch_kwargs():
    sc = importlib.import_module("spacecore")

    for name, base in sc.BackendOps.__dict__.items():
        if name.startswith("_") or isinstance(base, property) or not callable(base):
            continue
        if not hasattr(sc.TorchOps, name):
            continue

        base_params = [param for param in inspect.signature(base).parameters if param != "self"]
        torch_params = inspect.signature(getattr(sc.TorchOps, name)).parameters

        assert set(base_params).issubset(torch_params)

    expected_torch_kwargs = {
        "empty": {"out", "layout", "device", "requires_grad", "pin_memory", "memory_format"},
        "zeros": {"out", "layout", "device", "requires_grad"},
        "zeros_like": {"layout", "device", "requires_grad", "memory_format"},
        "arange": {"out", "layout", "device", "requires_grad"},
        "sum": {"out"},
        "matmul": {"out"},
        "sparse_matmul": {"reduce"},
        "eigh": {"UPLO", "out"},
        "norm": {"dtype", "out"},
        "solve": {"left", "out"},
        "svd": {"driver", "out"},
        "cholesky": {"upper", "out"},
        "where": {"out"},
        "concatenate": {"out"},
    }
    for name, params in expected_torch_kwargs.items():
        signature_params = inspect.signature(getattr(sc.TorchOps, name)).parameters
        assert params.issubset(signature_params)


def test_torch_ops_agree_with_numpy_on_dense_math():
    sc = importlib.import_module("spacecore")

    dt = torch_real_dtype()
    np_ops = sc.NumpyOps()
    th_ops = sc.TorchOps()

    x_np = np_ops.asarray([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x_th = th_ops.asarray(x_np, dtype=dt)
    y_np = np_ops.asarray([3.0, -1.0, 2.0], dtype=np.float32)
    y_th = th_ops.asarray(y_np, dtype=dt)
    A_np = np_ops.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    A_th = th_ops.asarray(A_np, dtype=dt)
    b_np = np_ops.asarray([1.0, 2.0], dtype=np.float32)
    b_th = th_ops.asarray(b_np, dtype=dt)

    assert np.allclose(to_numpy(th_ops.reshape(x_th, (3, 2))), np.reshape(x_np, (3, 2)))
    assert np.allclose(to_numpy(th_ops.sum(x_th, axis=0)), np.sum(x_np, axis=0))
    assert np.allclose(to_numpy(th_ops.mean(x_th, axis=1)), np.mean(x_np, axis=1))
    assert np.allclose(to_numpy(th_ops.matmul(x_th, y_th)), x_np @ y_np)
    assert np.allclose(to_numpy(th_ops.solve(A_th, b_th)), np.linalg.solve(A_np, b_np))
    assert np.allclose(to_numpy(th_ops.cholesky(A_th)), np.linalg.cholesky(A_np))


def test_torch_ops_sparse_conversion_and_matmul():
    sc = importlib.import_module("spacecore")
    import scipy.sparse as sps

    ops = sc.TorchOps()
    dense = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    sparse = ops.assparse(sps.csr_matrix(dense), format="coo")
    x = ops.asarray([4.0, 5.0], dtype=sparse.dtype)

    assert ops.is_sparse(sparse)
    assert np.allclose(to_numpy(ops.sparse_matmul(sparse, x)), dense @ to_numpy(x))


def test_torch_ops_preserve_autograd_for_tensor_ops():
    sc = importlib.import_module("spacecore")
    import torch

    ops = sc.TorchOps()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = ops.sum(ops.asarray(x) * ops.asarray(x))
    y.backward()

    assert np.allclose(x.grad.detach().numpy(), np.array([2.0, 4.0, 6.0]))


def test_torch_ops_reject_complex_to_real_casts():
    sc = importlib.import_module("spacecore")
    ops = sc.TorchOps()
    x = ops.asarray([1.0 + 1.0j], dtype=torch_complex_dtype())

    with pytest.raises(TypeError, match="rejected complex-valued input"):
        ops.asarray(x, dtype=torch_real_dtype())
    with pytest.raises(TypeError, match="rejected complex-valued input"):
        ops.astype(x, torch_real_dtype())

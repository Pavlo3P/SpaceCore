"""TorchOps-specific tests.

Generic operation conformance lives in :mod:`tests.backend.test_operations`;
this module covers behavior that is *only* meaningful for ``TorchOps``:

* the ``torch`` family identifier and the ``array_api_compat.torch`` xp
  namespace;
* default representation dtype tracking ``torch.get_default_dtype()``;
* native ``torch.vmap`` / ``torch.func.vmap`` integration;
* autograd preservation through SpaceCore ops;
* torch-specific keyword arguments (``device=``, ``layout=``, ``out=``,
  ``requires_grad=``, ``UPLO`` for eigh, ``upper`` for cholesky, ``left``
  for solve);
* torch sparse-tensor format conversion (``coo``, ``csr``, ``csc``);
* ``index_add`` last-write-wins semantics on repeated indices;
* ``__eq__`` / ``__hash__`` / ``__repr__`` of ``TorchOps`` instances.

Skipped wholesale when torch is not importable.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests._helpers import has_torch, to_numpy, torch_real_dtype

pytestmark = pytest.mark.skipif(not has_torch(), reason="torch is not installed")


@pytest.fixture
def ops():
    import spacecore as sc

    return sc.TorchOps()


# ---------------------------------------------------------------------------
# Family and capability flags
# ---------------------------------------------------------------------------
def test_torch_ops_family_string(ops):
    assert ops.family == "torch"


def test_torch_ops_allow_sparse_is_true(ops):
    assert ops.allow_sparse is True


def test_torch_ops_has_native_vmap_is_true(ops):
    assert ops.has_native_vmap is True


def test_torch_ops_xp_is_array_api_compat_torch():
    import spacecore as sc

    assert sc.TorchOps.xp.__name__ == "array_api_compat.torch"


# ---------------------------------------------------------------------------
# Dtype defaulting
# ---------------------------------------------------------------------------
def test_torch_ops_default_dtype_follows_torch_default(ops):
    import torch

    assert ops.sanitize_dtype(None) == torch.get_default_dtype()


def test_torch_ops_eps_default(ops):
    import torch

    dt = torch_real_dtype()
    assert ops.eps(dt) == pytest.approx(float(torch.finfo(dt).eps))


# ---------------------------------------------------------------------------
# Equality, hash, repr
# ---------------------------------------------------------------------------
def test_torch_ops_equality_and_hash():
    import spacecore as sc

    a = sc.TorchOps()
    b = sc.TorchOps()
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1, b: 2} == {a: 2}


def test_torch_ops_repr():
    import spacecore as sc

    assert "TorchOps" in repr(sc.TorchOps())
    assert "family='torch'" in repr(sc.TorchOps())


# ---------------------------------------------------------------------------
# Torch-specific kwargs in the signature (device=, out=, requires_grad=, ...)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method,required_kwargs",
    [
        ("empty", {"out", "layout", "device", "requires_grad", "pin_memory", "memory_format"}),
        ("zeros", {"out", "layout", "device", "requires_grad"}),
        ("zeros_like", {"layout", "device", "requires_grad", "memory_format"}),
        ("arange", {"out", "layout", "device", "requires_grad"}),
        ("sum", {"out"}),
        ("matmul", {"out"}),
        ("sparse_matmul", {"reduce"}),
        ("eigh", {"UPLO", "out"}),
        ("norm", {"dtype", "out"}),
        ("solve", {"left", "out"}),
        ("svd", {"driver", "out"}),
        ("cholesky", {"upper", "out"}),
        ("where", {"out"}),
        ("concatenate", {"out"}),
    ],
)
def test_torch_ops_exposes_backend_specific_kwargs(method, required_kwargs):
    import spacecore as sc

    params = inspect.signature(getattr(sc.TorchOps, method)).parameters
    assert required_kwargs.issubset(params)


def test_torch_ops_signatures_extend_backendops():
    """Every TorchOps override accepts at least the BackendOps parameters."""
    import spacecore as sc

    for name, base in sc.BackendOps.__dict__.items():
        if name.startswith("_") or isinstance(base, property) or not callable(base):
            continue
        if not hasattr(sc.TorchOps, name):
            continue
        base_params = [p for p in inspect.signature(base).parameters if p != "self"]
        torch_params = inspect.signature(getattr(sc.TorchOps, name)).parameters
        assert set(base_params).issubset(torch_params), (
            f"TorchOps.{name} drops base parameters: "
            f"{set(base_params) - set(torch_params)}"
        )


def test_torch_ops_zeros_device_kwarg_passthrough(ops):
    import torch

    x = ops.zeros((3,), dtype=torch.float32, device="cpu")
    assert x.device.type == "cpu"


def test_torch_ops_zeros_requires_grad_kwarg(ops):
    x = ops.zeros((3,), requires_grad=True)
    assert x.requires_grad is True


# ---------------------------------------------------------------------------
# Autograd preservation
# ---------------------------------------------------------------------------
def test_torch_ops_preserves_autograd(ops):
    import torch

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = ops.sum(ops.asarray(x) * ops.asarray(x))
    y.backward()
    np.testing.assert_allclose(x.grad.detach().numpy(), np.asarray([2.0, 4.0, 6.0]))


# ---------------------------------------------------------------------------
# Sparse format (COO / CSR / CSC)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt,expected_layout_attr", [
    ("coo", "sparse_coo"),
    ("csr", "sparse_csr"),
    ("csc", "sparse_csc"),
])
def test_torch_ops_assparse_format(ops, fmt, expected_layout_attr):
    import torch

    dense_np = np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    dense_be = ops.asarray(dense_np)
    sparse = ops.assparse(dense_be, format=fmt)
    assert sparse.layout == getattr(torch, expected_layout_attr)


def test_torch_ops_sparse_matmul_from_scipy_input(ops):
    import scipy.sparse as sps

    dense = np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    sparse = ops.assparse(sps.csr_matrix(dense), format="coo")
    x = ops.asarray([4.0, 5.0], dtype=sparse.dtype)
    expected = dense @ to_numpy(x)
    np.testing.assert_allclose(to_numpy(ops.sparse_matmul(sparse, x)), expected)


# ---------------------------------------------------------------------------
# index_add: torch implements scatter-assign, not np.add.at-style accumulate.
# ---------------------------------------------------------------------------
def test_torch_ops_index_add_is_scatter_assign_at_repeated_indices(ops):
    """At repeated indices, TorchOps.index_add behaves as last-write-wins.

    The generic conformance suite skips torch for accumulate semantics;
    this test pins the actual torch behavior so a future refactor cannot
    silently change it without flagging here.
    """
    x = ops.asarray(np.zeros(5, dtype=np.float32))
    out = ops.index_add(x, [1, 1, 3], ops.asarray(np.asarray([2.0, 3.0, 5.0], dtype=np.float32)))
    # ``y[index] = y[index] + values`` — both writes to idx=1 are evaluated
    # in source order on the same right-hand side ``y[index]==0``, so the
    # final value at idx=1 is 3.0 (the last write), not 5.0 (the sum).
    np.testing.assert_allclose(to_numpy(out), np.asarray([0.0, 3.0, 0.0, 5.0, 0.0]))


# ---------------------------------------------------------------------------
# Constants and astype(None)
# ---------------------------------------------------------------------------
def test_torch_ops_constants_are_cached(ops):
    assert ops.inf is ops.inf
    assert ops.nan is ops.nan
    assert ops.pi is ops.pi
    assert ops.e is ops.e


def test_torch_ops_astype_none_is_identity(ops):
    x = ops.asarray(np.asarray([1.0, 2.0], dtype=np.float32))
    assert ops.astype(x, None) is x

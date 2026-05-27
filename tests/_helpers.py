from __future__ import annotations
import importlib.util
from functools import lru_cache
import numpy as np


def has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


@lru_cache
def has_cupy() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        cupy.asarray([0]).sum()
    except Exception:
        return False
    return True


def jax_real_dtype():
    if not has_jax():
        return np.float32
    import jax
    return np.float64 if bool(jax.config.read("jax_enable_x64")) else np.float32


def jax_complex_dtype():
    return np.complex128 if jax_real_dtype() == np.float64 else np.complex64


def torch_real_dtype():
    if not has_torch():
        return np.float32
    import torch
    return torch.get_default_dtype()


def torch_complex_dtype():
    if not has_torch():
        return np.complex64
    import torch
    return torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64


def cupy_real_dtype():
    return np.float64


def cupy_complex_dtype():
    return np.complex128


def to_numpy(x):
    if isinstance(x, tuple):
        return tuple(to_numpy(xi) for xi in x)
    if has_cupy():
        import cupy
        if isinstance(x, cupy.ndarray):
            return cupy.asnumpy(x)
        try:
            import cupyx.scipy.sparse as cupy_sparse
            sparse_types = tuple(
                typ
                for typ in (
                    getattr(cupy_sparse, "spmatrix", None),
                    getattr(cupy_sparse, "csr_matrix", None),
                    getattr(cupy_sparse, "csc_matrix", None),
                    getattr(cupy_sparse, "coo_matrix", None),
                )
                if typ is not None
            )
            if sparse_types and isinstance(x, sparse_types):
                return cupy.asnumpy(x.toarray())
        except Exception:
            pass
    if has_torch():
        import torch
        if isinstance(x, torch.Tensor):
            if x.layout != torch.strided:
                x = x.to_dense()
            return x.detach().cpu().numpy()
    return np.asarray(x)


def prod(shape):
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)

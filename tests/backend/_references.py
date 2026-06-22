"""Native-library reference adapter for backend conformance tests.

Each ``BackendOps`` method is supposed to be a thin wrapper around a function
in its underlying native library (``numpy`` for ``NumpyOps``, ``jax.numpy``
for ``JaxOps``, ``torch`` for ``TorchOps``, ``cupy`` for ``CuPyOps``). The
:class:`ReferenceOps` adapter calls those native functions directly and is
the truth a backend's behavior is compared against.

Tests that parametrize over the ``backend_ops`` fixture obtain the matching
reference via :func:`native_reference`::

    def test_some_op(backend_ops, conformance_dtype):
        ref = native_reference(backend_ops)
        expected = ref.zeros((3, 3), dtype=conformance_dtype)
        actual = backend_ops.zeros((3, 3), dtype=conformance_dtype)
        assert_matches_reference("zeros", actual, expected, dtype=conformance_dtype)

The adapter never delegates back into :class:`spacecore.backend.BackendOps`;
if it did, the conformance suite would compare ``BackendOps`` against
itself. Per-family signature quirks (Torch's ``dim``/``keepdim``, JAX's
``jax.lax`` control flow, JAX's experimental sparse) are handled inside the
adapter so test bodies stay backend-agnostic.
"""
from __future__ import annotations

from typing import Any, Sequence


class ReferenceOps:
    """Calls the native library a backend wraps. The truth, not a wrapper.

    Parameters
    ----------
    family
        ``"numpy"``, ``"jax"``, ``"torch"``, or ``"cupy"``.

    Notes
    -----
    Methods mirror :class:`spacecore.backend.BackendOps` method names but
    are implemented by direct calls into the underlying library:

    * ``"numpy"`` → ``numpy``, ``numpy.linalg``, ``scipy.special``, ``scipy.sparse``
    * ``"jax"``   → ``jax.numpy``, ``jax.numpy.linalg``, ``jax.scipy.special``,
                    ``jax.lax``, ``jax.experimental.sparse``
    * ``"torch"`` → ``torch``, ``torch.linalg``, ``torch.sparse``
    * ``"cupy"``  → ``cupy``, ``cupy.linalg``, ``cupyx.scipy.special``,
                    ``cupyx.scipy.sparse``
    """

    def __init__(self, family: str) -> None:
        self.family = family
        if family == "numpy":
            import numpy as np
            import scipy.special
            import scipy.sparse

            self._xp = np
            self._la = np.linalg
            self._special = scipy.special
            self._sparse = scipy.sparse
        elif family == "jax":
            import jax
            import jax.numpy as jnp
            import jax.scipy.special as jss
            from jax.experimental import sparse as jsparse

            self._xp = jnp
            self._la = jnp.linalg
            self._special = jss
            self._sparse = jsparse
            self._lax = jax.lax
            self._jax = jax
        elif family == "torch":
            import torch

            self._torch = torch
            self._la = torch.linalg
        elif family == "cupy":
            import cupy as cp
            import cupyx.scipy.special as css
            import cupyx.scipy.sparse as cssp

            self._xp = cp
            self._la = cp.linalg
            self._special = css
            self._sparse = cssp
        else:
            raise ValueError(f"Unknown backend family {family!r}")

    # ------------------------------------------------------------------
    # Dtype helpers
    # ------------------------------------------------------------------
    def _dtype(self, dtype: Any) -> Any:
        """Convert a NumPy-style dtype to the backend's native dtype.

        NumPy, JAX, and CuPy accept ``numpy.dtype`` objects directly. Torch
        requires a ``torch.dtype``.
        """
        if dtype is None:
            return None
        if self.family == "torch":
            return _np_to_torch_dtype(self._torch, dtype)
        if self.family == "jax":
            return self._xp.dtype(dtype)
        return dtype

    # ------------------------------------------------------------------
    # Expected property values (no native equivalent — pinned knowledge)
    # ------------------------------------------------------------------
    @property
    def expected_family(self) -> str:
        return self.family

    @property
    def expected_allow_sparse(self) -> bool:
        return self.family in ("numpy", "torch", "cupy", "jax")

    @property
    def expected_has_native_vmap(self) -> bool:
        # JaxOps and TorchOps return True; NumpyOps and CuPyOps inherit False.
        return self.family in ("jax", "torch")

    @property
    def dense_array_type(self) -> Any:
        if self.family == "numpy":
            return self._xp.ndarray
        if self.family == "jax":
            return self._jax.Array
        if self.family == "torch":
            return self._torch.Tensor
        if self.family == "cupy":
            return self._xp.ndarray
        raise AssertionError(self.family)

    @property
    def default_dtype(self) -> Any:
        """The dtype ``BackendOps.sanitize_dtype(None)`` should return."""
        if self.family == "numpy":
            return self._xp.float64
        if self.family == "jax":
            x64 = bool(self._jax.config.read("jax_enable_x64"))
            return self._xp.float64 if x64 else self._xp.float32
        if self.family == "torch":
            return self._torch.get_default_dtype()
        if self.family == "cupy":
            return self._xp.float64
        raise AssertionError(self.family)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def asarray(self, x: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.as_tensor(x, dtype=self._dtype(dtype))
        return self._xp.asarray(x, dtype=self._dtype(dtype))

    def astype(self, x: Any, dtype: Any) -> Any:
        if dtype is None:
            return x
        if self.family == "torch":
            return x.to(dtype=self._dtype(dtype))
        return x.astype(self._dtype(dtype))

    def empty(self, shape: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.empty(shape, dtype=self._dtype(dtype))
        return self._xp.empty(shape, dtype=self._dtype(dtype))

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.zeros(shape, dtype=self._dtype(dtype))
        return self._xp.zeros(shape, dtype=self._dtype(dtype))

    def ones(self, shape: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.ones(shape, dtype=self._dtype(dtype))
        return self._xp.ones(shape, dtype=self._dtype(dtype))

    def zeros_like(self, x: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.zeros_like(x, dtype=self._dtype(dtype))
        return self._xp.zeros_like(x, dtype=self._dtype(dtype))

    def ones_like(self, x: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.ones_like(x, dtype=self._dtype(dtype))
        return self._xp.ones_like(x, dtype=self._dtype(dtype))

    def full_like(self, x: Any, value: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.full_like(x, value, dtype=self._dtype(dtype))
        return self._xp.full_like(x, value, dtype=self._dtype(dtype))

    def arange(self, start: int, stop: int | None = None, step: int | None = None,
               dtype: Any = None) -> Any:
        dt = self._dtype(dtype)
        if self.family == "torch":
            ns = self._torch
        else:
            ns = self._xp
        if stop is None:
            return ns.arange(start, dtype=dt)
        if step is None:
            return ns.arange(start, stop, dtype=dt)
        return ns.arange(start, stop, step, dtype=dt)

    def full(self, shape: Any, value: Any, dtype: Any = None) -> Any:
        if self.family == "torch":
            return self._torch.full(shape, value, dtype=self._dtype(dtype))
        return self._xp.full(shape, value, dtype=self._dtype(dtype))

    def eye(self, n: int, m: int | None = None, dtype: Any = None) -> Any:
        if self.family == "torch":
            if m is None:
                return self._torch.eye(n, dtype=self._dtype(dtype))
            return self._torch.eye(n, m, dtype=self._dtype(dtype))
        return self._xp.eye(n, m, dtype=self._dtype(dtype))

    # ------------------------------------------------------------------
    # Reshape / layout
    # ------------------------------------------------------------------
    def ravel(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.ravel(x)
        return self._xp.ravel(x)

    def reshape(self, x: Any, shape: Any) -> Any:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if self.family == "torch":
            return x.reshape(shape)
        return self._xp.reshape(x, shape)

    def transpose(self, x: Any, axes: Sequence[int] | None = None) -> Any:
        if axes is None:
            axes = tuple(reversed(range(x.ndim)))
        if self.family == "torch":
            return x.permute(*axes)
        return self._xp.transpose(x, axes)

    def swapaxes(self, x: Any, axis1: int, axis2: int) -> Any:
        if self.family == "torch":
            return self._torch.swapaxes(x, axis1, axis2)
        return self._xp.swapaxes(x, axis1, axis2)

    def broadcast_to(self, x: Any, shape: Any) -> Any:
        if self.family == "torch":
            return self._torch.broadcast_to(x, shape)
        return self._xp.broadcast_to(x, shape)

    def expand_dims(self, x: Any, axis: int | Sequence[int]) -> Any:
        if isinstance(axis, int):
            if self.family == "torch":
                return self._torch.unsqueeze(x, axis)
            return self._xp.expand_dims(x, axis)
        out = x
        ndim = x.ndim + len(tuple(axis))
        for ax in sorted(a + ndim if a < 0 else a for a in axis):
            out = self.expand_dims(out, ax)
        return out

    def squeeze(self, x: Any, axis: Any = None) -> Any:
        if axis is None:
            if self.family == "torch":
                return self._torch.squeeze(x)
            return self._xp.squeeze(x)
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        if self.family == "torch":
            for ax in sorted(axis, reverse=True):
                x = self._torch.squeeze(x, ax)
            return x
        return self._xp.squeeze(x, axis=axis)

    def moveaxis(self, x: Any, source: Any, destination: Any) -> Any:
        if self.family == "torch":
            return self._torch.moveaxis(x, source, destination)
        return self._xp.moveaxis(x, source, destination)

    def stack(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        if self.family == "torch":
            return self._torch.stack(tuple(arrays), dim=axis)
        return self._xp.stack(tuple(arrays), axis=axis)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0, dtype: Any = None) -> Any:
        if self.family == "torch":
            out = self._torch.cat(tuple(arrays), dim=axis)
        else:
            out = self._xp.concatenate(tuple(arrays), axis=axis)
        return self.astype(out, dtype) if dtype is not None else out

    def take(self, x: Any, indices: Any, axis: int | None = None) -> Any:
        if self.family == "torch":
            if axis is None:
                return self._torch.take(x, indices)
            return self._torch.index_select(x, axis, indices)
        return self._xp.take(x, indices, axis=axis)

    # ------------------------------------------------------------------
    # Elementwise
    # ------------------------------------------------------------------
    def conj(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.conj_physical(x) if self._torch.is_complex(x) else x
        return self._xp.conj(x)

    def real(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.real(x) if self._torch.is_complex(x) else x
        return self._xp.real(x)

    def imag(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.imag(x) if self._torch.is_complex(x) else self._torch.zeros_like(x)
        return self._xp.imag(x)

    def abs(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.abs(x)
        return self._xp.abs(x)

    def sign(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.sgn(x) if self._torch.is_complex(x) else self._torch.sign(x)
        return self._xp.sign(x)

    def sqrt(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.sqrt(x)
        return self._xp.sqrt(x)

    def exp(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.exp(x)
        return self._xp.exp(x)

    def log(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.log(x)
        return self._xp.log(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        if self.family == "torch":
            return self._torch.where(condition, x, y)
        return self._xp.where(condition, x, y)

    def maximum(self, x: Any, y: Any) -> Any:
        if self.family == "torch":
            return self._torch.maximum(x, y)
        return self._xp.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        if self.family == "torch":
            return self._torch.minimum(x, y)
        return self._xp.minimum(x, y)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        if self.family == "torch":
            return self._torch.clamp(x, min=a_min, max=a_max)
        return self._xp.clip(x, a_min, a_max)

    def isfinite(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.isfinite(x)
        return self._xp.isfinite(x)

    def isnan(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.isnan(x)
        return self._xp.isnan(x)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, x: Any, axis: Any = None, keepdims: bool = False, dtype: Any = None) -> Any:
        if self.family == "torch":
            kwargs: dict[str, Any] = {"keepdim": keepdims}
            # Torch needs an explicit dim list to honor keepdim=True over the
            # full reduction; passing dim=None silently drops the keepdim.
            if axis is None:
                kwargs["dim"] = tuple(range(x.ndim))
            else:
                kwargs["dim"] = axis if isinstance(axis, int) else tuple(axis)
            if dtype is not None:
                kwargs["dtype"] = self._dtype(dtype)
            return self._torch.sum(x, **kwargs)
        axis_arg = axis if isinstance(axis, int) or axis is None else tuple(axis)
        return self._xp.sum(x, axis=axis_arg, keepdims=keepdims, dtype=self._dtype(dtype))

    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            dim = tuple(range(x.ndim)) if axis is None else (
                axis if isinstance(axis, int) else tuple(axis)
            )
            return self._torch.mean(x, dim=dim, keepdim=keepdims)
        axis_arg = axis if isinstance(axis, int) or axis is None else tuple(axis)
        return self._xp.mean(x, axis=axis_arg, keepdims=keepdims)

    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            dim = tuple(range(x.ndim)) if axis is None else (
                axis if isinstance(axis, int) else tuple(axis)
            )
            return self._torch.amin(x, dim=dim, keepdim=keepdims)
        axis_arg = axis if isinstance(axis, int) or axis is None else tuple(axis)
        return self._xp.min(x, axis=axis_arg, keepdims=keepdims)

    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            dim = tuple(range(x.ndim)) if axis is None else (
                axis if isinstance(axis, int) else tuple(axis)
            )
            return self._torch.amax(x, dim=dim, keepdim=keepdims)
        axis_arg = axis if isinstance(axis, int) or axis is None else tuple(axis)
        return self._xp.max(x, axis=axis_arg, keepdims=keepdims)

    def prod(self, x: Any, axis: Any = None, keepdims: bool = False, dtype: Any = None) -> Any:
        if self.family == "torch":
            kwargs: dict[str, Any] = {}
            if dtype is not None:
                kwargs["dtype"] = self._dtype(dtype)
            if axis is None:
                # Reduce over all axes; preserve keepdim by per-axis reduction.
                if keepdims:
                    out = x
                    for ax in range(x.ndim - 1, -1, -1):
                        out = self._torch.prod(out, dim=ax, keepdim=True, **kwargs)
                    return out
                return self._torch.prod(x, **kwargs)
            if isinstance(axis, int):
                return self._torch.prod(x, dim=axis, keepdim=keepdims, **kwargs)
            # tuple axis — torch.prod takes only one dim; iterate from
            # highest axis so axis numbering stays stable.
            out = x
            for ax in sorted(tuple(axis), reverse=True):
                out = self._torch.prod(out, dim=ax, keepdim=keepdims, **kwargs)
            return out
        axis_arg = axis if isinstance(axis, int) or axis is None else tuple(axis)
        return self._xp.prod(x, axis=axis_arg, keepdims=keepdims, dtype=self._dtype(dtype))

    def trace(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.trace(x)
        return self._xp.trace(x)

    # ------------------------------------------------------------------
    # Index reductions
    # ------------------------------------------------------------------
    def argsort(self, x: Any, axis: int = -1) -> Any:
        if self.family == "torch":
            return self._torch.argsort(x, dim=axis)
        return self._xp.argsort(x, axis=axis)

    def sort(self, x: Any, axis: int = -1) -> Any:
        if self.family == "torch":
            values, _ = self._torch.sort(x, dim=axis)
            return values
        return self._xp.sort(x, axis=axis)

    def argmin(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            if axis is None:
                # torch.argmin with dim=None returns a scalar tensor.
                return self._torch.argmin(x)
            return self._torch.argmin(x, dim=axis, keepdim=keepdims)
        return self._xp.argmin(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            if axis is None:
                return self._torch.argmax(x)
            return self._torch.argmax(x, dim=axis, keepdim=keepdims)
        return self._xp.argmax(x, axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------
    def vdot(self, x: Any, y: Any) -> Any:
        if self.family == "torch":
            return self._torch.vdot(x.ravel(), y.ravel())
        return self._xp.vdot(x.ravel(), y.ravel())

    def matmul(self, a: Any, b: Any) -> Any:
        if self.family == "torch":
            return self._torch.matmul(a, b)
        return self._xp.matmul(a, b)

    def kron(self, a: Any, b: Any) -> Any:
        if self.family == "torch":
            return self._torch.kron(a, b)
        return self._xp.kron(a, b)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        if self.family == "torch":
            return self._torch.einsum(subscripts, *operands)
        return self._xp.einsum(subscripts, *operands)

    def eigh(self, x: Any) -> Any:
        return self._la.eigh(x)

    def eigvalsh(self, x: Any) -> Any:
        return self._la.eigvalsh(x)

    def svd(self, x: Any, full_matrices: bool = True) -> Any:
        return self._la.svd(x, full_matrices=full_matrices)

    def solve(self, A: Any, b: Any) -> Any:
        return self._la.solve(A, b)

    def cholesky(self, A: Any) -> Any:
        return self._la.cholesky(A)

    def norm(self, x: Any, ord: Any = None, axis: Any = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            return self._torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)
        return self._la.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------
    def diag(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.diag(x)
        return self._xp.diag(x)

    def diagonal(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.diagonal(x)
        return self._xp.diagonal(x)

    def tril(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.tril(x)
        return self._xp.tril(x)

    def triu(self, x: Any) -> Any:
        if self.family == "torch":
            return self._torch.triu(x)
        return self._xp.triu(x)

    # ------------------------------------------------------------------
    # allclose
    # ------------------------------------------------------------------
    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8,
                 equal_nan: bool = False) -> bool:
        if self.family == "torch":
            return bool(self._torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
        return bool(self._xp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    # ------------------------------------------------------------------
    # Special
    # ------------------------------------------------------------------
    def logsumexp(self, a: Any, axis: Any = None, b: Any = None, keepdims: bool = False) -> Any:
        if self.family == "torch":
            # Torch's torch.logsumexp does not accept ``b`` weights; fall back
            # to the math when needed so the reference is honest.
            dim = tuple(range(a.ndim)) if axis is None else axis
            if b is None:
                return self._torch.logsumexp(a, dim=dim, keepdim=keepdims)
            m = self._torch.amax(a, dim=dim, keepdim=True)
            total = self._torch.sum(b * self._torch.exp(a - m), dim=dim, keepdim=True)
            result = self._torch.log(self._torch.abs(total)) + m
            if not keepdims:
                if isinstance(dim, int):
                    result = result.squeeze(dim)
                else:
                    for d in sorted(dim, reverse=True):
                        result = result.squeeze(d)
            return result
        return self._special.logsumexp(a, axis=axis, b=b, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Index ops
    # ------------------------------------------------------------------
    def index_set(self, x: Any, index: Any, values: Any) -> Any:
        """Out-of-place set ``y = x.copy(); y[index] = values; return y``."""
        if self.family == "torch":
            y = x.clone()
            y[index] = values
            return y
        if self.family == "jax":
            return x.at[index].set(values)
        # numpy, cupy
        y = x.copy()
        y[index] = values
        return y

    def index_add(self, x: Any, index: Any, values: Any) -> Any:
        """Out-of-place ``y = x.copy(); np.add.at(y, index, values); return y``."""
        if self.family == "numpy":
            y = x.copy()
            self._xp.add.at(y, index, values)
            return y
        if self.family == "jax":
            return x.at[index].add(values)
        if self.family == "cupy":
            y = x.copy()
            self._xp.add.at(y, index, values)
            return y
        if self.family == "torch":
            # torch lacks np.add.at; emulate by sequential assignment-add.
            y = x.clone()
            y[index] = y[index] + values
            return y
        raise AssertionError(self.family)

    def ix_(self, *args: Any) -> Any:
        if self.family == "torch":
            tensors = tuple(
                a if isinstance(a, self._torch.Tensor) else self.asarray(a) for a in args
            )
            return self._torch.meshgrid(*tensors, indexing="ij")
        return self._xp.ix_(*args)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------
    def fori_loop(self, lower: int, upper: int, body_fun, init_val):
        """Python-loop reference; correct semantics, not the JAX-traced one."""
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    def while_loop(self, cond_fun, body_fun, init_val):
        val = init_val
        while bool(cond_fun(val)):
            val = body_fun(val)
        return val

    def cond(self, pred: bool, true_fun, false_fun, *operands):
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    # ------------------------------------------------------------------
    # Sparse
    # ------------------------------------------------------------------
    def assparse_csr(self, dense_2d: Any) -> Any:
        """Convert a 2-D dense array to a sparse CSR-equivalent matrix.

        Returns a backend-native sparse type (scipy CSR for numpy/cupy,
        torch CSR for torch, JAX BCOO for jax — JAX has no first-class CSR).
        Tests compare via dense round-trip rather than direct sparse equality.
        """
        if self.family == "numpy":
            return self._sparse.csr_matrix(dense_2d)
        if self.family == "torch":
            return dense_2d.to_sparse_csr()
        if self.family == "cupy":
            return self._sparse.csr_matrix(dense_2d)
        if self.family == "jax":
            return self._sparse.BCOO.fromdense(dense_2d)
        raise AssertionError(self.family)

    def sparse_to_dense(self, x: Any) -> Any:
        """Densify a backend sparse object for cross-backend comparison."""
        if self.family == "numpy":
            return x.toarray()
        if self.family == "torch":
            return x.to_dense()
        if self.family == "cupy":
            return x.toarray()
        if self.family == "jax":
            return x.todense()
        raise AssertionError(self.family)

    def sparse_matmul(self, a_sparse: Any, b_dense: Any) -> Any:
        """Matrix-multiply ``a_sparse @ b_dense`` using native sparse paths."""
        if self.family == "numpy":
            return a_sparse @ b_dense
        if self.family == "torch":
            if b_dense.ndim == 1:
                return self._torch.sparse.mm(a_sparse, b_dense[:, None])[:, 0]
            return self._torch.sparse.mm(a_sparse, b_dense)
        if self.family == "cupy":
            return a_sparse @ b_dense
        if self.family == "jax":
            return a_sparse @ b_dense
        raise AssertionError(self.family)


def native_reference(backend_ops: Any) -> ReferenceOps:
    """Return the :class:`ReferenceOps` adapter for ``backend_ops``'s family."""
    return ReferenceOps(backend_ops.family)


def _np_to_torch_dtype(torch_mod: Any, dtype: Any) -> Any:
    """Translate a numpy-style dtype specifier into a ``torch.dtype``.

    The mapping is intentionally narrow: SpaceCore's conformance suite only
    needs the standard real/complex/int/bool dtypes.
    """
    import numpy as np

    if isinstance(dtype, torch_mod.dtype):
        return dtype
    np_dtype = np.dtype(dtype)
    table = {
        np.dtype("bool"): torch_mod.bool,
        np.dtype("uint8"): torch_mod.uint8,
        np.dtype("int8"): torch_mod.int8,
        np.dtype("int16"): torch_mod.int16,
        np.dtype("int32"): torch_mod.int32,
        np.dtype("int64"): torch_mod.int64,
        np.dtype("float16"): torch_mod.float16,
        np.dtype("float32"): torch_mod.float32,
        np.dtype("float64"): torch_mod.float64,
        np.dtype("complex64"): torch_mod.complex64,
        np.dtype("complex128"): torch_mod.complex128,
    }
    try:
        return table[np_dtype]
    except KeyError as exc:
        raise TypeError(f"No torch dtype for {dtype!r}") from exc

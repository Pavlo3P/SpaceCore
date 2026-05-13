from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Type

import numpy as np

from .._family import BackendFamily
from .._ops import BackendOps
from ...types import ArrayLike, DenseArray, DType, Index, SparseArray, T, X, Y, R, Carry


class TorchOps(BackendOps):
    """
    BackendOps implementation for PyTorch tensors.

    This backend uses PyTorch for dense and sparse tensor operations.

    Dense arrays
        torch.Tensor with strided layout

    Sparse arrays
        torch.Tensor with a PyTorch sparse layout

    Methods
        Most methods mirror the corresponding PyTorch public API signatures and
        delegate to ``torch`` or ``torch.linalg``. Backend-specific behavior,
        dtype promotion, broadcasting, device placement, autograd tracking, and
        error modes therefore follow PyTorch semantics.

    Backend handles
      - torch : module
            PyTorch module stored on the class and available through instances
            as ``ops.torch``. Advanced users may use it when SpaceCore's
            portable API does not expose a required PyTorch feature.

    Notes
        Code intended to remain backend-portable should prefer ``BackendOps``
        methods. Direct use of ``ops.torch`` is an explicit PyTorch-specific
        escape hatch.

        ``TorchOps`` follows PyTorch dtype defaults. When no dtype is provided,
        ``sanitize_dtype(None)`` returns ``torch.get_default_dtype()``. Python
        ``complex`` maps to ``torch.complex64`` or ``torch.complex128`` based
        on the active default floating dtype, and NumPy dtype specifiers are
        mapped to their corresponding PyTorch dtypes when supported.

        Array creation and conversion methods may accept a backend-specific
        ``device=`` keyword. Existing tensors stay on their device unless an
        explicit device conversion is requested. Dense conversion and ordinary
        math operations do not detach tensors; autograd metadata is preserved
        according to normal PyTorch rules.
    """

    import torch

    _family = BackendFamily.torch.value.lower()
    _allow_sparse = True

    _sparse_layouts = (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    )

    @property
    def dense_array(self) -> Type[Any]:
        return self.torch.Tensor

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        return (self.torch.Tensor,)

    def is_dense(self, x: Any) -> bool:
        return isinstance(x, self.torch.Tensor) and x.layout == self.torch.strided

    def is_sparse(self, x: Any) -> bool:
        return isinstance(x, self.torch.Tensor) and x.layout in self._sparse_layouts

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        if dtype is None:
            return self.torch.get_default_dtype()
        if isinstance(dtype, self.torch.dtype):
            return dtype
        if dtype is float:
            return self.torch.get_default_dtype()
        if dtype is complex:
            return (
                self.torch.complex128
                if self.torch.get_default_dtype() == self.torch.float64
                else self.torch.complex64
            )
        if dtype is int:
            return self.torch.int64
        if dtype is bool:
            return self.torch.bool

        try:
            np_dtype = np.dtype(dtype)
        except Exception as e:
            raise TypeError(f"Invalid dtype specifier for PyTorch: {dtype!r}.") from e

        mapping = {
            np.dtype("bool"): self.torch.bool,
            np.dtype("uint8"): self.torch.uint8,
            np.dtype("int8"): self.torch.int8,
            np.dtype("int16"): self.torch.int16,
            np.dtype("int32"): self.torch.int32,
            np.dtype("int64"): self.torch.int64,
            np.dtype("float16"): self.torch.float16,
            np.dtype("float32"): self.torch.float32,
            np.dtype("float64"): self.torch.float64,
            np.dtype("complex64"): self.torch.complex64,
            np.dtype("complex128"): self.torch.complex128,
        }
        if np_dtype in mapping:
            return mapping[np_dtype]
        raise TypeError(f"Dtype {np_dtype!r} is not supported by PyTorch.")

    def get_dtype(self, x: Any) -> DType:
        if self.is_array(x):
            return x.dtype
        raise TypeError(f"Expected PyTorch tensor, got {type(x)}.")

    def shape(self, x: Any) -> tuple[int, ...]:
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        return int(x.ndim)

    def size(self, x: Any) -> int:
        return int(x.numel())

    @property
    def inf(self):
        return self.torch.tensor(float("inf"))

    @property
    def nan(self):
        return self.torch.tensor(float("nan"))

    @property
    def pi(self):
        return self.torch.tensor(np.pi)

    @property
    def e(self):
        return self.torch.tensor(np.e)

    @property
    def eps(self):
        return self.torch.tensor(self.torch.finfo(self.torch.float64).eps)

    def asarray(
        self,
        a: Any,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        copy: bool | None = None,
    ) -> DenseArray:
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if self.is_sparse(a):
            a = a.to_dense()
        out = self.torch.as_tensor(a, dtype=dtype, device=device)
        if copy:
            out = out.clone()
        return out

    def astype(self, x: DenseArray, dtype: DType, copy: bool = True) -> DenseArray:
        return x.to(dtype=self.sanitize_dtype(dtype), copy=copy)

    def assparse(
        self,
        x: Any,
        *,
        format: Literal["coo", "csr", "csc"] = "coo",
        dtype: DType | None = None,
        device: Any | None = None,
    ) -> SparseArray:
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if self.is_sparse(x):
            y = x.to(dtype=dtype, device=device) if dtype is not None or device is not None else x
            if format == "coo":
                return y.to_sparse_coo()
            if format == "csr":
                return y.to_sparse_csr()
            if format == "csc":
                return y.to_sparse_csc()
            raise ValueError(f"Unknown sparse format: {format!r}")

        try:
            import scipy.sparse as sps
        except Exception:
            sps = None

        if sps is not None and sps.issparse(x):
            coo = x.tocoo()
            indices = self.torch.as_tensor(
                np.vstack((coo.row, coo.col)),
                dtype=self.torch.int64,
                device=device,
            )
            values = self.torch.as_tensor(coo.data, dtype=dtype, device=device)
            out = self.torch.sparse_coo_tensor(indices, values, coo.shape, device=device)
        else:
            out = self.asarray(x, dtype=dtype, device=device).to_sparse_coo()

        if format == "coo":
            return out.coalesce()
        if format == "csr":
            return out.to_sparse_csr()
        if format == "csc":
            return out.to_sparse_csc()
        raise ValueError(f"Unknown sparse format: {format!r}")

    def empty(self, shape: int | Tuple[int, ...], dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.empty(shape, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def zeros(self, shape: int | Tuple[int, ...], dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.zeros(shape, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def ones(self, shape: int | Tuple[int, ...], dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.ones(shape, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def zeros_like(self, x: DenseArray, dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.zeros_like(x, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def ones_like(self, x: DenseArray, dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.ones_like(x, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def full_like(self, x: DenseArray, value: Any, dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.full_like(x, value, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def arange(self, start: int, stop: int | None = None, step: int | None = None, dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if stop is None:
            return self.torch.arange(start, dtype=dtype, device=device)
        if step is None:
            return self.torch.arange(start, stop, dtype=dtype, device=device)
        return self.torch.arange(start, stop, step, dtype=dtype, device=device)

    def full(self, shape: int | Tuple[int, ...], fill_value: Any, dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        return self.torch.full(shape, fill_value, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, device=device)

    def eye(self, N: int, M: int | None = None, k: int = 0, dtype: DType | None = None, *, device: Any | None = None) -> DenseArray:
        M = N if M is None else M
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        out = self.torch.zeros((N, M), dtype=dtype, device=device)
        if k == 0:
            diag_len = min(N, M)
            out[self.torch.arange(diag_len, device=device), self.torch.arange(diag_len, device=device)] = 1
            return out
        diag_len = min(N, M - k) if k > 0 else min(N + k, M)
        if diag_len <= 0:
            return out
        rows = self.torch.arange(diag_len, device=device)
        cols = rows + k
        if k < 0:
            rows = rows - k
            cols = self.torch.arange(diag_len, device=device)
        out[rows, cols] = 1
        return out

    def ravel(self, x: DenseArray) -> DenseArray:
        return self.torch.ravel(x)

    def reshape(self, x: DenseArray, shape: int | Tuple[int, ...], *, copy: bool | None = None) -> DenseArray:
        if copy:
            x = x.clone()
        return self.torch.reshape(x, shape if isinstance(shape, tuple) else (shape,))

    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        if axes is None:
            axes = tuple(reversed(range(x.ndim)))
        return x.permute(tuple(axes))

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        return self.torch.swapaxes(x, axis1, axis2)

    def broadcast_to(self, x: DenseArray, shape: int | Tuple[int, ...]) -> DenseArray:
        return self.torch.broadcast_to(x, shape)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        if isinstance(axis, int):
            return self.torch.unsqueeze(x, axis)
        ndim = x.ndim + len(axis)
        axes = sorted(a + ndim if a < 0 else a for a in axis)
        out = x
        for ax in axes:
            out = self.torch.unsqueeze(out, ax)
        return out

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        if axis is None:
            return self.torch.squeeze(x)
        if isinstance(axis, int):
            return self.torch.squeeze(x, dim=axis)
        out = x
        for ax in sorted(axis, reverse=True):
            out = self.torch.squeeze(out, dim=ax)
        return out

    def moveaxis(self, x: DenseArray, source: int | Sequence[int], destination: int | Sequence[int]) -> DenseArray:
        return self.torch.moveaxis(x, source, destination)

    def stack(self, arrays: Sequence[DenseArray], axis: int = 0, out: DenseArray | None = None) -> DenseArray:
        return self.torch.stack(tuple(arrays), dim=axis, out=out)

    def conj(self, x: DenseArray) -> DenseArray:
        return self.torch.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        return self.torch.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        return self.torch.imag(x)

    def abs(self, x: DenseArray) -> DenseArray:
        return self.torch.abs(x)

    def sign(self, x: DenseArray) -> DenseArray:
        return self.torch.sign(x)

    def sqrt(self, x: DenseArray) -> DenseArray:
        return self.torch.sqrt(x)

    def sum(self, x: DenseArray, axis: int | Sequence[int] | None = None, dtype: DType | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        return self.torch.sum(x, dim=axis, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, keepdim=keepdims)

    def mean(self, x: DenseArray, axis: int | Sequence[int] | None = None, dtype: DType | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        return self.torch.mean(x, dim=axis, dtype=self.sanitize_dtype(dtype) if dtype is not None else None, keepdim=keepdims)

    def min(self, x: DenseArray, axis: int | Sequence[int] | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        return self.torch.amin(x, dim=axis, keepdim=keepdims) if axis is not None else self.torch.min(x)

    def max(self, x: DenseArray, axis: int | Sequence[int] | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        return self.torch.amax(x, dim=axis, keepdim=keepdims) if axis is not None else self.torch.max(x)

    def prod(self, x: DenseArray, axis: int | Sequence[int] | None = None, dtype: DType | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if axis is None:
            return self.torch.prod(x, dtype=dtype)
        if isinstance(axis, int):
            return self.torch.prod(x, dim=axis, dtype=dtype, keepdim=keepdims)
        out = x
        for ax in sorted(axis, reverse=True):
            out = self.torch.prod(out, dim=ax, dtype=dtype, keepdim=keepdims)
        return out

    def trace(self, x: DenseArray, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: DType | None = None, **_: Any) -> DenseArray:
        return self.sum(self.diagonal(x, offset=offset, axis1=axis1, axis2=axis2), dtype=dtype)

    def argsort(self, x: DenseArray, axis: int = -1, stable: bool = False, descending: bool = False, **_: Any) -> DenseArray:
        return self.torch.argsort(x, dim=axis, stable=stable, descending=descending)

    def sort(self, x: DenseArray, axis: int = -1, stable: bool = False, descending: bool = False, **_: Any) -> DenseArray:
        return self.torch.sort(x, dim=axis, stable=stable, descending=descending).values

    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        return self.torch.argmin(x, dim=axis, keepdim=keepdims)

    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False, **_: Any) -> DenseArray:
        return self.torch.argmax(x, dim=axis, keepdim=keepdims)

    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
        return self.torch.vdot(self.ravel(x), self.ravel(y))

    def matmul(self, a: DenseArray, b: DenseArray, **_: Any) -> DenseArray:
        return self.torch.matmul(a, b)

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        if b.ndim == 1:
            return self.torch.sparse.mm(a, b[:, None])[:, 0]
        return self.torch.sparse.mm(a, b)

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        return self.torch.kron(a, b)

    def einsum(self, subscripts: str, *operands: DenseArray, **_: Any) -> DenseArray:
        return self.torch.einsum(subscripts, *operands)

    def eigh(self, x: DenseArray) -> tuple[DenseArray, DenseArray]:
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        return self.torch.linalg.eigh(x)

    def norm(self, x: DenseArray, ord: int | str | None = None, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DenseArray:
        return self.torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)

    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        return self.torch.linalg.solve(A, b)

    def eigvalsh(self, A: DenseArray) -> DenseArray:
        return self.torch.linalg.eigvalsh(A)

    def svd(self, A: DenseArray, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False) -> DenseArray | tuple[DenseArray, DenseArray, DenseArray]:
        if hermitian:
            raise NotImplementedError("PyTorch svd does not expose a hermitian option.")
        if not compute_uv:
            return self.torch.linalg.svdvals(A)
        return self.torch.linalg.svd(A, full_matrices=full_matrices)

    def cholesky(self, A: DenseArray) -> DenseArray:
        return self.torch.linalg.cholesky(A)

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None, keepdims: bool = False, return_sign: bool = False) -> DenseArray | tuple[DenseArray, DenseArray]:
        dim = tuple(range(a.ndim)) if axis is None else axis
        if b is None and not return_sign:
            return self.torch.logsumexp(a, dim=dim, keepdim=keepdims)
        weights = self.ones_like(a) if b is None else b
        m = self.torch.amax(a, dim=dim, keepdim=True)
        total = self.sum(weights * self.torch.exp(a - m), axis=dim, keepdims=True)
        sign = self.torch.sign(total)
        out = self.torch.log(self.torch.abs(total)) + m
        if not keepdims:
            out = self.squeeze(out, axis)
            sign = self.squeeze(sign, axis)
        return (out, sign) if return_sign else out

    def exp(self, x: DenseArray) -> DenseArray:
        return self.torch.exp(x)

    def log(self, x: DenseArray) -> DenseArray:
        return self.torch.log(x)

    def where(self, condition: DenseArray | bool, x: ArrayLike, y: ArrayLike) -> DenseArray:
        return self.torch.where(condition, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        return self.torch.maximum(x, y if isinstance(y, self.torch.Tensor) else self.asarray(y, dtype=x.dtype, device=x.device))

    def minimum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        return self.torch.minimum(x, y if isinstance(y, self.torch.Tensor) else self.asarray(y, dtype=x.dtype, device=x.device))

    def clip(self, x: DenseArray, a_min: ArrayLike, a_max: ArrayLike) -> DenseArray:
        return self.torch.clamp(x, min=a_min, max=a_max)

    def isfinite(self, x: DenseArray) -> DenseArray:
        return self.torch.isfinite(x)

    def isnan(self, x: DenseArray) -> DenseArray:
        return self.torch.isnan(x)

    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType | None = None) -> DenseArray:
        out = self.torch.cat(tuple(arrays), dim=axis)
        return self.astype(out, dtype) if dtype is not None else out

    def take(self, x: DenseArray, indices: DenseArray, axis: int | None = None, **_: Any) -> DenseArray:
        if axis is None:
            return self.torch.take(x, indices)
        return self.torch.index_select(x, dim=axis, index=indices)

    def diag(self, x: DenseArray, k: int = 0) -> DenseArray:
        return self.torch.diag(x, diagonal=k)

    def diagonal(self, x: DenseArray, offset: int = 0, axis1: int = 0, axis2: int = 1) -> DenseArray:
        return self.torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)

    def tril(self, x: DenseArray, k: int = 0) -> DenseArray:
        return self.torch.tril(x, diagonal=k)

    def triu(self, x: DenseArray, k: int = 0) -> DenseArray:
        return self.torch.triu(x, diagonal=k)

    def index_set(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        y = x.clone() if copy else x
        y[index] = values
        return y

    def index_add(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        y = x.clone() if copy else x
        y[index] = y[index] + values
        return y

    def ix_(self, *args: Any) -> Any:
        tensors = tuple(arg if isinstance(arg, self.torch.Tensor) else self.asarray(arg) for arg in args)
        return self.torch.meshgrid(*tensors, indexing="ij")

    def fori_loop(self, lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T) -> T:
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    def while_loop(self, cond_fun: Callable[[T], bool], body_fun: Callable[[T], T], init_val: T) -> T:
        val = init_val
        while bool(cond_fun(val)):
            val = body_fun(val)
        return val

    def _tree_map(self, f: Callable[[Any], Any], tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: self._tree_map(f, v) for k, v in tree.items()}
        if isinstance(tree, tuple):
            return tuple(self._tree_map(f, v) for v in tree)
        if isinstance(tree, list):
            return [self._tree_map(f, v) for v in tree]
        return f(tree)

    def _tree_multimap(self, f: Callable[..., Any], *trees: Any) -> Any:
        t0 = trees[0]
        if isinstance(t0, dict):
            return {k: self._tree_multimap(f, *(t[k] for t in trees)) for k in t0.keys()}
        if isinstance(t0, tuple):
            return tuple(self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0)))
        if isinstance(t0, list):
            return [self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0))]
        return f(*trees)

    def _tree_take0(self, xs: Any) -> Any:
        if isinstance(xs, dict):
            return self._tree_take0(next(iter(xs.values())))
        if isinstance(xs, (tuple, list)):
            return self._tree_take0(xs[0])
        return xs

    def _tree_index(self, xs: Any, i: int) -> Any:
        def _idx(a: Any) -> Any:
            try:
                return a[i]
            except Exception:
                return a

        return self._tree_map(_idx, xs)

    def _tree_stack(self, ys_list: Sequence[Any]) -> Any:
        if not ys_list:
            return ()
        return self._tree_multimap(lambda *leaves: self.stack(leaves, axis=0), *ys_list)

    def scan(
        self,
        f: Callable[[Carry, X], Tuple[Carry, Y]],
        init: Carry,
        xs: X,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
    ) -> Tuple[Carry, Y]:
        carry = init
        if xs is None:
            if length is None:
                raise ValueError("scan(xs=None) requires an explicit `length`.")
            indices = range(int(length) - 1, -1, -1) if reverse else range(int(length))
            ys_steps = []
            for _ in indices:
                carry, y = f(carry, None)  # type: ignore[arg-type]
                ys_steps.append(y)
        else:
            n = int(length) if length is not None else int(self._tree_take0(xs).shape[0])
            indices = range(n - 1, -1, -1) if reverse else range(n)
            ys_steps = []
            for i in indices:
                carry, y = f(carry, self._tree_index(xs, i))
                ys_steps.append(y)
        if reverse:
            ys_steps.reverse()
        return carry, self._tree_stack(ys_steps)

    def cond(self, pred: bool, true_fun: Callable[[T], R], false_fun: Callable[[T], R], *operands: Any) -> R:
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    def allclose(self, a: DenseArray, b: DenseArray, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
        return bool(self.torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    def allclose_sparse(self, a: SparseArray, b: SparseArray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError("allclose_sparse expects two sparse tensors.")
        return self.allclose(a.to_dense(), b.to_dense(), rtol=rtol, atol=atol)

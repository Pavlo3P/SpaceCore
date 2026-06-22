from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, Tuple, Type

import numpy as np

from .._eager import EagerControlFlowMixin
from .._family import BackendFamily
from .._ops import BackendOps, LazyNamespace
from ...types import DenseArray, DType, Index, SparseArray


class TorchOps(EagerControlFlowMixin, BackendOps):
    """
    BackendOps implementation for PyTorch tensors.

    This backend uses PyTorch for dense and sparse tensor operations.

    Dense arrays
        torch.Tensor with strided layout

    Sparse arrays
        torch.Tensor with a PyTorch sparse layout

    Methods
    -------
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
    -----
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

    xp = LazyNamespace("array_api_compat.torch")

    _family = BackendFamily.torch.value.lower()
    _allow_sparse = True

    _sparse_layouts = (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    )

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _defined_kwargs(**kwargs: Any) -> dict[str, Any]:
        return {key: value for key, value in kwargs.items() if value is not None}

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using PyTorch.

        Returns
        -------
            Concrete dense tensor class accepted by this backend.

        See:
            https://docs.pytorch.org/docs/stable/tensors.html
        """
        return self.torch.Tensor

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using PyTorch.

        Returns
        -------
            Tensor class accepted by this backend for sparse tensor layouts.

        See:
            https://docs.pytorch.org/docs/stable/sparse.html
        """
        return (self.torch.Tensor,)

    def is_dense(self, x: Any) -> bool:
        """
        Check whether an object is a dense PyTorch tensor.

        Input:
            x: Object to inspect.

        Output:
            Boolean indicating whether x is a strided PyTorch tensor.

        See:
            https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-layout
        """
        return isinstance(x, self.torch.Tensor) and x.layout == self.torch.strided

    def is_sparse(self, x: Any) -> bool:
        """
        Check whether an object is a sparse PyTorch tensor.

        Input:
            x: Object to inspect.

        Output:
            Boolean indicating whether x is a PyTorch tensor with a sparse layout.

        See:
            https://docs.pytorch.org/docs/stable/sparse.html
        """
        return isinstance(x, self.torch.Tensor) and x.layout in self._sparse_layouts

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize a dtype specifier using PyTorch.

        Input:
            dtype: Optional dtype requested by SpaceCore or the caller.

        Output:
            Backend dtype object accepted by PyTorch tensor constructors.

        See:
            https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype

        Backend-specific notes:
            ``None`` follows ``torch.get_default_dtype()``. NumPy dtype
            specifiers are mapped to equivalent PyTorch dtypes when supported.
        """
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

    def assparse(
        self,
        x: Any,
        *,
        format: Literal["coo", "csr", "csc"] = "coo",
        dtype: DType | None = None,
        device: Any | None = None,
    ) -> SparseArray:
        """
        Convert input to a sparse tensor using PyTorch.

        Input:
            x: Dense, sparse, or SciPy sparse input plus sparse format, dtype, and device.

        Output:
            Sparse backend tensor in COO, CSR, or CSC format.

        See:
            https://docs.pytorch.org/docs/stable/sparse.html

        Backend-specific notes:
            SciPy sparse inputs are converted through COO indices and values.
            Dense inputs are converted through PyTorch's sparse COO conversion.
        """
        self._reject_complex_to_real(x, dtype, operation="assparse")
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

    def asarray(
        self,
        x: Any,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        copy: bool | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        self._reject_complex_to_real(x, dtype, operation="asarray")
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        if device is not None:
            kwargs["device"] = device
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if self.is_sparse(x):
            x = x.to_dense()
        out = self.torch.as_tensor(x, dtype=dtype, **kwargs)
        return out.clone() if copy else out

    def astype(
        self,
        x: DenseArray,
        dtype: DType | None,
        *,
        copy: bool = True,
        non_blocking: bool = False,
        memory_format: Any | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        if dtype is None:
            return x
        self._reject_complex_to_real(x, dtype, operation="astype")
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(memory_format=memory_format))
        return x.to(
            dtype=self.sanitize_dtype(dtype),
            non_blocking=non_blocking,
            copy=copy,
            **kwargs,
        )

    def empty(
        self,
        shape: Tuple[int, ...],
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
        memory_format: Any | None = None,
    ) -> DenseArray:
        return self.torch.empty(
            shape,
            out=out,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
            **self._defined_kwargs(layout=layout, device=device, memory_format=memory_format),
        )

    def zeros(
        self,
        shape: Tuple[int, ...],
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        return self.torch.zeros(
            shape,
            out=out,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device),
        )

    def zeros_like(
        self,
        x: DenseArray,
        dtype: DType | None = None,
        *,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
        memory_format: Any | None = None,
    ) -> DenseArray:
        return self.torch.zeros_like(
            x,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device, memory_format=memory_format),
        )

    def arange(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        kwargs = self._defined_kwargs(out=out, layout=layout, device=device)
        kwargs["requires_grad"] = requires_grad
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if stop is None:
            return self.torch.arange(start, dtype=dtype, **kwargs)
        if step is None:
            return self.torch.arange(start, stop, dtype=dtype, **kwargs)
        return self.torch.arange(start, stop, step, dtype=dtype, **kwargs)

    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        kwargs = {"dim": self._to_axis_tuple(axis), "keepdim": keepdims}
        if dtype is not None:
            kwargs["dtype"] = self.sanitize_dtype(dtype)
        if out is not None:
            kwargs["out"] = out
        return self.torch.sum(x, **kwargs)

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        if out is not None:
            kwargs["out"] = out
        return self.torch.matmul(a, b, **kwargs)

    def sparse_matmul(
        self,
        a: SparseArray,
        b: DenseArray,
        *,
        reduce: Literal["sum", "mean", "amax", "amin"] = "sum",
    ) -> DenseArray:
        """
        Matrix-multiply a sparse tensor by a dense tensor using PyTorch.

        Input:
            a: Sparse backend tensor; b: Dense backend tensor or vector.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html
        """
        kwargs = {"reduce": reduce} if reduce != "sum" else {}
        if b.ndim == 1:
            return self.torch.sparse.mm(a, b[:, None], **kwargs)[:, 0]
        return self.torch.sparse.mm(a, b, **kwargs)

    def vmap(
        self,
        fn: Callable,
        in_axes: int | Sequence[int | None] | None = 0,
        out_axes: int | Sequence[int | None] | None = 0,
    ) -> Callable:
        """Vectorize a function using PyTorch's native vmap when available."""
        vmap = getattr(self.torch, "vmap", None)
        if vmap is None and hasattr(self.torch, "func"):
            vmap = getattr(self.torch.func, "vmap", None)
        if vmap is None:
            return super().vmap(fn, in_axes=in_axes, out_axes=out_axes)
        return vmap(fn, in_dims=in_axes, out_dims=out_axes)

    @property
    def has_native_vmap(self) -> bool:
        """Return ``True`` because supported PyTorch versions provide native ``vmap``."""
        return True

    def eigh(
        self,
        x: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
        UPLO: Literal["L", "U"] = "L",
        *,
        out: tuple[DenseArray, DenseArray] | None = None,
    ) -> tuple[DenseArray, DenseArray]:
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(out=out))
        return self.torch.linalg.eigh(x, UPLO=UPLO, **kwargs)

    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        dtype: DType | None = None,
        out: DenseArray | None = None,
    ) -> DenseArray:
        return self.torch.linalg.norm(
            x,
            ord=ord,
            dim=axis,
            keepdim=keepdims,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            out=out,
        )

    def solve(
        self,
        A: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
        *,
        left: bool = True,
        out: DenseArray | None = None,
    ) -> DenseArray:
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(out=out))
        return self.torch.linalg.solve(A, b, left=left, **kwargs)

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        backend_kwargs: dict[str, Any] | None = None,
        *,
        driver: str | None = None,
        out: DenseArray | tuple[DenseArray, DenseArray, DenseArray] | None = None,
    ) -> DenseArray | tuple[DenseArray, DenseArray, DenseArray]:
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(driver=driver, out=out))
        return self.torch.linalg.svd(A, full_matrices=full_matrices, **kwargs)

    def cholesky(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
        *,
        upper: bool = False,
        out: DenseArray | None = None,
    ) -> DenseArray:
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(out=out))
        return self.torch.linalg.cholesky(A, upper=upper, **kwargs)

    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray | tuple[DenseArray, DenseArray]:
        """
        Compute log-sum-exp using PyTorch.

        Input:
            a: Dense backend tensor; axis, b, keepdims, return_sign: Reduction controls.

        Output:
            Dense backend tensor, or ``(value, sign)`` when ``return_sign`` is true.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html

        Backend-specific notes:
            Weighted and signed variants are implemented in SpaceCore because
            PyTorch's public ``logsumexp`` does not expose SciPy-style ``b`` or
            ``return_sign`` parameters.
        """
        dim = tuple(range(a.ndim)) if axis is None else axis
        if b is None and not return_sign:
            return self.torch.logsumexp(a, dim=dim, keepdim=keepdims, out=out)
        weights = self.ones_like(a) if b is None else b
        m = self.torch.amax(a, dim=dim, keepdim=True)
        total = self.sum(weights * self.torch.exp(a - m), axis=dim, keepdims=True)
        sign = self.torch.sign(total)
        result = self.torch.log(self.torch.abs(total)) + m
        if not keepdims:
            result = self.squeeze(result, axis)
            sign = self.squeeze(sign, axis)
        if return_sign:
            return result, sign
        if out is not None:
            out.copy_(result)
            return out
        return result

    def where(
        self,
        condition: DenseArray | bool,
        x: DenseArray,
        y: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        if out is None:
            return self.torch.where(condition, x, y)
        return self.torch.where(condition, x, y, out=out)

    def concatenate(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        if out is None:
            result = self.torch.cat(tuple(arrays), dim=axis)
        else:
            result = self.torch.cat(tuple(arrays), dim=axis, out=out)
        return self.astype(result, dtype) if dtype is not None else result

    def _copy(self, x: DenseArray) -> DenseArray:
        """Return a PyTorch clone of ``x`` (mutation primitive for index ops)."""
        return x.clone()

    def _scatter_add_inplace(self, y: DenseArray, index: Index, values: DenseArray) -> None:
        """Add ``values`` into ``y`` at ``index`` in place.

        Unlike NumPy's ``add.at``, plain indexed assignment does not accumulate
        repeated indices; this matches PyTorch's prior ``index_add`` behavior.
        """
        y[index] = y[index] + values

    def ix_(self, *args: Any) -> Any:
        """
        Construct open mesh indices using PyTorch.

        Input:
            args: One-dimensional index arrays or array-like objects.

        Output:
            Tuple of broadcastable index tensors.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.meshgrid.html
        """
        tensors = tuple(
            arg if isinstance(arg, self.torch.Tensor) else self.asarray(arg) for arg in args
        )
        return self.torch.meshgrid(*tensors, indexing="ij")

    def allclose_sparse(
        self, a: SparseArray, b: SparseArray, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """
        Compare sparse tensors elementwise within tolerances using PyTorch.

        Input:
            a, b: Sparse backend tensors; rtol and atol: Comparison controls.

        Output:
            Boolean indicating whether sparse tensors are close.

        See:
            https://docs.pytorch.org/docs/stable/sparse.html

        Backend-specific notes:
            Sparse tensors are compared by converting both operands to dense
            tensors before calling ``allclose``.
        """
        self._require_two_sparse(a, b, noun="sparse tensors")
        return self.allclose(a.to_dense(), b.to_dense(), rtol=rtol, atol=atol)

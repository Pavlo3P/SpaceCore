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

    @staticmethod
    def _defined_kwargs(**kwargs: Any) -> dict[str, Any]:
        return {key: value for key, value in kwargs.items() if value is not None}

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using PyTorch.

        Returns:
            Concrete dense tensor class accepted by this backend.

        See:
            https://docs.pytorch.org/docs/stable/tensors.html
        """
        return self.torch.Tensor

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using PyTorch.

        Returns:
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

    def get_dtype(self, x: Any) -> DType:
        """
        Return a tensor dtype using PyTorch.

        Input:
            x: Dense or sparse backend tensor.

        Output:
            Backend dtype associated with x.

        See:
            https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
        """
        if self.is_array(x):
            return x.dtype
        raise TypeError(f"Expected PyTorch tensor, got {type(x)}.")

    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return tensor shape metadata using PyTorch.

        Input:
            x: Dense or sparse backend tensor.

        Output:
            Tuple describing the logical shape of x.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.Tensor.shape.html
        """
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        """
        Return tensor rank metadata using PyTorch.

        Input:
            x: Dense or sparse backend tensor.

        Output:
            Number of dimensions in x.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.Tensor.ndim.html
        """
        return int(x.ndim)

    def size(self, x: Any) -> int:
        """
        Return logical element count using PyTorch.

        Input:
            x: Dense or sparse backend tensor.

        Output:
            Total number of logical dense elements.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.numel.html
        """
        return int(x.numel())

    @property
    def inf(self):
        """
        Positive infinity scalar using PyTorch.

        Returns:
            Backend tensor scalar representing positive infinity.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
        """
        return self.torch.tensor(float("inf"))

    @property
    def nan(self):
        """
        NaN scalar using PyTorch.

        Returns:
            Backend tensor scalar representing NaN.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
        """
        return self.torch.tensor(float("nan"))

    @property
    def pi(self):
        """
        Pi scalar using PyTorch.

        Returns:
            Backend tensor scalar representing pi.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
        """
        return self.torch.tensor(np.pi)

    @property
    def e(self):
        """
        Euler number scalar using PyTorch.

        Returns:
            Backend tensor scalar representing Euler's number.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
        """
        return self.torch.tensor(np.e)

    @property
    def eps(self):
        """
        Machine epsilon scalar using PyTorch.

        Returns:
            Backend tensor scalar for float64 machine epsilon.

        See:
            https://docs.pytorch.org/docs/stable/type_info.html#torch.finfo
        """
        return self.torch.tensor(self.torch.finfo(self.torch.float64).eps)

    def asarray(
        self,
        x: Any,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        copy: bool | None = None,
    ) -> DenseArray:
        """
        Convert input to a dense tensor using PyTorch.

        Input:
            x/a: Array-like input and optional dtype, device, or copy controls.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html

        Backend-specific notes:
            Sparse tensors are densified. Existing tensors keep autograd
            metadata according to normal PyTorch conversion rules.
        """
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if self.is_sparse(x):
            x = x.to_dense()
        out = self.torch.as_tensor(x, dtype=dtype, device=device)
        if copy:
            out = out.clone()
        return out

    def astype(
        self,
        x: DenseArray,
        dtype: DType,
        copy: bool = True,
        *,
        non_blocking: bool = False,
        memory_format: Any | None = None,
    ) -> DenseArray:
        """
        Cast a tensor to a dtype using PyTorch.

        Input:
            x: Dense backend tensor; dtype: Target dtype; copy: Whether to force a copy.

        Output:
            Tensor with the requested dtype.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html
        """
        return x.to(
            dtype=self.sanitize_dtype(dtype),
            non_blocking=non_blocking,
            copy=copy,
            **self._defined_kwargs(memory_format=memory_format),
        )

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

    def empty(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
        memory_format: Any | None = None,
    ) -> DenseArray:
        """
        Create an uninitialized dense tensor using PyTorch.

        Input:
            shape: Output shape; dtype and device: Optional construction parameters.

        Output:
            Dense backend tensor with uninitialized values.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.empty.html
        """
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
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        """
        Create a dense tensor filled with zeros using PyTorch.

        Input:
            shape: Output shape; dtype and device: Optional construction parameters.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.zeros.html
        """
        return self.torch.zeros(
            shape,
            out=out,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device),
        )

    def ones(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        """
        Create a dense tensor filled with ones using PyTorch.

        Input:
            shape: Output shape; dtype and device: Optional construction parameters.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.ones.html
        """
        return self.torch.ones(
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
        """
        Create a zero tensor matching another tensor using PyTorch.

        Input:
            x: Reference tensor; dtype and device: Optional overrides.

        Output:
            Dense backend tensor with shape matching x.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html
        """
        return self.torch.zeros_like(
            x,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device, memory_format=memory_format),
        )

    def ones_like(
        self,
        x: DenseArray,
        dtype: DType | None = None,
        *,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
        memory_format: Any | None = None,
    ) -> DenseArray:
        """
        Create a one tensor matching another tensor using PyTorch.

        Input:
            x: Reference tensor; dtype and device: Optional overrides.

        Output:
            Dense backend tensor with shape matching x.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.ones_like.html
        """
        return self.torch.ones_like(
            x,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device, memory_format=memory_format),
        )

    def full_like(
        self,
        x: DenseArray,
        value: Any,
        dtype: DType | None = None,
        *,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
        memory_format: Any | None = None,
    ) -> DenseArray:
        """
        Create a filled tensor matching another tensor using PyTorch.

        Input:
            x: Reference tensor; value: Fill value; dtype and device: Optional overrides.

        Output:
            Dense backend tensor with shape matching x.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.full_like.html
        """
        return self.torch.full_like(
            x,
            value,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device, memory_format=memory_format),
        )

    def arange(
        self,
        start: int | float = 0,
        stop: int | float | None = None,
        step: int | float | None = None,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        """
        Create a range tensor using PyTorch.

        Input:
            start, stop, step: Range parameters; dtype and device: Optional construction parameters.

        Output:
            Dense backend tensor containing evenly spaced values.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.arange.html
        """
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        kwargs = self._defined_kwargs(out=out, layout=layout, device=device)
        kwargs["requires_grad"] = requires_grad
        if stop is None:
            return self.torch.arange(start, dtype=dtype, **kwargs)
        if step is None:
            return self.torch.arange(start, stop, dtype=dtype, **kwargs)
        return self.torch.arange(start, stop, step, dtype=dtype, **kwargs)

    def full(
        self,
        shape: int | Tuple[int, ...],
        fill_value: Any,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        """
        Create a filled dense tensor using PyTorch.

        Input:
            shape: Output shape; fill_value: Fill value; dtype and device: Optional parameters.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.full.html
        """
        return self.torch.full(
            shape,
            fill_value,
            out=out,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            requires_grad=requires_grad,
            **self._defined_kwargs(layout=layout, device=device),
        )

    def eye(
        self,
        n: int,
        m: int | None = None,
        k: int = 0,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
        layout: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> DenseArray:
        """
        Create a two-dimensional identity-like tensor using PyTorch.

        Input:
            n, m: Matrix dimensions; k: Diagonal offset; dtype and device: Optional parameters.

        Output:
            Dense backend tensor with ones on the requested diagonal.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.eye.html

        Backend-specific notes:
            PyTorch ``torch.eye`` has no diagonal offset parameter, so SpaceCore
            constructs the offset diagonal explicitly.
        """
        m = n if m is None else m
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if k == 0:
            return self.torch.eye(
                n,
                m,
                out=out,
                dtype=dtype,
                requires_grad=requires_grad,
                **self._defined_kwargs(layout=layout, device=device),
            )
        out = self.torch.zeros(
            (n, m),
            out=out,
            dtype=dtype,
            requires_grad=False,
            **self._defined_kwargs(layout=layout, device=device),
        )
        diag_len = min(n, m - k) if k > 0 else min(n + k, m)
        if diag_len <= 0:
            return out
        rows = self.torch.arange(diag_len, device=device)
        cols = rows + k
        if k < 0:
            rows = rows - k
            cols = self.torch.arange(diag_len, device=device)
        out[rows, cols] = 1
        if requires_grad:
            out.requires_grad_()
        return out

    def ravel(self, x: DenseArray) -> DenseArray:
        """
        Flatten a tensor using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            One-dimensional tensor view or copy following PyTorch semantics.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.ravel.html
        """
        return self.torch.ravel(x)

    def reshape(self, x: DenseArray, shape: int | Tuple[int, ...], *, copy: bool | None = None) -> DenseArray:
        """
        Reshape a tensor using PyTorch.

        Input:
            x: Dense backend tensor; shape: Target shape; copy: Whether to clone first.

        Output:
            Reshaped dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.reshape.html
        """
        if copy:
            x = x.clone()
        return self.torch.reshape(x, shape if isinstance(shape, tuple) else (shape,))

    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        """
        Permute tensor axes using PyTorch.

        Input:
            x: Dense backend tensor; axes: Optional axis order.

        Output:
            Tensor with permuted axes.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.permute.html
        """
        if axes is None:
            axes = tuple(reversed(range(x.ndim)))
        return x.permute(tuple(axes))

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """
        Swap two tensor axes using PyTorch.

        Input:
            x: Dense backend tensor; axis1, axis2: Axes to swap.

        Output:
            Tensor with the requested axes swapped.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.swapaxes.html
        """
        return self.torch.swapaxes(x, axis1, axis2)

    def broadcast_to(self, x: DenseArray, shape: int | Tuple[int, ...]) -> DenseArray:
        """
        Broadcast a tensor to a shape using PyTorch.

        Input:
            x: Dense backend tensor; shape: Target broadcast shape.

        Output:
            Broadcasted tensor view following PyTorch broadcasting rules.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html
        """
        return self.torch.broadcast_to(x, shape)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """
        Insert singleton dimensions using PyTorch.

        Input:
            x: Dense backend tensor; axis: Axis or axes where dimensions are inserted.

        Output:
            Tensor with inserted singleton dimensions.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
        """
        if isinstance(axis, int):
            return self.torch.unsqueeze(x, axis)
        ndim = x.ndim + len(axis)
        axes = sorted(a + ndim if a < 0 else a for a in axis)
        out = x
        for ax in axes:
            out = self.torch.unsqueeze(out, ax)
        return out

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Remove singleton dimensions using PyTorch.

        Input:
            x: Dense backend tensor; axis: Optional axis or axes to squeeze.

        Output:
            Tensor with singleton dimensions removed.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html
        """
        if axis is None:
            return self.torch.squeeze(x)
        if isinstance(axis, int):
            return self.torch.squeeze(x, dim=axis)
        out = x
        for ax in sorted(axis, reverse=True):
            out = self.torch.squeeze(out, dim=ax)
        return out

    def moveaxis(self, x: DenseArray, source: int | Sequence[int], destination: int | Sequence[int]) -> DenseArray:
        """
        Move tensor axes to new positions using PyTorch.

        Input:
            x: Dense backend tensor; source and destination: Axis positions.

        Output:
            Tensor with axes moved.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.moveaxis.html
        """
        return self.torch.moveaxis(x, source, destination)

    def stack(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Stack tensors along a new axis using PyTorch.

        Input:
            arrays: Sequence of tensors; axis: New axis; out: Optional output tensor.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.stack.html
        """
        arrays = tuple(arrays)
        if out is None:
            return self.torch.stack(arrays, dim=axis)
        return self.torch.stack(arrays, dim=axis, out=out)

    def conj(self, x: DenseArray) -> DenseArray:
        """
        Return the complex conjugate using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Tensor containing complex conjugates.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.conj.html
        """
        return self.torch.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        """
        Return the real part of a tensor using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Tensor view or value containing real components.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.real.html
        """
        return self.torch.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """
        Return the imaginary part of a tensor using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Tensor view or value containing imaginary components.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.imag.html
        """
        return self.torch.imag(x)

    def abs(
        self,
        x: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise absolute value using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.abs.html
        """
        if out is None:
            return self.torch.abs(x)
        return self.torch.abs(x, out=out)

    def sign(
        self,
        x: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise sign using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.sign.html
        """
        if out is None:
            return self.torch.sign(x)
        return self.torch.sign(x, out=out)

    def sqrt(
        self,
        x: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise square root using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.sqrt.html
        """
        if out is None:
            return self.torch.sqrt(x)
        return self.torch.sqrt(x, out=out)

    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Sum tensor elements using PyTorch.

        Input:
            x: Dense backend tensor; axis, dtype, keepdims: Reduction controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.sum.html
        """
        if dtype is None:
            if out is None:
                if axis is None and not keepdims:
                    return self.torch.sum(x)
                return self.torch.sum(x, dim=axis, keepdim=keepdims)
            return self.torch.sum(x, dim=axis, keepdim=keepdims, out=out)
        dtype = self.sanitize_dtype(dtype)
        if out is None:
            if axis is None and not keepdims:
                return self.torch.sum(x, dtype=dtype)
            return self.torch.sum(x, dim=axis, keepdim=keepdims, dtype=dtype)
        return self.torch.sum(
            x,
            dim=axis,
            keepdim=keepdims,
            dtype=dtype,
            out=out,
        )

    def mean(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Average tensor elements using PyTorch.

        Input:
            x: Dense backend tensor; axis, dtype, keepdims: Reduction controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.mean.html
        """
        if dtype is None:
            if out is None:
                if axis is None and not keepdims:
                    return self.torch.mean(x)
                return self.torch.mean(x, dim=axis, keepdim=keepdims)
            return self.torch.mean(x, dim=axis, keepdim=keepdims, out=out)
        dtype = self.sanitize_dtype(dtype)
        if out is None:
            if axis is None and not keepdims:
                return self.torch.mean(x, dtype=dtype)
            return self.torch.mean(x, dim=axis, keepdim=keepdims, dtype=dtype)
        return self.torch.mean(
            x,
            dim=axis,
            keepdim=keepdims,
            dtype=dtype,
            out=out,
        )

    def min(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute minimum values using PyTorch.

        Input:
            x: Dense backend tensor; axis and keepdims: Reduction controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.amin.html
        """
        if out is None:
            if axis is None and not keepdims:
                return self.torch.amin(x)
            return self.torch.amin(x, dim=axis, keepdim=keepdims)
        return self.torch.amin(x, dim=axis, keepdim=keepdims, out=out)

    def max(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute maximum values using PyTorch.

        Input:
            x: Dense backend tensor; axis and keepdims: Reduction controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.amax.html
        """
        if out is None:
            if axis is None and not keepdims:
                return self.torch.amax(x)
            return self.torch.amax(x, dim=axis, keepdim=keepdims)
        return self.torch.amax(x, dim=axis, keepdim=keepdims, out=out)

    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Multiply tensor elements using PyTorch.

        Input:
            x: Dense backend tensor; axis, dtype, keepdims: Reduction controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.prod.html

        Backend-specific notes:
            Multiple-axis products are applied one axis at a time because
            PyTorch's ``torch.prod`` reduces a single dimension per call.
        """
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        if axis is None:
            result = self.torch.prod(x) if dtype is None else self.torch.prod(x, dtype=dtype)
            if out is not None:
                out.copy_(result)
                return out
            return result
        if isinstance(axis, int):
            if out is None:
                if dtype is None:
                    return self.torch.prod(x, dim=axis, keepdim=keepdims)
                return self.torch.prod(x, dim=axis, dtype=dtype, keepdim=keepdims)
            return self.torch.prod(x, dim=axis, dtype=dtype, keepdim=keepdims, out=out)
        result = x
        for ax in sorted(axis, reverse=True):
            result = self.torch.prod(result, dim=ax, dtype=dtype, keepdim=keepdims)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def trace(
        self,
        x: DenseArray,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DType | None = None,
    ) -> DenseArray:
        """
        Sum diagonal entries using PyTorch.

        Input:
            x: Dense backend tensor; offset, axis1, axis2, dtype: Diagonal controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.diagonal.html
        """
        return self.sum(self.diagonal(x, offset=offset, axis1=axis1, axis2=axis2), dtype=dtype)

    def argsort(
        self,
        x: DenseArray,
        axis: int = -1,
        stable: bool = False,
        descending: bool = False,
    ) -> DenseArray:
        """
        Return sorting indices using PyTorch.

        Input:
            x: Dense backend tensor; axis, stable, descending: Sorting controls.

        Output:
            Integer tensor of indices.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.argsort.html
        """
        return self.torch.argsort(x, dim=axis, stable=stable, descending=descending)

    def sort(
        self,
        x: DenseArray,
        axis: int = -1,
        stable: bool = False,
        descending: bool = False,
        *,
        out: tuple[DenseArray, DenseArray] | None = None,
    ) -> DenseArray:
        """
        Sort tensor values using PyTorch.

        Input:
            x: Dense backend tensor; axis, stable, descending: Sorting controls.

        Output:
            Dense backend tensor of sorted values.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.sort.html
        """
        return self.torch.sort(x, dim=axis, stable=stable, descending=descending, out=out).values

    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """
        Return indices of minimum values using PyTorch.

        Input:
            x: Dense backend tensor; axis and keepdims: Reduction controls.

        Output:
            Integer tensor of indices.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.argmin.html
        """
        return self.torch.argmin(x, dim=axis, keepdim=keepdims)

    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """
        Return indices of maximum values using PyTorch.

        Input:
            x: Dense backend tensor; axis and keepdims: Reduction controls.

        Output:
            Integer tensor of indices.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
        """
        return self.torch.argmax(x, dim=axis, keepdim=keepdims)

    def vdot(
        self,
        x: DenseArray,
        y: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute conjugating vector dot product using PyTorch.

        Input:
            x, y: Dense backend tensors.

        Output:
            Scalar tensor containing the vector dot product.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.vdot.html
        """
        x1 = x if x.ndim == 1 else self.torch.ravel(x)
        y1 = y if y.ndim == 1 else self.torch.ravel(y)
        if out is None:
            return self.torch.vdot(x1, y1)
        return self.torch.vdot(x1, y1, out=out)

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Matrix-multiply tensors using PyTorch.

        Input:
            a, b: Dense backend tensors.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
        """
        if out is None:
            return self.torch.matmul(a, b)
        return self.torch.matmul(a, b, out=out)

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

    def kron(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute the Kronecker product using PyTorch.

        Input:
            a, b: Dense backend tensors.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.kron.html
        """
        return self.torch.kron(a, b, out=out)

    def einsum(self, subscripts: str, *operands: DenseArray) -> DenseArray:
        """
        Evaluate an Einstein summation using PyTorch.

        Input:
            subscripts: Einsum expression; operands: Dense backend tensors.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.einsum.html
        """
        return self.torch.einsum(subscripts, *operands)

    def eigh(
        self,
        x: DenseArray,
        UPLO: Literal["L", "U"] = "L",
        *,
        out: tuple[DenseArray, DenseArray] | None = None,
    ) -> tuple[DenseArray, DenseArray]:
        """
        Compute Hermitian eigenvalues and eigenvectors using PyTorch.

        Input:
            x: Dense Hermitian or symmetric backend tensor.

        Output:
            Tuple of eigenvalues and eigenvectors.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html
        """
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        return self.torch.linalg.eigh(x, UPLO=UPLO, out=out)

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
        """
        Compute vector or matrix norms using PyTorch.

        Input:
            x: Dense backend tensor; ord, axis, keepdims: Norm controls.

        Output:
            Dense backend tensor or scalar tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.linalg.norm.html
        """
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
        *,
        left: bool = True,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Solve a linear system using PyTorch.

        Input:
            A: Coefficient tensor; b: Right-hand side tensor.

        Output:
            Dense backend tensor solving ``A @ x = b``.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html
        """
        return self.torch.linalg.solve(A, b, left=left, out=out)

    def eigvalsh(
        self,
        A: DenseArray,
        UPLO: Literal["L", "U"] = "L",
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute Hermitian eigenvalues using PyTorch.

        Input:
            A: Dense Hermitian or symmetric backend tensor.

        Output:
            Dense backend tensor of eigenvalues.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html
        """
        return self.torch.linalg.eigvalsh(A, UPLO=UPLO, out=out)

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        compute_uv: bool = True,
        hermitian: bool = False,
        *,
        driver: str | None = None,
        out: DenseArray | tuple[DenseArray, DenseArray, DenseArray] | None = None,
    ) -> DenseArray | tuple[DenseArray, DenseArray, DenseArray]:
        """
        Compute singular value decomposition using PyTorch.

        Input:
            A: Dense backend tensor; full_matrices, compute_uv, hermitian: SVD controls.

        Output:
            Singular values or tuple ``(U, S, Vh)``.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html

        Backend-specific notes:
            PyTorch does not expose a ``hermitian`` option for SVD. When
            ``compute_uv`` is false, this delegates to ``torch.linalg.svdvals``.
        """
        if hermitian:
            raise NotImplementedError("PyTorch svd does not expose a hermitian option.")
        if not compute_uv:
            return self.torch.linalg.svdvals(A, driver=driver, out=out)
        return self.torch.linalg.svd(A, full_matrices=full_matrices, driver=driver, out=out)

    def cholesky(
        self,
        A: DenseArray,
        *,
        upper: bool = False,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute a Cholesky factorization using PyTorch.

        Input:
            A: Positive-definite dense backend tensor.

        Output:
            Dense backend tensor containing the Cholesky factor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html
        """
        return self.torch.linalg.cholesky(A, upper=upper, out=out)

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

    def exp(
        self,
        x: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise exponential using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.exp.html
        """
        if out is None:
            return self.torch.exp(x)
        return self.torch.exp(x, out=out)

    def log(
        self,
        x: DenseArray,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise natural logarithm using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.log.html
        """
        if out is None:
            return self.torch.log(x)
        return self.torch.log(x, out=out)

    def where(
        self,
        condition: DenseArray | bool,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Select values conditionally using PyTorch.

        Input:
            condition: Boolean tensor or scalar; x, y: Values to select.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.where.html
        """
        if x is None and y is None:
            return self.torch.where(condition)
        if x is None or y is None:
            raise TypeError("where requires both x and y when either is provided.")
        if out is None:
            return self.torch.where(condition, x, y)
        return self.torch.where(condition, x, y, out=out)

    def maximum(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise maximum using PyTorch.

        Input:
            x, y: Array-like operands.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.maximum.html
        """
        y = y if isinstance(y, self.torch.Tensor) else self.asarray(y, dtype=x.dtype, device=x.device)
        if out is None:
            return self.torch.maximum(x, y)
        return self.torch.maximum(
            x,
            y,
            out=out,
        )

    def minimum(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute elementwise minimum using PyTorch.

        Input:
            x, y: Array-like operands.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.minimum.html
        """
        y = y if isinstance(y, self.torch.Tensor) else self.asarray(y, dtype=x.dtype, device=x.device)
        if out is None:
            return self.torch.minimum(x, y)
        return self.torch.minimum(
            x,
            y,
            out=out,
        )

    def clip(
        self,
        x: DenseArray,
        a_min: ArrayLike | None = None,
        a_max: ArrayLike | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Clip tensor values using PyTorch.

        Input:
            x: Dense backend tensor; a_min, a_max: Lower and upper bounds.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.clamp.html
        """
        if out is None:
            return self.torch.clamp(x, min=a_min, max=a_max)
        return self.torch.clamp(x, min=a_min, max=a_max, out=out)

    def isfinite(self, x: DenseArray) -> DenseArray:
        """
        Test finiteness elementwise using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Boolean dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
        """
        return self.torch.isfinite(x)

    def isnan(self, x: DenseArray) -> DenseArray:
        """
        Test NaN values elementwise using PyTorch.

        Input:
            x: Dense backend tensor.

        Output:
            Boolean dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.isnan.html
        """
        return self.torch.isnan(x)

    def concatenate(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        dtype: DType | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Concatenate tensors using PyTorch.

        Input:
            arrays: Sequence of dense backend tensors; axis: Concatenation axis; dtype: Optional cast.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.cat.html
        """
        arrays = tuple(arrays)
        if out is None:
            result = self.torch.cat(arrays, dim=axis)
        else:
            result = self.torch.cat(arrays, dim=axis, out=out)
        return self.astype(result, dtype) if dtype is not None else result

    def take(
        self,
        x: DenseArray,
        indices: DenseArray,
        axis: int | None = None,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Take tensor elements by index using PyTorch.

        Input:
            x: Dense backend tensor; indices: Integer indices; axis: Optional selection axis.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.take.html
        """
        if axis is None:
            result = self.torch.take(x, indices)
            if out is not None:
                out.copy_(result)
                return out
            return result
        return self.torch.index_select(x, dim=axis, index=indices, out=out)

    def diag(
        self,
        x: DenseArray,
        k: int = 0,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Extract or construct a diagonal tensor using PyTorch.

        Input:
            x: Dense backend tensor; k: Diagonal offset.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.diag.html
        """
        return self.torch.diag(x, diagonal=k, out=out)

    def diagonal(self, x: DenseArray, offset: int = 0, axis1: int = 0, axis2: int = 1) -> DenseArray:
        """
        Return a tensor diagonal using PyTorch.

        Input:
            x: Dense backend tensor; offset, axis1, axis2: Diagonal controls.

        Output:
            Dense backend tensor view or value.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.diagonal.html
        """
        return self.torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)

    def tril(
        self,
        x: DenseArray,
        k: int = 0,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Return the lower triangular part using PyTorch.

        Input:
            x: Dense backend tensor; k: Diagonal offset.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.tril.html
        """
        return self.torch.tril(x, diagonal=k, out=out)

    def triu(
        self,
        x: DenseArray,
        k: int = 0,
        *,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Return the upper triangular part using PyTorch.

        Input:
            x: Dense backend tensor; k: Diagonal offset.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.triu.html
        """
        return self.torch.triu(x, diagonal=k, out=out)

    def index_set(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Set indexed tensor values using PyTorch.

        Input:
            x: Dense backend tensor; index: Index expression; values: Replacement values.

        Output:
            Tensor with indexed values replaced.

        See:
            https://docs.pytorch.org/docs/stable/tensor_view.html

        Backend-specific notes:
            When ``copy`` is true, this clones ``x`` before assignment.
            Otherwise the assignment mutates ``x`` in place.
        """
        y = x.clone() if copy else x
        y[index] = values
        return y

    def index_add(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Add values at indexed tensor positions using PyTorch.

        Input:
            x: Dense backend tensor; index: Index expression; values: Values to add.

        Output:
            Tensor with indexed values incremented.

        See:
            https://docs.pytorch.org/docs/stable/tensor_view.html

        Backend-specific notes:
            When ``copy`` is true, this clones ``x`` before assignment.
            Otherwise the assignment mutates ``x`` in place.
        """
        y = x.clone() if copy else x
        y[index] = y[index] + values
        return y

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
        tensors = tuple(arg if isinstance(arg, self.torch.Tensor) else self.asarray(arg) for arg in args)
        return self.torch.meshgrid(*tensors, indexing="ij")

    def fori_loop(self, lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T) -> T:
        """
        Run a counted loop eagerly in Python for PyTorch.

        Input:
            lower, upper: Integer loop bounds; body_fun: Loop body; init_val: Initial value.

        Output:
            Final loop value.

        See:
            https://docs.python.org/3/reference/compound_stmts.html#the-for-statement

        Backend-specific notes:
            This is an eager Python loop, not a compiled PyTorch control-flow
            primitive. Tensor operations inside ``body_fun`` follow PyTorch
            autograd semantics.
        """
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    def while_loop(self, cond_fun: Callable[[T], bool], body_fun: Callable[[T], T], init_val: T) -> T:
        """
        Run a while loop eagerly in Python for PyTorch.

        Input:
            cond_fun: Loop predicate; body_fun: Loop body; init_val: Initial value.

        Output:
            Final loop value.

        See:
            https://docs.python.org/3/reference/compound_stmts.html#the-while-statement

        Backend-specific notes:
            This is an eager Python loop. The predicate is converted to a
            Python bool each iteration.
        """
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
        """
        Run a scan-style loop eagerly in Python for PyTorch.

        Input:
            f: Scan body; init: Initial carry; xs: Per-step inputs plus scan options.

        Output:
            Tuple of final carry and stacked outputs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

        Backend-specific notes:
            PyTorch has no direct eager equivalent to ``jax.lax.scan`` in this
            backend. SpaceCore implements a Python loop and stacks tensor
            leaves at the end.
        """
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
        """
        Run conditional branch selection eagerly in Python for PyTorch.

        Input:
            pred: Predicate; true_fun and false_fun: Branch functions; operands: Branch inputs.

        Output:
            Result returned by the selected branch.

        See:
            https://docs.python.org/3/reference/expressions.html#conditional-expressions

        Backend-specific notes:
            This uses Python eager branching, not a staged or compiled control
            flow primitive.
        """
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    def allclose(self, a: DenseArray, b: DenseArray, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
        """
        Compare dense tensors elementwise within tolerances using PyTorch.

        Input:
            a, b: Dense backend tensors; rtol, atol, equal_nan: Comparison controls.

        Output:
            Boolean indicating whether tensors are close.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.allclose.html
        """
        return bool(self.torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    def allclose_sparse(self, a: SparseArray, b: SparseArray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
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
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError("allclose_sparse expects two sparse tensors.")
        return self.allclose(a.to_dense(), b.to_dense(), rtol=rtol, atol=atol)

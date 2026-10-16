from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, Tuple, Type, cast

import numpy as np

from .._eager import EagerControlFlowMixin
from .._family import BackendFamily
from .._ops import BackendOps
from ...types import ArrayLike, DenseArray, DType, Index, SparseArray


if TYPE_CHECKING:
    from torch import Tensor
    NativeArray = Tensor
else:
    NativeArray = DenseArray


class TorchOps(EagerControlFlowMixin, BackendOps[NativeArray]):
    """
    Backend operations on PyTorch tensors (strided dense, or any sparse layout).

    Methods delegate to ``torch`` and ``torch.linalg``, so dtypes, devices and autograd
    follow PyTorch; ``sanitize_dtype(None)`` is ``torch.get_default_dtype()``.
    ``ops.torch`` is an explicit PyTorch-only escape hatch.
    """

    import torch as _torch
    import array_api_compat.torch as xp

    # Concrete library handle exposed as ``Any`` so the portable protocols
    # (DenseArray/SparseArray) can be passed to typed PyTorch calls without a
    # cast at every boundary; mirrors the base ``xp: ClassVar[Any]`` design.
    torch: Any = _torch

    _family = BackendFamily.torch.value
    _allow_sparse = True

    _sparse_layouts = (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    )

    @classmethod
    def install_pytree_protocol(cls) -> None:
        """
        Register SpaceCore containers as torch pytree nodes.

        Torch takes children as a list and calls ``unflatten(children, context)``, the
        reverse of :class:`~spacecore.backend._container.PyTreeNode`'s order; this adapter
        translates. ``torch.utils._pytree`` is private API. Registering there also covers
        the C++ pytree, and a breaking change degrades to a registry warning.
        """
        import torch.utils._pytree as torch_pytree

        from .._container import registry

        # torch's pytree wants flatten -> (list, aux) and unflatten(children, aux).
        def registrar(klass: type) -> None:
            def flatten(obj: Any) -> tuple[list[Any], Any]:
                children, aux = obj.tree_flatten()
                return list(children), aux

            def unflatten(children: Any, aux: Any) -> Any:
                return klass.tree_unflatten(aux, tuple(children))

            torch_pytree.register_pytree_node(klass, flatten, unflatten)

        registry.register_backend("torch", registrar)

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
        """Normalize a dtype for torch; ``None`` gives ``torch.get_default_dtype()`` and NumPy dtypes map to their torch equivalents."""
        # Torch dtypes and Python types map directly; the rest go through NumPy dtype names.
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

        # NumPy and torch share the names of the dtypes they both support.
        torch_dtype = getattr(self.torch, np_dtype.name, None)
        if isinstance(torch_dtype, self.torch.dtype):
            return torch_dtype
        raise TypeError(f"Dtype {np_dtype!r} is not supported by PyTorch.")

    def assparse(
        self,
        x: Any,
        *,
        format: Literal["coo", "csr", "csc"] = "coo",
        dtype: DType | None = None,
        device: Any | None = None,
    ) -> SparseArray:
        """Convert to a torch sparse tensor (COO, CSR or CSC); SciPy inputs go through COO indices and values."""
        self._reject_complex_to_real(x, dtype, operation="assparse")
        dtype = self.sanitize_dtype(dtype) if dtype is not None else None
        # Already a torch sparse tensor: move/cast, then convert the layout.
        if self.is_sparse(x):
            y = x.to(dtype=dtype, device=device) if dtype is not None or device is not None else x
            if format == "coo":
                return y.to_sparse_coo()
            if format == "csr":
                return y.to_sparse_csr()
            if format == "csc":
                return y.to_sparse_csc()
            raise ValueError(f"Unknown sparse format: {format!r}")

        # SciPy is optional; its sparse input converts via COO triplets, dense input directly.
        try:
            import scipy.sparse as sps
        except Exception:
            sps = None

        if sps is not None and sps.issparse(x):
            coo = cast(Any, x).tocoo()
            indices = self.torch.as_tensor(
                np.vstack((coo.row, coo.col)),
                dtype=self.torch.int64,
                device=device,
            )
            values = self.torch.as_tensor(coo.data, dtype=dtype, device=device)
            out = self.torch.sparse_coo_tensor(indices, values, coo.shape, device=device)
        else:
            out = cast(Any, self.asarray(x, dtype=dtype, device=device)).to_sparse_coo()

        # Built as COO; convert to the requested layout.
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
        **extra_kwargs: Any,
    ) -> NativeArray:
        self._reject_complex_to_real(x, dtype, operation="asarray")
        # Merge backend kwargs; densify sparse input; copy only on request.
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(extra_kwargs)
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
        **extra_kwargs: Any,
    ) -> NativeArray:
        if dtype is None:
            return cast(NativeArray, x)
        self._reject_complex_to_real(x, dtype, operation="astype")
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(extra_kwargs)
        kwargs.update(self._defined_kwargs(memory_format=memory_format))
        return cast(Any, x).to(
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
    ) -> NativeArray:
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
    ) -> NativeArray:
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
    ) -> NativeArray:
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
    ) -> NativeArray:
        # NumPy's arange overloads, forwarding torch-only options.
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
    ) -> NativeArray:
        # NumPy keyword names mapped to torch's dim/keepdim.
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
    ) -> NativeArray:
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
    ) -> NativeArray:
        """
        Matrix-multiply a sparse tensor by a dense tensor using PyTorch.

        Input:
            a: Sparse backend tensor; b: Dense backend tensor or vector.

        Output:
            Dense backend tensor.

        See:
            https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html
        """
        # torch.sparse.mm wants a matrix: lift a vector to one column, then drop it.
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
        """Vectorize a function using ``torch.vmap``."""
        return self.torch.vmap(fn, in_dims=in_axes, out_dims=out_axes)

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
    ) -> tuple[NativeArray, NativeArray]:
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(out=out))
        eigenvalues, eigenvectors = self.torch.linalg.eigh(x, UPLO=UPLO, **kwargs)
        # Same ordering/gauge normalization as the neutral method: Torch's raw
        # complex Hermitian eigenvectors come back as -1 times NumPy's and JAX's.
        return self._order_eigenpairs(eigenvalues, eigenvectors)

    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        dtype: DType | None = None,
        out: DenseArray | None = None,
    ) -> NativeArray:
        return self.torch.linalg.norm(
            x,
            ord=ord,
            dim=axis,
            keepdim=keepdims,
            dtype=self.sanitize_dtype(dtype) if dtype is not None else None,
            out=out,
        )

    def diagonal(self, x: DenseArray) -> NativeArray:
        # torch.diagonal spells the axes dim1/dim2; same trailing-two-axes
        # convention as the neutral method.
        return self.torch.diagonal(x, dim1=-2, dim2=-1)

    def _take_along_axis(self, x: DenseArray, indices: DenseArray, axis: int) -> NativeArray:
        return self.torch.take_along_dim(x, indices, dim=axis)

    def solve(
        self,
        A: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
        *,
        left: bool = True,
        out: DenseArray | None = None,
    ) -> NativeArray:
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
    ) -> tuple[NativeArray, NativeArray, NativeArray]:
        kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        kwargs.update(self._defined_kwargs(driver=driver, out=out))
        U, s, Vh = self.torch.linalg.svd(A, full_matrices=full_matrices, **kwargs)
        return self._order_svd(U, s, Vh)

    def cholesky(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
        *,
        upper: bool = False,
        out: DenseArray | None = None,
    ) -> NativeArray:
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
    ) -> NativeArray | tuple[NativeArray, NativeArray]:
        """Compute log-sum-exp, with SciPy's ``b`` and ``return_sign`` options, which torch lacks, implemented here."""
        dim = tuple(range(a.ndim)) if axis is None else axis
        # Fast path: plain torch.logsumexp.
        if b is None and not return_sign:
            return self.torch.logsumexp(a, dim=dim, keepdim=keepdims, out=out)
        # Weighted form: log|sum b exp(a - m)| + m with m = max a; the sign is kept apart.
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
            cast(Any, out).copy_(result)
            return cast(NativeArray, out)
        return result

    def where(
        self,
        condition: DenseArray | bool,
        x: ArrayLike | float | complex,
        y: ArrayLike | float | complex,
        *,
        out: DenseArray | None = None,
    ) -> NativeArray:
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
    ) -> NativeArray:
        if out is None:
            result = self.torch.cat(tuple(arrays), dim=axis)
        else:
            result = self.torch.cat(tuple(arrays), dim=axis, out=out)
        return self.astype(result, dtype) if dtype is not None else result

    def _copy(self, x: DenseArray) -> NativeArray:
        """Return a PyTorch clone of ``x`` (mutation primitive for index ops)."""
        return cast(Any, x).clone()

    def _scatter_add_inplace(
        self, y: DenseArray, index: Index, values: ArrayLike | float | complex
    ) -> None:
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
        # ix_ is meshgrid with 'ij' indexing on 1-D tensors.
        tensors = tuple(
            arg if isinstance(arg, self.torch.Tensor) else self.asarray(arg) for arg in args
        )
        return self.torch.meshgrid(*tensors, indexing="ij")

    def allclose_sparse(
        self, a: SparseArray, b: SparseArray, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """Compare two torch sparse tensors within tolerances, by densifying both."""
        self._require_two_sparse(a, b, noun="sparse tensors")
        return self.allclose(cast(Any, a).to_dense(), cast(Any, b).to_dense(), rtol=rtol, atol=atol)

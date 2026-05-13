from __future__ import annotations

from typing import Any, Sequence, Tuple, Literal, Callable, Optional, Type
import inspect

from .._family import BackendFamily
from .._ops import BackendOps
from ...types import DenseArray, SparseArray, DType, Index, X, T, Y, R, Carry


class NumpyOps(BackendOps):
    """
    BackendOps implementation for the NumPy/SciPy ecosystem.

    This backend uses NumPy for dense array operations and SciPy for sparse
    array operations.

    Dense arrays
        numpy.ndarray

    Sparse arrays
        scipy.sparse.spmatrix
        scipy.sparse.sparray

    Methods
        Most methods mirror the corresponding NumPy or SciPy signatures and
        delegate directly to NumPy/SciPy implementations. Backend-specific
        behavior, dtype promotion, broadcasting, memory layout, and error modes
        therefore follow NumPy/SciPy semantics.

    Backend handles
      - np : module
        NumPy module stored on the class and available through instances as
        `ops.np`. Advanced users may use it when SpaceCore's portable API
        does not expose a required NumPy feature.

      - sp : module
        SciPy module stored on the class and available through instances as
        `ops.sp`. Advanced users may use it for SciPy-specific functionality.

    Notes
        Code intended to remain backend-portable should prefer `BackendOps`
        methods. Direct use of `ops.np` or `ops.sp` is an explicit
        NumPy/SciPy-specific escape hatch.

        NumPy's `device` keyword is present for Array API interoperability.
        When supplied, it must be `"cpu"` or `None`; see the corresponding NumPy
        documentation for each method.
    """
    import numpy as np
    import scipy as sp

    _family = BackendFamily.numpy.value.lower()
    _allow_sparse = True

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using NumPy.

        Returns:
            Concrete dense array class accepted by this backend.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
        """
        return self.np.ndarray

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using SciPy.

        Returns:
            Concrete sparse array classes accepted by this backend, or None.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html
        """
        sparse = self.sp.sparse
        types: list[type[Any]] = []
        if hasattr(sparse, "spmatrix"):
            types.append(sparse.spmatrix)
        if hasattr(sparse, "sparray"):
            types.append(sparse.sparray)
        return tuple(types)


    def __init__(self) -> None:
        self._reshape_supports_copy = "copy" in inspect.signature(self.np.reshape).parameters

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize a dtype specifier using NumPy.

        Input:
            dtype: Optional dtype requested by SpaceCore or the caller.

        Output:
            Backend dtype object accepted by array constructors.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.dtype.html
        """

        if dtype is None:
            return self.np.float64
        return self.np.dtype(dtype)

    def get_dtype(self, x: Any) -> DType:
        """
        Return an array dtype using NumPy.

        Input:
            x: Dense or sparse backend array.

        Output:
            Backend dtype associated with x.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dtype.html
        """
        if self.is_dense(x):
            return x.dtype
        elif self.is_sparse(x):
            return x.dtype
        else:
            raise TypeError(f'Expected Numpy ndarray or SciPy sparse array, got {type(x)}.')

    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return array shape metadata using NumPy.

        Input:
            x: Dense or sparse backend array.

        Output:
            Tuple describing the logical shape of x.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
        """
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        """
        Return array rank metadata using NumPy.

        Input:
            x: Dense or sparse backend array.

        Output:
            Number of dimensions in x.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
        """
        return int(x.ndim)

    def size(self, x: Any) -> int:
        """
        Return logical element count using NumPy.

        Input:
            x: Dense or sparse backend array.

        Output:
            Total number of logical dense elements.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html

        Backend-specific notes:
            SciPy sparse inputs are reported by logical dense size, not stored entries.
        """
        return int(self.np.prod(self.shape(x), dtype=self.np.intp))

    @property
    def inf(self):
        """
        Positive infinity scalar using NumPy.

        Returns:
            Backend scalar representing positive infinity.

        See:
            https://numpy.org/doc/stable/reference/constants.html
        """
        return self.np.array(self.np.inf)

    @property
    def nan(self):
        """
        NaN scalar using NumPy.

        Returns:
            Backend scalar representing NaN.

        See:
            https://numpy.org/doc/stable/reference/constants.html
        """
        return self.np.array(self.np.nan)

    @property
    def pi(self):
        """
        Pi scalar using NumPy.

        Returns:
            Backend scalar representing pi.

        See:
            https://numpy.org/doc/stable/reference/constants.html
        """
        return self.np.array(self.np.pi)

    @property
    def e(self):
        """
        Euler number scalar using NumPy.

        Returns:
            Backend scalar representing Euler's number.

        See:
            https://numpy.org/doc/stable/reference/constants.html
        """
        return self.np.array(self.np.e)

    @property
    def eps(self):
        """
        Machine epsilon scalar using NumPy.

        Returns:
            Backend scalar for float64 machine epsilon.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
        """
        return self.np.array(self.np.finfo(self.np.float64).eps)

    def asarray(
        self,
        a: Any,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] | None = None,
        *,
        device: str | None = None,
        copy: bool | None = None,
        like: DenseArray | None = None,
    ) -> DenseArray:
        """
        Convert input to a dense array using NumPy.

        Input:
            x/a: Array-like input and optional dtype or backend conversion parameters.

        Output:
            Dense backend array.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
        """
        return self.np.asarray(
            a,
            dtype=dtype,
            order=order,
            device=device,
            copy=copy,
            like=like,
        )

    def astype(
        self,
        x: DenseArray,
        dtype: DType,
        order: Literal["C", "F", "A", "K"] = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> DenseArray:
        """
        Cast an array to a dtype using NumPy.

        Input:
            x: Dense backend array; dtype: target dtype and optional casting controls.

        Output:
            Dense backend array with the requested dtype.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
        """
        return x.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)

    def assparse(self, x: Any, *, format: Literal["csr", "csc", "coo"] = "csr", dtype: DType | None = None) -> SparseArray:
        """
        Convert input to a sparse array using SciPy.

        Input:
            x: Dense, sparse, or array-like input plus sparse-format options.

        Output:
            Sparse backend array.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

        Backend-specific notes:
            SpaceCore currently converts dense inputs to 2-D SciPy sparse matrices in the requested format.
        """
        sparse = self.sp.sparse

        if self.is_sparse(x):
            if format == "csr":
                return x.tocsr()
            if format == "csc":
                return x.tocsc()
            if format == "coo":
                return x.tocoo()
            raise ValueError(f"Unknown sparse format: {format!r}")

        x_arr = self.asarray(x)

        if x_arr.ndim != 2:
            raise ValueError("NumPy/SciPy sparse conversion currently expects a 2D array.")

        if format == "csr":
            return sparse.csr_matrix(x_arr)
        if format == "csc":
            return sparse.csc_matrix(x_arr)
        if format == "coo":
            return sparse.coo_matrix(x_arr)

        raise ValueError(f"Unknown sparse format: {format!r}")

    def empty(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = float,
        order: Literal["C", "F"] = "C",
        *,
        device: str | None = None,
        like: DenseArray | None = None,
    ) -> DenseArray:
        """
        Create an uninitialized dense array using NumPy.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array with uninitialized values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.empty.html
        """

        return self.np.empty(
            shape,
            dtype=dtype,
            order=order,
            device=device,
            like=like,
        )

    def zeros(
            self,
            shape: int | Tuple[int, ...],
            dtype: DType | None = None,
            order: Literal["C", "F"] = "C",
            *,
            device: str | None = None,
            like: DenseArray | None = None,
    ) -> DenseArray:
        """
        Create a zero-filled dense array using NumPy.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array filled with zeros.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
        """
        return self.np.zeros(
            shape,
            dtype=dtype,
            order=order,
            device=device,
            like=like,
        )

    def ones(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        order: Literal["C", "F"] = "C",
        *,
        device: str | None = None,
        like: DenseArray | None = None,
    ) -> DenseArray:
        """
        Create a one-filled dense array using NumPy.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array filled with ones.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        """
        return self.np.ones(
            shape,
            dtype=dtype,
            order=order,
            device=device,
            like=like,
        )

    def zeros_like(
        self,
        x: DenseArray,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] = "K",
        subok: bool = True,
        shape: int | Tuple[int, ...] | None = None,
    ) -> DenseArray:
        """
        Create zeros shaped like another array using NumPy.

        Input:
            x: Prototype dense array; dtype, shape, and placement options are backend-specific.

        Output:
            Dense backend array of zeros.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
        """
        return self.np.zeros_like(x, dtype=dtype, order=order, subok=subok, shape=shape)

    def ones_like(
        self,
        x: DenseArray,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] = "K",
        subok: bool = True,
        shape: int | Tuple[int, ...] | None = None,
    ) -> DenseArray:
        """
        Create ones shaped like another array using NumPy.

        Input:
            x: Prototype dense array; dtype, shape, and placement options are backend-specific.

        Output:
            Dense backend array of ones.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
        """
        return self.np.ones_like(x, dtype=dtype, order=order, subok=subok, shape=shape)

    def full_like(
        self,
        x: DenseArray,
        value: Any,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] = "K",
        subok: bool = True,
        shape: int | Tuple[int, ...] | None = None,
    ) -> DenseArray:
        """
        Create filled values shaped like another array using NumPy.

        Input:
            x: Prototype dense array; value/fill_value and dtype options are backend-specific.

        Output:
            Dense backend array filled with the requested value.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.full_like.html
        """
        return self.np.full_like(x, value, dtype=dtype, order=order, subok=subok, shape=shape)

    def arange(self,
               start: int, stop: int | None = None,
               step: int | None = None,
               dtype: DType | None = None,
               *,
               device: str | None = None,
               like: DenseArray | None = None,
               ) -> DenseArray:
        """
        Create evenly spaced integer-range values using NumPy.

        Input:
            start, stop, step: Range parameters; dtype and placement options are backend-specific.

        Output:
            One-dimensional dense backend array.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        """
        return self.np.arange(
            start,
            stop,
            step,
            dtype=dtype,
            device=device,
            like=like,
        )

    def full(
        self,
        shape: int | Tuple[int, ...],
        fill_value: Any,
        dtype: DType | None = None,
        order: Literal["C", "F"] = "C",
        *,
        device: str | None = None,
        like: DenseArray | None = None,
    ) -> DenseArray:
        """
        Create a filled dense array using NumPy.

        Input:
            shape: Output shape; fill_value and dtype options are backend-specific.

        Output:
            Dense backend array filled with fill_value.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.full.html
        """
        return self.np.full(
            shape,
            fill_value,
            dtype=dtype,
            order=order,
            device=device,
            like=like,
        )

    def eye(
        self,
        N: int,
        M: int | None = None,
        k: int = 0,
        dtype: DType | None = float,
        order: Literal["C", "F"] = "C",
        *,
        device: str | None = None,
        like: DenseArray | None = None,
    ) -> DenseArray:
        """
        Create a dense identity-like matrix using NumPy.

        Input:
            n and optional m: Matrix dimensions; dtype and placement options are backend-specific.

        Output:
            Two-dimensional dense backend array.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.eye.html
        """
        return self.np.eye(
            N,
            M=M,
            k=k,
            dtype=dtype,
            order=order,
            device=device,
            like=like,
        )

    def ravel(self, a: DenseArray, order: Literal["C", "F", "A", "K"] = "C") -> DenseArray:
        """
        Flatten an array using NumPy.

        Input:
            x: Dense backend array plus optional order parameters.

        Output:
            One-dimensional dense backend array view or copy.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
        """
        return self.np.ravel(a, order=order)

    def reshape(
        self,
        a: DenseArray,
        shape: int | Tuple[int, ...],
        order: Literal["C", "F", "A", "K"] = "C",
        copy: bool | None = None,
    ) -> DenseArray:
        """
        Reshape an array using NumPy.

        Input:
            x: Dense backend array; shape: New shape plus backend-specific options.

        Output:
            Dense backend array with the requested shape.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        """
        if self._reshape_supports_copy:
            return self.np.reshape(a, shape, order=order, copy=copy)
        if copy:
            return self.np.array(a, copy=True).reshape(shape, order=order)
        return self.np.reshape(a, shape, order=order)

    def transpose(self, a: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        """
        Permute array axes using NumPy.

        Input:
            x: Dense backend array; axes: Optional axis order.

        Output:
            Dense backend array with permuted axes.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        """
        return self.np.transpose(a, axes=axes)

    def swapaxes(self, a: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """
        Interchange two axes using NumPy.

        Input:
            x: Dense backend array; axis1 and axis2: Axes to swap.

        Output:
            Dense backend array with the two axes exchanged.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html
        """
        return self.np.swapaxes(a, axis1, axis2)

    def broadcast_to(
        self,
        x: DenseArray,
        shape: int | Tuple[int, ...],
        subok: bool = False,
    ) -> DenseArray:
        """
        Broadcast an array to a shape using NumPy.

        Input:
            x: Dense backend array; shape: Target broadcast shape.

        Output:
            Dense backend array with broadcast shape.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
        """
        return self.np.broadcast_to(x, shape, subok=subok)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """
        Insert length-one axes using NumPy.

        Input:
            x: Dense backend array; axis: Position or positions to insert.

        Output:
            Dense backend array with expanded rank.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        """
        return self.np.expand_dims(x, axis=axis)

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Remove length-one axes using NumPy.

        Input:
            x: Dense backend array; axis: Optional axes to squeeze.

        Output:
            Dense backend array with selected singleton dimensions removed.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        """
        return self.np.squeeze(x, axis=axis)

    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """
        Move axes to new positions using NumPy.

        Input:
            x: Dense backend array; source and destination: Axis positions.

        Output:
            Dense backend array with moved axes.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html
        """
        return self.np.moveaxis(x, source=source, destination=destination)

    def stack(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        out: DenseArray | None = None,
        *,
        dtype: DType | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> DenseArray:
        """
        Stack arrays along a new axis using NumPy.

        Input:
            arrays: Sequence of dense backend arrays; axis: New axis position.

        Output:
            Dense backend array containing stacked inputs.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        """
        return self.np.stack(arrays, axis=axis, out=out, dtype=dtype, casting=casting)

    def conj(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute complex conjugates using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array with conjugated values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.conj.html
        """
        return self.np.conj(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def abs(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute absolute values using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of absolute values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.abs.html
        """
        return self.np.abs(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def sign(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute signs elementwise using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of signs.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.sign.html
        """
        return self.np.sign(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def sqrt(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute square roots elementwise using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of square roots.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        return self.np.sqrt(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def real(self, x: DenseArray) -> DenseArray:
        """
        Extract real components using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array containing real components.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.real.html
        """
        return self.np.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """
        Extract imaginary components using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array containing imaginary components.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.imag.html
        """
        return self.np.imag(x)

    def sum(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: DenseArray | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | bool = True,
    ) -> DenseArray:
        """
        Reduce by summation using NumPy.

        Input:
            x: Dense backend array; axis, keepdims, and dtype control the reduction.

        Output:
            Dense backend array or scalar containing sums.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        return self.np.sum(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def mean(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: DenseArray | None = None,
        keepdims: bool = False,
        where: DenseArray | bool = True,
    ) -> DenseArray:
        """
        Reduce by arithmetic mean using NumPy.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing means.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        return self.np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    def min(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        out: DenseArray | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | bool = True,
    ) -> DenseArray:
        """
        Reduce by minimum using NumPy.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing minima.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.min.html
        """
        return self.np.min(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def max(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        out: DenseArray | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | bool = True,
    ) -> DenseArray:
        """
        Reduce by maximum using NumPy.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing maxima.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.max.html
        """
        return self.np.max(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def prod(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: DenseArray | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | bool = True,
    ) -> DenseArray:
        """
        Reduce by product using NumPy.

        Input:
            x: Dense backend array; axis, keepdims, and dtype control the reduction.

        Output:
            Dense backend array or scalar containing products.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.prod.html
        """
        return self.np.prod(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def trace(
        self,
        a: DenseArray,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DType | None = None,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """
        Sum diagonal entries using NumPy.

        Input:
            x: Dense backend array plus optional diagonal and axis controls.

        Output:
            Dense backend array or scalar containing trace values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.trace.html
        """
        return self.np.trace(
            a,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            dtype=dtype,
            out=out,
        )

    def argsort(
        self,
        a: DenseArray,
        axis: int = -1,
        kind: Literal["quicksort", "mergesort", "heapsort", "stable"] | None = None,
        order: str | Sequence[str] | None = None,
        *,
        stable: bool | None = None,
    ) -> DenseArray:
        """
        Return sorting indices using NumPy.

        Input:
            x: Dense backend array; axis and ordering options are backend-specific.

        Output:
            Dense integer backend array of indices.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        """
        return self.np.argsort(a, axis=axis, kind=kind, order=order, stable=stable)

    def sort(
        self,
        a: DenseArray,
        axis: int = -1,
        kind: Literal["quicksort", "mergesort", "heapsort", "stable"] | None = None,
        order: str | Sequence[str] | None = None,
        *,
        stable: bool | None = None,
    ) -> DenseArray:
        """
        Sort values using NumPy.

        Input:
            x: Dense backend array; axis and ordering options are backend-specific.

        Output:
            Dense backend array with sorted values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        return self.np.sort(a, axis=axis, kind=kind, order=order, stable=stable)

    def argmin(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: DenseArray | None = None,
        *,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Return indices of minimum values using NumPy.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense integer backend array or scalar of indices.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        """
        return self.np.argmin(a, axis=axis, out=out, keepdims=keepdims)

    def argmax(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: DenseArray | None = None,
        *,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Return indices of maximum values using NumPy.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense integer backend array or scalar of indices.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        return self.np.argmax(a, axis=axis, out=out, keepdims=keepdims)

    def vdot(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """
        Compute a conjugating vector dot product using NumPy.

        Input:
            x, y: Dense backend arrays accepted by the backend vdot operation.

        Output:
            Backend scalar or dense array containing the dot product.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
        """
        return self.np.vdot(a, b)

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute matrix products using NumPy.

        Input:
            a, b: Dense backend arrays with matrix-multiplication-compatible shapes.

        Output:
            Dense backend array containing the product.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        """
        return self.np.matmul(
            a,
            b,
            out=out,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok
        )

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """
        Multiply sparse and dense arrays using SciPy.

        Input:
            a: Sparse backend array; b: Dense backend array.

        Output:
            Dense backend array containing the product.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

        Backend-specific notes:
            Uses SciPy sparse multiplication before returning a dense NumPy result when applicable.
        """
        if not self.is_sparse(a):
            raise TypeError("sparse_matmul expects `a` to be a SciPy sparse matrix/array.")
        if not self.is_dense(b):
            raise TypeError("sparse_matmul expects `b` to be a Numpy dense object.")
        return a @ b

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """
        Compute a Kronecker product using NumPy.

        Input:
            a, b: Dense backend arrays.

        Output:
            Dense backend array containing the Kronecker product.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.kron.html
        """
        return self.np.kron(a, b)

    def einsum(
        self,
        subscripts: str,
        *operands: DenseArray,
        out: DenseArray | None = None,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "safe",
        optimize: bool | str | Sequence[Any] = False,
    ) -> DenseArray:
        """
        Evaluate an Einstein summation expression using NumPy.

        Input:
            subscripts: Einstein summation string; operands: Dense backend arrays.

        Output:
            Dense backend array containing the contraction result.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        return self.np.einsum(
            subscripts,
            *operands,
            out=out,
            dtype=dtype,
            order=order,
            casting=casting,
            optimize=optimize,
        )

    def eigh(self, a: DenseArray, UPLO: Literal["L", "U"] = "L") -> Tuple[DenseArray, DenseArray]:
        """
        Compute Hermitian eigenpairs using NumPy.

        Input:
            x: Dense Hermitian or symmetric backend array.

        Output:
            Tuple of dense backend arrays containing eigenvalues and eigenvectors.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
        """
        if self.is_sparse(a):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        return self.np.linalg.eigh(a, UPLO=UPLO)

    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Compute vector or matrix norms using NumPy.

        Input:
            x: Dense backend array; ord, axis, and keepdims select the norm.

        Output:
            Dense backend array or scalar containing norm values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        return self.np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        """
        Solve dense linear systems using NumPy.

        Input:
            A: Dense coefficient array; b: Dense right-hand side array.

        Output:
            Dense backend array solving A @ x = b.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        """
        return self.np.linalg.solve(A, b)

    def eigvalsh(self, A: DenseArray, UPLO: Literal["L", "U"] = "L") -> DenseArray:
        """
        Compute Hermitian eigenvalues using NumPy.

        Input:
            A: Dense Hermitian or symmetric backend array.

        Output:
            Dense backend array containing eigenvalues.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html
        """
        return self.np.linalg.eigvalsh(A, UPLO=UPLO)

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        compute_uv: bool = True,
        hermitian: bool = False,
    ) -> DenseArray | Tuple[DenseArray, DenseArray, DenseArray]:
        """
        Compute singular value decompositions using NumPy.

        Input:
            A: Dense backend array plus SVD options.

        Output:
            Dense backend arrays containing singular vectors and/or singular values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        """
        return self.np.linalg.svd(
            A,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
        )

    def cholesky(self, A: DenseArray) -> DenseArray:
        """
        Compute Cholesky factors using NumPy.

        Input:
            A: Dense Hermitian positive-definite backend array.

        Output:
            Dense backend array containing a triangular factor.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
        """
        return self.np.linalg.cholesky(A)

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """
        Compute a stable log-sum-exp reduction using SciPy.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        See:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        return self.sp.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign)

    def exp(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute exponentials elementwise using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of exponentials.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        """
        return self.np.exp(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def log(
            self,
            x: DenseArray,
            /,
            out: DenseArray | None = None,
            *,
            where: DenseArray | bool = True,
            casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
            order: Literal["C", "F", "A", "K"] = "K",
            dtype: DType | None = None,
            subok: bool = True,
    ) -> DenseArray:
        """
        Compute natural logarithms elementwise using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of logarithms.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.log.html
        """
        return self.np.log(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def maximum(
            self,
            x: DenseArray,
            y: DenseArray,
            /,
            out: DenseArray | None = None,
            *,
            where: DenseArray | bool = True,
            casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
            order: Literal["C", "F", "A", "K"] = "K",
            dtype: DType | None = None,
            subok: bool = True,
    ) -> DenseArray:
        """
        Compute elementwise maxima using NumPy.

        Input:
            x, y: Arrays or scalars accepted by backend broadcasting.

        Output:
            Dense backend array containing maxima.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        """
        return self.np.maximum(
            x,
            y,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def minimum(
        self,
        x: DenseArray,
        y: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Compute elementwise minima using NumPy.

        Input:
            x, y: Arrays or scalars accepted by backend broadcasting.

        Output:
            Dense backend array containing minima.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        """
        return self.np.minimum(
            x,
            y,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def clip(
        self,
        x: DenseArray,
        a_min: DenseArray,
        a_max: DenseArray,
        out: DenseArray | None = None,
        **kwargs: Any,
    ) -> DenseArray:
        """
        Clip values into an interval using NumPy.

        Input:
            x: Dense backend array; a_min and a_max: Broadcastable bounds.

        Output:
            Dense backend array with clipped values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        """
        return self.np.clip(x, a_min, a_max, out=out, **kwargs)

    def isfinite(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Test finiteness elementwise using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Boolean dense backend array.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
        """
        return self.np.isfinite(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def isnan(
        self,
        x: DenseArray,
        /,
        out: DenseArray | None = None,
        *,
        where: DenseArray | bool = True,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
        order: Literal["C", "F", "A", "K"] = "K",
        dtype: DType | None = None,
        subok: bool = True,
    ) -> DenseArray:
        """
        Test NaN values elementwise using NumPy.

        Input:
            x: Dense backend array.

        Output:
            Boolean dense backend array.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        """
        return self.np.isnan(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def where(self, condition: DenseArray | bool, x: DenseArray, y: DenseArray) -> DenseArray:
        """
        Select values by condition using NumPy.

        Input:
            condition: Boolean array or scalar; x and y: Values to choose between.

        Output:
            Dense backend array containing selected values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """
        return self.np.where(condition, x, y)

    def concatenate(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        out: DenseArray | None = None,
        *,
        dtype: DType | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> DenseArray:
        """
        Join arrays along an existing axis using NumPy.

        Input:
            arrays: Sequence of dense backend arrays; axis and dtype options are backend-specific.

        Output:
            Dense backend array containing concatenated inputs.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        """
        return self.np.concatenate(arrays, axis=axis, out=out, dtype=dtype, casting=casting)

    def take(
        self,
        x: DenseArray,
        indices: DenseArray,
        axis: int | None = None,
        out: DenseArray | None = None,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> DenseArray:
        """
        Take values by integer indices using NumPy.

        Input:
            x: Dense backend array; indices: Integer indices; axis and mode options are backend-specific.

        Output:
            Dense backend array containing selected values.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.take.html
        """
        return self.np.take(x, indices, axis=axis, out=out, mode=mode)

    def diag(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Extract or build a diagonal using NumPy.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array containing a diagonal view/copy or matrix.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        """
        return self.np.diag(x, k=k)

    def diagonal(
        self,
        x: DenseArray,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> DenseArray:
        """
        Return selected diagonals using NumPy.

        Input:
            x: Dense backend array plus offset and axis controls.

        Output:
            Dense backend array containing selected diagonals.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
        """
        return self.np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    def tril(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return lower-triangular values using NumPy.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array with upper entries zeroed.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.tril.html
        """
        return self.np.tril(x, k=k)

    def triu(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return upper-triangular values using NumPy.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array with lower entries zeroed.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.triu.html
        """
        return self.np.triu(x, k=k)

    def index_set(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Set indexed values using NumPy.

        Input:
            x: Dense backend array; index: Selection; values: Replacement values; copy controls mutation policy.

        Output:
            Dense backend array with indexed values set.

        See:
            https://numpy.org/doc/stable/user/basics.indexing.html

        Backend-specific notes:
            With copy=True this copies before assignment; with copy=False it mutates the input array.
        """
        if copy:
            y = x.copy()
            y[index] = values
            return y
        else:
            x[index] = values
            return x

    def ix_(self, *args: Any) -> Any:
        """
        Build open mesh index arrays using NumPy.

        Input:
            args: One-dimensional index arrays or sequences.

        Output:
            Tuple of dense backend arrays usable for open-mesh indexing.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ix\\_.html
        """
        return self.np.ix_(*args)

    def fori_loop(
            self,
            lower: int,
            upper: int,
            body_fun: Callable[[int, T], T],
            init_val: T,
    ) -> T:
        """
        Run a counted loop primitive using NumPy.

        Input:
            lower, upper: Loop bounds; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html

        Backend-specific notes:
            NumPy executes this eagerly as a Python for-loop, without JAX tracing semantics.
        """
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    def while_loop(
            self,
            cond_fun: Callable[[T], bool],
            body_fun: Callable[[T], T],
            init_val: T,
    ) -> T:
        """
        Run a while-loop primitive using NumPy.

        Input:
            cond_fun: Loop condition; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html

        Backend-specific notes:
            NumPy executes this eagerly as a Python while-loop, without JAX tracing semantics.
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
        # assumes matching structure
        t0 = trees[0]
        if isinstance(t0, dict):
            return {k: self._tree_multimap(f, *(t[k] for t in trees)) for k in t0.keys()}
        if isinstance(t0, tuple):
            return tuple(self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0)))
        if isinstance(t0, list):
            return [self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0))]
        return f(*trees)

    def _tree_take0(self, xs: Any) -> Any:
        """
        Grab a representative leaf to infer leading length.
        """
        if isinstance(xs, dict):
            return self._tree_take0(next(iter(xs.values())))
        if isinstance(xs, (tuple, list)):
            return self._tree_take0(xs[0])
        return xs

    def _tree_index(self, xs: Any, i: int) -> Any:
        """
        Take per-step slice xs[i] along axis=0 for each leaf.
        """

        def _idx(a: Any) -> Any:
            # If it's an ndarray-like with leading axis, slice it; else treat as scalar leaf.
            try:
                return a[i]
            except Exception:
                return a

        return self._tree_map(_idx, xs)

    def _tree_stack(self, ys_list: Sequence[Any]) -> Any:
        """
        Stack a list of per-step outputs into a single pytree of arrays
        by stacking leaves along axis=0.
        """
        if not ys_list:
            # JAX would return empty stacked outputs when length == 0
            # Here we return an empty tuple (could also raise).
            return ()

        def _stack_leaves(*leaves: Any) -> Any:
            # Prefer self.np.stack when possible; fallback to array() if stacking fails.
            try:
                return self.np.stack(leaves, axis=0)
            except Exception:
                return self.np.array(leaves)

        # Reduce by structure: stack corresponding leaves across time
        return self._tree_multimap(_stack_leaves, *ys_list)

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
        Run a scan primitive using NumPy.

        Input:
            f: Scan body; init: Initial carry; xs: Per-step inputs plus scan options.

        Output:
            Tuple of final carry and stacked outputs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

        Backend-specific notes:
            NumPy executes this eagerly and accepts unroll only for API parity.
        """
        carry = init

        if xs is None:
            if length is None:
                raise ValueError("scan(xs=None) requires an explicit `length`.")
            n = int(length)
            indices = range(n - 1, -1, -1) if reverse else range(n)
            ys_steps: list[Any] = []
            for _i in indices:
                carry, y = f(carry, None)  # type: ignore[arg-type]
                ys_steps.append(y)
            if reverse:
                ys_steps.reverse()
            return carry, self._tree_stack(ys_steps)

        # infer length from xs if not provided
        if length is None:
            leaf0 = self._tree_take0(xs)
            try:
                n = int(leaf0.shape[0])  # ndarray-like
            except Exception as e:
                raise ValueError(
                    "Could not infer scan length from `xs`; pass `length=` explicitly."
                ) from e
        else:
            n = int(length)

        indices = range(n - 1, -1, -1) if reverse else range(n)
        ys_steps = []
        for i in indices:
            x_i = self._tree_index(xs, i)
            carry, y = f(carry, x_i)
            ys_steps.append(y)

        if reverse:
            ys_steps.reverse()

        return carry, self._tree_stack(ys_steps)

    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        # Eager branch selection (NumPy has no tracing semantics)
        """
        Run conditional branch selection using NumPy.

        Input:
            pred: Predicate; true_fun and false_fun: Branch functions; operands: Branch inputs.

        Output:
            Result returned by the selected branch.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html

        Backend-specific notes:
            NumPy chooses the branch eagerly with Python truth-value conversion.
        """
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    def index_add(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
        """
        Add into indexed values using NumPy.

        Input:
            x: Dense backend array; index: Selection; values: Values to add; copy controls mutation policy.

        Output:
            Dense backend array with indexed values incremented.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html

        Backend-specific notes:
            Uses numpy.add.at so repeated indices accumulate in NumPy order.
        """
        y = x.copy() if copy else x
        self.np.add.at(y, index, values)
        return y

    def allclose(
        self,
        a: DenseArray,
        b: DenseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        """
        Compare dense arrays elementwise within tolerances using NumPy.

        Input:
            a, b: Dense backend arrays; rtol, atol, and equal_nan configure comparison.

        Output:
            Boolean indicating whether arrays are close.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        return self.np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """
        Compare sparse arrays elementwise within tolerances using SciPy.

        Input:
            a, b: Sparse backend arrays; rtol and atol configure comparison.

        Output:
            Boolean indicating whether sparse arrays are close.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

        Backend-specific notes:
            Sparse inputs are converted to CSR and compared by logical sparse difference.
        """
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError("allclose_sparse expects two sparse arrays.")

        a = a.tocsr()
        b = b.tocsr()

        if a.shape != b.shape:
            return False

        diff = (a - b).tocsr()

        if diff.nnz == 0:
            return True

        a_abs = abs(a).tocsr()
        b_abs = abs(b).tocsr()
        scale = self.sp.sparse.csr_matrix.maximum(a_abs, b_abs)

        # tolerance_ij = atol + rtol * max(|a_ij|, |b_ij|)
        tol = scale.multiply(rtol)
        tol.data += atol

        bad = abs(diff) > tol
        return bad.nnz == 0

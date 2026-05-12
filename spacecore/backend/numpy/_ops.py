from __future__ import annotations

from typing import Any, Sequence, Tuple, Literal, Callable, Optional, Type
import inspect

from .._family import BackendFamily
from .._ops import BackendOps
from ...types import DenseArray, SparseArray, DType, Index, X, T, Y, R, Carry


class NumpyOps(BackendOps):
    import numpy as np
    import scipy as sp
    
    """
    BackendOps implementation for the NumPy ecosystem.

    Dense arrays:
      - numpy.ndarray

    Sparse arrays:
      - scipy.sparse.spmatrix and scipy.sparse.sparray (when SciPy is installed)

    Methods mirror NumPy signatures and delegate to NumPy/SciPy.

    Notes on `device`:
      - NumPy's `device` keyword is present for Array-API interoperability and,
        when passed, must be "cpu" (or None). See NumPy docs for each function.
    """

    _family = BackendFamily.numpy.value.lower()
    _allow_sparse = True

    @property
    def dense_array(self) -> Type[Any]:
        return self.np.ndarray

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
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
        """Normalize dtype to a NumPy dtype object. See: numpy.dtype."""

        if dtype is None:
            return self.np.float64
        return self.np.dtype(dtype)

    def get_dtype(self, x: Any) -> DType:
        if self.is_dense(x):
            return x.dtype
        elif self.is_sparse(x):
            return x.dtype
        else:
            raise TypeError(f'Expected Numpy ndarray or SciPy sparse array, got {type(x)}.')

    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return `x.shape` as a tuple.

        NumPy/SciPy expose eager Python shape metadata. For SciPy sparse arrays this
        reports logical dense shape, not stored entries.
        """
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        """
        Return the number of dimensions of `x`.

        NumPy arrays and SciPy sparse arrays expose eager Python metadata.
        """
        return int(x.ndim)

    def size(self, x: Any) -> int:
        """
        Return the logical dense element count of `x`.

        SciPy sparse `.size` may reflect stored data for some sparse classes, so this
        method computes the product of the logical shape.
        """
        return int(self.np.prod(self.shape(x), dtype=self.np.intp))

    @property
    def inf(self):
        return self.np.array(self.np.inf)

    @property
    def nan(self):
        return self.np.array(self.np.nan)

    @property
    def pi(self):
        return self.np.array(self.np.pi)

    @property
    def e(self):
        return self.np.array(self.np.e)

    @property
    def eps(self):
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
        Convert the input to an array. See: numpy.asarray.

        Signature mirrors:
          numpy.asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)
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
        Copy `x` cast to `dtype`. See: numpy.ndarray.astype.

        NumPy dtype promotion and casting safety are controlled by `casting`; device
        placement is always host CPU for NumPy arrays.
        """
        return x.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)

    def assparse(self, x: Any, *, format: Literal["csr", "csc", "coo"] = "csr", dtype: DType | None = None) -> SparseArray:
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
        Return a new array of given shape and type, without initializing entries.
        See: numpy.empty.

        Signature mirrors:
          numpy.empty(shape, dtype=float, order='C', *, device=None, like=None)
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
        Return a new array of given shape and type, filled with zeros.
        See: numpy.zeros.

        Signature mirrors:
          numpy.zeros(shape, dtype=None, order='C', *, device=None, like=None)
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
        Return a new array of given shape and type, filled with ones.
        See: numpy.ones.

        Signature mirrors:
          numpy.ones(shape, dtype=None, order='C', *, device=None, like=None)
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
        Return an array of zeros with shape and dtype like `x`. See: numpy.zeros_like.

        NumPy may preserve subclasses when `subok=True`; the portable API should not
        rely on subclass-specific behavior.
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
        Return an array of ones with shape and dtype like `x`. See: numpy.ones_like.

        NumPy may preserve subclasses when `subok=True`; dtype inference follows
        NumPy scalar promotion rules.
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
        Return an array filled with `value` and shaped like `x`. See: numpy.full_like.

        NumPy determines the result dtype from `x`, `value`, and explicit `dtype`.
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
        Return a new array of given shape and type, filled with `fill_value`.
        See: numpy.full.

        Signature mirrors:
          numpy.full(shape, fill_value, dtype=None, order='C', *, device=None, like=None)
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
        Return a 2-D array with ones on the diagonal and zeros elsewhere.
        See: numpy.eye.

        Signature mirrors:
          numpy.eye(N, M=None, k=0, dtype=float, order='C', *, device=None, like=None)
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
        """Return a contiguous flattened array. See: numpy.ravel."""
        return self.np.ravel(a, order=order)

    def reshape(
        self,
        a: DenseArray,
        shape: int | Tuple[int, ...],
        order: Literal["C", "F", "A", "K"] = "C",
        copy: bool | None = None,
    ) -> DenseArray:
        """Give a new shape to an array without changing its data. See: numpy.reshape."""
        if self._reshape_supports_copy:
            return self.np.reshape(a, shape, order=order, copy=copy)
        if copy:
            return self.np.array(a, copy=True).reshape(shape, order=order)
        return self.np.reshape(a, shape, order=order)

    def transpose(self, a: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        """Permute the dimensions of an array. See: numpy.transpose."""
        return self.np.transpose(a, axes=axes)

    def swapaxes(self, a: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """Interchange two axes of an array. See: numpy.swapaxes."""
        return self.np.swapaxes(a, axis1, axis2)

    def broadcast_to(
        self,
        x: DenseArray,
        shape: int | Tuple[int, ...],
        subok: bool = False,
    ) -> DenseArray:
        """
        Broadcast `x` to `shape`. See: numpy.broadcast_to.

        NumPy usually returns a readonly view; callers must not assume mutability.
        """
        return self.np.broadcast_to(x, shape, subok=subok)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """Insert new axes into `x`. See: numpy.expand_dims."""
        return self.np.expand_dims(x, axis=axis)

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Remove length-one axes from `x`. See: numpy.squeeze.

        NumPy may return a view; mutability aliases the original array when it does.
        """
        return self.np.squeeze(x, axis=axis)

    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """
        Move axes to new positions. See: numpy.moveaxis.

        NumPy returns a view when possible, so callers should avoid relying on copy
        or mutability behavior in portable code.
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
        Join a sequence of arrays along a new axis. See: numpy.stack.

        Signature mirrors (NumPy >= 1.24):
          numpy.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')
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
        """Return the complex conjugate, element-wise. See: numpy.conjugate / numpy.conj."""
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
        """Compute the absolute value element-wise. See: numpy.absolute / numpy.abs."""
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
        """Return an element-wise indication of the sign. See: numpy.sign."""
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
        """Return the non-negative square-root element-wise. See: numpy.sqrt."""
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
        """Return the real part of the complex argument. See: numpy.real."""
        return self.np.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """Return the imaginary part of the complex argument. See: numpy.imag."""
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
        """Sum of array elements over a given axis. See: numpy.sum."""
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
        Compute the arithmetic mean over an axis. See: numpy.mean.

        NumPy chooses accumulator dtype according to NumPy reduction rules unless
        `dtype` is supplied.
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
        Compute minimum values over an axis. See: numpy.min.

        Empty reductions, `initial`, `where`, and NaN behavior follow NumPy semantics.
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
        Compute maximum values over an axis. See: numpy.max.

        Empty reductions, `initial`, `where`, and NaN behavior follow NumPy semantics.
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
        """Product of array elements over a given axis. See: numpy.prod."""
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
        Return the sum along diagonals of the array.
        Works for N-D arrays via `axis1`/`axis2`. See: numpy.trace.
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
        """Return indices that would sort an array. See: numpy.argsort."""
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
        """Return a sorted copy of an array. See: numpy.sort."""
        return self.np.sort(a, axis=axis, kind=kind, order=order, stable=stable)

    def argmin(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: DenseArray | None = None,
        *,
        keepdims: bool = False,
    ) -> DenseArray:
        """Return indices of the minimum values along an axis. See: numpy.argmin."""
        return self.np.argmin(a, axis=axis, out=out, keepdims=keepdims)

    def argmax(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: DenseArray | None = None,
        *,
        keepdims: bool = False,
    ) -> DenseArray:
        """Return indices of the maximum values along an axis. See: numpy.argmax."""
        return self.np.argmax(a, axis=axis, out=out, keepdims=keepdims)

    def vdot(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """Return the dot product of two vectors. See: numpy.vdot."""
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
        Matrix product of two arrays (supports batched matmul semantics).
        See: numpy.matmul.
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
        Multiply a SciPy sparse matrix/array `a` by a NumPy dense array `b`.

        Notes:
          - SciPy sparse objects are 2-D; this method follows SciPy's `@` rules.

        See:
          - scipy.sparse.issparse
          - SciPy sparse `@` multiplication
        """
        if not self.is_sparse(a):
            raise TypeError("sparse_matmul expects `a` to be a SciPy sparse matrix/array.")
        if not self.is_dense(b):
            raise TypeError("sparse_matmul expects `b` to be a Numpy dense object.")
        return a @ b

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """Kronecker product of two arrays. See: numpy.kron."""
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
        """Evaluate the Einstein summation convention. See: numpy.einsum."""
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
        Return eigenvalues and eigenvectors of a Hermitian/symmetric array.

        Notes:
          - Supports stacked inputs of shape (..., M, M).

        See: numpy.linalg.eigh.
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
        Compute a vector or matrix norm. See: numpy.linalg.norm.

        Supported `ord` values and return dtype follow NumPy's linear algebra rules.
        """
        return self.np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        """
        Solve a dense linear system. See: numpy.linalg.solve.

        NumPy raises `LinAlgError` for singular input and uses host CPU LAPACK routines.
        """
        return self.np.linalg.solve(A, b)

    def eigvalsh(self, A: DenseArray, UPLO: Literal["L", "U"] = "L") -> DenseArray:
        """
        Return Hermitian/symmetric eigenvalues. See: numpy.linalg.eigvalsh.

        NumPy uses the selected triangle via `UPLO`; batching and precision follow NumPy.
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
        Compute singular value decomposition. See: numpy.linalg.svd.

        The portable path uses `compute_uv=True`; when `compute_uv=False`, NumPy returns
        only singular values and that is a NumPy-specific extension here.
        """
        return self.np.linalg.svd(
            A,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
        )

    def cholesky(self, A: DenseArray) -> DenseArray:
        """
        Compute the lower Cholesky factor. See: numpy.linalg.cholesky.

        NumPy raises `LinAlgError` when the input is not positive definite.
        """
        return self.np.linalg.cholesky(A)

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """ See: scipy.special.logsumexp. """
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
        """See: numpy.exp. """
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
        """See: numpy.log. """
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
        """See: numpy.maximum. """
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
        """See: numpy.minimum. """
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
        Clip values to an interval. See: numpy.clip.

        Broadcasting, dtype promotion, and optional NumPy-only keyword behavior follow
        NumPy. Portable code should pass only `x`, `a_min`, and `a_max`.
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
        Test elementwise finiteness. See: numpy.isfinite.

        NumPy supports ufunc keywords such as `where` and `out`; these are not part of
        the backend-neutral contract.
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
        Test elementwise NaN values. See: numpy.isnan.

        NumPy supports ufunc keywords such as `where` and `out`; these are not part of
        the backend-neutral contract.
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
        """ See: numpy.where. """
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
        Signature mirrors (NumPy >= 1.24):
          numpy.concatenate(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')
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
        Take elements by integer index. See: numpy.take.

        Portable code should pass valid indices because out-of-bounds modes differ
        from JAX defaults.
        """
        return self.np.take(x, indices, axis=axis, out=out, mode=mode)

    def diag(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Extract or construct a diagonal. See: numpy.diag.

        The portable signature uses the main diagonal; `k` is a NumPy-compatible extension.
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
        Return selected diagonals. See: numpy.diagonal.

        NumPy may return a view. Portable code should not rely on mutability or aliasing.
        """
        return self.np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    def tril(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return the lower triangle of `x`. See: numpy.tril.

        Entries above the selected diagonal are filled with zero of the result dtype.
        """
        return self.np.tril(x, k=k)

    def triu(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return the upper triangle of `x`. See: numpy.triu.

        Entries below the selected diagonal are filled with zero of the result dtype.
        """
        return self.np.triu(x, k=k)

    def index_set(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        if copy:
            y = x.copy()
            y[index] = values
            return y
        else:
            x[index] = values
            return x

    def ix_(self, *args: Any) -> Any:
        """ See: numpy.ix_. """
        return self.np.ix_(*args)

    def fori_loop(
            self,
            lower: int,
            upper: int,
            body_fun: Callable[[int, T], T],
            init_val: T,
    ) -> T:
        """NumPy implementation of jax.lax.fori_loop."""
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
        """NumPy implementation of jax.lax.while_loop."""
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
        NumPy implementation of jax.lax.scan.

        Notes:
          - Supports xs as a pytree (dict/list/tuple nesting) of arrays with a leading axis.
          - If xs is None, you must provide `length`, and x passed to f is None each step.
          - `unroll` is accepted for API parity; it does not change semantics here.
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
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    def index_add(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
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
        return self.np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
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

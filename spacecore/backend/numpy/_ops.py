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
        self.family = BackendFamily.NUMPY
        self._reshape_supports_copy = "copy" in inspect.signature(self.np.reshape).parameters

    def sanitize_dtype(self, dtype: DType | None) -> DType | None:
        """Normalize dtype to a NumPy dtype object. See: numpy.dtype."""

        if dtype is None:
            return None
        return self.np.dtype(dtype)

    def get_dtype(self, x: Any) -> DType:
        if self.is_dense(x):
            return x.dtype
        elif self.is_sparse(x):
            return x.dtype
        else:
            raise TypeError(f'Expected Numpy ndarray or SciPy sparse array, got {type(x)}.')

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

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """ See: scipy.special.logsumexp. """
        sp = self._sp
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
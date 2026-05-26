from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Any, Sequence, Tuple, Callable, Optional, Type, ClassVar

from ..types import DenseArray, SparseArray, DType, ArrayLike, Index, T, X, Y, R, Carry


class LazyNamespace:
    def __init__(self, module_name: str) -> None:
        self.__name__ = module_name
        self.__isabstractmethod__ = False
        self._module_name = module_name
        self._module: Any | None = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    @property
    def is_loaded(self) -> bool:
        return self._module is not None


class BackendOps(ABC):
    """
    Public numerical contract for SpaceCore backends.

    Common dense-array operations delegate to the backend's Array API-compatible
    ``xp`` namespace. Subclasses provide backend-specific sparse conversion,
    dtype policy, indexing mutation, control flow, device/autograd behavior, and
    operations not covered by the Array API.
    """

    _family: ClassVar[str]
    _allow_sparse: ClassVar[bool]
    xp: ClassVar[Any]

    def __init__(self) -> None:
        self._constant_cache: dict[str, DenseArray] = {}

    @property
    def family(self) -> str:
        """Backend family identifier."""
        return type(self)._family

    @property
    def allow_sparse(self) -> bool:
        """Whether this backend supports sparse arrays."""
        return self._allow_sparse

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BackendOps):
            return self.family == other.family
        return False

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.family))

    @property
    @abstractmethod
    def dense_array(self) -> Type[Any]:
        """Dense array type accepted by this backend."""
        ...

    @property
    @abstractmethod
    def sparse_array(self) -> Tuple[Type[Any], ...] | None:
        """Sparse array types accepted by this backend, or None."""
        ...

    @abstractmethod
    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """Normalize a dtype specifier for this backend."""
        ...

    def is_dense(self, x: Any) -> bool:
        """Return whether x is a dense array for this backend."""
        return isinstance(x, self.dense_array)

    def is_sparse(self, x: Any) -> bool:
        """Return whether x is a sparse array for this backend."""
        return self.sparse_array is not None and isinstance(x, self.sparse_array)

    def is_array(self, x: Any) -> bool:
        """Return whether x is any array for this backend."""
        return self.is_dense(x) or self.is_sparse(x)

    @abstractmethod
    def assparse(self, x: Any, dtype: DType | None = None) -> SparseArray:
        """Convert input to a backend sparse array."""
        ...

    @abstractmethod
    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """Multiply a sparse array by a dense array."""
        ...

    @abstractmethod
    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """Compute a stable log-sum-exp reduction."""
        ...

    @abstractmethod
    def index_set(
            self,
            x: DenseArray,
            index: Index,
            values: ArrayLike,
            *,
            copy: bool = True,
    ) -> DenseArray:
        """Set indexed values using backend mutation semantics."""

    @abstractmethod
    def index_add(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
        """Add values into indexed positions using backend mutation semantics."""
        ...

    @abstractmethod
    def ix_(self, *args: Any) -> Any:
        """Build open-mesh index arrays."""
        ...

    @abstractmethod
    def fori_loop(
            self,
            lower: int,
            upper: int,
            body_fun: Callable[[int, T], T],
            init_val: T,
    ) -> T:
        """Run a counted loop primitive."""

    @abstractmethod
    def while_loop(
            self,
            cond_fun: Callable[[T], bool],
            body_fun: Callable[[T], T],
            init_val: T,
    ) -> T:
        """Run a while-loop primitive."""

    @abstractmethod
    def scan(
            self,
            f: Callable[[Carry, X], Tuple[Carry, Y]],
            init: Carry,
            xs: X,
            length: Optional[int] = None,
            reverse: bool = False,
            unroll: int = 1,
    ) -> Tuple[Carry, Y]:
        """Run a scan primitive."""

    @abstractmethod
    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        """Run backend-compatible conditional branch selection."""
        ...

    @abstractmethod
    def allclose_sparse(
            self,
            a: SparseArray,
            b: SparseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
    ) -> bool:
        """Compare sparse arrays elementwise within tolerances."""
        ...

    def _dtype_arg(self, dtype: DType | None) -> DType | None:
        return None if dtype is None else self.sanitize_dtype(dtype)

    def _to_axis_tuple(self, axis: int | Sequence[int] | None) -> int | tuple[int, ...] | None:
        if axis is None or isinstance(axis, int):
            return axis
        return tuple(axis)

    def _permute_dims(self, x: DenseArray, axes: Sequence[int]) -> DenseArray:
        axes = tuple(axes)
        if hasattr(self.xp, "permute_dims"):
            return self.xp.permute_dims(x, axes)
        if hasattr(self.xp, "permute"):
            return self.xp.permute(x, axes)
        return self.xp.transpose(x, axes=axes)

    def _move_axis_order(
        self,
        ndim: int,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> tuple[int, ...]:
        src = (source,) if isinstance(source, int) else tuple(source)
        dst = (destination,) if isinstance(destination, int) else tuple(destination)
        src = tuple(axis + ndim if axis < 0 else axis for axis in src)
        dst = tuple(axis + ndim if axis < 0 else axis for axis in dst)
        order = [axis for axis in range(ndim) if axis not in src]
        for dest, axis in sorted(zip(dst, src, strict=True)):
            order.insert(dest, axis)
        return tuple(order)

    @property
    def inf(self) -> DenseArray:
        """Positive infinity as a cached backend scalar."""
        return self._constant("inf", float("inf"))

    @property
    def nan(self) -> DenseArray:
        """NaN as a cached backend scalar."""
        return self._constant("nan", float("nan"))

    @property
    def pi(self) -> DenseArray:
        """Pi as a cached backend scalar."""
        return self._constant("pi", 3.141592653589793)

    @property
    def e(self) -> DenseArray:
        """Euler's number as a cached backend scalar."""
        return self._constant("e", 2.718281828459045)

    def _constant(self, name: str, value: float) -> DenseArray:
        if name not in self._constant_cache:
            self._constant_cache[name] = self.asarray(value)
        return self._constant_cache[name]

    def eps(self, dtype: DType) -> float:
        """Machine epsilon for dtype."""
        return float(self.xp.finfo(self.sanitize_dtype(dtype)).eps)

    def is_complex_dtype(self, dtype: DType) -> bool:
        """
        Return whether ``dtype`` is a complex floating type.

        Parameters
        ----------
        dtype:
            Backend or portable dtype specifier to inspect.

        Returns
        -------
        bool
            ``True`` when ``dtype`` represents complex floating values.
        """
        dtype = self.sanitize_dtype(dtype)
        return getattr(dtype, "kind", None) == "c" or str(dtype).startswith("torch.complex")

    def real_dtype(self, dtype: DType) -> DType:
        """
        Return the real floating dtype with the same precision as ``dtype``.

        Parameters
        ----------
        dtype:
            Backend or portable dtype specifier.

        Returns
        -------
        DType
            ``dtype`` itself when it is already real-valued; otherwise
            ``float32`` for complex64 and ``float64`` for complex128.
        """
        dtype = self.sanitize_dtype(dtype)
        if not self.is_complex_dtype(dtype):
            return dtype
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is None:
            dtype_text = str(dtype)
            if "complex64" in dtype_text:
                return self.sanitize_dtype("float32")
            return self.sanitize_dtype("float64")
        return self.sanitize_dtype("float32" if itemsize <= 8 else "float64")

    def get_dtype(self, x: Any) -> DType:
        """Return x.dtype after verifying x is a backend array."""
        if self.is_array(x):
            return x.dtype
        raise TypeError(f"Expected {self.family} array, got {type(x)}.")

    def shape(self, x: Any) -> tuple[int, ...]:
        """Return x.shape as a tuple."""
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        """Return the number of dimensions of x."""
        return int(x.ndim)

    def size(self, x: Any) -> int:
        """Return the total number of elements in x."""
        result = 1
        for dim in self.shape(x):
            result *= int(dim)
        return result

    def asarray(self, x: Any, dtype: DType | None = None, **backend_kwargs: Any) -> DenseArray:
        """Convert input to a dense backend array (delegates to xp.asarray)."""
        if self.is_sparse(x) and hasattr(x, "to_dense"):
            x = x.to_dense()
        dtype = self._dtype_arg(dtype)
        if hasattr(self.xp, "asarray"):
            return self.xp.asarray(x, dtype=dtype, **backend_kwargs)
        return self.xp.as_tensor(x, dtype=dtype, **backend_kwargs)

    def astype(self, x: DenseArray, dtype: DType | None, **backend_kwargs: Any) -> DenseArray:
        """Cast x to dtype, returning x unchanged when dtype is None."""
        if dtype is None:
            return x
        dtype = self.sanitize_dtype(dtype)
        if hasattr(x, "astype"):
            return x.astype(dtype, **backend_kwargs)
        return x.to(dtype=dtype, **backend_kwargs)

    def empty(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        """Create an uninitialized array (delegates to xp.empty)."""
        return self.xp.empty(shape, dtype=self._dtype_arg(dtype))

    def zeros(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        """Create a zero-filled array (delegates to xp.zeros)."""
        return self.xp.zeros(shape, dtype=self._dtype_arg(dtype))

    def ones(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        """Create a one-filled array (delegates to xp.ones)."""
        return self.xp.ones(shape, dtype=self._dtype_arg(dtype))

    def zeros_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        """Create a zero-filled array like x (delegates to xp.zeros_like)."""
        return self.xp.zeros_like(x, dtype=self._dtype_arg(dtype))

    def ones_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        """Create a one-filled array like x (delegates to xp.ones_like)."""
        return self.xp.ones_like(x, dtype=self._dtype_arg(dtype))

    def full_like(self, x: DenseArray, value: Any, dtype: DType | None = None) -> DenseArray:
        """Create a value-filled array like x (delegates to xp.full_like)."""
        return self.xp.full_like(x, value, dtype=self._dtype_arg(dtype))

    def arange(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
        dtype: DType | None = None,
    ) -> DenseArray:
        """Create an evenly spaced range (delegates to xp.arange)."""
        dtype = self._dtype_arg(dtype)
        if stop is None:
            return self.xp.arange(start, dtype=dtype)
        if step is None:
            return self.xp.arange(start, stop, dtype=dtype)
        return self.xp.arange(start, stop, step, dtype=dtype)

    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None) -> DenseArray:
        """Create a value-filled array (delegates to xp.full)."""
        return self.xp.full(shape, fill_value, dtype=self._dtype_arg(dtype))

    def eye(self, n: int, m: int | None = None, dtype: DType | None = None) -> DenseArray:
        """Create an identity-like matrix (delegates to xp.eye)."""
        return self.xp.eye(n, m, dtype=self._dtype_arg(dtype))

    def ravel(self, x: DenseArray) -> DenseArray:
        """Flatten x to one dimension."""
        if hasattr(self.xp, "ravel"):
            return self.xp.ravel(x)
        return self.reshape(x, (-1,))

    def reshape(self, x: DenseArray, shape: Tuple[int, ...] | int) -> DenseArray:
        """Reshape x (delegates to xp.reshape)."""
        shape_arg = (shape,) if isinstance(shape, int) else shape
        return self.xp.reshape(x, shape_arg)

    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        """Permute dimensions of x."""
        if axes is None:
            axes = tuple(reversed(range(self.ndim(x))))
        return self._permute_dims(x, axes)

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """Swap two axes of x."""
        if hasattr(self.xp, "swapaxes"):
            return self.xp.swapaxes(x, axis1, axis2)
        axes = list(range(self.ndim(x)))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self._permute_dims(x, axes)

    def broadcast_to(self, x: DenseArray, shape: Tuple[int, ...]) -> DenseArray:
        """Broadcast x to shape (delegates to xp.broadcast_to)."""
        return self.xp.broadcast_to(x, shape)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """Insert singleton dimensions into x."""
        if isinstance(axis, int):
            if hasattr(self.xp, "expand_dims"):
                return self.xp.expand_dims(x, axis=axis)
            return self.xp.unsqueeze(x, axis)
        out = x
        ndim = self.ndim(x) + len(axis)
        for ax in sorted(a + ndim if a < 0 else a for a in axis):
            out = self.expand_dims(out, ax)
        return out

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """Remove singleton dimensions from x."""
        if axis is None:
            axis = tuple(i for i, dim in enumerate(self.shape(x)) if dim == 1)
            if not axis:
                return x
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return self.xp.squeeze(x, axis=axis)

    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """Move axes of x to new positions."""
        if hasattr(self.xp, "moveaxis"):
            return self.xp.moveaxis(x, source, destination)
        return self._permute_dims(x, self._move_axis_order(self.ndim(x), source, destination))

    def stack(self, arrays: Sequence[DenseArray], axis: int = 0) -> DenseArray:
        """Stack arrays along a new axis (delegates to xp.stack)."""
        return self.xp.stack(tuple(arrays), axis=axis)

    def vmap(
        self,
        fn: Callable,
        in_axes: int | Sequence[int | None] | None = 0,
        out_axes: int | Sequence[int | None] | None = 0,
    ) -> Callable:
        """Vectorize ``fn`` over array axes using a Python-loop fallback."""

        def axis_for_arg(i: int) -> int | Sequence[int | None] | None:
            if isinstance(in_axes, tuple) or isinstance(in_axes, list):
                return in_axes[i]
            return in_axes

        def normalize_axis(axis: int, ndim: int) -> int:
            return axis + ndim if axis < 0 else axis

        def tree_size(x: Any, axis: Any) -> int | None:
            if axis is None:
                return None
            if isinstance(x, tuple):
                axes = axis if isinstance(axis, (tuple, list)) else (axis,) * len(x)
                for xi, ai in zip(x, axes):
                    size = tree_size(xi, ai)
                    if size is not None:
                        return size
                return None
            shape = tuple(getattr(x, "shape", ()))
            axis = normalize_axis(int(axis), len(shape))
            return int(shape[axis])

        def tree_take(x: Any, axis: Any, i: int) -> Any:
            if axis is None:
                return x
            if isinstance(x, tuple):
                axes = axis if isinstance(axis, (tuple, list)) else (axis,) * len(x)
                return tuple(tree_take(xi, ai, i) for xi, ai in zip(x, axes))
            shape = tuple(getattr(x, "shape", ()))
            axis = normalize_axis(int(axis), len(shape))
            index = [slice(None)] * len(shape)
            index[axis] = i
            return x[tuple(index)]

        def tree_stack(xs: Sequence[Any], axis: Any) -> Any:
            first = xs[0]
            if isinstance(first, tuple):
                axes = axis if isinstance(axis, (tuple, list)) else (axis,) * len(first)
                return tuple(
                    tree_stack(tuple(x[i] for x in xs), ai)
                    for i, ai in enumerate(axes)
                )
            if axis is None:
                return first
            return self.stack(xs, axis=int(axis))

        def mapped(*args: Any) -> Any:
            axes = tuple(axis_for_arg(i) for i in range(len(args)))
            size = None
            for arg, axis in zip(args, axes):
                size = tree_size(arg, axis)
                if size is not None:
                    break
            if size is None:
                return fn(*args)
            outputs = tuple(
                fn(*(tree_take(arg, axis, i) for arg, axis in zip(args, axes)))
                for i in range(size)
            )
            return tree_stack(outputs, out_axes)

        return mapped

    def conj(self, x: DenseArray) -> DenseArray:
        """Complex conjugate of x (delegates to xp.conj)."""
        return self.xp.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        """Real component of x (delegates to xp.real)."""
        return self.xp.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """Imaginary component of x (delegates to xp.imag)."""
        return self.xp.imag(x)

    def abs(self, x: DenseArray) -> DenseArray:
        """Absolute value of x (delegates to xp.abs)."""
        return self.xp.abs(x)

    def sign(self, x: DenseArray) -> DenseArray:
        """Elementwise sign of x (delegates to xp.sign)."""
        return self.xp.sign(x)

    def sqrt(self, x: DenseArray) -> DenseArray:
        """Elementwise square root of x (delegates to xp.sqrt)."""
        return self.xp.sqrt(x)

    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        """Sum over given axes (delegates to xp.sum)."""
        return self.xp.sum(
            x,
            axis=self._to_axis_tuple(axis),
            dtype=self._dtype_arg(dtype),
            keepdims=keepdims,
        )

    def mean(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """Mean over given axes (delegates to xp.mean)."""
        return self.xp.mean(x, axis=self._to_axis_tuple(axis), keepdims=keepdims)

    def min(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """Minimum over given axes (delegates to xp.min)."""
        return self.xp.min(x, axis=self._to_axis_tuple(axis), keepdims=keepdims)

    def max(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """Maximum over given axes (delegates to xp.max)."""
        return self.xp.max(x, axis=self._to_axis_tuple(axis), keepdims=keepdims)

    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        """Product over given axes (delegates to xp.prod)."""
        return self.xp.prod(
            x,
            axis=self._to_axis_tuple(axis),
            dtype=self._dtype_arg(dtype),
            keepdims=keepdims,
        )

    def trace(self, x: DenseArray) -> DenseArray:
        """Trace of a matrix (delegates to xp.trace when available)."""
        if hasattr(self.xp, "trace"):
            return self.xp.trace(x)
        return self.sum(self.diagonal(x))

    def argsort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        """Indices that sort x (delegates to xp.argsort)."""
        return self.xp.argsort(x, axis=axis)

    def sort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        """Sort x along an axis (delegates to xp.sort)."""
        return self.xp.sort(x, axis=axis)

    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """Indices of minima (delegates to xp.argmin)."""
        return self.xp.argmin(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """Indices of maxima (delegates to xp.argmax)."""
        return self.xp.argmax(x, axis=axis, keepdims=keepdims)

    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """
        Returns sum(conj(x) * y). Matches numpy/jax/torch vdot and Array API
        vecdot. DenseLinOp.rapply relies on this for complex inputs.
        """
        x_flat = self.ravel(x)
        y_flat = self.ravel(y)
        if hasattr(self.xp, "vdot"):
            return self.xp.vdot(x_flat, y_flat)
        return self.xp.vecdot(x_flat, y_flat)

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        """Matrix product (delegates to xp.matmul)."""
        return self.xp.matmul(a, b, **({} if backend_kwargs is None else backend_kwargs))

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """Kronecker product (delegates to xp.kron)."""
        return self.xp.kron(a, b)

    def einsum(self, subscripts: str, *operands: DenseArray) -> DenseArray:
        """Einstein summation (delegates to xp.einsum)."""
        return self.xp.einsum(subscripts, *operands)

    def eigh(
        self,
        x: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DenseArray, DenseArray]:
        """Eigenpairs of a Hermitian dense matrix (delegates to xp.linalg.eigh)."""
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        return self.xp.linalg.eigh(x, **({} if backend_kwargs is None else backend_kwargs))

    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """Vector or matrix norm (delegates to xp.linalg.norm)."""
        return self.xp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(
        self,
        A: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        """Solve a dense linear system (delegates to xp.linalg.solve)."""
        return self.xp.linalg.solve(A, b, **({} if backend_kwargs is None else backend_kwargs))

    def eigvalsh(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        """Eigenvalues of a Hermitian dense matrix (delegates to xp.linalg.eigvalsh)."""
        return self.xp.linalg.eigvalsh(A, **({} if backend_kwargs is None else backend_kwargs))

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DenseArray, DenseArray, DenseArray]:
        """Singular value decomposition (delegates to xp.linalg.svd)."""
        return self.xp.linalg.svd(
            A,
            full_matrices=full_matrices,
            **({} if backend_kwargs is None else backend_kwargs),
        )

    def cholesky(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        """Cholesky factorization (delegates to xp.linalg.cholesky)."""
        return self.xp.linalg.cholesky(A, **({} if backend_kwargs is None else backend_kwargs))

    def exp(self, x: DenseArray) -> DenseArray:
        """Elementwise exponential (delegates to xp.exp)."""
        return self.xp.exp(x)

    def log(self, x: DenseArray) -> DenseArray:
        """Elementwise natural logarithm (delegates to xp.log)."""
        return self.xp.log(x)

    def where(self, condition: DenseArray | bool, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """Select between x and y by condition (delegates to xp.where)."""
        return self.xp.where(condition, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """Elementwise maximum (delegates to xp.maximum)."""
        return self.xp.maximum(x, y)

    def minimum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """Elementwise minimum (delegates to xp.minimum)."""
        return self.xp.minimum(x, y)

    def clip(self, x: DenseArray, a_min: ArrayLike, a_max: ArrayLike) -> DenseArray:
        """Clip x into [a_min, a_max] (delegates to xp.clip)."""
        return self.xp.clip(x, a_min, a_max)

    def isfinite(self, x: DenseArray) -> DenseArray:
        """Elementwise finite check (delegates to xp.isfinite)."""
        return self.xp.isfinite(x)

    def isnan(self, x: DenseArray) -> DenseArray:
        """Elementwise NaN check (delegates to xp.isnan)."""
        return self.xp.isnan(x)

    def concatenate(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        dtype: DType | None = None,
    ) -> DenseArray:
        """Concatenate arrays along an existing axis (delegates to xp.concat)."""
        if hasattr(self.xp, "concat"):
            result = self.xp.concat(tuple(arrays), axis=axis)
        else:
            result = self.xp.concatenate(tuple(arrays), axis=axis)
        return self.astype(result, dtype) if dtype is not None else result

    def take(
        self,
        x: DenseArray,
        indices: DenseArray,
        axis: int | None = None,
    ) -> DenseArray:
        """Take entries from x by integer indices (delegates to xp.take)."""
        return self.xp.take(x, indices, axis=axis)

    def diag(self, x: DenseArray) -> DenseArray:
        """Extract or construct a diagonal (delegates to xp.diag)."""
        return self.xp.diag(x)

    def diagonal(self, x: DenseArray) -> DenseArray:
        """Return the main diagonal of x (delegates to xp.diagonal)."""
        return self.xp.diagonal(x)

    def tril(self, x: DenseArray) -> DenseArray:
        """Lower triangle of x (delegates to xp.tril)."""
        return self.xp.tril(x)

    def triu(self, x: DenseArray) -> DenseArray:
        """Upper triangle of x (delegates to xp.triu)."""
        return self.xp.triu(x)

    def allclose(
            self,
            a: DenseArray,
            b: DenseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            equal_nan: bool = False,
    ) -> bool:
        """Return whether dense arrays are close within tolerances."""
        return bool(self.xp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    def __repr__(self) -> str:
        xp = type(self).xp
        xp_state = ""
        if isinstance(xp, LazyNamespace):
            xp_state = f", xp_loaded={xp.is_loaded!r}"
        return f"{type(self).__name__}(family={self.family!r}{xp_state})"

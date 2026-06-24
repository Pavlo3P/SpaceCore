from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Any, Sequence, Tuple, Callable, Optional, Type, ClassVar, cast

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

    @property
    def has_native_vmap(self) -> bool:
        """Whether ``vmap`` is implemented by the backend rather than a Python loop."""
        return False

    def free_memory_bytes(self) -> int | None:
        """Return currently free device memory in bytes, or ``None`` if unknown.

        The kernel dispatcher (ADR-016) uses this to gate *materializing* fast
        paths — those that allocate more than ``O(1)`` extra memory — against a
        memory budget before selecting them. The base implementation returns
        ``None`` (unknown): a backend that can cheaply query free memory (e.g.
        a GPU runtime) overrides this. When the budget is unknown the dispatcher
        treats any cost-carrying spec as unaffordable, so reporting ``None`` is
        always safe and never routes to a materializing kernel.
        """
        return None

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
    def assparse(self, x: Any, *, dtype: DType | None = None) -> SparseArray:
        """Convert input to a backend sparse array."""
        ...

    @abstractmethod
    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """Multiply a sparse array by a dense array."""
        ...

    @abstractmethod
    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
    ) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """Compute a stable log-sum-exp reduction."""
        ...

    def _copy(self, x: DenseArray) -> DenseArray:
        """Return a mutable copy of ``x``.

        Mutable backends override this with their native copy (``x.copy()`` for
        NumPy/CuPy, ``x.clone()`` for PyTorch). Immutable backends such as JAX
        override :meth:`index_set` / :meth:`index_add` directly and never call
        this helper.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _copy.")

    def _scatter_add_inplace(self, y: DenseArray, index: Index, values: ArrayLike) -> None:
        """Accumulate ``values`` into ``y`` at ``index`` in place.

        Backend mutation primitive used by the default :meth:`index_add`.
        Repeated-index accumulation is backend-specific (NumPy/CuPy use
        ``add.at`` and accumulate duplicate indices; PyTorch does not).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _scatter_add_inplace."
        )

    def index_set(
        self,
        x: DenseArray,
        index: Index,
        values: ArrayLike,
        *,
        copy: bool = True,
    ) -> DenseArray:
        """Return ``x`` with ``x[index]`` set to ``values``.

        With ``copy=True`` a mutable copy is updated and returned; with
        ``copy=False`` ``x`` is mutated in place. Immutable backends override
        this method.
        """
        y = self._copy(x) if copy else x
        y[index] = values
        return y

    def index_add(
        self,
        x: DenseArray,
        index: Index,
        values: DenseArray,
        *,
        copy: bool = True,
    ) -> DenseArray:
        """Return ``x`` with ``values`` accumulated into ``x[index]``.

        With ``copy=True`` a mutable copy is updated and returned; with
        ``copy=False`` ``x`` is mutated in place. Immutable backends override
        this method.
        """
        y = self._copy(x) if copy else x
        self._scatter_add_inplace(y, index, values)
        return y

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

    def _require_two_sparse(self, a: Any, b: Any, *, noun: str = "sparse arrays") -> None:
        """Raise ``TypeError`` unless both ``a`` and ``b`` are sparse for this backend.

        Shared guard for :meth:`allclose_sparse`. ``noun`` names the expected
        operands in the error message (e.g. ``"sparse tensors"``).
        """
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError(f"allclose_sparse expects two {noun}.")

    def _dtype_arg(self, dtype: DType | None) -> DType | None:
        return None if dtype is None else self.sanitize_dtype(dtype)

    def _source_is_complex(self, x: Any) -> bool:
        """Return whether ``x`` carries a complex representation."""
        dtype = getattr(x, "dtype", None)
        if dtype is not None:
            return getattr(dtype, "kind", None) == "c" or "complex" in str(dtype)
        if isinstance(x, complex):
            return True
        data = getattr(x, "data", None)
        if data is not None and data is not x:
            data_dtype = getattr(data, "dtype", None)
            if data_dtype is not None:
                return getattr(data_dtype, "kind", None) == "c" or "complex" in str(data_dtype)
        if isinstance(x, (list, tuple)):
            return any(self._source_is_complex(value) for value in x)
        return False

    def _reject_complex_to_real(
        self,
        x: Any,
        dtype: DType | None,
        *,
        operation: str,
    ) -> None:
        """Reject implicit loss of a complex representation during conversion."""
        if dtype is None:
            return
        target_dtype = self.sanitize_dtype(dtype)
        if not self.is_complex_dtype(target_dtype) and self._source_is_complex(x):
            raise TypeError(
                f"{operation} rejected complex-valued input for non-complex dtype "
                f"{target_dtype}. Explicitly discard the imaginary part first, for "
                "example with `x.real` or a backend real-part operation, then convert."
            )

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
        self._reject_complex_to_real(x, dtype, operation="asarray")
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
        self._reject_complex_to_real(x, dtype, operation="astype")
        dtype = self.sanitize_dtype(dtype)
        if hasattr(x, "astype"):
            return cast(Any, x).astype(dtype, **backend_kwargs)
        return cast(Any, x).to(dtype=dtype, **backend_kwargs)

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

    def full(
        self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None
    ) -> DenseArray:
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

    def hstack(self, arrays: Sequence[DenseArray]) -> DenseArray:
        """Stack arrays horizontally / column-wise (delegates to xp.hstack).

        Concatenates along axis 0 for 1-D inputs and along axis 1 otherwise,
        matching NumPy ``hstack`` semantics.
        """
        return self.xp.hstack(tuple(arrays))

    def vstack(self, arrays: Sequence[DenseArray]) -> DenseArray:
        """Stack arrays vertically / row-wise (delegates to xp.vstack).

        Inputs are promoted to at least 2-D and concatenated along axis 0,
        matching NumPy ``vstack`` semantics.
        """
        return self.xp.vstack(tuple(arrays))

    def dstack(self, arrays: Sequence[DenseArray]) -> DenseArray:
        """Stack arrays depth-wise along the third axis (delegates to xp.dstack).

        Inputs are promoted to at least 3-D and concatenated along axis 2,
        matching NumPy ``dstack`` semantics.
        """
        return self.xp.dstack(tuple(arrays))

    def column_stack(self, arrays: Sequence[DenseArray]) -> DenseArray:
        """Stack 1-D arrays as columns into a 2-D array (delegates to xp.column_stack).

        1-D inputs become columns; higher-dimensional inputs are concatenated
        along axis 1, matching NumPy ``column_stack`` semantics.
        """
        return self.xp.column_stack(tuple(arrays))

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
            return int(cast(Any, shape[axis]))

        def tree_take(x: Any, axis: Any, i: int) -> Any:
            if axis is None:
                return x
            if isinstance(x, tuple):
                axes = axis if isinstance(axis, (tuple, list)) else (axis,) * len(x)
                return tuple(tree_take(xi, ai, i) for xi, ai in zip(x, axes))
            shape = tuple(getattr(x, "shape", ()))
            axis = normalize_axis(int(axis), len(shape))
            index: list[Any] = [slice(None)] * len(shape)
            index[axis] = i
            return x[tuple(index)]

        def tree_stack(xs: Sequence[Any], axis: Any) -> Any:
            first = xs[0]
            if isinstance(first, tuple):
                axes = axis if isinstance(axis, (tuple, list)) else (axis,) * len(first)
                return tuple(tree_stack(tuple(x[i] for x in xs), ai) for i, ai in enumerate(axes))
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
                fn(*(tree_take(arg, axis, i) for arg, axis in zip(args, axes))) for i in range(size)
            )
            return tree_stack(outputs, out_axes)

        return mapped

    def vectorize(
        self,
        pyfunc: Callable,
        *,
        excluded: Sequence[int] | None = None,
        signature: str | None = None,
    ) -> Callable:
        """Vectorize a scalar Python function over its array arguments.

        Returns a callable that applies ``pyfunc`` elementwise, broadcasting
        the array arguments against one another and preserving the broadcast
        shape. Mirrors :func:`numpy.vectorize`.

        Parameters
        ----------
        pyfunc:
            Function called on scalar elements of the (broadcast) inputs.
        excluded:
            Positional argument indices passed through to ``pyfunc`` unchanged
            instead of being vectorized.
        signature:
            Generalized-ufunc signature (e.g. ``"(n),(n)->()"``). Supported only
            on backends that provide a native ``vectorize``.

        Notes
        -----
        Delegates to the backend's native ``vectorize`` (NumPy, JAX, CuPy) when
        available; otherwise applies a portable Python-loop fallback. The
        fallback does not support ``signature``.
        """
        if hasattr(self.xp, "vectorize"):
            kwargs: dict[str, Any] = {}
            if excluded is not None:
                kwargs["excluded"] = excluded
            if signature is not None:
                kwargs["signature"] = signature
            return self.xp.vectorize(pyfunc, **kwargs)
        return self._vectorize_loop(pyfunc, excluded=excluded, signature=signature)

    def _vectorize_loop(
        self,
        pyfunc: Callable,
        *,
        excluded: Sequence[int] | None = None,
        signature: str | None = None,
    ) -> Callable:
        """Portable ``vectorize`` fallback for backends without a native one."""
        if signature is not None:
            raise NotImplementedError(
                "The vectorize fallback does not support gufunc signatures; use a "
                "backend that provides a native vectorize (NumPy, JAX, CuPy)."
            )
        excluded_set = set() if excluded is None else set(excluded)

        def vectorized(*args: Any) -> Any:
            positions = [i for i in range(len(args)) if i not in excluded_set]
            if not positions:
                return self.asarray(pyfunc(*args))
            mapped = {i: self.asarray(args[i]) for i in positions}
            out_shape = self._broadcast_shapes(*(self.shape(mapped[i]) for i in positions))
            flat = {i: self.ravel(self.broadcast_to(mapped[i], out_shape)) for i in positions}
            count = 1
            for dim in out_shape:
                count *= int(dim)
            outputs = []
            for k in range(count):
                call_args = list(args)
                for i in positions:
                    call_args[i] = flat[i][k]
                outputs.append(self.asarray(pyfunc(*call_args)))
            stacked = self.stack(outputs, axis=0)
            return self.reshape(stacked, out_shape)

        return vectorized

    @staticmethod
    def _broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the NumPy broadcast of several shapes (pure-Python)."""
        ndim = max((len(shape) for shape in shapes), default=0)
        result = [1] * ndim
        for shape in shapes:
            offset = ndim - len(shape)
            for axis, dim in enumerate(shape):
                pos = offset + axis
                if dim == 1 or dim == result[pos]:
                    continue
                if result[pos] == 1:
                    result[pos] = dim
                else:
                    raise ValueError(f"shapes {shapes} are not broadcast-compatible")
        return tuple(result)

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
        """Return indices that sort ``x`` along an axis."""
        return self.xp.argsort(x, axis=axis)

    def sort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        """Sort x along an axis (delegates to xp.sort)."""
        return self.xp.sort(x, axis=axis)

    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """Return indices of minima along an axis."""
        return self.xp.argmin(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """Return indices of maxima along an axis."""
        return self.xp.argmax(x, axis=axis, keepdims=keepdims)

    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """Return ``sum(conj(x) * y)`` over flattened inputs.

        Matches NumPy, JAX, and Torch ``vdot`` semantics. ``DenseLinOp.rapply``
        relies on this convention for complex inputs.
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

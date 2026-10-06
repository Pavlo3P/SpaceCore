from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import math
from typing import Any, Sequence, Tuple, Callable, Optional, Type, ClassVar, Generic, cast

from ..types._array import DenseT

from ..types import DenseArray, SparseArray, DType, ArrayLike, Index, T, X, Y, R, Carry


class LazyNamespace:
    def __init__(self, module_name: str) -> None:
        # Set eagerly, so ABC checks and introspection do not trigger the import.
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


class BackendOps(ABC, Generic[DenseT]):
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
        self._constant_cache: dict[str, DenseT] = {}

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

    @classmethod
    def install_pytree_protocol(cls) -> None:
        """
        Register SpaceCore containers with this backend's pytree machinery; a no-op by default.

        Backends with transforms (JAX, torch) override this and import their modules inside
        the method, so a missing dependency cannot break ``import spacecore``.
        """
        return

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BackendOps) and self.family == other.family

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
    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseT:
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
    ) -> DenseT | Tuple[DenseT, DenseT]:
        """Compute a stable log-sum-exp reduction."""
        ...

    def _copy(self, x: DenseArray) -> DenseT:
        """Return a mutable copy of ``x``.

        Mutable backends override this with their native copy (``x.copy()`` for
        NumPy/CuPy, ``x.clone()`` for PyTorch). Immutable backends such as JAX
        override :meth:`index_set` / :meth:`index_add` directly and never call
        this helper.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _copy.")

    def _scatter_add_inplace(self, y: DenseArray, index: Index, values: ArrayLike | float | complex) -> None:
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
        values: ArrayLike | float | complex,
        *,
        copy: bool = True,
    ) -> DenseT:
        """Return ``x`` with ``x[index]`` set to ``values``.

        With ``copy=True`` a mutable copy is updated and returned; with
        ``copy=False`` ``x`` is mutated in place. Immutable backends override
        this method.
        """
        y = self._copy(x) if copy else x
        y[index] = values
        return cast(DenseT, y)

    def index_add(
        self,
        x: DenseArray,
        index: Index,
        values: DenseArray,
        *,
        copy: bool = True,
    ) -> DenseT:
        """Return ``x`` with ``values`` accumulated into ``x[index]``.

        With ``copy=True`` a mutable copy is updated and returned; with
        ``copy=False`` ``x`` is mutated in place. Immutable backends override
        this method.
        """
        y = self._copy(x) if copy else x
        self._scatter_add_inplace(y, index, values)
        return cast(DenseT, y)

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
        # Arrays report a dtype; else check Python complex, a sparse .data, then sequences.
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
        # Complex to real would silently drop the imaginary part.
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

    def _as_array(self, x: Any) -> DenseT:
        """Return ``x`` as a backend array, promoting a host scalar to 0-d.

        SpaceCore's convention is that every array-valued method returns a
        backend *array*, never a host scalar. NumPy (and CuPy) reductions
        return ``np.float64``-style scalars for which :meth:`is_array` is
        False and :meth:`get_dtype` raises; JAX and Torch already return 0-d
        arrays. Wrapping here makes the four backends agree.

        The ``is_array`` guard keeps this free on backends that never produce
        host scalars, and keeps JAX tracers (which are ``jax.Array``
        instances) untouched, so the call is safe inside ``jit``.
        """
        return x if self.is_array(x) else self.xp.asarray(x)

    def _take_along_axis(self, x: DenseArray, indices: DenseArray, axis: int) -> DenseT:
        """Gather along ``axis`` using per-position ``indices``.

        Used by the spectral-ordering normalization. Torch spells this
        ``take_along_dim`` and overrides this method.
        """
        return self.xp.take_along_axis(x, indices, axis=axis)

    def _permute_dims(self, x: DenseArray, axes: Sequence[int]) -> DenseT:
        axes = tuple(axes)
        # Array API name, then torch's, then NumPy's.
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
        # Normalize both axis lists to non-negative tuples.
        src = (source,) if isinstance(source, int) else tuple(source)
        dst = (destination,) if isinstance(destination, int) else tuple(destination)
        src = tuple(axis + ndim if axis < 0 else axis for axis in src)
        dst = tuple(axis + ndim if axis < 0 else axis for axis in dst)
        # Untouched axes keep their order; each moved axis is inserted at its destination.
        order = [axis for axis in range(ndim) if axis not in src]
        for dest, axis in sorted(zip(dst, src, strict=True)):
            order.insert(dest, axis)
        return tuple(order)

    @property
    def inf(self) -> DenseT:
        """Positive infinity as a cached backend scalar."""
        return self._constant("inf", float("inf"))

    @property
    def nan(self) -> DenseT:
        """NaN as a cached backend scalar."""
        return self._constant("nan", float("nan"))

    @property
    def pi(self) -> DenseT:
        """Pi as a cached backend scalar."""
        return self._constant("pi", 3.141592653589793)

    @property
    def e(self) -> DenseT:
        """Euler's number as a cached backend scalar."""
        return self._constant("e", 2.718281828459045)

    def _constant(self, name: str, value: float) -> DenseT:
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
        # complex64 pairs with float32 and complex128 with float64; use the name if there is no itemsize.
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is None:
            dtype_text = str(dtype)
            if "complex64" in dtype_text:
                return self.sanitize_dtype("float32")
            return self.sanitize_dtype("float64")
        return self.sanitize_dtype("float32" if itemsize <= 8 else "float64")

    def complex_dtype(self, dtype: DType) -> DType:
        """
        Return the complex dtype of the same precision as ``dtype`` (``float32 -> complex64``).

        Needed because ``result_type(float32, 1j)`` is ``complex128`` on NumPy but
        ``complex64`` on JAX and torch.
        """
        dtype = self.sanitize_dtype(dtype)
        if self.is_complex_dtype(dtype):
            return dtype
        # The complex dtype whose real part is this dtype; else let promotion decide.
        for candidate in (self.xp.complex64, self.xp.complex128):
            if self.real_dtype(candidate) == dtype:
                return candidate
        return self.xp.result_type(dtype, self.xp.asarray(1j))

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
        return math.prod(map(int, self.shape(x)))

    def asarray(self, x: Any, dtype: DType | None = None, **backend_kwargs: Any) -> DenseT:
        """Convert input to a dense backend array (delegates to xp.asarray)."""
        self._reject_complex_to_real(x, dtype, operation="asarray")
        # Densify sparse input, then Array API asarray or torch's as_tensor.
        if self.is_sparse(x) and hasattr(x, "to_dense"):
            x = x.to_dense()
        dtype = self._dtype_arg(dtype)
        if hasattr(self.xp, "asarray"):
            return self.xp.asarray(x, dtype=dtype, **backend_kwargs)
        return self.xp.as_tensor(x, dtype=dtype, **backend_kwargs)

    def astype(self, x: DenseArray, dtype: DType | None, **backend_kwargs: Any) -> DenseT:
        """Cast x to dtype, returning x unchanged when dtype is None."""
        if dtype is None:
            return cast(DenseT, x)
        self._reject_complex_to_real(x, dtype, operation="astype")
        dtype = self.sanitize_dtype(dtype)
        # NumPy-style astype, else torch's .to.
        if hasattr(x, "astype"):
            return cast(Any, x).astype(dtype, **backend_kwargs)
        return cast(Any, x).to(dtype=dtype, **backend_kwargs)

    def empty(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseT:
        """Create an uninitialized array (delegates to xp.empty)."""
        return self.xp.empty(shape, dtype=self._dtype_arg(dtype))

    def zeros(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseT:
        """Create a zero-filled array (delegates to xp.zeros)."""
        return self.xp.zeros(shape, dtype=self._dtype_arg(dtype))

    def ones(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseT:
        """Create a one-filled array (delegates to xp.ones)."""
        return self.xp.ones(shape, dtype=self._dtype_arg(dtype))

    def zeros_like(self, x: DenseArray, dtype: DType | None = None) -> DenseT:
        """Create a zero-filled array like x (delegates to xp.zeros_like)."""
        return self.xp.zeros_like(x, dtype=self._dtype_arg(dtype))

    def ones_like(self, x: DenseArray, dtype: DType | None = None) -> DenseT:
        """Create a one-filled array like x (delegates to xp.ones_like)."""
        return self.xp.ones_like(x, dtype=self._dtype_arg(dtype))

    def full_like(self, x: DenseArray, value: Any, dtype: DType | None = None) -> DenseT:
        """Create a value-filled array like x (delegates to xp.full_like)."""
        return self.xp.full_like(x, value, dtype=self._dtype_arg(dtype))

    def arange(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
        dtype: DType | None = None,
    ) -> DenseT:
        """Create an evenly spaced range (delegates to xp.arange)."""
        dtype = self._dtype_arg(dtype)
        # Mirror the arange(stop), arange(start, stop), arange(start, stop, step) overloads.
        if stop is None:
            return self.xp.arange(start, dtype=dtype)
        if step is None:
            return self.xp.arange(start, stop, dtype=dtype)
        return self.xp.arange(start, stop, step, dtype=dtype)

    def full(
        self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None
    ) -> DenseT:
        """Create a value-filled array (delegates to xp.full)."""
        return self.xp.full(shape, fill_value, dtype=self._dtype_arg(dtype))

    def eye(self, n: int, m: int | None = None, dtype: DType | None = None) -> DenseT:
        """Create an identity-like matrix (delegates to xp.eye)."""
        return self.xp.eye(n, m, dtype=self._dtype_arg(dtype))

    # -- Flattening order --------------------------------------------------
    #
    # SpaceCore flattens in **C order** (row-major: the last axis varies
    # fastest), on every backend, always. This is the vec convention that the
    # coordinate layer is built on -- ``CoordinateSpace.flatten``,
    # ``LinOp.to_matrix`` and the Kronecker identities all
    # assume it, and vec(A x) = (I ⊗ A) vec(x) rather than (Aᵀ ⊗ I) is exactly
    # this choice. It is a *declared* convention rather than a forwarded
    # ``order=`` argument because Torch has no Fortran-order reshape at all:
    # offering the parameter would mean offering it on three backends of four.
    #
    # The order is over *index* positions, not memory layout, so it is
    # unaffected by whether the input is C-contiguous, Fortran-contiguous, or
    # a non-contiguous view -- a transposed array flattens by its logical
    # index order, not its buffer.

    def ravel(self, x: DenseArray) -> DenseT:
        """Flatten x to one dimension in C order (last axis varies fastest)."""
        if hasattr(self.xp, "ravel"):
            return self.xp.ravel(x)
        return self.reshape(x, (-1,))

    def reshape(self, x: DenseArray, shape: Tuple[int, ...] | int) -> DenseT:
        """Reshape x in C order (delegates to xp.reshape; see the note above)."""
        shape_arg = (shape,) if isinstance(shape, int) else shape
        return self.xp.reshape(x, shape_arg)

    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseT:
        """Permute dimensions of x."""
        if axes is None:
            axes = tuple(reversed(range(self.ndim(x))))
        return self._permute_dims(x, axes)

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseT:
        """Swap two axes of x."""
        if hasattr(self.xp, "swapaxes"):
            return self.xp.swapaxes(x, axis1, axis2)
        # No native swapaxes: permute with the two axes exchanged.
        axes = list(range(self.ndim(x)))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self._permute_dims(x, axes)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseT:
        """Insert singleton dimensions into x."""
        if isinstance(axis, int):
            if hasattr(self.xp, "expand_dims"):
                return self.xp.expand_dims(x, axis=axis)
            return self.xp.unsqueeze(x, axis)
        # Several axes: insert one at a time, in increasing final position.
        out = cast(DenseT, x)
        ndim = self.ndim(x) + len(axis)
        for ax in sorted(a + ndim if a < 0 else a for a in axis):
            out = self.expand_dims(out, ax)
        return out

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseT:
        """Remove singleton dimensions from x."""
        if axis is None:
            # All size-1 axes, passed explicitly since not every backend accepts axis=None.
            axis = tuple(i for i, dim in enumerate(self.shape(x)) if dim == 1)
            if not axis:
                return cast(DenseT, x)
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return self.xp.squeeze(x, axis=axis)

    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseT:
        """Move axes of x to new positions."""
        if hasattr(self.xp, "moveaxis"):
            return self.xp.moveaxis(x, source, destination)
        return self._permute_dims(x, self._move_axis_order(self.ndim(x), source, destination))

    def stack(self, arrays: Sequence[DenseArray], axis: int = 0) -> DenseT:
        """Stack arrays along a new axis (delegates to xp.stack)."""
        return self.xp.stack(tuple(arrays), axis=axis)

    def hstack(self, arrays: Sequence[DenseArray]) -> DenseT:
        """Stack arrays horizontally / column-wise (delegates to xp.hstack).

        Concatenates along axis 0 for 1-D inputs and along axis 1 otherwise,
        matching NumPy ``hstack`` semantics.
        """
        return self.xp.hstack(tuple(arrays))

    def vstack(self, arrays: Sequence[DenseArray]) -> DenseT:
        """Stack arrays vertically / row-wise (delegates to xp.vstack).

        Inputs are promoted to at least 2-D and concatenated along axis 0,
        matching NumPy ``vstack`` semantics.
        """
        return self.xp.vstack(tuple(arrays))

    def dstack(self, arrays: Sequence[DenseArray]) -> DenseT:
        """Stack arrays depth-wise along the third axis (delegates to xp.dstack).

        Inputs are promoted to at least 3-D and concatenated along axis 2,
        matching NumPy ``dstack`` semantics.
        """
        return self.xp.dstack(tuple(arrays))

    def column_stack(self, arrays: Sequence[DenseArray]) -> DenseT:
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

        # Fallback vmap: loop over the mapped axis in Python and stack the outputs.
        # in_axes is one axis for all arguments or one per argument.
        def axis_for_arg(i: int) -> int | Sequence[int | None] | None:
            if isinstance(in_axes, tuple) or isinstance(in_axes, list):
                return in_axes[i]
            return in_axes

        def normalize_axis(axis: int, ndim: int) -> int:
            return axis + ndim if axis < 0 else axis

        # Batch size of the first mapped leaf; tuples are walked recursively.
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

        # The i-th slice of every mapped leaf.
        def tree_take(x: Any, axis: Any, i: int) -> Any:
            if axis is None:
                return cast(DenseT, x)
            if isinstance(x, tuple):
                axes = axis if isinstance(axis, (tuple, list)) else (axis,) * len(x)
                return tuple(tree_take(xi, ai, i) for xi, ai in zip(x, axes))
            shape = tuple(getattr(x, "shape", ()))
            axis = normalize_axis(int(axis), len(shape))
            index: list[Any] = [slice(None)] * len(shape)
            index[axis] = i
            return x[tuple(index)]

        # Stack the per-index outputs along out_axes, recursing into tuples.
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
            # No argument is mapped: one plain call.
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
        """
        Vectorize a scalar function over broadcast array arguments, like :func:`numpy.vectorize`.

        Uses the backend's native ``vectorize`` when present; the Python-loop fallback does
        not support ``signature``.
        """
        # Native vectorize when available; otherwise a Python loop.
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
            # Broadcast the non-excluded arguments, call pyfunc per element, reshape back.
            positions = [i for i in range(len(args)) if i not in excluded_set]
            if not positions:
                return self.asarray(pyfunc(*args))
            mapped = {i: self.asarray(args[i]) for i in positions}
            out_shape = self._broadcast_shapes(*(self.shape(mapped[i]) for i in positions))
            flat = {i: self.ravel(self.broadcast_to(mapped[i], out_shape)) for i in positions}
            count = math.prod(map(int, out_shape))
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
        # NumPy rules: align from the right; size 1 stretches, any other mismatch fails.
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

    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseT:
        """Sum over given axes (delegates to xp.sum). Returns a 0-d array when total."""
        return self._as_array(
            self.xp.sum(
                x,
                axis=self._to_axis_tuple(axis),
                dtype=self._dtype_arg(dtype),
                keepdims=keepdims,
            )
        )

    def mean(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseT:
        """Mean over given axes (delegates to xp.mean). Returns a 0-d array when total."""
        return self._as_array(self.xp.mean(x, axis=self._to_axis_tuple(axis), keepdims=keepdims))

    def min(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseT:
        """Minimum over given axes (delegates to xp.min). Returns a 0-d array when total."""
        return self._as_array(self.xp.min(x, axis=self._to_axis_tuple(axis), keepdims=keepdims))

    def max(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseT:
        """Maximum over given axes (delegates to xp.max). Returns a 0-d array when total."""
        return self._as_array(self.xp.max(x, axis=self._to_axis_tuple(axis), keepdims=keepdims))

    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseT:
        """Product over given axes (delegates to xp.prod). Returns a 0-d array when total."""
        return self._as_array(
            self.xp.prod(
                x,
                axis=self._to_axis_tuple(axis),
                dtype=self._dtype_arg(dtype),
                keepdims=keepdims,
            )
        )

    def trace(self, x: DenseArray) -> DenseT:
        """Trace over the **trailing two axes**; batched input traces each matrix.

        Deliberately *not* ``xp.trace(x)``: NumPy and JAX default to
        ``axis1=0, axis2=1``, which on a ``(batch, n, n)`` stack traces across
        the batch axis and silently returns a length-``n`` vector, while
        ``torch.trace`` rejects anything but a 2-D input. SpaceCore's batched
        convention is leading batch axes, so the trailing two are the matrix.
        """
        return self.sum(self.diagonal(x), axis=-1)

    def argsort(self, x: DenseArray, axis: int = -1) -> DenseT:
        """Return indices that sort ``x`` along an axis."""
        return self._as_array(self.xp.argsort(x, axis=axis))

    def sort(self, x: DenseArray, axis: int = -1) -> DenseT:
        """Sort x along an axis (delegates to xp.sort)."""
        return self._as_array(self.xp.sort(x, axis=axis))

    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseT:
        """Return indices of minima along an axis. Returns a 0-d array when total."""
        return self._as_array(self.xp.argmin(x, axis=axis, keepdims=keepdims))

    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseT:
        """Return indices of maxima along an axis. Returns a 0-d array when total."""
        return self._as_array(self.xp.argmax(x, axis=axis, keepdims=keepdims))

    def vdot(self, x: DenseArray, y: DenseArray) -> DenseT:
        """Return ``sum(conj(x) * y)`` over flattened inputs.

        Matches NumPy, JAX, and Torch ``vdot`` semantics. ``DenseLinOp.rapply``
        relies on this convention for complex inputs. Returns a 0-d array.
        """
        # Flattened conj(x) . y; vecdot is the Array API spelling of vdot.
        x_flat = self.ravel(x)
        y_flat = self.ravel(y)
        if hasattr(self.xp, "vdot"):
            return self._as_array(self.xp.vdot(x_flat, y_flat))
        return self._as_array(self.xp.vecdot(x_flat, y_flat))

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseT:
        """Matrix product (delegates to xp.matmul)."""
        return self._as_array(
            self.xp.matmul(a, b, **({} if backend_kwargs is None else backend_kwargs))
        )

    # -- Spectral presentation conventions ---------------------------------
    #
    # A decomposition is determined only up to a choice of ordering and, for
    # each vector, a sign (real) or unit phase (complex). Four backends are
    # free to make those choices differently -- and do: Torch returns complex
    # Hermitian eigenvectors that are -1 times NumPy's and JAX's. SpaceCore
    # therefore *declares* the presentation and normalizes to it here rather
    # than inheriting whatever the underlying library returned:
    #
    #   * eigenvalues ascending, singular values descending;
    #   * each eigenvector / singular vector scaled so that its entry of
    #     largest magnitude is real and positive.
    #
    # NOT normalized, and not normalizable: the basis of a degenerate
    # eigenspace (multiplicity > 1) is arbitrary, so callers must not depend
    # on it. Ties in "entry of largest magnitude" are broken by first
    # occurrence, which all backends agree on.
    #
    # Everything below is pure array arithmetic -- no data-dependent Python
    # branch, no host sync -- so it stays safe under jit, vmap and compile.
    # Cost is O(n^2) on top of an O(n^3) decomposition.

    def _column_phase(self, vectors: DenseArray) -> DenseT:
        """Return the unit phase of each column's largest entry, shaped ``(..., 1, k)``."""
        # Per column: the entry of largest modulus and its unit phase.
        magnitude = self.abs(vectors)
        pivot_row = self.xp.argmax(magnitude, axis=-2)
        pivot_row = self.expand_dims(pivot_row, -2)
        # Gather the pivot with a one-hot mask, which stays jittable.
        rows = self.reshape(self.arange(self.shape(vectors)[-2]), (-1, 1))
        selector = self.astype(cast(Any, rows) == pivot_row, self.get_dtype(vectors))
        pivot = self.sum(vectors * selector, axis=-2, keepdims=True)
        scale = self.abs(pivot)
        # A unit-norm vector cannot be all zeros; the guard only keeps the
        # division defined for degenerate inputs rather than changing a result.
        safe_scale = self.where(cast(Any, scale) > 0, scale, self.ones_like(scale))
        return self.where(cast(Any, scale) > 0, pivot / safe_scale, self.ones_like(pivot))

    def _gauge_columns(self, vectors: DenseArray) -> DenseT:
        """Scale each column so its largest-magnitude entry is real positive."""
        return vectors / self._column_phase(vectors)

    def _order_eigenpairs(
        self,
        eigenvalues: DenseArray,
        eigenvectors: DenseArray,
    ) -> tuple[DenseT, DenseT]:
        """Sort eigenpairs ascending by eigenvalue and gauge the eigenvectors."""
        # Ascending eigenvalues, columns permuted to match, then a fixed phase gauge.
        order = self.xp.argsort(self.real(eigenvalues), axis=-1)
        eigenvalues = self._take_along_axis(eigenvalues, order, -1)
        eigenvectors = self._take_along_axis(eigenvectors, self.expand_dims(order, -2), -1)
        return eigenvalues, self._gauge_columns(eigenvectors)

    def eigh(
        self,
        x: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DenseT, DenseT]:
        """Eigenpairs of a Hermitian dense matrix, in SpaceCore's presentation.

        Eigenvalues ascending; each eigenvector scaled so its largest-magnitude
        entry is real and positive. See the conventions note above -- this is
        deliberately not raw ``xp.linalg.eigh`` output.
        """
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        eigenvalues, eigenvectors = self.xp.linalg.eigh(
            x, **({} if backend_kwargs is None else backend_kwargs)
        )
        return self._order_eigenpairs(eigenvalues, eigenvectors)

    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseT:
        """Vector or matrix norm (delegates to xp.linalg.norm). Returns a 0-d array when total."""
        return self._as_array(self.xp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims))

    def solve(
        self,
        A: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseT:
        """Solve a dense linear system (delegates to xp.linalg.solve)."""
        return self.xp.linalg.solve(A, b, **({} if backend_kwargs is None else backend_kwargs))

    def eigvalsh(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseT:
        """Eigenvalues of a Hermitian dense matrix, ascending (SpaceCore convention)."""
        eigenvalues = self.xp.linalg.eigvalsh(
            A, **({} if backend_kwargs is None else backend_kwargs)
        )
        return self.sort(self.real(eigenvalues), axis=-1)

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DenseT, DenseT, DenseT]:
        """Singular value decomposition, in SpaceCore's presentation.

        Singular values descending; each of the leading ``k = min(m, n)``
        columns of ``U`` scaled so its largest-magnitude entry is real and
        positive, with the compensating phase applied to the matching row of
        ``Vh`` so that ``U @ diag(s) @ Vh`` still reconstructs ``A`` exactly.

        With ``full_matrices=True`` the trailing columns of ``U`` (and rows of
        ``Vh``) span a null space whose basis is arbitrary and backend-specific
        -- they are left untouched, and are not comparable across backends.
        Pass ``full_matrices=False`` when you need agreement.
        """
        U, s, Vh = self.xp.linalg.svd(
            A,
            full_matrices=full_matrices,
            **({} if backend_kwargs is None else backend_kwargs),
        )
        return self._order_svd(U, s, Vh)

    def _order_svd(
        self,
        U: DenseArray,
        s: DenseArray,
        Vh: DenseArray,
    ) -> tuple[DenseT, DenseT, DenseT]:
        """Sort singular triples descending and gauge them, preserving ``U s Vh``."""
        # Sort singular values descending; permute the first k columns of U and rows of Vh.
        k = self.shape(s)[-1]
        order = self.xp.argsort(-s, axis=-1)
        s = self._take_along_axis(s, order, -1)

        column_order = self.expand_dims(order, -2)
        U_main = self._take_along_axis(U[..., :, :k], column_order, -1)
        Vh_main = self._take_along_axis(Vh[..., :k, :], self.expand_dims(order, -1), -2)

        # Fix the gauge: U columns / phase and Vh rows * phase leave U diag(s) Vh unchanged.
        phase = self._column_phase(U_main)
        U_main = U_main / phase
        Vh_main = Vh_main * self.swapaxes(phase, -1, -2)

        # Columns beyond k (full_matrices=True) are reattached unchanged.
        U = self.concatenate([U_main, U[..., :, k:]], axis=-1)
        Vh = self.concatenate([Vh_main, Vh[..., k:, :]], axis=-2)
        return U, s, Vh

    def cholesky(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseT:
        """Cholesky factorization (delegates to xp.linalg.cholesky)."""
        return self.xp.linalg.cholesky(A, **({} if backend_kwargs is None else backend_kwargs))

    def where(
        self,
        condition: DenseArray | bool,
        x: ArrayLike | float | complex,
        y: ArrayLike | float | complex,
    ) -> DenseT:
        """Select between x and y by condition (delegates to xp.where)."""
        return self.xp.where(condition, x, y)

    def concatenate(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        dtype: DType | None = None,
    ) -> DenseT:
        """Concatenate arrays along an existing axis (delegates to xp.concat)."""
        # Array API concat, else NumPy's concatenate.
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
    ) -> DenseT:
        """Take entries from x by integer indices (delegates to xp.take)."""
        return self.xp.take(x, indices, axis=axis)

    def diagonal(self, x: DenseArray) -> DenseT:
        """Return the main diagonal over the **trailing two axes**.

        Same correction as :meth:`trace`: the array libraries default to
        ``axis1=0, axis2=1``, which on a ``(batch, n, n)`` stack diagonalizes
        across the batch axis. Identical to the library default for 2-D input.
        """
        return self.xp.diagonal(x, axis1=-2, axis2=-1)

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
        # Report whether the lazy namespace has been imported yet.
        xp_state = ""
        if isinstance(xp, LazyNamespace):
            xp_state = f", xp_loaded={xp.is_loaded!r}"
        return f"{type(self).__name__}(family={self.family!r}{xp_state})"

    def conj(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.conj(...)`` as a backend array."""
        return self._as_array(self.xp.conj(x, **kwargs))

    def real(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.real(...)`` as a backend array."""
        return self._as_array(self.xp.real(x, **kwargs))

    def imag(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.imag(...)`` as a backend array."""
        return self._as_array(self.xp.imag(x, **kwargs))

    def abs(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.abs(...)`` as a backend array."""
        return self._as_array(self.xp.abs(x, **kwargs))

    def sign(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.sign(...)`` as a backend array."""
        return self._as_array(self.xp.sign(x, **kwargs))

    def sqrt(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.sqrt(...)`` as a backend array."""
        return self._as_array(self.xp.sqrt(x, **kwargs))

    def exp(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.exp(...)`` as a backend array."""
        return self._as_array(self.xp.exp(x, **kwargs))

    def log(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.log(...)`` as a backend array."""
        return self._as_array(self.xp.log(x, **kwargs))

    def isfinite(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.isfinite(...)`` as a backend array."""
        return self._as_array(self.xp.isfinite(x, **kwargs))

    def isnan(self, x: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.isnan(...)`` as a backend array."""
        return self._as_array(self.xp.isnan(x, **kwargs))

    def broadcast_to(self, x: DenseArray, shape: Sequence[int], **kwargs: Any) -> DenseT:
        """Return ``xp.broadcast_to(...)`` as a backend array."""
        return self._as_array(self.xp.broadcast_to(x, shape, **kwargs))

    def kron(self, a: DenseArray, b: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.kron(...)`` as a backend array."""
        return self._as_array(self.xp.kron(a, b, **kwargs))

    def einsum(self, subscripts: str, *operands: DenseArray, **kwargs: Any) -> DenseT:
        """Return ``xp.einsum(...)`` as a backend array."""
        return self._as_array(self.xp.einsum(subscripts, *operands, **kwargs))

    def maximum(self, x1: DenseArray | float, x2: DenseArray | float, **kwargs: Any) -> DenseT:
        """Return ``xp.maximum(...)`` as a backend array."""
        return self._as_array(self.xp.maximum(x1, x2, **kwargs))

    def minimum(self, x1: DenseArray | float, x2: DenseArray | float, **kwargs: Any) -> DenseT:
        """Return ``xp.minimum(...)`` as a backend array."""
        return self._as_array(self.xp.minimum(x1, x2, **kwargs))

    def clip(self, x: DenseArray, min: Any = None, max: Any = None, **kwargs: Any) -> DenseT:
        """Return ``xp.clip(...)`` as a backend array."""
        return self._as_array(self.xp.clip(x, min, max, **kwargs))

    def diag(self, x: DenseArray, k: int = 0, **kwargs: Any) -> DenseT:
        """Return ``xp.diag(...)`` as a backend array."""
        return self._as_array(self.xp.diag(x, k, **kwargs))

    def tril(self, x: DenseArray, k: int = 0, **kwargs: Any) -> DenseT:
        """Return ``xp.tril(...)`` as a backend array."""
        return self._as_array(self.xp.tril(x, k=k, **kwargs))

    def triu(self, x: DenseArray, k: int = 0, **kwargs: Any) -> DenseT:
        """Return ``xp.triu(...)`` as a backend array."""
        return self._as_array(self.xp.triu(x, k=k, **kwargs))

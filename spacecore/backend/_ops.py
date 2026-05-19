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


class BackendOps(ABC):
    """
    Backend-agnostic numerical ops interface (portable core).

    Contract:
      - This base class exposes only the portable subset used by library internals.
      - Concrete backends (NumPy/JAX/Torch) may extend these methods with additional
        optional keyword parameters (e.g., `order=`, `out=`, `where=`, `like=`, ...).
    """

    _family: ClassVar[str]
    _allow_sparse: ClassVar[bool]
    xp: ClassVar[Any]

    @property
    def family(self) -> str:
        """
        Generic backend-agnostic wrapper to backend family identifier.

        Input:
            None.

        Output:
            String naming the concrete backend family.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        return type(self)._family

    @property
    def allow_sparse(self) -> bool:
        """
        Generic backend-agnostic wrapper to sparse-array support flag.

        Input:
            None.

        Output:
            Boolean indicating whether this backend supports sparse arrays.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        return self._allow_sparse

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BackendOps):
            return self.family == other.family
        return False

    @property
    @abstractmethod
    def dense_array(self) -> Type[Any]:
        """
        Generic backend-agnostic wrapper to dense array type.

        Input:
            None.

        Output:
            Concrete dense array class accepted by this backend.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @property
    @abstractmethod
    def sparse_array(self) -> Tuple[Type[Any], ...] | None:
        """
        Generic backend-agnostic wrapper to sparse array type tuple.

        Input:
            None.

        Output:
            Concrete sparse array classes accepted by this backend, or None.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Generic backend-agnostic wrapper to normalize a dtype specifier.

        Input:
            dtype: Optional dtype requested by SpaceCore or the caller.

        Output:
            Backend dtype object accepted by array constructors.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    def is_dense(self, x: Any) -> bool:
        """
        Generic backend-agnostic wrapper to test for a dense backend array.

        Input:
            x: Object to test.

        Output:
            True when x is an instance of the backend dense array type.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        return isinstance(x, self.dense_array)

    def is_sparse(self, x: Any) -> bool:
        """
        Generic backend-agnostic wrapper to test for a sparse backend array.

        Input:
            x: Object to test.

        Output:
            True when x is an instance of a backend sparse array type.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        return self.sparse_array is not None and isinstance(x, self.sparse_array)

    def is_array(self, x: Any) -> bool:
        """
        Generic backend-agnostic wrapper to test for any backend array.

        Input:
            x: Object to test.

        Output:
            True when x is dense or sparse for this backend.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        return self.is_dense(x) or self.is_sparse(x)

    @abstractmethod
    def assparse(self, x: Any, dtype: DType | None = None) -> SparseArray:
        """
        Generic backend-agnostic wrapper to convert input to a sparse array.

        Input:
            x: Dense, sparse, or array-like input plus sparse-format options.

        Output:
            Sparse backend array.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to multiply sparse and dense arrays.

        Input:
            a: Sparse backend array; b: Dense backend array.

        Output:
            Dense backend array containing the product.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """
        Generic backend-agnostic wrapper to compute a stable log-sum-exp reduction.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
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
        """
        Generic backend-agnostic wrapper to set indexed values.

        Input:
            x: Dense backend array; index: Selection; values: Replacement values; copy controls mutation policy.

        Output:
            Dense backend array with indexed values set.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def index_add(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to add into indexed values.

        Input:
            x: Dense backend array; index: Selection; values: Values to add; copy controls mutation policy.

        Output:
            Dense backend array with indexed values incremented.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def ix_(self, *args: Any) -> Any:
        """
        Generic backend-agnostic wrapper to build open mesh index arrays.

        Input:
            args: One-dimensional index arrays or sequences.

        Output:
            Tuple of dense backend arrays usable for open-mesh indexing.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def fori_loop(
            self,
            lower: int,
            upper: int,
            body_fun: Callable[[int, T], T],
            init_val: T,
    ) -> T:
        """
        Generic backend-agnostic wrapper to run a counted loop primitive.

        Input:
            lower, upper: Loop bounds; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def while_loop(
            self,
            cond_fun: Callable[[T], bool],
            body_fun: Callable[[T], T],
            init_val: T,
    ) -> T:
        """
        Generic backend-agnostic wrapper to run a while-loop primitive.

        Input:
            cond_fun: Loop condition; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

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
        """
        Generic backend-agnostic wrapper to run a scan primitive.

        Input:
            f: Scan body; init: Initial carry; xs: Per-step inputs plus scan options.

        Output:
            Tuple of final carry and stacked outputs.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        """
        Generic backend-agnostic wrapper to run conditional branch selection.

        Input:
            pred: Predicate; true_fun and false_fun: Branch functions; operands: Branch inputs.

        Output:
            Result returned by the selected branch.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def allclose_sparse(
            self,
            a: SparseArray,
            b: SparseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
    ) -> bool:
        """
        Generic backend-agnostic wrapper to compare sparse arrays elementwise within tolerances.

        Input:
            a, b: Sparse backend arrays; rtol and atol configure comparison.

        Output:
            Boolean indicating whether sparse arrays are close.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
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
        return self.asarray(float("inf"))

    @property
    def nan(self) -> DenseArray:
        return self.asarray(float("nan"))

    @property
    def pi(self) -> DenseArray:
        return self.asarray(3.141592653589793)

    @property
    def e(self) -> DenseArray:
        return self.asarray(2.718281828459045)

    @property
    def eps(self) -> DenseArray:
        return self.asarray(self.xp.finfo(self.sanitize_dtype(None)).eps)

    def get_dtype(self, x: Any) -> DType:
        if self.is_array(x):
            return x.dtype
        raise TypeError(f"Expected {self.family} array, got {type(x)}.")

    def shape(self, x: Any) -> tuple[int, ...]:
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        return int(x.ndim)

    def size(self, x: Any) -> int:
        result = 1
        for dim in self.shape(x):
            result *= int(dim)
        return result

    def asarray(self, x: Any, dtype: DType | None = None, **backend_kwargs: Any) -> DenseArray:
        if self.is_sparse(x) and hasattr(x, "to_dense"):
            x = x.to_dense()
        dtype = self._dtype_arg(dtype)
        if hasattr(self.xp, "asarray"):
            return self.xp.asarray(x, dtype=dtype, **backend_kwargs)
        return self.xp.as_tensor(x, dtype=dtype, **backend_kwargs)

    def astype(self, x: DenseArray, dtype: DType, **backend_kwargs: Any) -> DenseArray:
        dtype = self.sanitize_dtype(dtype)
        if hasattr(x, "astype"):
            return x.astype(dtype, **backend_kwargs)
        return x.to(dtype=dtype, **backend_kwargs)

    def empty(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        return self.xp.empty(shape, dtype=self._dtype_arg(dtype))

    def zeros(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        return self.xp.zeros(shape, dtype=self._dtype_arg(dtype))

    def ones(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        return self.xp.ones(shape, dtype=self._dtype_arg(dtype))

    def zeros_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        return self.xp.zeros_like(x, dtype=self._dtype_arg(dtype))

    def ones_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        return self.xp.ones_like(x, dtype=self._dtype_arg(dtype))

    def full_like(self, x: DenseArray, value: Any, dtype: DType | None = None) -> DenseArray:
        return self.xp.full_like(x, value, dtype=self._dtype_arg(dtype))

    def arange(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
        dtype: DType | None = None,
    ) -> DenseArray:
        dtype = self._dtype_arg(dtype)
        if stop is None:
            return self.xp.arange(start, dtype=dtype)
        if step is None:
            return self.xp.arange(start, stop, dtype=dtype)
        return self.xp.arange(start, stop, step, dtype=dtype)

    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None) -> DenseArray:
        return self.xp.full(shape, fill_value, dtype=self._dtype_arg(dtype))

    def eye(self, n: int, m: int | None = None, dtype: DType | None = None) -> DenseArray:
        return self.xp.eye(n, m, dtype=self._dtype_arg(dtype))

    def ravel(self, x: DenseArray) -> DenseArray:
        if hasattr(self.xp, "ravel"):
            return self.xp.ravel(x)
        return self.reshape(x, (-1,))

    def reshape(self, x: DenseArray, shape: Tuple[int, ...] | int) -> DenseArray:
        shape_arg = (shape,) if isinstance(shape, int) else shape
        return self.xp.reshape(x, shape_arg)

    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        if axes is None:
            axes = tuple(reversed(range(self.ndim(x))))
        return self._permute_dims(x, axes)

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        if hasattr(self.xp, "swapaxes"):
            return self.xp.swapaxes(x, axis1, axis2)
        axes = list(range(self.ndim(x)))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self._permute_dims(x, axes)

    def broadcast_to(self, x: DenseArray, shape: Tuple[int, ...]) -> DenseArray:
        return self.xp.broadcast_to(x, shape)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
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
        if hasattr(self.xp, "moveaxis"):
            return self.xp.moveaxis(x, source, destination)
        return self._permute_dims(x, self._move_axis_order(self.ndim(x), source, destination))

    def stack(self, arrays: Sequence[DenseArray], axis: int = 0) -> DenseArray:
        return self.xp.stack(tuple(arrays), axis=axis)

    def conj(self, x: DenseArray) -> DenseArray:
        return self.xp.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        return self.xp.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        return self.xp.imag(x)

    def abs(self, x: DenseArray) -> DenseArray:
        return self.xp.abs(x)

    def sign(self, x: DenseArray) -> DenseArray:
        return self.xp.sign(x)

    def sqrt(self, x: DenseArray) -> DenseArray:
        return self.xp.sqrt(x)

    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
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
        return self.xp.mean(x, axis=self._to_axis_tuple(axis), keepdims=keepdims)

    def min(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        return self.xp.min(x, axis=self._to_axis_tuple(axis), keepdims=keepdims)

    def max(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        return self.xp.max(x, axis=self._to_axis_tuple(axis), keepdims=keepdims)

    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        return self.xp.prod(
            x,
            axis=self._to_axis_tuple(axis),
            dtype=self._dtype_arg(dtype),
            keepdims=keepdims,
        )

    def trace(self, x: DenseArray) -> DenseArray:
        if hasattr(self.xp, "trace"):
            return self.xp.trace(x)
        return self.sum(self.diagonal(x))

    def argsort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        return self.xp.argsort(x, axis=axis)

    def sort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        return self.xp.sort(x, axis=axis)

    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        return self.xp.argmin(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        return self.xp.argmax(x, axis=axis, keepdims=keepdims)

    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
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
        return self.xp.matmul(a, b, **({} if backend_kwargs is None else backend_kwargs))

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        return self.xp.kron(a, b)

    def einsum(self, subscripts: str, *operands: DenseArray) -> DenseArray:
        return self.xp.einsum(subscripts, *operands)

    def eigh(
        self,
        x: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DenseArray, DenseArray]:
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
        return self.xp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(
        self,
        A: DenseArray,
        b: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        return self.xp.linalg.solve(A, b, **({} if backend_kwargs is None else backend_kwargs))

    def eigvalsh(
        self,
        A: DenseArray,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> DenseArray:
        return self.xp.linalg.eigvalsh(A, **({} if backend_kwargs is None else backend_kwargs))

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DenseArray, DenseArray, DenseArray]:
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
        return self.xp.linalg.cholesky(A, **({} if backend_kwargs is None else backend_kwargs))

    def exp(self, x: DenseArray) -> DenseArray:
        return self.xp.exp(x)

    def log(self, x: DenseArray) -> DenseArray:
        return self.xp.log(x)

    def where(self, condition: DenseArray | bool, x: ArrayLike, y: ArrayLike) -> DenseArray:
        return self.xp.where(condition, x, y)

    def maximum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        return self.xp.maximum(x, y)

    def minimum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        return self.xp.minimum(x, y)

    def clip(self, x: DenseArray, a_min: ArrayLike, a_max: ArrayLike) -> DenseArray:
        return self.xp.clip(x, a_min, a_max)

    def isfinite(self, x: DenseArray) -> DenseArray:
        return self.xp.isfinite(x)

    def isnan(self, x: DenseArray) -> DenseArray:
        return self.xp.isnan(x)

    def concatenate(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        dtype: DType | None = None,
    ) -> DenseArray:
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
        return self.xp.take(x, indices, axis=axis)

    def diag(self, x: DenseArray) -> DenseArray:
        return self.xp.diag(x)

    def diagonal(self, x: DenseArray) -> DenseArray:
        return self.xp.diagonal(x)

    def tril(self, x: DenseArray) -> DenseArray:
        return self.xp.tril(x)

    def triu(self, x: DenseArray) -> DenseArray:
        return self.xp.triu(x)

    def allclose(
            self,
            a: DenseArray,
            b: DenseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            equal_nan: bool = False,
    ) -> bool:
        return bool(self.xp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    def __repr__(self):
        return f"{type(self).__name__}"

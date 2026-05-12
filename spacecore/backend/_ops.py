from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple, Callable, Optional, Type, ClassVar

from ..types import DenseArray, SparseArray, DType, ArrayLike, Index, T, X, Y, R, Carry


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

    @property
    def family(self) -> str:
        return type(self)._family

    @property
    def allow_sparse(self) -> bool:
        return self._allow_sparse

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BackendOps):
            return self.family == other.family
        return False

    @property
    @abstractmethod
    def dense_array(self) -> Type[Any]:
        ...

    @property
    @abstractmethod
    def sparse_array(self) -> Tuple[Type[Any], ...] | None:
        ...

    @abstractmethod
    def sanitize_dtype(self, dtype: DType | None) -> DType:
        ...

    def is_dense(self, x: Any) -> bool:
        return isinstance(x, self.dense_array)

    def is_sparse(self, x: Any) -> bool:
        return self.sparse_array is not None and isinstance(x, self.sparse_array)

    def is_array(self, x: Any) -> bool:
        return self.is_dense(x) or self.is_sparse(x)

    @abstractmethod
    def get_dtype(self, x: Any) -> DType:
        ...

    @abstractmethod
    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return the shape tuple for an array-like object.

        Shape metadata is expected to be backend-static for normal arrays. Backends
        with tracing or dynamic-shape modes may return symbolic dimension objects;
        callers should avoid relying on Python-side shape values inside traced code.
        """

    @abstractmethod
    def ndim(self, x: Any) -> int:
        """
        Return the number of dimensions of an array-like object.

        This is metadata only and does not copy data. For traced backends, the value
        must be available from the abstract array shape.
        """

    @abstractmethod
    def size(self, x: Any) -> int:
        """
        Return the total number of logical elements in an array-like object.

        Sparse backends should report logical dense size, not stored element count.
        Dynamic-shape backends may expose non-plain Python integers.
        """

    @property
    @abstractmethod
    def inf(self) -> DenseArray:
        """Positive infinity (backend scalar)."""

    @property
    @abstractmethod
    def nan(self) -> DenseArray:
        """NaN (backend scalar)."""

    @property
    @abstractmethod
    def pi(self) -> DenseArray:
        """π as backend scalar."""

    @property
    @abstractmethod
    def e(self) -> DenseArray:
        """Euler's number as backend scalar."""

    @property
    @abstractmethod
    def eps(self) -> DenseArray:
        """Machine epsilon for default float dtype."""

    @abstractmethod
    def asarray(self, x: Any, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def astype(self, x: DenseArray, dtype: DType) -> DenseArray:
        """
        Return `x` converted to `dtype`.

        Dtype availability, promotion, and precision are backend-dependent. JAX may
        canonicalize or reject dtypes according to its global configuration and device.
        """

    @abstractmethod
    def assparse(self, x: Any, dtype: DType | None = None) -> SparseArray:
        ...

    @abstractmethod
    def empty(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def zeros_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        """
        Return an array of zeros with the shape and dtype of `x` unless `dtype` is provided.

        Device placement and sharding follow backend defaults for like-constructors.
        """

    @abstractmethod
    def ones_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        """
        Return an array of ones with the shape and dtype of `x` unless `dtype` is provided.

        Device placement and sharding follow backend defaults for like-constructors.
        """

    @abstractmethod
    def full_like(self, x: DenseArray, value: Any, dtype: DType | None = None) -> DenseArray:
        """
        Return an array filled with `value` and shaped like `x`.

        Dtype inference and scalar promotion are backend-dependent, especially for
        mixed Python scalar and array inputs.
        """

    @abstractmethod
    def arange(self, start: int, stop: int | None = None, step: int | None = None, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def eye(self, n: int, m: int | None = None, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def ravel(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def reshape(self, x: DenseArray, shape: Tuple[int, ...] | int) -> DenseArray:
        ...

    @abstractmethod
    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        ...

    @abstractmethod
    def broadcast_to(self, x: DenseArray, shape: Tuple[int, ...]) -> DenseArray:
        """
        Broadcast `x` to `shape` without changing values.

        The returned array may be a view, a lazy/traced value, or an immutable array
        depending on the backend; callers must not rely on writeability.
        """

    @abstractmethod
    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """
        Insert one or more length-one axes at `axis`.

        Axis validation follows the backend implementation. JAX requires axis values
        to be static under JIT.
        """

    @abstractmethod
    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Remove length-one axes from `x`.

        If `axis` is provided, backends raise when a selected axis is not length one.
        JAX requires axis values to be static under JIT.
        """

    @abstractmethod
    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """
        Move axes from `source` positions to `destination` positions.

        Axis arguments are metadata and may need to be Python-static for tracing
        backends such as JAX.
        """

    @abstractmethod
    def stack(self, arrays: Sequence[DenseArray], axis: int = 0) -> DenseArray:
        ...

    @abstractmethod
    def conj(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def real(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def imag(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def abs(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sign(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sqrt(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        ...

    @abstractmethod
    def mean(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Compute the arithmetic mean over `axis`.

        Accumulator dtype, integer promotion, and NaN handling follow the backend.
        JAX requires reduction axes to be static under JIT.
        """

    @abstractmethod
    def min(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Compute the minimum values over `axis`.

        Empty reductions, NaN ordering, and dtype promotion follow the backend.
        JAX requires reduction axes to be static under JIT.
        """

    @abstractmethod
    def max(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Compute the maximum values over `axis`.

        Empty reductions, NaN ordering, and dtype promotion follow the backend.
        JAX requires reduction axes to be static under JIT.
        """

    @abstractmethod
    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        ...

    @abstractmethod
    def trace(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def argsort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        ...

    @abstractmethod
    def sort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        ...

    @abstractmethod
    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        ...

    @abstractmethod
    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        ...

    @abstractmethod
    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def matmul(self, a: DenseArray, b: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def einsum(self, subscripts: str, *operands: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def eigh(self, x: DenseArray) -> tuple[DenseArray, DenseArray]:
        ...

    @abstractmethod
    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Compute a vector or matrix norm.

        Supported `ord` values and dtype precision follow the backend linear algebra
        implementation. JAX requires `ord` and `axis` to be static under JIT.
        """

    @abstractmethod
    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        """
        Solve a dense linear system `A @ x = b`.

        Singular-matrix behavior, batching support, and numerical precision are
        backend-dependent. Sparse inputs are outside this portable contract.
        """

    @abstractmethod
    def eigvalsh(self, A: DenseArray) -> DenseArray:
        """
        Return eigenvalues of a Hermitian or symmetric dense matrix.

        Backends may differ in UPLO defaults, symmetrization details, batching support,
        and floating-point precision.
        """

    @abstractmethod
    def svd(self, A: DenseArray, full_matrices: bool = True) -> tuple[DenseArray, DenseArray, DenseArray]:
        """
        Compute the singular value decomposition of a dense array.

        The portable contract returns `(u, s, vh)` with `compute_uv=True`. Sign choices,
        rank-deficient behavior, and precision are backend-dependent.
        """

    @abstractmethod
    def cholesky(self, A: DenseArray) -> DenseArray:
        """
        Compute a Cholesky factor of a Hermitian positive-definite dense matrix.

        The portable contract returns the lower-triangular factor. Failure modes for
        non-positive-definite input are backend-dependent.
        """

    @abstractmethod
    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        ...

    @abstractmethod
    def exp(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def log(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def where(self, condition: DenseArray | bool, x: ArrayLike, y: ArrayLike) -> DenseArray:
        ...

    @abstractmethod
    def maximum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        ...

    @abstractmethod
    def minimum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """
        Return the elementwise minimum of `x` and `y`.

        Broadcasting, scalar promotion, signed-zero handling, and NaN behavior follow
        the concrete backend.
        """

    @abstractmethod
    def clip(self, x: DenseArray, a_min: ArrayLike, a_max: ArrayLike) -> DenseArray:
        """
        Clip values in `x` to the inclusive interval [`a_min`, `a_max`].

        Broadcasting, dtype promotion, and behavior when bounds contain NaN follow the
        backend. Mutating `out` variants are not part of the portable signature.
        """

    @abstractmethod
    def isfinite(self, x: DenseArray) -> DenseArray:
        """
        Return a boolean array indicating finite values.

        Complex inputs are finite only when both real and imaginary parts are finite.
        Backend dtype support follows the concrete array library.
        """

    @abstractmethod
    def isnan(self, x: DenseArray) -> DenseArray:
        """
        Return a boolean array indicating NaN values.

        Complex-input behavior follows the backend library. Integer and boolean inputs
        typically return all-false arrays.
        """

    @abstractmethod
    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def take(
        self,
        x: DenseArray,
        indices: DenseArray,
        axis: int | None = None,
    ) -> DenseArray:
        """
        Take elements from `x` at integer `indices` along `axis`.

        Out-of-bounds handling differs across backends; the portable contract assumes
        valid indices. JAX requires `axis` to be static under JIT.
        """

    @abstractmethod
    def diag(self, x: DenseArray) -> DenseArray:
        """
        Extract a diagonal from a 2-D array or construct a diagonal matrix from a 1-D array.

        Offset variants are intentionally outside the portable signature. Backend
        behavior for higher-dimensional inputs may differ.
        """

    @abstractmethod
    def diagonal(self, x: DenseArray) -> DenseArray:
        """
        Return the main diagonal using the backend's default diagonal axes.

        Axis and offset variants are intentionally outside the portable signature.
        Returned mutability/view semantics are backend-dependent.
        """

    @abstractmethod
    def tril(self, x: DenseArray) -> DenseArray:
        """
        Return the lower triangle of an array with entries above the main diagonal zeroed.

        Offset variants are intentionally outside the portable signature. Dtype and
        zero-fill behavior follow the backend.
        """

    @abstractmethod
    def triu(self, x: DenseArray) -> DenseArray:
        """
        Return the upper triangle of an array with entries below the main diagonal zeroed.

        Offset variants are intentionally outside the portable signature. Dtype and
        zero-fill behavior follow the backend.
        """

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
        Set `values` at `index` in `x`.

        If copy=True:
            Return a new array y with y[index] = values and y != x.

        If copy=False:
            Mutate x in-place if the backend supports it.
            If the backend does NOT support mutation, raise NotImplementedError.
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
        ...

    @abstractmethod
    def ix_(self, *args: Any) -> Any:
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
        Signature matches jax.lax.fori_loop(lower, upper, body_fun, init_val).
        """

    @abstractmethod
    def while_loop(
            self,
            cond_fun: Callable[[T], bool],
            body_fun: Callable[[T], T],
            init_val: T,
    ) -> T:
        """
        Signature matches jax.lax.while_loop(cond_fun, body_fun, init_val).
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
        Signature matches jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1).
        Note: in JAX, `xs` is a pytree with a leading axis; `ys` matches `xs` structure
        (per-step outputs stacked along the leading axis).
        """

    @abstractmethod
    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        ...

    @abstractmethod
    def allclose(
            self,
            a: DenseArray,
            b: DenseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            equal_nan: bool = False,
    ) -> bool:
        ...

    @abstractmethod
    def allclose_sparse(
            self,
            a: SparseArray,
            b: SparseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
    ) -> bool:
        ...

    def __repr__(self):
        return f"{type(self).__name__}"

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
    def get_dtype(self, x: Any) -> DType:
        """
        Generic backend-agnostic wrapper to return an array dtype.

        Input:
            x: Dense or sparse backend array.

        Output:
            Backend dtype associated with x.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Generic backend-agnostic wrapper to return array shape metadata.

        Input:
            x: Dense or sparse backend array.

        Output:
            Tuple describing the logical shape of x.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def ndim(self, x: Any) -> int:
        """
        Generic backend-agnostic wrapper to return array rank metadata.

        Input:
            x: Dense or sparse backend array.

        Output:
            Number of dimensions in x.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def size(self, x: Any) -> int:
        """
        Generic backend-agnostic wrapper to return logical element count.

        Input:
            x: Dense or sparse backend array.

        Output:
            Total number of logical dense elements.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @property
    @abstractmethod
    def inf(self) -> DenseArray:
        """
        Generic backend-agnostic wrapper to positive infinity scalar.

        Input:
            None.

        Output:
            Backend scalar representing positive infinity.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @property
    @abstractmethod
    def nan(self) -> DenseArray:
        """
        Generic backend-agnostic wrapper to access a NaN scalar.

        Input:
            None.

        Output:
            Backend scalar representing NaN.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @property
    @abstractmethod
    def pi(self) -> DenseArray:
        """
        Generic backend-agnostic wrapper to pi scalar.

        Input:
            None.

        Output:
            Backend scalar representing pi.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @property
    @abstractmethod
    def e(self) -> DenseArray:
        """
        Generic backend-agnostic wrapper to access Euler's number scalar.

        Input:
            None.

        Output:
            Backend scalar representing Euler's number.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @property
    @abstractmethod
    def eps(self) -> DenseArray:
        """
        Generic backend-agnostic wrapper to machine epsilon scalar.

        Input:
            None.

        Output:
            Backend scalar for float64 machine epsilon.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def asarray(self, x: Any, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to convert input to a dense array.

        Input:
            x/a: Array-like input and optional dtype or backend conversion parameters.

        Output:
            Dense backend array.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def astype(self, x: DenseArray, dtype: DType) -> DenseArray:
        """
        Generic backend-agnostic wrapper to cast an array to a dtype.

        Input:
            x: Dense backend array; dtype: target dtype and optional casting controls.

        Output:
            Dense backend array with the requested dtype.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

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
    def empty(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create an uninitialized dense array.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array with uninitialized values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create a zero-filled dense array.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array filled with zeros.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create a one-filled dense array.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array filled with ones.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def zeros_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create zeros shaped like another array.

        Input:
            x: Prototype dense array; dtype, shape, and placement options are backend-specific.

        Output:
            Dense backend array of zeros.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def ones_like(self, x: DenseArray, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create ones shaped like another array.

        Input:
            x: Prototype dense array; dtype, shape, and placement options are backend-specific.

        Output:
            Dense backend array of ones.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def full_like(self, x: DenseArray, value: Any, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create filled values shaped like another array.

        Input:
            x: Prototype dense array; value/fill_value and dtype options are backend-specific.

        Output:
            Dense backend array filled with the requested value.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def arange(self, start: int, stop: int | None = None, step: int | None = None, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create evenly spaced integer-range values.

        Input:
            start, stop, step: Range parameters; dtype and placement options are backend-specific.

        Output:
            One-dimensional dense backend array.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create a filled dense array.

        Input:
            shape: Output shape; fill_value and dtype options are backend-specific.

        Output:
            Dense backend array filled with fill_value.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def eye(self, n: int, m: int | None = None, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to create a dense identity-like matrix.

        Input:
            n and optional m: Matrix dimensions; dtype and placement options are backend-specific.

        Output:
            Two-dimensional dense backend array.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def ravel(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to flatten an array.

        Input:
            x: Dense backend array plus optional order parameters.

        Output:
            One-dimensional dense backend array view or copy.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def reshape(self, x: DenseArray, shape: Tuple[int, ...] | int) -> DenseArray:
        """
        Generic backend-agnostic wrapper to reshape an array.

        Input:
            x: Dense backend array; shape: New shape plus backend-specific options.

        Output:
            Dense backend array with the requested shape.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to permute array axes.

        Input:
            x: Dense backend array; axes: Optional axis order.

        Output:
            Dense backend array with permuted axes.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """
        Generic backend-agnostic wrapper to interchange two axes.

        Input:
            x: Dense backend array; axis1 and axis2: Axes to swap.

        Output:
            Dense backend array with the two axes exchanged.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def broadcast_to(self, x: DenseArray, shape: Tuple[int, ...]) -> DenseArray:
        """
        Generic backend-agnostic wrapper to broadcast an array to a shape.

        Input:
            x: Dense backend array; shape: Target broadcast shape.

        Output:
            Dense backend array with broadcast shape.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """
        Generic backend-agnostic wrapper to insert length-one axes.

        Input:
            x: Dense backend array; axis: Position or positions to insert.

        Output:
            Dense backend array with expanded rank.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to remove length-one axes.

        Input:
            x: Dense backend array; axis: Optional axes to squeeze.

        Output:
            Dense backend array with selected singleton dimensions removed.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to move axes to new positions.

        Input:
            x: Dense backend array; source and destination: Axis positions.

        Output:
            Dense backend array with moved axes.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def stack(self, arrays: Sequence[DenseArray], axis: int = 0) -> DenseArray:
        """
        Generic backend-agnostic wrapper to stack arrays along a new axis.

        Input:
            arrays: Sequence of dense backend arrays; axis: New axis position.

        Output:
            Dense backend array containing stacked inputs.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def conj(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute complex conjugates.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array with conjugated values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def real(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to extract real components.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array containing real components.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def imag(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to extract imaginary components.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array containing imaginary components.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def abs(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute absolute values.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of absolute values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def sign(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute signs elementwise.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of signs.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def sqrt(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute square roots elementwise.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of square roots.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to reduce by summation.

        Input:
            x: Dense backend array; axis, keepdims, and dtype control the reduction.

        Output:
            Dense backend array or scalar containing sums.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def mean(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to reduce by arithmetic mean.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing means.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def min(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to reduce by minimum.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing minima.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def max(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to reduce by maximum.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing maxima.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to reduce by product.

        Input:
            x: Dense backend array; axis, keepdims, and dtype control the reduction.

        Output:
            Dense backend array or scalar containing products.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def trace(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to sum diagonal entries.

        Input:
            x: Dense backend array plus optional diagonal and axis controls.

        Output:
            Dense backend array or scalar containing trace values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def argsort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        """
        Generic backend-agnostic wrapper to return sorting indices.

        Input:
            x: Dense backend array; axis and ordering options are backend-specific.

        Output:
            Dense integer backend array of indices.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def sort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        """
        Generic backend-agnostic wrapper to sort values.

        Input:
            x: Dense backend array; axis and ordering options are backend-specific.

        Output:
            Dense backend array with sorted values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """
        Generic backend-agnostic wrapper to return indices of minimum values.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense integer backend array or scalar of indices.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        """
        Generic backend-agnostic wrapper to return indices of maximum values.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense integer backend array or scalar of indices.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute a conjugating vector dot product.

        Input:
            x, y: Dense backend arrays accepted by the backend vdot operation.

        Output:
            Backend scalar or dense array containing the dot product.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def matmul(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute matrix products.

        Input:
            a, b: Dense backend arrays with matrix-multiplication-compatible shapes.

        Output:
            Dense backend array containing the product.

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
    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute a Kronecker product.

        Input:
            a, b: Dense backend arrays.

        Output:
            Dense backend array containing the Kronecker product.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def einsum(self, subscripts: str, *operands: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to evaluate an Einstein summation expression.

        Input:
            subscripts: Einstein summation string; operands: Dense backend arrays.

        Output:
            Dense backend array containing the contraction result.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def eigh(self, x: DenseArray) -> tuple[DenseArray, DenseArray]:
        """
        Generic backend-agnostic wrapper to compute Hermitian eigenpairs.

        Input:
            x: Dense Hermitian or symmetric backend array.

        Output:
            Tuple of dense backend arrays containing eigenvalues and eigenvectors.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
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
        Generic backend-agnostic wrapper to compute vector or matrix norms.

        Input:
            x: Dense backend array; ord, axis, and keepdims select the norm.

        Output:
            Dense backend array or scalar containing norm values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to solve dense linear systems.

        Input:
            A: Dense coefficient array; b: Dense right-hand side array.

        Output:
            Dense backend array solving A @ x = b.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def eigvalsh(self, A: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute Hermitian eigenvalues.

        Input:
            A: Dense Hermitian or symmetric backend array.

        Output:
            Dense backend array containing eigenvalues.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def svd(self, A: DenseArray, full_matrices: bool = True) -> tuple[DenseArray, DenseArray, DenseArray]:
        """
        Generic backend-agnostic wrapper to compute singular value decompositions.

        Input:
            A: Dense backend array plus SVD options.

        Output:
            Dense backend arrays containing singular vectors and/or singular values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def cholesky(self, A: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute Cholesky factors.

        Input:
            A: Dense Hermitian positive-definite backend array.

        Output:
            Dense backend array containing a triangular factor.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

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
    def exp(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute exponentials elementwise.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of exponentials.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def log(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute natural logarithms elementwise.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of logarithms.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def where(self, condition: DenseArray | bool, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """
        Generic backend-agnostic wrapper to select values by condition.

        Input:
            condition: Boolean array or scalar; x and y: Values to choose between.

        Output:
            Dense backend array containing selected values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def maximum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute elementwise maxima.

        Input:
            x, y: Arrays or scalars accepted by backend broadcasting.

        Output:
            Dense backend array containing maxima.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def minimum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        """
        Generic backend-agnostic wrapper to compute elementwise minima.

        Input:
            x, y: Arrays or scalars accepted by backend broadcasting.

        Output:
            Dense backend array containing minima.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def clip(self, x: DenseArray, a_min: ArrayLike, a_max: ArrayLike) -> DenseArray:
        """
        Generic backend-agnostic wrapper to clip values into an interval.

        Input:
            x: Dense backend array; a_min and a_max: Broadcastable bounds.

        Output:
            Dense backend array with clipped values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def isfinite(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to test finiteness elementwise.

        Input:
            x: Dense backend array.

        Output:
            Boolean dense backend array.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def isnan(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to test NaN values elementwise.

        Input:
            x: Dense backend array.

        Output:
            Boolean dense backend array.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType | None = None) -> DenseArray:
        """
        Generic backend-agnostic wrapper to join arrays along an existing axis.

        Input:
            arrays: Sequence of dense backend arrays; axis and dtype options are backend-specific.

        Output:
            Dense backend array containing concatenated inputs.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """
        ...

    @abstractmethod
    def take(
        self,
        x: DenseArray,
        indices: DenseArray,
        axis: int | None = None,
    ) -> DenseArray:
        """
        Generic backend-agnostic wrapper to take values by integer indices.

        Input:
            x: Dense backend array; indices: Integer indices; axis and mode options are backend-specific.

        Output:
            Dense backend array containing selected values.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def diag(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to extract or build a diagonal.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array containing a diagonal view/copy or matrix.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def diagonal(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to return selected diagonals.

        Input:
            x: Dense backend array plus offset and axis controls.

        Output:
            Dense backend array containing selected diagonals.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def tril(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to return lower-triangular values.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array with upper entries zeroed.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
        """

    @abstractmethod
    def triu(self, x: DenseArray) -> DenseArray:
        """
        Generic backend-agnostic wrapper to return upper-triangular values.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array with lower entries zeroed.

        This declaration only specifies the portable SpaceCore interface.
        See the concrete backend implementation for backend-specific behavior.
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
    def allclose(
            self,
            a: DenseArray,
            b: DenseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            equal_nan: bool = False,
    ) -> bool:
        """
        Generic backend-agnostic wrapper to compare dense arrays elementwise within tolerances.

        Input:
            a, b: Dense backend arrays; rtol, atol, and equal_nan configure comparison.

        Output:
            Boolean indicating whether arrays are close.

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

    def __repr__(self):
        return f"{type(self).__name__}"

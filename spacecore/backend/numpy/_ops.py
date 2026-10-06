from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Tuple, Literal, Type, cast

from .._family import BackendFamily
from .._eager import EagerControlFlowMixin
from .._ops import BackendOps
from ...types import ArrayLike, DenseArray, SparseArray, DType, Index


if TYPE_CHECKING:
    from numpy.typing import NDArray
    NativeArray = NDArray[Any]
else:
    NativeArray = DenseArray


class NumpyOps(EagerControlFlowMixin, BackendOps[NativeArray]):
    """
    Backend operations on NumPy arrays, with SciPy sparse matrices and arrays.

    Methods delegate to NumPy and SciPy and follow their semantics; a ``device``
    argument must be ``"cpu"`` or ``None``. ``ops.np`` and ``ops.sp`` are an explicit
    escape hatch.
    """

    import numpy as _np
    import scipy as _sp
    import array_api_compat.numpy as xp

    # Concrete library handles exposed as ``Any`` so the portable protocols
    # can flow into typed NumPy/SciPy calls without per-boundary casts; mirrors
    # the base ``xp: ClassVar[Any]`` design.
    np: Any = _np
    sp: Any = _sp

    _family = BackendFamily.numpy.value
    _allow_sparse = True

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using NumPy.

        Returns
        -------
            Concrete dense array class accepted by this backend.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
        """
        return self.np.ndarray

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using SciPy.

        Returns
        -------
            Concrete sparse array classes accepted by this backend, or None.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html
        """
        # SciPy has spmatrix (legacy) and sparray (newer); accept whichever exist.
        sparse = self.sp.sparse
        types: list[type[Any]] = []
        if hasattr(sparse, "spmatrix"):
            types.append(sparse.spmatrix)
        if hasattr(sparse, "sparray"):
            types.append(sparse.sparray)
        return tuple(types)

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

    def assparse(
        self, x: Any, *, format: Literal["csr", "csc", "coo"] = "csr", dtype: DType | None = None
    ) -> SparseArray:
        """Convert to a 2-D SciPy sparse matrix in the requested format."""
        sparse = self.sp.sparse
        self._reject_complex_to_real(x, dtype, operation="assparse")

        # Already sparse: cast if needed, then convert the format.
        if self.is_sparse(x):
            if dtype is not None and self.get_dtype(x) != self.sanitize_dtype(dtype):
                x = x.astype(self.sanitize_dtype(dtype))
            if format == "csr":
                return x.tocsr()
            if format == "csc":
                return x.tocsc()
            if format == "coo":
                return x.tocoo()
            raise ValueError(f"Unknown sparse format: {format!r}")

        # Dense input: only 2-D arrays have a SciPy sparse form.
        x_arr = self.asarray(x, dtype=dtype)

        if x_arr.ndim != 2:
            raise ValueError("NumPy/SciPy sparse conversion currently expects a 2D array.")

        if format == "csr":
            return sparse.csr_matrix(x_arr)
        if format == "csc":
            return sparse.csc_matrix(x_arr)
        if format == "coo":
            return sparse.coo_matrix(x_arr)

        raise ValueError(f"Unknown sparse format: {format!r}")

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> NativeArray:
        """Multiply a SciPy sparse matrix by a dense array."""
        if not self.is_sparse(a):
            raise TypeError("sparse_matmul expects `a` to be a SciPy sparse matrix/array.")
        if not self.is_dense(b):
            raise TypeError("sparse_matmul expects `b` to be a Numpy dense object.")
        return a @ b

    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
    ) -> NativeArray | Tuple[NativeArray, NativeArray]:
        """
        Compute a stable log-sum-exp reduction using SciPy.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        See:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        result = self.sp.special.logsumexp(
            a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign
        )

        if return_sign:
            return self._as_array(result[0]), self._as_array(result[1])
        return self._as_array(result)

    def _copy(self, x: DenseArray) -> NativeArray:
        """Return a NumPy copy of ``x`` (mutation primitive for index ops)."""
        return cast(Any, x).copy()

    def _scatter_add_inplace(
        self, y: DenseArray, index: Index, values: ArrayLike | float | complex
    ) -> None:
        """Accumulate ``values`` into ``y`` at ``index`` via ``numpy.add.at``."""
        self.np.add.at(y, index, values)

    def ix_(self, *args: Any) -> Any:
        r"""
        Build open mesh index arrays using NumPy.

        Input:
            args: One-dimensional index arrays or sequences.

        Output:
            Tuple of dense backend arrays usable for open-mesh indexing.

        See:
            `Open-mesh indexing documentation <https://numpy.org/doc/stable/reference/generated/numpy.ix_.html>`_
        """
        return self.np.ix_(*args)

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Compare two SciPy sparse matrices within tolerances, via their CSR difference."""
        self._require_two_sparse(a, b)

        # Compare in CSR: different shapes are never close, an empty difference always is.
        a_csr = cast(Any, a).tocsr()
        b_csr = cast(Any, b).tocsr()

        if a_csr.shape != b_csr.shape:
            return False

        diff = (a_csr - b_csr).tocsr()

        if diff.nnz == 0:
            return True

        # NaN differences are never "close": ``abs(NaN) > tol`` is False, which
        # would otherwise mask a NaN-vs-finite (or NaN-vs-NaN) entry as equal and
        # wrongly report two different operators as close. This matches the
        # torch/cupy paths (dense ``allclose`` with ``equal_nan=False``).
        if self.np.isnan(diff.data).any():
            return False

        # Entrywise max(|a|, |b|) scales the relative tolerance.
        a_abs = abs(a_csr).tocsr()
        b_abs = abs(b_csr).tocsr()
        scale = self.sp.sparse.csr_matrix.maximum(a_abs, b_abs)

        # tolerance_ij = atol + rtol * max(|a_ij|, |b_ij|)
        tol = scale.multiply(rtol)
        tol.data += atol

        # Close iff no stored entry of |a - b| exceeds its tolerance.
        bad = abs(diff) > tol
        return bad.nnz == 0

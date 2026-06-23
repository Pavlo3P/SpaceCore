from __future__ import annotations

from typing import Any, Sequence, Tuple, Literal, Type, cast

from .._family import BackendFamily
from .._eager import EagerControlFlowMixin
from .._ops import BackendOps
from ...types import ArrayLike, DenseArray, SparseArray, DType, Index


class NumpyOps(EagerControlFlowMixin, BackendOps):
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
    -------
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
    -----
        Code intended to remain backend-portable should prefer `BackendOps`
        methods. Direct use of `ops.np` or `ops.sp` is an explicit
        NumPy/SciPy-specific escape hatch.

        NumPy's `device` keyword is present for Array API interoperability.
        When supplied, it must be `"cpu"` or `None`; see the corresponding NumPy
        documentation for each method.
    """

    import numpy as _np
    import scipy as _sp
    import array_api_compat.numpy as xp

    # Concrete library handles exposed as ``Any`` so the portable protocols
    # can flow into typed NumPy/SciPy calls without per-boundary casts; mirrors
    # the base ``xp: ClassVar[Any]`` design.
    np: Any = _np
    sp: Any = _sp

    _family = BackendFamily.numpy.value.lower()
    _allow_sparse = True

    def __init__(self) -> None:
        super().__init__()

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
        self._reject_complex_to_real(x, dtype, operation="assparse")

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

    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
    ) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """
        Compute a stable log-sum-exp reduction using SciPy.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        See:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        return self.sp.special.logsumexp(
            a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign
        )

    def _copy(self, x: DenseArray) -> DenseArray:
        """Return a NumPy copy of ``x`` (mutation primitive for index ops)."""
        return cast(Any, x).copy()

    def _scatter_add_inplace(self, y: DenseArray, index: Index, values: ArrayLike) -> None:
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
            https://numpy.org/doc/stable/reference/generated/numpy.ix\\_.html
        """
        return self.np.ix_(*args)

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
        self._require_two_sparse(a, b)

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

        a_abs = abs(a_csr).tocsr()
        b_abs = abs(b_csr).tocsr()
        scale = self.sp.sparse.csr_matrix.maximum(a_abs, b_abs)

        # tolerance_ij = atol + rtol * max(|a_ij|, |b_ij|)
        tol = scale.multiply(rtol)
        tol.data += atol

        bad = abs(diff) > tol
        return bad.nnz == 0

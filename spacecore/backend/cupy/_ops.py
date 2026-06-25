from __future__ import annotations

from typing import Any, Literal, Sequence, Tuple, Type, cast

from .._eager import EagerControlFlowMixin
from .._family import BackendFamily
from .._ops import BackendOps
from ...types import ArrayLike, DenseArray, DType, Index, SparseArray


class CuPyOps(EagerControlFlowMixin, BackendOps):
    """
    BackendOps implementation for CuPy GPU arrays.

    This backend uses CuPy for dense array operations and ``cupyx.scipy.sparse``
    for sparse arrays. Most operations follow CuPy's NumPy-compatible API and
    execute on the active CUDA device.

    Dense arrays
        ``cupy.ndarray``

    Sparse arrays
        ``cupyx.scipy.sparse`` matrix types such as CSR, CSC, and COO.
    """

    import cupy as _cp  # pyright: ignore[reportMissingImports]
    import cupyx.scipy as _cpx_scipy  # pyright: ignore[reportMissingImports]
    import cupyx.scipy.sparse as _cpx_sparse  # pyright: ignore[reportMissingImports]

    # CuPy ships no usable type stubs; expose the handles as ``Any`` so the
    # portable type checks do not depend on the optional GPU backend.
    cp: Any = _cp
    cpx_scipy: Any = _cpx_scipy
    cpx_sparse: Any = _cpx_sparse

    xp = cp

    _family = BackendFamily.cupy.value.lower()
    _allow_sparse = True

    @property
    def dense_array(self) -> Type[Any]:
        """Dense CuPy array type."""
        return self.cp.ndarray

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """Sparse CuPy array type tuple."""
        sparse = self.cpx_sparse
        types: list[type[Any]] = []
        for name in ("spmatrix", "csr_matrix", "csc_matrix", "coo_matrix"):
            typ = getattr(sparse, name, None)
            if typ is not None:
                types.append(typ)
        return tuple(types)

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize a dtype specifier using CuPy.

        ``None`` follows NumPy/CuPy's float64 default.
        """
        if dtype is None:
            return self.cp.float64
        return self.cp.dtype(dtype)

    def assparse(
        self,
        x: Any,
        *,
        format: Literal["csr", "csc", "coo"] = "csr",
        dtype: DType | None = None,
    ) -> SparseArray:
        """
        Convert input to a CuPy sparse matrix.

        Dense inputs must be two-dimensional. Existing sparse inputs are
        converted to the requested sparse format.
        """
        sparse = self.cpx_sparse
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
            raise ValueError("CuPy sparse conversion currently expects a 2D array.")

        if format == "csr":
            return sparse.csr_matrix(x_arr)
        if format == "csc":
            return sparse.csc_matrix(x_arr)
        if format == "coo":
            return sparse.coo_matrix(x_arr)
        raise ValueError(f"Unknown sparse format: {format!r}")

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """Multiply a CuPy sparse matrix by a CuPy dense array."""
        if not self.is_sparse(a):
            raise TypeError("sparse_matmul expects a CuPy sparse matrix.")
        if not self.is_dense(b):
            raise TypeError("sparse_matmul expects a CuPy dense array.")
        return a @ b

    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
    ) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """Compute log-sum-exp using ``cupyx.scipy.special``."""
        return self.cpx_scipy.special.logsumexp(
            a,
            axis=axis,
            b=b,
            keepdims=keepdims,
            return_sign=return_sign,
        )

    def _copy(self, x: DenseArray) -> DenseArray:
        """Return a CuPy copy of ``x`` (mutation primitive for index ops)."""
        return cast(Any, x).copy()

    def _scatter_add_inplace(self, y: DenseArray, index: Index, values: ArrayLike) -> None:
        """Accumulate ``values`` into ``y`` at ``index`` via ``cupy.add.at``."""
        self.cp.add.at(y, index, values)

    def ix_(self, *args: Any) -> Any:
        """Build open-mesh indices using CuPy."""
        return self.cp.ix_(*args)

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Compare two CuPy sparse matrices by dense values."""
        self._require_two_sparse(a, b, noun="CuPy sparse matrices")
        return bool(
            self.cp.asnumpy(self.cp.allclose(cast(Any, a).toarray(), cast(Any, b).toarray(), rtol=rtol, atol=atol))
        )

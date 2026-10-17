from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, cast

from .._family import BackendFamily
from ..numpy import NumpyOps
from ...types import DType, SparseArray


class CuPyOps(NumpyOps):
    """
    BackendOps implementation for CuPy GPU arrays.

    CuPy and ``cupyx.scipy`` mirror NumPy and SciPy, so this backend is
    :class:`~spacecore.backend.NumpyOps` with the library handles swapped:
    ``ops.cp`` (also ``ops.np``) is ``cupy`` and ``ops.sp`` is ``cupyx.scipy``.
    Operations execute on the active CUDA device.

    Dense arrays
        ``cupy.ndarray``

    Sparse arrays
        ``cupyx.scipy.sparse`` matrix types such as CSR, CSC, and COO.
    """

    import cupy as _cp  # pyright: ignore[reportMissingImports]
    import cupyx.scipy as _cpx_scipy  # pyright: ignore[reportMissingImports]
    import cupyx.scipy.sparse  # pyright: ignore[reportMissingImports]  # noqa: F401 - binds sp.sparse
    import cupyx.scipy.special  # pyright: ignore[reportMissingImports]  # noqa: F401 - binds sp.special
    import array_api_compat.cupy as xp  # pyright: ignore[reportMissingImports]

    # CuPy ships no usable type stubs; expose the handles as ``Any``.
    cp: Any = _cp
    np: Any = _cp
    sp: Any = _cpx_scipy

    _family = BackendFamily.cupy.value

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Compare two CuPy sparse matrices by dense values."""
        # Densify on device and compare; equal_nan stays False.
        self._require_two_sparse(a, b, noun="CuPy sparse matrices")
        return bool(
            self.cp.asnumpy(self.cp.allclose(cast(Any, a).toarray(), cast(Any, b).toarray(), rtol=rtol, atol=atol))
        )

    def assparse(
        self, x: Any, *, format: Literal["csr", "csc", "coo"] = "csr", dtype: DType | None = None
    ) -> SparseArray:
        """Convert to a 2-D ``cupyx.scipy.sparse`` matrix; host SciPy sparse input is copied to the device."""
        # cupy.asarray cannot densify a SciPy sparse matrix, so move it to the device as CSR first.
        import scipy.sparse as sps

        if sps.issparse(x):
            x = self.sp.sparse.csr_matrix(sps.csr_matrix(x))
        return super().assparse(x, format=format, dtype=dtype)

    def vectorize(
        self,
        pyfunc: Callable,
        *,
        excluded: Sequence[int] | None = None,
        signature: str | None = None,
    ) -> Callable:
        """Like :meth:`NumpyOps.vectorize`; ``excluded`` uses the portable loop (``cupy.vectorize`` lacks it)."""
        if excluded:
            return self._vectorize_loop(pyfunc, excluded=excluded, signature=signature)
        return super().vectorize(pyfunc, excluded=excluded, signature=signature)

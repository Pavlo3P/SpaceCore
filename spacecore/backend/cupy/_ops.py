from __future__ import annotations

from typing import Any, cast

from .._family import BackendFamily
from ..numpy import NumpyOps
from ...types import SparseArray


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

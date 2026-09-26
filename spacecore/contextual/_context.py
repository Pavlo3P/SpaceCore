from dataclasses import dataclass
from typing import Any

from ..backend import BackendOps
from ..types import DenseArray, SparseArray, DType, ArrayLike


@dataclass(frozen=True, slots=True, init=False)
class Context:
    """
    Backend operations plus the default dtype for new arrays.

    A context owns no arrays; it creates and converts them. It is frozen, and
    ``check_level`` is deliberately not part of it (see
    :class:`spacecore.contextual.ContextBound`).

    Parameters
    ----------
    ops : BackendOps
        Backend operations, e.g. :class:`spacecore.backend.NumpyOps`.
    dtype : dtype-like or None, optional
        Dtype for :meth:`asarray` and :meth:`assparse`; normalized by
        ``ops.sanitize_dtype``. It does not fix a space's scalar field.

    Examples
    --------
    Create a NumPy context and convert a Python list to a backend array.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> x = ctx.asarray([1.0, 2.0])
    >>> x.dtype == np.dtype("float64")
    True
    """

    ops: BackendOps
    dtype: DType | None

    def __init__(self, ops: BackendOps, dtype: DType | None = None) -> None:
        """
        Validate and normalize the context after dataclass initialization.

        Raises
        ------
        TypeError
            If ``ops`` is not a :class:`BackendOps` instance.
        """
        from ._state import normalize_ops

        try:
            ops = normalize_ops(ops)
        except TypeError:
            raise TypeError("Unknown ops type.")
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "dtype", self.ops.sanitize_dtype(dtype))

    def assert_dense(self, x: Any) -> DenseArray:
        """Return ``x`` if it is a dense array of this backend; raise ``TypeError`` otherwise."""
        if not self.ops.is_dense(x):
            raise TypeError(f"Expected dense array for {self.ops.family}, got {type(x).__name__}")
        return x

    def assert_sparse(self, x: Any) -> SparseArray:
        """Return ``x`` if it is a sparse array of this backend; raise ``TypeError`` otherwise."""
        if not self.ops.allow_sparse:
            raise TypeError("Sparse objects are disallowed by this backend.")
        if not self.ops.is_sparse(x):
            raise TypeError(f"Expected sparse array for {self.ops.family}, got {type(x).__name__}")
        return x

    def asarray(self, x: Any) -> DenseArray:
        """
        Convert ``x`` to a dense backend array of dtype ``self.dtype``.

        Raises ``TypeError`` rather than silently dropping an imaginary part.
        """
        return self.ops.asarray(x, dtype=self.dtype)

    def assparse(self, x: Any) -> SparseArray:
        """
        Convert ``x`` to a sparse backend array of dtype ``self.dtype``.

        Raises ``TypeError`` if the backend has no sparse support, or if the conversion
        would drop an imaginary part.
        """
        return self.ops.assparse(x, dtype=self.dtype)

    def convert(self, x: Any) -> ArrayLike:
        """Convert a dense or sparse backend array to this context; raise ``NotImplementedError`` for anything else."""
        if self.ops.is_dense(x):
            return self.asarray(x)
        elif self.ops.is_sparse(x):
            return self.assparse(x)
        else:
            raise NotImplementedError

    def same_math(self, other: Any) -> bool:
        """
        Return whether ``other`` has equal backend ops and dtype.

        The first-tier gate of every context-bound ``__eq__``: it guarantees later value
        comparisons run on same-backend, same-dtype arrays. Not policy-configurable.

        Parameters
        ----------
        other : Any
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when ``other`` is a ``Context`` with equal ops and dtype.
        """
        if isinstance(other, Context):
            return self.ops == other.ops and self.dtype == other.dtype
        return False

    def same_backend(self, other: Any) -> bool:
        """
        Return whether ``other`` is a ``Context`` with equal ops, ignoring dtype.

        The loosest context relation, used by context resolution: operands on one backend
        combine, and their dtypes are reconciled by conversion.
        """
        if isinstance(other, Context):
            return self.ops == other.ops
        return False

    def __eq__(self, other: Any) -> bool:
        """Return whether ``other`` is a ``Context`` with equal ops and dtype."""
        if isinstance(other, Context):
            return self.ops == other.ops and self.dtype == other.dtype
        return False

    def __repr__(self) -> str:
        return f"Context(ops={self.ops!r}, dtype={self.dtype!r})"

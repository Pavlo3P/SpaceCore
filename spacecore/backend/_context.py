from dataclasses import dataclass
from typing import Any

from ._ops import BackendOps
from ..types import DenseArray, SparseArray, DType, ArrayLike
from .._contextual import normalize_ops


@dataclass(frozen=True, slots=True)
class Context:
    """
    Backend execution context for SpaceCore objects.

    A context collects the backend operations object, default dtype, and runtime
    validation policy used by spaces, linear operators, and context-bound
    values. It is intentionally small: it does not own arrays, but it defines
    how new arrays are created and how existing arrays are checked or converted
    for a backend family.

    Parameters
    ----------
    ops:
        Backend operations implementation. This must be an instance of
        :class:`spacecore.backend.BackendOps`, such as
        :class:`spacecore.backend.NumpyOps` or
        :class:`spacecore.backend.JaxOps`.
    dtype:
        Default dtype used by :meth:`asarray` and :meth:`assparse`. The value is
        normalized through ``ops.sanitize_dtype`` during initialization.
    enable_checks:
        Whether spaces and linear operators using this context should perform
        membership and compatibility checks before operations.

    Notes
    -----
    ``Context`` is frozen and slot-based. Methods that convert values return new
    backend arrays or sparse objects; they do not mutate the context itself.

    Equality compares backend family and ``enable_checks``. It currently does
    not compare ``dtype``.
    """

    ops: BackendOps
    dtype: DType | None = None
    enable_checks: bool = True

    def __post_init__(self):
        """
        Validate and normalize the context after dataclass initialization.

        Raises
        ------
        TypeError
            If ``ops`` is not a :class:`BackendOps` instance.
        """
        try:
            ops = normalize_ops(self.ops)
        except TypeError:
            raise TypeError("Unknown ops type.")
        object.__setattr__(self, "ops", ops)

        sanitized = self.ops.sanitize_dtype(self.dtype)
        object.__setattr__(self, "dtype", sanitized)

    def assert_dense(self, x: Any) -> DenseArray:
        """
        Return ``x`` after verifying that it is a dense array for this backend.

        Parameters
        ----------
        x:
            Object to validate.

        Returns
        -------
        DenseArray
            The original object, typed as a dense array.

        Raises
        ------
        TypeError
            If ``x`` is not recognized as a dense array by ``self.ops``.
        """
        if not self.ops.is_dense(x):
            raise TypeError(f"Expected dense array for {self.ops.family}, got {type(x).__name__}")
        return x

    def assert_sparse(self, x: Any) -> SparseArray:
        """
        Return ``x`` after verifying that it is a sparse array for this backend.

        Parameters
        ----------
        x:
            Object to validate.

        Returns
        -------
        SparseArray
            The original object, typed as a sparse array.

        Raises
        ------
        TypeError
            If the backend does not allow sparse arrays or if ``x`` is not
            recognized as a sparse array by ``self.ops``.
        """
        if not self.ops.allow_sparse:
            raise TypeError("Sparse objects are disallowed by this backend.")
        if not self.ops.is_sparse(x):
            raise TypeError(f"Expected sparse array for {self.ops.family}, got {type(x).__name__}")
        return x

    def asarray(self, x: Any) -> DenseArray:
        """
        Convert ``x`` to a dense backend array using this context's dtype.

        Parameters
        ----------
        x:
            Array-like object accepted by the backend implementation.

        Returns
        -------
        DenseArray
            Backend-native dense array with dtype ``self.dtype``.
        """
        return self.ops.asarray(x, dtype=self.dtype)

    def assparse(self, x: Any) -> SparseArray:
        """
        Convert ``x`` to a sparse backend object using this context's dtype.

        Parameters
        ----------
        x:
            Array-like or sparse object accepted by the backend implementation.

        Returns
        -------
        SparseArray
            Backend-native sparse object with dtype ``self.dtype``.

        Raises
        ------
        TypeError
            If the backend implementation does not support sparse conversion.
        """
        return self.ops.assparse(x, dtype=self.dtype)

    def convert(self, x: Any) -> ArrayLike:
        """
        Convert an existing backend array to this context.

        Dense inputs are converted with :meth:`asarray`; sparse inputs are
        converted with :meth:`assparse`.

        Parameters
        ----------
        x:
            Dense or sparse backend array recognized by ``self.ops``.

        Returns
        -------
        ArrayLike
            Converted dense or sparse backend object.

        Raises
        ------
        NotImplementedError
            If ``x`` is neither dense nor sparse according to this backend.
        """
        if self.ops.is_dense(x):
            return self.asarray(x)
        elif self.ops.is_sparse(x):
            return self.assparse(x)
        else:
            raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        """
        Return whether another object has the same effective backend policy.

        Parameters
        ----------
        other:
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when ``other`` is a ``Context`` with equal backend
            operations and equal ``enable_checks``.
        """
        if isinstance(other, Context):
            return self.ops == other.ops and self.enable_checks == other.enable_checks
        return False

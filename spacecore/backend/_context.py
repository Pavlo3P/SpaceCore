from dataclasses import dataclass
from typing import Any

from .._check_policy import CheckLevel, normalize_check_level
from ._ops import BackendOps
from ..types import DenseArray, SparseArray, DType, ArrayLike


@dataclass(frozen=True, slots=True, init=False)
class Context:
    """
    Select backend operations, representation dtype, and validation policy.

    A context collects the backend operations object, default dtype, and runtime
    validation policy used by spaces, linear operators, and context-bound
    values. It is intentionally small: it does not own arrays, but it defines
    how new arrays are created and how existing arrays are checked or converted
    for a backend family.

    Parameters
    ----------
    ops : BackendOps
        Backend operations implementation. This must be an instance of
        :class:`spacecore.backend.BackendOps`, such as
        :class:`spacecore.backend.NumpyOps` or
        :class:`spacecore.backend.JaxOps`.
    dtype : dtype-like or None, optional
        Default array representation dtype used by :meth:`asarray` and
        :meth:`assparse`. The value is normalized through
        ``ops.sanitize_dtype`` during initialization. It does not independently
        define a mathematical scalar field; spaces expose that contract through
        :attr:`spacecore.space.Space.field`.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy. The default is ``"standard"``.
    enable_checks : bool or None, optional
        Deprecated compatibility alias. ``True`` maps to ``"standard"`` and
        ``False`` maps to ``"none"``. Passing both policy arguments is an error.

    Attributes
    ----------
    ops : BackendOps
        Normalized backend operations instance.
    dtype : dtype-like
        Backend-native dtype used by array constructors.
    Notes
    -----
    ``Context`` is frozen and slot-based. Methods that convert values return new
    backend arrays or sparse objects; they do not mutate the context itself.

    Equality compares backend family, dtype, and ``check_level``.

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
    check_level: CheckLevel

    def __init__(
        self,
        ops: BackendOps,
        dtype: DType | None = None,
        enable_checks: bool | None = None,
        *,
        check_level: CheckLevel | None = None,
    ) -> None:
        """
        Validate and normalize the context after dataclass initialization.

        Raises
        ------
        TypeError
            If ``ops`` is not a :class:`BackendOps` instance.
        """
        from .._contextual._state import normalize_ops

        try:
            ops = normalize_ops(ops)
        except TypeError:
            raise TypeError("Unknown ops type.")
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "dtype", self.ops.sanitize_dtype(dtype))
        object.__setattr__(
            self,
            "check_level",
            normalize_check_level(
                check_level,
                enable_checks=enable_checks,
                warn_legacy=enable_checks is not None,
            ),
        )

    @property
    def enable_checks(self) -> bool:
        """Deprecated Boolean view of :attr:`check_level`."""
        return self.check_level != "none"

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

        Raises
        ------
        TypeError
            If conversion to ``self.dtype`` would discard a complex
            representation. Extract the real part explicitly before conversion
            when that loss is intentional.
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
            If the backend implementation does not support sparse conversion,
            or conversion to ``self.dtype`` would discard a complex
            representation.
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
        Return whether another object has the same execution context.

        Parameters
        ----------
        other:
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when ``other`` is a ``Context`` with equal backend
            operations, dtype, and ``check_level``.
        """
        if isinstance(other, Context):
            return (
                self.ops == other.ops
                and self.dtype == other.dtype
                and self.check_level == other.check_level
            )
        return False

    def __repr__(self) -> str:
        return (
            f"Context(ops={self.ops!r}, dtype={self.dtype!r}, "
            f"check_level={self.check_level!r})"
        )

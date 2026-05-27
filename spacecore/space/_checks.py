from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class SpaceValidationError(ValueError, TypeError):
    """Raised when an object is not a member of a space."""


def _shape_of(space: Any, x: Any) -> tuple[int, ...] | None:
    """Return the backend-visible shape of ``x`` when available."""
    try:
        return tuple(space.ops.shape(x))
    except Exception:
        maybe_shape = getattr(x, "shape", None)
        return tuple(maybe_shape) if maybe_shape is not None else None


def _dtype_of(space: Any, x: Any) -> Any:
    """Return the backend-visible dtype of ``x`` when available."""
    try:
        return space.ops.get_dtype(x)
    except Exception:
        return getattr(x, "dtype", None)


@dataclass(frozen=True)
class SpaceCheck(ABC):
    """
    Define a membership check for :class:`Space` objects.

    Parameters
    ----------
    name : str
        Human-readable check name used in diagnostics.
    """

    name: str

    def __call__(self, space: Any, x: Any) -> None:
        """Raise :class:`SpaceValidationError` when ``x`` is invalid."""
        if not self.is_valid(space, x):
            raise SpaceValidationError(self.error_message(space, x))

    @abstractmethod
    def is_valid(self, space: Any, x: Any) -> bool:
        """Return whether ``x`` is valid for ``space``."""
        ...

    @abstractmethod
    def error_message(self, space: Any, x: Any) -> str:
        """Return a diagnostic for an invalid ``x``."""
        ...


@dataclass(frozen=True)
class BackendCheck(SpaceCheck):
    """
    Check that a value is a dense array for a space backend.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"backend"``.
    """

    name: str = "backend"

    def is_valid(self, space: Any, x: Any) -> bool:
        return bool(space.ops.is_dense(x))

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected dense array for {space.ops.family}, got {type(x).__name__}"


@dataclass(frozen=True)
class ShapeCheck(SpaceCheck):
    """
    Check that a value has the canonical shape of a space.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"shape"``.
    """

    name: str = "shape"

    def is_valid(self, space: Any, x: Any) -> bool:
        return _shape_of(space, x) == tuple(space.shape)

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected shape {tuple(space.shape)}, got {_shape_of(space, x)}"


@dataclass(frozen=True)
class DTypeCheck(SpaceCheck):
    """
    Check that a value has the dtype required by a space context.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"dtype"``.
    """

    name: str = "dtype"

    def is_valid(self, space: Any, x: Any) -> bool:
        return _dtype_of(space, x) == space.dtype

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected dtype {space.dtype}, got {_dtype_of(space, x)}"


@dataclass(frozen=True)
class SquareMatrixCheck(SpaceCheck):
    """
    Check that a value has square trailing matrix axes.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"square_matrix"``.
    """

    name: str = "square_matrix"

    def is_valid(self, space: Any, x: Any) -> bool:
        shape = _shape_of(space, x)
        return shape is not None and len(shape) >= 2 and shape[-1] == shape[-2]

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected square matrix, got shape {_shape_of(space, x)}"


@dataclass(frozen=True)
class HermitianCheck(SpaceCheck):
    """
    Check that a value is Hermitian within tolerances.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"hermitian"``.
    atol : float, optional
        Absolute tolerance for Hermitian comparison.
    rtol : float, optional
        Relative tolerance for Hermitian comparison.
    enforce : bool, optional
        Whether to enforce the Hermitian comparison.
    """

    name: str = "hermitian"
    atol: float = 1e-8
    rtol: float = 1e-8
    enforce: bool = True

    def is_valid(self, space: Any, x: Any) -> bool:
        if not self.enforce:
            return True

        ops = space.ops
        x_adj = ops.conj(ops.swapaxes(x, -1, -2))
        return bool(ops.allclose(x, x_adj, atol=self.atol, rtol=self.rtol))

    def error_message(self, space: Any, x: Any) -> str:
        return (
            "Expected Hermitian matrix; input is not Hermitian. "
            "Expected x satisfying x = x.conj().T "
            f"within atol={self.atol} and "
            f"rtol={self.rtol}. "
            f"Got shape {_shape_of(space, x)}."
        )


@dataclass(frozen=True)
class ProductStructureCheck(SpaceCheck):
    """
    Check that a product-space value is a tuple of the right length.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"product_structure"``.
    """

    name: str = "product_structure"

    def is_valid(self, space: Any, x: Any) -> bool:
        return isinstance(x, tuple) and len(x) == space.arity

    def error_message(self, space: Any, x: Any) -> str:
        if not isinstance(x, tuple):
            return f"ProductSpace element must be a tuple, got {type(x).__name__}"
        return f"Expected tuple of length {space.arity}, got {len(x)}"


@dataclass(frozen=True)
class ProductComponentCheck(SpaceCheck):
    """
    Check each component of a product-space value.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"product_components"``.
    """

    name: str = "product_components"

    def is_valid(self, space: Any, x: Any) -> bool:
        if not isinstance(x, tuple) or len(x) != space.arity:
            return False

        for subspace, component in zip(space.spaces, x):
            try:
                subspace.check_member(component)
            except Exception:
                return False
        return True

    def error_message(self, space: Any, x: Any) -> str:
        if not isinstance(x, tuple):
            return f"ProductSpace element must be a tuple, got {type(x).__name__}"
        if len(x) != space.arity:
            return f"Expected tuple of length {space.arity}, got {len(x)}"

        for i, (subspace, component) in enumerate(zip(space.spaces, x)):
            try:
                subspace.check_member(component)
            except Exception as exc:
                return (
                    f"Invalid component {i} for spaces[{i}] "
                    f"({type(subspace).__name__}): {exc}"
                )
        return "Invalid product-space component."

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


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
    core_rank: ClassVar[int] = 0
    enforce_core_shape: ClassVar[bool] = False

    def __call__(self, space: Any, x: Any) -> None:
        """Raise :class:`SpaceValidationError` when ``x`` is invalid."""
        if not self.validate(space, x, allow_leading=False):
            raise SpaceValidationError(
                self.validation_message(space, x, allow_leading=False)
            )

    def core_shape(self, space: Any) -> tuple[int, ...]:
        """Return the trailing shape that defines one element for this check."""
        if self.core_rank == 0:
            return ()
        return tuple(space.shape)[-self.core_rank:]

    def leading_dims(self, x: Any, space: Any) -> tuple[int, ...] | None:
        """Return leading batch dimensions before this check's core axes."""
        shape = _shape_of(space, x)
        if shape is None:
            return None
        core_shape = self.core_shape(space)
        core_rank = len(core_shape)
        if core_rank == 0:
            return shape
        if len(shape) < core_rank:
            return None
        return shape[:-core_rank]

    def validate(self, space: Any, x: Any, *, allow_leading: bool) -> bool:
        """Return whether ``x`` is valid under member or batched shape policy."""
        if self.enforce_core_shape:
            shape = _shape_of(space, x)
            core_shape = self.core_shape(space)
            core_rank = len(core_shape)
            trailing_matches = core_rank == 0 or (
                shape is not None
                and len(shape) >= core_rank
                and shape[-core_rank:] == core_shape
            )
            if shape is None or not trailing_matches:
                return False
            if not allow_leading and self.leading_dims(x, space) != ():
                return False
        return self.is_valid(space, x)

    def validation_message(self, space: Any, x: Any, *, allow_leading: bool) -> str:
        """Return a diagnostic for an invalid validation result."""
        return self.error_message(space, x)

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
    core_rank: ClassVar[int] = 0

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
    enforce_core_shape: ClassVar[bool] = True

    def core_shape(self, space: Any) -> tuple[int, ...]:
        """Return the whole canonical shape as the trailing element shape."""
        return tuple(space.shape)

    def is_valid(self, space: Any, x: Any) -> bool:
        shape = _shape_of(space, x)
        if shape is None:
            return False

        core_shape = self.core_shape(space)
        core_rank = len(core_shape)
        if core_rank == 0:
            return True
        return len(shape) >= core_rank and shape[-core_rank:] == core_shape

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected shape {tuple(space.shape)}, got {_shape_of(space, x)}"

    def validation_message(self, space: Any, x: Any, *, allow_leading: bool) -> str:
        if allow_leading:
            return (
                f"Batched value trailing shape must be {tuple(space.shape)}, "
                f"got {_shape_of(space, x)}."
            )
        return self.error_message(space, x)


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
    core_rank: ClassVar[int] = 0

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
    core_rank: ClassVar[int] = 2
    enforce_core_shape: ClassVar[bool] = True

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
    core_rank: ClassVar[int] = 2
    enforce_core_shape: ClassVar[bool] = True
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
    Check that a value is valid for the configured ProductSpace structure.

    Parameters
    ----------
    name : str, optional
        Check name. Default is ``"product_structure"``.
    """

    name: str = "product_structure"
    core_rank: ClassVar[int] = 0

    def is_valid(self, space: Any, x: Any) -> bool:
        try:
            space._structure.to_components(x, arity=space.arity)
        except Exception:
            return False
        return True

    def error_message(self, space: Any, x: Any) -> str:
        return self.validation_message(space, x, allow_leading=False)

    def validation_message(self, space: Any, x: Any, *, allow_leading: bool) -> str:
        try:
            space._structure.to_components(x, arity=space.arity)
        except Exception as exc:
            if allow_leading:
                return f"Invalid batched product structure: {exc}"
            return str(exc)
        return "Invalid product-space structure."


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
    core_rank: ClassVar[int] = 0

    def is_valid(self, space: Any, x: Any) -> bool:
        return self.validate(space, x, allow_leading=True)

    def validate(self, space: Any, x: Any, *, allow_leading: bool) -> bool:
        try:
            parts = space._structure.to_components(x, arity=space.arity)
        except Exception:
            return False

        for subspace, component in zip(space.spaces, parts):
            try:
                _run_checks(subspace, component, allow_leading=allow_leading)
            except Exception:
                return False
        return True

    def error_message(self, space: Any, x: Any) -> str:
        return self.validation_message(space, x, allow_leading=False)

    def validation_message(self, space: Any, x: Any, *, allow_leading: bool) -> str:
        try:
            parts = space._structure.to_components(x, arity=space.arity)
        except Exception as exc:
            return str(exc)

        for i, (subspace, component) in enumerate(zip(space.spaces, parts)):
            try:
                _run_checks(subspace, component, allow_leading=allow_leading)
            except Exception as exc:
                return (
                    f"Invalid component {i} for spaces[{i}] "
                    f"({type(subspace).__name__}): {exc}"
                )
        return "Invalid product-space component."


def _run_checks(space: Any, x: Any, *, allow_leading: bool) -> None:
    """Run all membership checks with a shared member/batched shape policy."""
    for check in space.member_checks():
        if not check.validate(space, x, allow_leading=allow_leading):
            raise SpaceValidationError(
                check.validation_message(space, x, allow_leading=allow_leading)
            )

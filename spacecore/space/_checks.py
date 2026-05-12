from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class SpaceValidationError(ValueError, TypeError):
    """Raised when an object is not a member of a space."""


def _shape_of(space: Any, x: Any) -> tuple[int, ...] | None:
    try:
        return tuple(space.ops.shape(x))
    except Exception:
        maybe_shape = getattr(x, "shape", None)
        return tuple(maybe_shape) if maybe_shape is not None else None


def _dtype_of(space: Any, x: Any) -> Any:
    try:
        return space.ops.get_dtype(x)
    except Exception:
        return getattr(x, "dtype", None)


@dataclass(frozen=True)
class SpaceCheck(ABC):
    name: str

    def __call__(self, space: Any, x: Any) -> None:
        if not self.is_valid(space, x):
            raise SpaceValidationError(self.error_message(space, x))

    @abstractmethod
    def is_valid(self, space: Any, x: Any) -> bool:
        ...

    @abstractmethod
    def error_message(self, space: Any, x: Any) -> str:
        ...


@dataclass(frozen=True)
class BackendCheck(SpaceCheck):
    name: str = "backend"

    def is_valid(self, space: Any, x: Any) -> bool:
        return bool(space.ops.is_dense(x))

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected dense array for {space.ops.family}, got {type(x).__name__}"


@dataclass(frozen=True)
class ShapeCheck(SpaceCheck):
    name: str = "shape"

    def is_valid(self, space: Any, x: Any) -> bool:
        return _shape_of(space, x) == tuple(space.shape)

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected shape {tuple(space.shape)}, got {_shape_of(space, x)}"


@dataclass(frozen=True)
class DTypeCheck(SpaceCheck):
    name: str = "dtype"

    def is_valid(self, space: Any, x: Any) -> bool:
        return _dtype_of(space, x) == space.dtype

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected dtype {space.dtype}, got {_dtype_of(space, x)}"


@dataclass(frozen=True)
class SquareMatrixCheck(SpaceCheck):
    name: str = "square_matrix"

    def is_valid(self, space: Any, x: Any) -> bool:
        shape = _shape_of(space, x)
        return shape is not None and len(shape) >= 2 and shape[-1] == shape[-2]

    def error_message(self, space: Any, x: Any) -> str:
        return f"Expected square matrix, got shape {_shape_of(space, x)}"


@dataclass(frozen=True)
class HermitianCheck(SpaceCheck):
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
    name: str = "product_structure"

    def is_valid(self, space: Any, x: Any) -> bool:
        return isinstance(x, tuple) and len(x) == space.arity

    def error_message(self, space: Any, x: Any) -> str:
        if not isinstance(x, tuple):
            return f"ProductSpace element must be a tuple, got {type(x).__name__}"
        return f"Expected tuple of length {space.arity}, got {len(x)}"


@dataclass(frozen=True)
class ProductComponentCheck(SpaceCheck):
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

"""Composable validation checks for spaces and arrays."""

from ._base import (
    BackendCheck,
    DTypeCheck,
    HermitianCheck,
    ProductComponentCheck,
    ProductStructureCheck,
    ShapeCheck,
    SpaceCheck,
    SpaceValidationError,
    SquareMatrixCheck,
    _run_checks as _run_checks,
)

__all__ = [
    "BackendCheck",
    "DTypeCheck",
    "HermitianCheck",
    "ProductComponentCheck",
    "ProductStructureCheck",
    "ShapeCheck",
    "SpaceCheck",
    "SpaceValidationError",
    "SquareMatrixCheck",
    "_run_checks",
]

"""Vector space abstractions, concrete spaces, and validation checks."""

from ._checks import (
    BackendCheck,
    DTypeCheck,
    HermitianCheck,
    ProductComponentCheck,
    ProductStructureCheck,
    ShapeCheck,
    SpaceCheck,
    SpaceValidationError,
    SquareMatrixCheck,
)
from ._base import Space
from ._herm import HermitianSpace
from ._vector import VectorSpace
from ._product import ProductSpace

__all__ = [
    "BackendCheck",
    "DTypeCheck",
    "HermitianCheck",
    "ProductComponentCheck",
    "ProductStructureCheck",
    "ShapeCheck",
    "Space",
    "SpaceCheck",
    "SpaceValidationError",
    "SquareMatrixCheck",
    "HermitianSpace",
    "VectorSpace",
    "ProductSpace",
]

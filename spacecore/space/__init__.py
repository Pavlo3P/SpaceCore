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
from ._inner import EuclideanInnerProduct, InnerProduct, WeightedInnerProduct
from ._herm import HermitianSpace
from ._vector import VectorSpace
from ._product import ProductSpace
from ._structure import ProductStructure, TupleStructure, PytreeStructure
from ._stacked import StackedSpace

__all__ = [
    "BackendCheck",
    "DTypeCheck",
    "HermitianCheck",
    "ProductComponentCheck",
    "ProductStructureCheck",
    "ShapeCheck",
    "InnerProduct",
    "EuclideanInnerProduct",
    "WeightedInnerProduct",
    "Space",
    "SpaceCheck",
    "SpaceValidationError",
    "SquareMatrixCheck",
    "HermitianSpace",
    "VectorSpace",
    "ProductSpace",
    "ProductStructure",
    "TupleStructure",
    "PytreeStructure",
    "StackedSpace",
]

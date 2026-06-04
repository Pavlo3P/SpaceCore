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
from ._base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
    VectorSpace,
)
from ._inner import EuclideanInnerProduct, InnerProduct, WeightedInnerProduct
from ._herm import HermitianSpace
from ._vector import DenseCoordinateSpace, DenseVectorSpace
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
    "CoordinateSpace",
    "InnerProductSpace",
    "StarSpace",
    "JordanAlgebraSpace",
    "EuclideanJordanAlgebraSpace",
    "DenseCoordinateSpace",
    "DenseVectorSpace",
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

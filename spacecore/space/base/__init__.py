"""Base space abstractions and inner-product geometry."""

from ._space import Space
from ._vector import VectorSpace
from ._coordinate import CoordinateSpace
from ._inner_product import (
    EuclideanInnerProduct,
    InnerProduct,
    InnerProductSpace,
    WeightedInnerProduct,
)
from ._star import StarSpace
from ._jordan import JordanAlgebraSpace, EuclideanJordanAlgebraSpace

__all__ = [
    "Space",
    "VectorSpace",
    "CoordinateSpace",
    "InnerProduct",
    "EuclideanInnerProduct",
    "WeightedInnerProduct",
    "InnerProductSpace",
    "StarSpace",
    "JordanAlgebraSpace",
    "EuclideanJordanAlgebraSpace",
]

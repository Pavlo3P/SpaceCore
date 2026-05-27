"""Linear operator abstractions, concrete operators, and algebra helpers."""

from ._base import LinOp
from ._algebra import (
    ComposedLinOp,
    IdentityLinOp,
    MatrixFreeLinOp,
    ScaledLinOp,
    SumLinOp,
    ZeroLinOp,
    make_composed,
    make_scaled,
    make_sum,
)
from ._dense import DenseLinOp
from ._diagonal import DiagonalLinOp
from ._sparse import SparseLinOp
from .product import ProductLinOp, StackedLinOp, SumToSingleLinOp, BlockDiagonalLinOp

__all__ = [
    "LinOp",
    "ComposedLinOp",
    "DiagonalLinOp",
    "DenseLinOp",
    "IdentityLinOp",
    "MatrixFreeLinOp",
    "ScaledLinOp",
    "SparseLinOp",
    "SumLinOp",
    "ZeroLinOp",
    "make_composed",
    "make_scaled",
    "make_sum",
    "ProductLinOp",
    "SumToSingleLinOp",
    "BlockDiagonalLinOp",
    "StackedLinOp",
]

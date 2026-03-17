from ._base import ProductLinOp
from ._block import BlockDiagonalLinOp
from ._from_single import StackedLinOp
from ._to_single import SumToSingleLinOp

__all__ = [
    "ProductLinOp",
    "BlockDiagonalLinOp",
    "StackedLinOp",
    "SumToSingleLinOp",
]
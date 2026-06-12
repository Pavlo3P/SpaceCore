"""Linear operators that map to or from tree-structured spaces."""

from ._base import TreeLinOp
from ._block import BlockDiagonalLinOp
from ._from_single import StackedLinOp
from ._to_single import SumToSingleLinOp

__all__ = [
    "TreeLinOp",
    "BlockDiagonalLinOp",
    "StackedLinOp",
    "SumToSingleLinOp",
]

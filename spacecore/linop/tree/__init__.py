"""Linear operators that map to or from tree-structured spaces."""

from ._base import TreeLinOp
from ._block import BlockDiagonalLinOp, BlockMatrixLinOp

__all__ = [
    "TreeLinOp",
    "BlockDiagonalLinOp",
    "BlockMatrixLinOp",
]

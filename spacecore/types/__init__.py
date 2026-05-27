"""Common typing aliases and protocols used by SpaceCore."""

from ._array import ArrayLike, DenseArray, SparseArray
from ._dtype import DType
from ._misc import Index, T, X, Y, R, Carry

__all__ = [
    "DenseArray",
    "SparseArray",
    "ArrayLike",
    "DType",
    "Index",
    "T",
    "X",
    "Y",
    "R",
    "Carry",
]

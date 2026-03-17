from .backend import Context, BackendOps, JaxOps, NumpyOps, jax_pytree_class
from .linop import DenseLinOp, SparseLinOp, BlockDiagonalLinOp, SumToSingleLinOp, StackedLinOp
from .space import VectorSpace, HermitianSpace, Space, ProductSpace
from .types import DenseArray, SparseArray, ArrayLike

from _contextual.manager import set_context, register_ops

__all__ = [
    "Context",

    "BackendOps",
    "JaxOps",
    "jax_pytree_class",
    "NumpyOps",

    "DenseLinOp",
    "SparseLinOp",
    "BlockDiagonalLinOp",
    "SumToSingleLinOp",
    "StackedLinOp",

    "VectorSpace",
    "HermitianSpace",
    "ProductSpace",
    "Space",

    "DenseArray",
    "SparseArray",
    "ArrayLike",

    "set_context",
    "register_ops",
]

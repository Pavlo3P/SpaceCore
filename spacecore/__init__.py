__version__ = "0.1.1"


from .backend import Context, BackendOps, JaxOps, NumpyOps, jax_pytree_class
from .linop import DenseLinOp, SparseLinOp, BlockDiagonalLinOp, SumToSingleLinOp, StackedLinOp, LinOp
from .space import VectorSpace, HermitianSpace, Space, ProductSpace
from .types import DenseArray, SparseArray, ArrayLike

from ._contextual.manager import (
    set_context, get_context,
    register_ops,
    set_resolution_policy, set_dtype_resolution_policy,
    get_resolution_policy, get_dtype_resolution_policy
)

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
    "get_context",
    "register_ops",
    "set_resolution_policy",
    "set_dtype_resolution_policy",
    "get_resolution_policy",
    "get_dtype_resolution_policy",
]

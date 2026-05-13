__version__ = "0.1.3"


from .backend import Context, BackendOps, JaxOps, NumpyOps, jax_pytree_class
try:
    from .backend import TorchOps as TorchOps
except ImportError:
    pass
from .linop import DenseLinOp, SparseLinOp, BlockDiagonalLinOp, SumToSingleLinOp, StackedLinOp, LinOp
from .space import (
    BackendCheck,
    DTypeCheck,
    HermitianCheck,
    ProductComponentCheck,
    ProductSpace,
    ProductStructureCheck,
    ShapeCheck,
    Space,
    SpaceCheck,
    SpaceValidationError,
    SquareMatrixCheck,
    VectorSpace,
    HermitianSpace,
)
from .types import DenseArray, SparseArray, ArrayLike

from ._contextual import ContextBound
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

    "LinOp",
    "DenseLinOp",
    "SparseLinOp",
    "BlockDiagonalLinOp",
    "SumToSingleLinOp",
    "StackedLinOp",

    "BackendCheck",
    "DTypeCheck",
    "HermitianCheck",
    "ProductComponentCheck",
    "ProductStructureCheck",
    "ShapeCheck",
    "VectorSpace",
    "HermitianSpace",
    "ProductSpace",
    "Space",
    "SpaceCheck",
    "SpaceValidationError",
    "SquareMatrixCheck",

    "DenseArray",
    "SparseArray",
    "ArrayLike",

    "ContextBound",
    "set_context",
    "get_context",
    "register_ops",
    "set_resolution_policy",
    "set_dtype_resolution_policy",
    "get_resolution_policy",
    "get_dtype_resolution_policy",
]

if "TorchOps" in globals():
    __all__.append("TorchOps")

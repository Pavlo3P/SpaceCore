__version__ = "0.1.4"


from .backend import Context, BackendOps, JaxOps, NumpyOps, jax_pytree_class
try:
    from .backend import CuPyOps as CuPyOps
except ImportError:
    pass
try:
    from .backend import TorchOps as TorchOps
except ImportError:
    pass
from .linop import (
    BlockDiagonalLinOp,
    ComposedLinOp,
    DiagonalLinOp,
    DenseLinOp,
    IdentityLinOp,
    LinOp,
    MatrixFreeLinOp,
    ScaledLinOp,
    SparseLinOp,
    StackedLinOp,
    SumLinOp,
    SumToSingleLinOp,
    ZeroLinOp,
    make_composed,
    make_scaled,
    make_sum,
)
from .functional import (
    Functional,
    InnerProductFunctional,
    LinearFunctional,
    LinOpQuadraticForm,
    MatrixFreeLinearFunctional,
    QuadraticForm,
)
from .linalg import (
    CGResult,
    LSQRResult,
    PowerIterationResult,
    StochasticLanczosResult,
    cg,
    lsqr,
    power_iteration,
    stochastic_lanczos,
)
from .space import (
    BatchSpace,
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
    resolve_context_priority,
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
    "BlockDiagonalLinOp",
    "SumToSingleLinOp",
    "StackedLinOp",

    "Functional",
    "LinearFunctional",
    "InnerProductFunctional",
    "MatrixFreeLinearFunctional",
    "QuadraticForm",
    "LinOpQuadraticForm",

    "CGResult",
    "LSQRResult",
    "PowerIterationResult",
    "StochasticLanczosResult",
    "cg",
    "lsqr",
    "power_iteration",
    "stochastic_lanczos",

    "BackendCheck",
    "DTypeCheck",
    "HermitianCheck",
    "ProductComponentCheck",
    "ProductStructureCheck",
    "ShapeCheck",
    "BatchSpace",
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
    "resolve_context_priority",
    "register_ops",
    "set_resolution_policy",
    "set_dtype_resolution_policy",
    "get_resolution_policy",
    "get_dtype_resolution_policy",
]

if "TorchOps" in globals():
    __all__.append("TorchOps")
if "CuPyOps" in globals():
    __all__.append("CuPyOps")

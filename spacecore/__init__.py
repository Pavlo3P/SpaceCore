"""Backend-agnostic vector spaces, linear operators, and solvers."""

from importlib.metadata import version as _version

try:
    __version__ = _version("spacecore")
except Exception:
    __version__ = "0.0.0+unknown"


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
    ComposedFunctional,
    Functional,
    InnerProductFunctional,
    LinearFunctional,
    LinOpQuadraticForm,
    MatrixFreeLinearFunctional,
    QuadraticForm,
    make_functional_composed,
)
from .linalg import (
    CGResult,
    ExpmMultiplyResult,
    LanczosResult,
    LSQRResult,
    PowerIterationResult,
    cg,
    expm_multiply,
    lanczos_smallest,
    lsqr,
    power_iteration,
)
from .space import (
    BackendCheck,
    DTypeCheck,
    HermitianCheck,
    ProductComponentCheck,
    ProductSpace,
    ProductStructureCheck,
    ShapeCheck,
    InnerProduct,
    EuclideanInnerProduct,
    Space,
    SpaceCheck,
    SpaceValidationError,
    SquareMatrixCheck,
    VectorSpace,
    HermitianSpace,
)
from .types import DenseArray, SparseArray, ArrayLike

from ._checks import checked_method
from ._contextual import (
    ContextBound,
    set_context, get_context,
    resolve_context_priority,
    register_ops,
    normalize_ops, normalize_context,
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

    "ComposedFunctional",
    "Functional",
    "LinearFunctional",
    "InnerProductFunctional",
    "MatrixFreeLinearFunctional",
    "QuadraticForm",
    "LinOpQuadraticForm",
    "make_functional_composed",

    "CGResult",
    "ExpmMultiplyResult",
    "LanczosResult",
    "LSQRResult",
    "PowerIterationResult",
    "cg",
    "expm_multiply",
    "lanczos_smallest",
    "lsqr",
    "power_iteration",

    "BackendCheck",
    "DTypeCheck",
    "HermitianCheck",
    "ProductComponentCheck",
    "ProductStructureCheck",
    "ShapeCheck",
    "InnerProduct",
    "EuclideanInnerProduct",
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

    "checked_method",
    "ContextBound",
    "set_context",
    "get_context",
    "resolve_context_priority",
    "register_ops",
    "normalize_ops",
    "normalize_context",
]

if "TorchOps" in globals():
    __all__.append("TorchOps")
if "CuPyOps" in globals():
    __all__.append("CuPyOps")

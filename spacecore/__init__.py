"""Backend-agnostic vector spaces, linear operators, and solvers."""

from ._version import __version__

from .backend import Context, BackendOps, NumpyOps, jax_pytree_class
try:
    from .backend import JaxOps as JaxOps
except ImportError:
    pass
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
    CoordinateSpace,
    DenseCoordinateSpace,
    DenseVectorSpace,
    ElementwiseJordanSpace,
    EuclideanElementwiseJordanSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    ProductSpace,
    ProductSpectralDecomposition,
    ProductStructure,
    ProductStructureCheck,
    PytreeStructure,
    StackedSpace,
    ShapeCheck,
    InnerProduct,
    EuclideanInnerProduct,
    WeightedInnerProduct,
    Space,
    StarSpace,
    SpaceCheck,
    SpaceValidationError,
    SquareMatrixCheck,
    TupleStructure,
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
    "__version__",
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
    "WeightedInnerProduct",
    "VectorSpace",
    "HermitianSpace",
    "ProductSpace",
    "ProductStructure",
    "TupleStructure",
    "PytreeStructure",
    "StackedSpace",
    "CoordinateSpace",
    "InnerProductSpace",
    "StarSpace",
    "JordanAlgebraSpace",
    "EuclideanJordanAlgebraSpace",
    "DenseCoordinateSpace",
    "DenseVectorSpace",
    "ElementwiseJordanSpace",
    "EuclideanElementwiseJordanSpace",
    "ProductSpectralDecomposition",
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

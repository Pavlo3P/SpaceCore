from ._base import Functional
from ._linear import InnerProductFunctional, LinearFunctional, MatrixFreeLinearFunctional
from ._quadratic import LinOpQuadraticForm, QuadraticForm

__all__ = [
    "Functional",
    "InnerProductFunctional",
    "LinearFunctional",
    "LinOpQuadraticForm",
    "MatrixFreeLinearFunctional",
    "QuadraticForm",
]

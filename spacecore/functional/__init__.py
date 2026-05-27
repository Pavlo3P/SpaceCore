"""Scalar-valued functionals and composition helpers."""

from ._base import Functional
from ._composed import ComposedFunctional, make_functional_composed
from ._linear import InnerProductFunctional, LinearFunctional, MatrixFreeLinearFunctional
from ._quadratic import LinOpQuadraticForm, QuadraticForm

__all__ = [
    "ComposedFunctional",
    "Functional",
    "InnerProductFunctional",
    "LinearFunctional",
    "LinOpQuadraticForm",
    "MatrixFreeLinearFunctional",
    "QuadraticForm",
    "make_functional_composed",
]

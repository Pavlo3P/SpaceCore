"""Scalar-valued functionals and composition helpers.

The core functional contract (``Functional``, the linear and quadratic families,
and composition) lives in this package. The coordinate norms the prox rules use
live in the :mod:`.tools` subpackage and are re-exported here.
"""

from ._base import Functional
from ._algebra import (
    ConstantFunctional,
    ProductFunctional,
    ScaledFunctional,
    SumFunctional,
    ZeroFunctional,
    make_constant_functional,
    make_functional_product,
    make_functional_sum,
    make_scaled_functional,
    make_shifted_functional,
)
from ._composed import ComposedFunctional, make_functional_composed
from ._linear import InnerProductFunctional, LinearFunctional, MatrixFreeLinearFunctional
from ._quadratic import LinOpQuadraticForm, QuadraticForm
from ._indicator import IndicatorFunctional
from .tools import (
    L1NormFunctional,
    LpNormFunctional,
    SquaredL2NormFunctional,
)
from . import _prox  # noqa: F401 -- registers the optional closed-form rules

__all__ = [
    "ComposedFunctional",
    "ConstantFunctional",
    "Functional",
    "InnerProductFunctional",
    "IndicatorFunctional",
    "L1NormFunctional",
    "LinearFunctional",
    "LinOpQuadraticForm",
    "LpNormFunctional",
    "MatrixFreeLinearFunctional",
    "ProductFunctional",
    "QuadraticForm",
    "ScaledFunctional",
    "SquaredL2NormFunctional",
    "SumFunctional",
    "ZeroFunctional",
    "make_constant_functional",
    "make_functional_composed",
    "make_functional_product",
    "make_functional_sum",
    "make_scaled_functional",
    "make_shifted_functional",
]

"""Scalar-valued functionals and composition helpers.

The core functional contract (``Functional``, the linear and quadratic families,
and composition) lives in this package. The everyday named constructors of
ADR-019 -- least squares, coordinate and spectral norms, entropy objectives,
the Huber loss, and the proximal primitive -- live in the :mod:`.tools`
subpackage and are re-exported here for convenience.
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
from ._realified import RealifiedFunctional, realify
from ._indicator import IndicatorFunctional
from .tools import (
    SpectralFunctional,
    eigenvalue_space,
    spectralize,
    HuberFunctional,
    KLDivergenceFunctional,
    L1NormFunctional,
    LpNormFunctional,
    NegativeEntropyFunctional,
    NuclearNormFunctional,
    SquaredL2NormFunctional,
    generalized_shrinkage,
    least_squares,
    project_nonneg,
    prox_l1,
    prox_l2sq,
)
from . import _prox  # noqa: F401 -- registers the optional closed-form rules

__all__ = [
    "ComposedFunctional",
    "ConstantFunctional",
    "Functional",
    "HuberFunctional",
    "InnerProductFunctional",
    "IndicatorFunctional",
    "KLDivergenceFunctional",
    "L1NormFunctional",
    "LinearFunctional",
    "LinOpQuadraticForm",
    "LpNormFunctional",
    "MatrixFreeLinearFunctional",
    "NegativeEntropyFunctional",
    "NuclearNormFunctional",
    "ProductFunctional",
    "QuadraticForm",
    "RealifiedFunctional",
    "ScaledFunctional",
    "SpectralFunctional",
    "SquaredL2NormFunctional",
    "SumFunctional",
    "ZeroFunctional",
    "generalized_shrinkage",
    "least_squares",
    "make_constant_functional",
    "make_functional_composed",
    "make_functional_product",
    "make_functional_sum",
    "make_scaled_functional",
    "make_shifted_functional",
    "project_nonneg",
    "prox_l1",
    "prox_l2sq",
    "eigenvalue_space",
    "realify",
    "spectralize",
]

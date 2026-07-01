"""Scalar-valued functionals and composition helpers.

The core functional contract (``Functional``, the linear and quadratic families,
and composition) lives in this package. The everyday named constructors of
ADR-019 -- least squares, coordinate and spectral norms, entropy objectives,
the Huber loss, and the proximal primitive -- live in the :mod:`.tools`
subpackage and are re-exported here for convenience.
"""

from ._base import Functional
from ._algebra import ScaledFunctional, make_scaled_functional
from ._composed import ComposedFunctional, make_functional_composed
from ._linear import InnerProductFunctional, LinearFunctional, MatrixFreeLinearFunctional
from ._quadratic import LinOpQuadraticForm, QuadraticForm
from .tools import (
    HuberFunctional,
    KLDivergenceFunctional,
    L1NormFunctional,
    LpNormFunctional,
    NegativeEntropyFunctional,
    NuclearNormFunctional,
    SpectralLpNormFunctional,
    SquaredL2NormFunctional,
    generalized_shrinkage,
    least_squares,
    project_nonneg,
    prox_l1,
    prox_l2sq,
)

__all__ = [
    "ComposedFunctional",
    "Functional",
    "HuberFunctional",
    "InnerProductFunctional",
    "KLDivergenceFunctional",
    "L1NormFunctional",
    "LinearFunctional",
    "LinOpQuadraticForm",
    "LpNormFunctional",
    "MatrixFreeLinearFunctional",
    "NegativeEntropyFunctional",
    "NuclearNormFunctional",
    "QuadraticForm",
    "ScaledFunctional",
    "SpectralLpNormFunctional",
    "SquaredL2NormFunctional",
    "generalized_shrinkage",
    "least_squares",
    "make_functional_composed",
    "make_scaled_functional",
    "project_nonneg",
    "prox_l1",
    "prox_l2sq",
]

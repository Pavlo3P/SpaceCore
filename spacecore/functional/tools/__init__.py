"""Everyday functional and proximal toolbox (ADR-019).

Named constructors over the existing ``Functional`` machinery -- least squares,
coordinate and spectral norms, entropy objectives, the Huber loss, and a
closed-form metric-aware proximal primitive -- with no new core type hierarchy.
"""

from ._entropy import KLDivergenceFunctional, NegativeEntropyFunctional
from ._huber import HuberFunctional
from ._least_squares import least_squares
from ._norms import L1NormFunctional, LpNormFunctional, SquaredL2NormFunctional
from ._proximal import generalized_shrinkage, project_nonneg, prox_l1, prox_l2sq
from ._spectral import NuclearNormFunctional, SpectralLpNormFunctional

__all__ = [
    "HuberFunctional",
    "KLDivergenceFunctional",
    "L1NormFunctional",
    "LpNormFunctional",
    "NegativeEntropyFunctional",
    "NuclearNormFunctional",
    "SpectralLpNormFunctional",
    "SquaredL2NormFunctional",
    "generalized_shrinkage",
    "least_squares",
    "project_nonneg",
    "prox_l1",
    "prox_l2sq",
]

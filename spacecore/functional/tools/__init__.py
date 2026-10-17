"""Coordinate norm functionals used by the closed-form prox rules.

The named catalogue (least squares, entropy, Huber, spectral lifts, shrinkage
helpers) lives in the ``spacecore-optimize`` package.
"""

from ._norms import L1NormFunctional, LpNormFunctional, SquaredL2NormFunctional

__all__ = [
    "L1NormFunctional",
    "LpNormFunctional",
    "SquaredL2NormFunctional",
]

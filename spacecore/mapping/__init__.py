"""The lazy map algebra shared by linear operators and functionals."""

from ._base import (
    Algebraic,
    Mapping,
    Scalar,
    ScaledNode,
    SumNode,
    ZeroNode,
    make_scaled_of,
    make_sum_of,
    register,
    same_space_for_algebra,
)

__all__ = [
    "Algebraic",
    "Mapping",
    "Scalar",
    "ScaledNode",
    "SumNode",
    "ZeroNode",
    "make_scaled_of",
    "make_sum_of",
    "register",
    "same_space_for_algebra",
]

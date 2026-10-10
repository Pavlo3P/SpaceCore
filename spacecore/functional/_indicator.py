"""Indicators connect set membership, metric projection, and proximal steps."""
from __future__ import annotations

from typing import Any

from .._checks import checked_method
from .._errors import CapabilityError
from ..mapping import register
from ..sets import Set
from ._base import Functional


class IndicatorFunctional(Functional):
    """
    The extended-real indicator of a closed convex set.

    Parameters
    ----------
    constraint : Set
        A set declaring ``is_convex = True``. Its ambient space owns the metric.
    check_level : str or None, optional
        Validation policy; inherited from the set by default.

    Notes
    -----
    Values are zero on the set and positive infinity outside. A positive-step
    prox is exactly ``constraint.project(x)``. Missing membership or projection
    raises ``CapabilityError``; neither operation is synthesized numerically.
    """

    _fields = ("constraint", "check_level")
    _dynamic = ("constraint",)

    def __init__(self, constraint: Set, *, check_level=None):
        if not isinstance(constraint, Set) or not constraint.is_convex:
            raise CapabilityError("IndicatorFunctional requires a declared closed convex Set.")
        level = constraint.check_level if check_level is None else check_level
        super().__init__(constraint.ambient, check_level=level)
        self.constraint = constraint._with_check_level(self.check_level)

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return zero on the set and infinity outside; membership must exist."""
        member = self.constraint.contains(x)
        if member is None:
            raise CapabilityError("Indicator value requires set membership (contains).")
        return self.ops.where(member, self.codomain.ctx.asarray(0.0),
                              self.codomain.ctx.asarray(float("inf")))

register("prox", IndicatorFunctional, object, lambda f, geometry, x, step: f.constraint.project(x, geometry=geometry))

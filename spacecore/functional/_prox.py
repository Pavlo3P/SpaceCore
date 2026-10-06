"""Optional convex proximal rules in the domain's Hilbert geometry."""
from __future__ import annotations

import math
import numbers
from typing import Any

from .._errors import CapabilityError
from ..mapping._base import _find_rule, register
from ..space import EuclideanInnerProduct, WeightedInnerProduct
from ._algebra import ConstantFunctional, ScaledFunctional, SumFunctional, ZeroFunctional
from .tools._norms import LpNormFunctional, SquaredL2NormFunctional


def _soft_threshold(ops: Any, x: Any, threshold: Any) -> Any:
    """Shrink ``|x|`` by ``threshold`` keeping the phase (sign for real ``x``)."""
    radius = ops.abs(x)
    if ops.is_complex_dtype(ops.get_dtype(x)):
        safe = ops.where(radius > 0, radius, ops.ones_like(radius))
        phase = x / safe
    else:
        phase = ops.sign(x)
    return phase * ops.maximum(radius - threshold, ops.zeros_like(radius))


def validate_step(step: Any, owner: Any, operation: str, *, positive: bool = True) -> None:
    """Validate scalar steps eagerly; unchecked traced values remain caller-owned."""
    relation = "> 0" if positive else ">= 0"
    message = f"{operation} requires a finite real scalar step {relation}."
    if isinstance(step, numbers.Number):
        if (not isinstance(step, numbers.Real) or not math.isfinite(step)
                or (step <= 0 if positive else step < 0)):
            raise ValueError(message)
        return
    if getattr(step, "shape", None) != ():
        raise ValueError(message)
    if not owner.ops.xp.isdtype(step.dtype, ("real floating", "integral")):
        raise ValueError(message)
    if owner.check_level != "none":
        try:
            value = float(step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{operation}: use check_level='none' for traced steps.") from exc
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(message)


def apply_prox(functional: Any, x: Any, step: Any) -> Any:
    """Resolve an exact rule without coordinates, densification, or iteration."""
    validate_step(step, functional, "prox")
    X = functional.domain
    geometry = getattr(X, "geometry", None)
    rule = _find_rule("prox", functional, geometry)
    if rule is None:
        raise CapabilityError(
            f"prox: no closed-form rule for {type(functional).__name__} "
            f"on {type(X).__name__} with {type(geometry).__name__} geometry."
        )
    if functional.check_level != "none":
        X._check_member(x)
    result = rule(functional, geometry, x, step)
    if functional.check_level != "none":
        X._check_member(result)
    return result


def _squared_norm(f, geometry, x, step):
    return f.domain.scale(1 / (1 + step), x)


def _lp(f, geometry, x, step):
    if f.p != 1:
        raise CapabilityError("prox: only p=1 has a registered coordinate Lp rule.")
    threshold = step / f.ops.real(geometry.weights) if isinstance(geometry, WeightedInnerProduct) else step
    return _soft_threshold(f.ops, x, threshold)


def _constant(f, geometry, x, step):
    if isinstance(f, ConstantFunctional):
        # An infinite or non-real constant is not a proper convex objective.
        if not isinstance(f.constant, numbers.Number) and f.check_level == "none":
            if not f.ops.xp.isdtype(f.constant.dtype, ("real floating", "integral")):
                raise CapabilityError("prox requires a finite real constant.")
            return x
        try:
            value = complex(f.constant)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("prox requires a known finite real constant.") from exc
        if value.imag or not math.isfinite(value.real):
            raise CapabilityError("prox requires a finite real constant.")
    return x


def _scaled(f, geometry, x, step):
    # Positive scaling preserves convexity and multiplies the proximal step.
    validate_step(f.scalar, f, "scaled prox")
    return f.functional.prox(x, step * f.scalar)


def _sum(f, geometry, x, step):
    terms = []
    for term in f.parts:
        if isinstance(term, (ConstantFunctional, ZeroFunctional)):
            _constant(term, geometry, x, step)
        else:
            terms.append(term)
    if not terms:
        return x
    if len(terms) == 1:
        return terms[0].prox(x, step)
    raise CapabilityError("prox of a sum is unavailable; only finite constant shifts propagate.")


register("prox", SquaredL2NormFunctional, object, _squared_norm)
register("prox", LpNormFunctional, EuclideanInnerProduct, _lp)
register("prox", LpNormFunctional, WeightedInnerProduct, _lp)
register("prox", ConstantFunctional, object, _constant)
register("prox", ZeroFunctional, object, _constant)
register("prox", ScaledFunctional, object, _scaled)
register("prox", SumFunctional, object, _sum)

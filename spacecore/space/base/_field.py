"""Scalar fields, independent of finite coordinate representations."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

from ..._checks import CheckLevel
from ...contextual import Context
from ..checks import SpaceCheck
from ._inner_product import EuclideanInnerProduct, InnerProductSpace


@dataclass(frozen=True)
class ScalarShapeCheck(SpaceCheck):
    """Require a single scalar at standard validation and above."""

    name: str = "scalar_shape"
    minimum_level: ClassVar[CheckLevel] = "standard"

    def validate(self, space, x, *, allow_leading):
        # Scalars have no trailing axes; ordinary batched membership permits
        # arbitrary leading axes. Exact (N,) checks remain at the caller.
        return allow_leading or self.is_valid(space, x)

    def is_valid(self, space, x):
        return tuple(getattr(x, "shape", ())) == ()

    def error_message(self, space, x):
        return f"Expected scalar output with shape (), got {tuple(getattr(x, 'shape', ()))}."


@dataclass(frozen=True)
class RealScalarCheck(SpaceCheck):
    """Require real numeric values at cheap validation and above."""

    name: str = "real_scalar"
    minimum_level: ClassVar[CheckLevel] = "cheap"

    def is_valid(self, space, x):
        dtype = getattr(x, "dtype", None)
        if dtype is None:
            return isinstance(x, Real)
        kind = getattr(dtype, "kind", None)
        return not space.ops.is_complex_dtype(dtype) and (kind is None or kind in "biuf")

    def error_message(self, space, x):
        return f"Expected a real value, got {getattr(x, 'dtype', type(x).__name__)}."


class RealField(InnerProductSpace):
    """
    Real scalars, represented in a real floating dtype, with no coordinate or batching capability.

    Its context records the representation of values, independently of the
    storage dtype of a vector space they are computed on.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Backend context specification. Default is the ambient context.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    declared_scalar_field = "real"
    checks = (RealScalarCheck(), ScalarShapeCheck())
    geometry = EuclideanInnerProduct()

    def __init__(self, ctx=None, check_level=None):
        super().__init__(ctx, check_level=check_level)
        self._ctx = Context(self.ops, self.ops.real_dtype(self.dtype))

    def zeros(self):
        """Return scalar zero in the field's context."""
        return self.ctx.asarray(0.)

    def ones(self):
        """Return scalar one for deterministic probes."""
        return self.ctx.asarray(1.)

    def _add_core(self, a, b):
        """Return the sum of two scalars."""
        return a + b

    def _scale_core(self, s, a):
        """Return the product of two scalars."""
        return s * a

    def _inner_core(self, a, b):
        """Return ``a * b``."""
        return a * b

    def _convert(self, new_ctx):
        # The target chooses a backend and precision; the values stay real.
        return type(self)(Context(new_ctx.ops, new_ctx.ops.real_dtype(new_ctx.dtype)),
                          check_level=self.check_level)

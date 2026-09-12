"""Scalar fields, independent of finite coordinate representations."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from ..._check_policy import CheckLevel
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


class Field(InnerProductSpace):
    """Euclidean scalar field with no coordinate or batching capability.

    Field membership checks scalar shape at standard and strict levels.
    Its context records the representation of coefficients, independently of
    the storage dtype of a vector space using those coefficients.
    """

    checks = (ScalarShapeCheck(),)
    geometry = EuclideanInnerProduct()

    @property
    @abstractmethod
    def declared_scalar_field(self):
        """The real or complex coefficient field."""

    def zeros(self):
        """Return scalar zero in the field's context."""
        return self.ctx.asarray(0.)

    def ones(self):
        """Return scalar one for deterministic probes."""
        return self.ctx.asarray(1.)

    def add(self, a, b):
        """Return the sum of two scalars."""
        return a + b

    def scale(self, s, a):
        """Return the product of two scalars."""
        return s * a

    def inner(self, a, b):
        """Return ``conj(a) * b``."""
        return self.ops.conj(a) * b

    def _convert(self, new_ctx):
        # Endpoint binding chooses a backend and precision, not a different
        # mathematical field. In particular complex codomains survive binding
        # to functionals whose domains use real storage.
        dtype = (new_ctx.ops.real_dtype(new_ctx.dtype) if self.scalar_field == "real"
                 else new_ctx.ops.complex_dtype(new_ctx.dtype))
        return type(self)(Context(new_ctx.ops, dtype), check_level=self.check_level)


class RealField(Field):
    """Real scalars, represented in a real floating dtype."""

    declared_scalar_field = "real"

    def __init__(self, ctx=None, check_level=None):
        super().__init__(ctx, check_level=check_level)
        self._ctx = Context(self.ops, self.ops.real_dtype(self.dtype))


class ComplexField(Field):
    """Complex scalars; explicit contexts must use complex floating storage."""

    declared_scalar_field = "complex"

    def __init__(self, ctx=None, check_level=None):
        if ctx is None:
            from ...contextual import normalize_context

            default = normalize_context(None)
            ctx = Context(default.ops, default.ops.complex_dtype(default.dtype))
        super().__init__(ctx, check_level=check_level)

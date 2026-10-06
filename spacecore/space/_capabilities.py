"""Class-based space capability queries and operation requirements."""
from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from .._errors import CapabilityError
from .base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)

CapabilitySet = frozenset[type]
_CAP_COORDINATE = CoordinateSpace
_CAP_BATCH = CoordinateSpace  # Batching uses the coordinate surface (ADR-006).
_CAP_INNER = InnerProductSpace
_CAP_STAR = StarSpace
_CAP_JORDAN = JordanAlgebraSpace
_CAP_EUCLIDEAN_JORDAN = EuclideanJordanAlgebraSpace


def _space_capabilities(space: Space) -> CapabilitySet:
    """Return structural capabilities advertised by one leaf space."""
    # Capabilities are the capability classes the space is an instance of.
    return frozenset(
        cap
        for cap in (_CAP_COORDINATE, _CAP_INNER, _CAP_STAR, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN)
        if isinstance(space, cap)
    )


@runtime_checkable
class SupportsOnes(Protocol):
    """A space that can build a deterministic all-ones element.

    A declared capability type rather than an ad-hoc ``hasattr`` probe (ADR-005).
    It is structural because ``ones`` is optional on spaces that are otherwise
    unrelated -- ``TreeSpace`` and ``StackedSpace`` define it, and so may a
    user's non-coordinate space, which is the case that matters: without it,
    such a space can build no non-zero element at all and cannot be probed.
    """

    def ones(self) -> Any:
        """Return the all-ones element of this space."""


def probe_element(space: Space, ops: Any, dtype: Any) -> Any | None:
    """Return a deterministic non-zero element, or ``None`` if none can be built.

    Prefers the space's own ``ones``; falls back to unflattening a flat ones
    vector for a coordinate space. A bare :class:`VectorSpace` offers only
    ``zeros``, which is useless as a probe, so it yields ``None`` and the caller
    skips whatever check it wanted the element for.
    """
    # Prefer the space's own ones; a coordinate space unflattens a ones vector.
    if isinstance(space, SupportsOnes):
        return space.ones()
    if isinstance(space, CoordinateSpace):
        size = math.prod(map(int, space.shape))
        return space.unflatten(ops.ones((size,), dtype=dtype))
    return None


# Capabilities every member of a dispatch registry already has, so they carry no
# information for class selection. Both registration and lookup go through
# ``registry_key`` so that adding a capability to ``_space_capabilities`` cannot
# silently stop the registries from matching.
_NON_DISPATCH_CAPABILITIES = frozenset({_CAP_COORDINATE})


def registry_key(capabilities: CapabilitySet) -> CapabilitySet:
    """Return the capability-dispatch key for a capability set."""
    return capabilities - _NON_DISPATCH_CAPABILITIES


def require(space: Space, capability: type, operation: str) -> None:
    """Raise CapabilityError if ``space`` does not provide ``capability``."""
    if not isinstance(space, capability):
        raise CapabilityError(
            f"{operation} requires {capability.__name__}; "
            f"{type(space).__name__} lacks this capability."
        )

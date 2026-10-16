"""Build the capability-dispatch subclasses of a container space from its capability mixins.

``TreeSpace`` and ``StackedSpace`` pick a subclass from the capabilities their parts
share. Each subclass only combines capability mixins with the matching abstract
spaces, so it is generated from its capability set instead of written out.
"""
from __future__ import annotations

from typing import Any

from ..base import EuclideanJordanAlgebraSpace, InnerProductSpace, JordanAlgebraSpace, StarSpace
from .._capabilities import (
    _CAP_EUCLIDEAN_JORDAN, _CAP_INNER, _CAP_JORDAN, _CAP_STAR, CapabilitySet, registry_key,
)

# Mixin order in the bases, and the name piece each capability contributes.
_ORDER = ((_CAP_INNER, "InnerProduct", InnerProductSpace), (_CAP_STAR, "Star", StarSpace),
          (_CAP_JORDAN, "Jordan", JordanAlgebraSpace))

#: The capability sets a container dispatches on (Euclidean-Jordan implies inner and Jordan).
DISPATCH_SETS: tuple[CapabilitySet, ...] = (
    frozenset({_CAP_INNER}),
    frozenset({_CAP_STAR}),
    frozenset({_CAP_JORDAN}),
    frozenset({_CAP_INNER, _CAP_STAR}),
    frozenset({_CAP_INNER, _CAP_JORDAN}),
    frozenset({_CAP_STAR, _CAP_JORDAN}),
    frozenset({_CAP_INNER, _CAP_STAR, _CAP_JORDAN}),
    frozenset({_CAP_INNER, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}),
    frozenset({_CAP_INNER, _CAP_STAR, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}),
)


def capability_abcs(caps: CapabilitySet) -> tuple[type, ...]:
    """Return the most specific abstract spaces a subclass with ``caps`` derives from."""
    if _CAP_EUCLIDEAN_JORDAN in caps:
        return (EuclideanJordanAlgebraSpace,) + ((StarSpace,) if _CAP_STAR in caps else ())
    return tuple(abc for cap, _, abc in _ORDER if cap in caps)


def _capability_name(caps: CapabilitySet) -> str:
    # e.g. "InnerProductStar"; a lone Jordan is "JordanAlgebra", Euclidean-Jordan its own stem.
    if _CAP_EUCLIDEAN_JORDAN in caps:
        return "EuclideanJordan" + ("Star" if _CAP_STAR in caps else "Algebra")
    parts = [piece for cap, piece, _ in _ORDER if cap in caps]
    return "JordanAlgebra" if parts == ["Jordan"] else "".join(parts)


def build_dispatch_classes(
    host: type,
    mixins: dict[Any, type],
    *,
    prefix: str,
    module_globals: dict[str, Any],
    namespace: dict[str, Any] | None = None,
) -> dict[CapabilitySet, type]:
    """
    Create one subclass of ``host`` per entry of :data:`DISPATCH_SETS` and return the registry.

    Each class ``{prefix}{Capabilities}Space`` derives from the capability mixins, then
    ``host``, then :func:`capability_abcs`; it is bound in ``module_globals`` so it is
    importable by name. The registry maps ``registry_key(caps)`` to the class, plus the
    empty set to ``host``.
    """
    registry: dict[CapabilitySet, type] = {registry_key(frozenset()): host}
    for caps in DISPATCH_SETS:
        name = f"{prefix}{_capability_name(caps)}Space"
        bases = tuple(mixins[cap] for cap, _, _ in _ORDER if cap in caps) + (host,) + capability_abcs(caps)
        doc = f"{host.__name__} whose parts support: {', '.join(abc.__name__ for abc in capability_abcs(caps))}."
        attrs = {"__module__": module_globals["__name__"], "__qualname__": name, "__doc__": doc,
                 "_dispatch_caps": caps, **(namespace or {})}
        cls = type(name, bases, attrs)
        module_globals[name] = cls
        registry[registry_key(caps)] = cls
    return registry

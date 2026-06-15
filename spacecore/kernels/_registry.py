"""Registry that records every :class:`KernelSpec` in process.

The registry has two roles:

1. Make every optimized kernel discoverable by tests and benchmarks.
   ``KernelRegistry.all()`` is what the kernel-vs-generic test iterates
   over to enforce the policy.

2. Detect duplicate registrations and drift between the kernels that
   exist and the kernels documented by the policy doc.

It is intentionally a singleton: every kernel module registers at
import time and the registry must observe every spec exactly once. A
second registration with the same name is a programming error and
raises immediately.
"""
from __future__ import annotations

from typing import Iterator

from ._policy import KernelSpec


class KernelRegistry:
    """In-process collection of registered :class:`KernelSpec` objects."""

    def __init__(self) -> None:
        self._specs: dict[str, KernelSpec] = {}

    def register(self, spec: KernelSpec) -> KernelSpec:
        """Add ``spec`` to the registry.

        Returns ``spec`` so call sites can use ``spec = registry.register(
        KernelSpec(...))`` as a single assignment.
        """
        existing = self._specs.get(spec.name)
        if existing is not None and existing is not spec:
            raise ValueError(
                f"kernel name collision: {spec.name!r} already registered"
            )
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> KernelSpec:
        """Return the spec named ``name`` or raise ``KeyError``."""
        return self._specs[name]

    def all(self) -> tuple[KernelSpec, ...]:
        """Return every registered spec, in registration order."""
        return tuple(self._specs.values())

    def names(self) -> tuple[str, ...]:
        """Return every registered name, in registration order."""
        return tuple(self._specs.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self) -> Iterator[KernelSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


registry: KernelRegistry = KernelRegistry()
"""The process-wide singleton. Kernel modules register here at import."""

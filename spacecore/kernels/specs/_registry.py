"""Registry that records every :class:`KernelSpec` in process.

The registry has three roles:

1. Make every optimized kernel discoverable by tests and benchmarks.
   ``KernelRegistry.all()`` is what the kernel-vs-generic test iterates
   over to enforce the policy.

2. Detect duplicate registrations and drift between the kernels that
   exist and the kernels documented by the policy doc.

3. Index the **dispatch-eligible** specs by ``dispatch_key`` so the
   dispatcher (ADR-016) can select an applicable optimized spec by
   structural match. The index lists, per key, the eligible specs sorted
   by descending ``priority``. Two eligible specs that share a key *and* a
   priority are an ambiguous selection and raise at registration time.

It is intentionally a singleton: every kernel module registers at
import time and the registry must observe every spec exactly once. A
second registration with the same name is a programming error and
raises immediately.
"""
from __future__ import annotations

from typing import Iterator

from ._policy import KernelSpec


class DispatchAmbiguityError(ValueError):
    """Raised when two eligible specs share a ``dispatch_key`` and ``priority``.

    Selection must be deterministic: under one ``dispatch_key`` the
    dispatcher picks the highest-``priority`` applicable spec, so a tie at
    equal priority would be a silent, order-dependent choice. The registry
    rejects it at registration time instead.
    """


class KernelRegistry:
    """In-process collection of registered :class:`KernelSpec` objects."""

    def __init__(self) -> None:
        self._specs: dict[str, KernelSpec] = {}
        # dispatch_key -> eligible specs, kept sorted by descending priority.
        self._dispatch_index: dict[str, list[KernelSpec]] = {}

    def register(self, spec: KernelSpec) -> KernelSpec:
        """Add ``spec`` to the registry.

        Returns ``spec`` so call sites can use ``spec = registry.register(
        KernelSpec(...))`` as a single assignment. Re-registering the
        identical object is idempotent. A different spec under an existing
        name, or a dispatch-eligible spec colliding with another at the same
        ``(dispatch_key, priority)``, raises.
        """
        existing = self._specs.get(spec.name)
        if existing is spec:
            return spec  # idempotent: already indexed
        if existing is not None:
            raise ValueError(
                f"kernel name collision: {spec.name!r} already registered"
            )
        if spec.is_dispatch_eligible:
            self._index_for_dispatch(spec)
        self._specs[spec.name] = spec
        return spec

    def _index_for_dispatch(self, spec: KernelSpec) -> None:
        """Insert ``spec`` into the dispatch index, rejecting priority ties."""
        bucket = self._dispatch_index.setdefault(spec.dispatch_key, [])
        for other in bucket:
            if other.priority == spec.priority:
                raise DispatchAmbiguityError(
                    f"dispatch ambiguity: kernels {other.name!r} and "
                    f"{spec.name!r} share dispatch_key {spec.dispatch_key!r} "
                    f"at priority {spec.priority}; priorities must be distinct."
                )
        bucket.append(spec)
        bucket.sort(key=lambda s: s.priority, reverse=True)

    def get(self, name: str) -> KernelSpec:
        """Return the spec named ``name`` or raise ``KeyError``."""
        return self._specs[name]

    def all(self) -> tuple[KernelSpec, ...]:
        """Return every registered spec, in registration order."""
        return tuple(self._specs.values())

    def names(self) -> tuple[str, ...]:
        """Return every registered name, in registration order."""
        return tuple(self._specs.keys())

    def dispatch_candidates(self, key: str) -> tuple[KernelSpec, ...]:
        """Return dispatch-eligible specs under ``key``, highest priority first.

        Empty when no eligible spec names ``key``. This is the only ordering
        the dispatcher walks; it never consults the name map for selection.
        """
        return tuple(self._dispatch_index.get(key, ()))

    def dispatch_keys(self) -> tuple[str, ...]:
        """Return every ``dispatch_key`` that has at least one eligible spec."""
        return tuple(self._dispatch_index.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self) -> Iterator[KernelSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


registry: KernelRegistry = KernelRegistry()
"""The process-wide singleton. Kernel modules register here at import."""

"""Registry of backend implementations, keyed by backend family.

The single answer to "which ``BackendOps`` classes exist in this process". It is
seeded at construction from :func:`~spacecore.backend._optional.available_ops`
— what is *importable* in this environment — and can then be extended at runtime
via :meth:`OpsRegistry.register`, which is how a third-party backend joins
without an entry point.

Separation of concerns: this module answers *what backends exist*;
:mod:`spacecore.contextual` answers *which one is currently in effect*. Those are
different kinds of state — a registry is process-wide and shared by every thread,
while ambient policy is scoped and swappable — and holding them in one object
made the registry impossible to test without the ambient singleton. Compare
:mod:`spacecore.backend._container`, which applies the same split to pytree
registration.

Two deliberate differences from :class:`~spacecore.backend._container.PyTreeRegistry`:

* **Duplicate registration raises** here rather than being idempotent. Two
  different classes claiming ``"numpy"`` is a genuine conflict a caller should
  hear about, whereas re-wiring an already-wired ``(backend, class)`` pair is a
  no-op by construction.
* **Removal is supported** (:meth:`unregister`). ``PyTreeRegistry`` must stay
  monotonic because its two-axis back-fill depends on it; this registry is a
  plain keyed lookup with no derived state, so removing an entry is safe. It
  exists for tests and for backends registered dynamically.
"""
from __future__ import annotations

import threading
from types import MappingProxyType
from typing import Any, Mapping

from .._errors import ContextConflictError, UnknownBackendError
from ._family import BackendFamily
from ._ops import BackendOps

#: Anything that can name a backend: a family string, the enum, an ops instance
#: or class, or a Context (duck-typed via ``.ops`` to avoid importing contextual).
BackendKeyLike = Any


def backend_key(x: BackendKeyLike) -> str:
    """Normalize any backend-naming value to its lowercase family key.

    Accepts a family string (``"pytorch"`` is folded to ``"torch"``), a
    :class:`BackendFamily`, a :class:`BackendOps` instance or subclass, or any
    object exposing a ``.ops`` attribute (a ``Context``; duck-typed so this
    module need not import :mod:`spacecore.contextual`, which would be a cycle).
    """
    if isinstance(x, BackendOps):
        return backend_key(x.family)
    if isinstance(x, type) and issubclass(x, BackendOps):
        return backend_key(x._family)
    if isinstance(x, BackendFamily):
        return x.value.lower()
    if isinstance(x, str):
        key = x.lower()
        return "torch" if key == "pytorch" else key
    ops = getattr(x, "ops", None)
    if ops is not None and isinstance(ops, BackendOps):
        return backend_key(ops)
    raise TypeError(f"Unsupported backend key source: {type(x)!r}")


class OpsRegistry:
    """Process-wide map from backend family name to its ``BackendOps`` class.

    Instantiable so tests can exercise registration without touching the shared
    instance; library code uses the module-level :data:`registry`.

    Parameters
    ----------
    seed : iterable of type of BackendOps, optional
        Classes to register at construction, typically the result of
        ``available_ops()``.
    """

    def __init__(self, seed: Any = ()) -> None:
        self._ops: dict[str, type[BackendOps]] = {}
        self._lock = threading.RLock()
        for cls in seed:
            self.register(cls)

    def register(self, ops: type[BackendOps]) -> type[BackendOps]:
        """Register a backend class under its family name.

        Parameters
        ----------
        ops : type of BackendOps
            Backend class to register.

        Returns
        -------
        type of BackendOps
            The class, so this can be used as a decorator.

        Raises
        ------
        TypeError
            If ``ops`` is not a ``BackendOps`` subclass.
        ContextConflictError
            If the family is already registered — including by this same class.
            Re-registration is treated as a conflict rather than a no-op because
            it almost always means two implementations are competing for one
            family name.
        """
        if not isinstance(ops, type) or not issubclass(ops, BackendOps):
            raise TypeError(f"Expected type[BackendOps], got {type(ops)!r}")
        family = backend_key(ops)
        with self._lock:
            if family in self._ops:
                raise ContextConflictError(f"BackendOps {family} is already registered.")
            self._ops[family] = ops
        return ops

    def unregister(self, key: BackendKeyLike) -> type[BackendOps] | None:
        """Remove a family's registration and return it, or ``None`` if absent.

        Safe because this registry holds no derived state (contrast
        :class:`~spacecore.backend._container.PyTreeRegistry`, whose back-fill
        requires monotonicity). Intended for tests and dynamically scoped
        backends, not for routine use.
        """
        with self._lock:
            return self._ops.pop(backend_key(key), None)

    def get_class(self, key: BackendKeyLike) -> type[BackendOps]:
        """Return the registered class for ``key``.

        Raises
        ------
        UnknownBackendError
            If no backend is registered under that family.
        """
        family = backend_key(key)
        with self._lock:
            cls = self._ops.get(family)
            if cls is None:
                allowed = ", ".join(self._ops)
                raise UnknownBackendError(
                    f"Unknown backend: {family!r}. Expected one of: {allowed}"
                )
            return cls

    def get(self, key: BackendKeyLike) -> BackendOps:
        """Return a fresh instance of the backend registered under ``key``."""
        return self.get_class(key)()

    def classes(self) -> Mapping[str, type[BackendOps]]:
        """Return a read-only view of family name to registered class."""
        with self._lock:
            return MappingProxyType(dict(self._ops))

    def families(self) -> tuple[str, ...]:
        """Return the registered family names, in registration order."""
        with self._lock:
            return tuple(self._ops)

    def __contains__(self, key: BackendKeyLike) -> bool:
        try:
            return backend_key(key) in self._ops
        except TypeError:
            return False

    def match(self, x: Any) -> list[BackendOps]:
        """Return every registered backend that recognizes ``x`` as one of its arrays.

        The reverse lookup behind context inference. Instantiation or the
        ``is_array`` probe failing is treated as "not a match" rather than an
        error, keeping inference conservative: a backend that cannot answer
        simply does not claim the value.
        """
        matched: list[BackendOps] = []
        for cls in tuple(self.classes().values()):
            try:
                ops = cls()
                if ops.is_array(x):
                    matched.append(ops)
            except Exception:
                continue
        return matched


def _seeded_registry() -> OpsRegistry:
    """Build the process registry from whatever backends are importable."""
    from ._optional import available_ops

    return OpsRegistry(available_ops())


#: Process-wide backend registry used by :mod:`spacecore.contextual`.
registry = _seeded_registry()

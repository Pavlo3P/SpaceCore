"""
Registry wiring SpaceCore's pytree containers into each backend's tree protocol.

JAX and torch keep separate pytree registries, so every container class must be
registered with every backend. :class:`PyTreeRegistry` records both sides and wires
each new class or backend against the other, whatever the import order. It only
grows, and this module imports nothing, so it cannot form an import cycle.
"""
from __future__ import annotations

import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Self

#: Registers one class with one backend's pytree system. Supplied by that
#: backend's adapter, which owns the translation from our ``tree_flatten`` /
#: ``tree_unflatten`` pair into whatever shape the backend expects.
TreeRegistrar = Callable[[type], object]


class PyTreeRegistry:
    """Backend-keyed registry wiring container classes into backend tree protocols.

    Maintains the cross-product of registered classes and registered backends,
    guaranteeing each ``(backend, class)`` pair is wired exactly once regardless
    of the order the two arrive in.

    Instantiable so tests can exercise the wiring with a fake registrar and no
    backend installed; library code uses the module-level :data:`registry`.
    """

    def __init__(self) -> None:
        self._classes: list[type] = []
        self._seen: set[type] = set()
        self._backends: dict[str, TreeRegistrar] = {}
        self._done: set[tuple[str, type]] = set()
        # Registration normally happens during imports (already serialized), but
        # a backend may install its protocol lazily at runtime; an RLock keeps
        # the check-then-wire in _wire atomic without costing anything at import.
        self._lock = threading.RLock()

    def register_class(self, cls: type) -> None:
        """Record a container class and wire it into every backend known so far.

        Parameters
        ----------
        cls : type
            Concrete container class providing ``tree_flatten``/``tree_unflatten``.
        """
        # Remember the class once, then wire it into every backend seen so far.
        with self._lock:
            if cls not in self._seen:
                self._seen.add(cls)
                self._classes.append(cls)
            for name, registrar in self._backends.items():
                self._wire(name, registrar, cls)

    def register_backend(self, name: str, registrar: TreeRegistrar) -> None:
        """Record a backend's tree protocol and back-fill every class known so far.

        Parameters
        ----------
        name : str
            Backend family name, e.g. ``"jax"`` or ``"torch"``.
        registrar : callable
            Registers a single class with that backend's pytree system.
        """
        # Wire the new backend into every class seen so far.
        with self._lock:
            self._backends[name] = registrar
            for cls in self._classes:
                self._wire(name, registrar, cls)

    def registered_classes(self) -> tuple[type, ...]:
        """Return the container classes recorded so far, in registration order."""
        with self._lock:
            return tuple(self._classes)

    def backends(self) -> tuple[str, ...]:
        """Return the names of the backends whose tree protocol is installed."""
        with self._lock:
            return tuple(self._backends)

    def is_wired(self, name: str, cls: type) -> bool:
        """Return whether ``cls`` has been wired into backend ``name``."""
        with self._lock:
            return (name, cls) in self._done

    def _wire(self, name: str, registrar: TreeRegistrar, cls: type) -> None:
        """Register ``cls`` with backend ``name`` at most once.

        A failing registrar must never abort ``import spacecore`` — an unusable
        transform integration is a far smaller problem than an unimportable
        library, and hard-failing here would reintroduce exactly the eager-import
        fragility this design removes. The pair is marked done even on failure so
        a benign cause (the class already registered with that backend by other
        means, which several backends report by raising) cannot produce repeated
        attempts or warning spam.
        """
        # Each (backend, class) pair is wired at most once.
        key = (name, cls)
        if key in self._done:
            return
        self._done.add(key)
        # A failing registrar must not break import; the class then stays an opaque leaf.
        try:
            registrar(cls)
        except Exception as exc:  # noqa: BLE001 - see docstring: import must survive
            warnings.warn(
                f"Could not register {cls.__module__}.{cls.__qualname__} with the "
                f"{name!r} pytree protocol: {type(exc).__name__}: {exc}. "
                f"{name} transforms will treat instances as opaque leaves.",
                RuntimeWarning,
                stacklevel=3,
            )


#: Process-wide registry used by :class:`PyTreeNode` and the backend adapters.
registry = PyTreeRegistry()


class PyTreeNode(ABC):
    """Capability: a structured container exposed to backends' transform systems.

    Owns the flatten protocol the registry depends on, and auto-registers every
    concrete subclass with :data:`registry`. Compose it alongside the other
    capability mixins; it is deliberately independent of ``ContextBound`` — being
    bound to a backend context and being a flattenable container are orthogonal
    capabilities, and keeping them apart means adding a backend's tree protocol
    never reaches into the context/check-policy layer.

    Implementations must round-trip: ``cls.tree_unflatten(*reversed(obj.tree_flatten()))``
    reconstructs an equal object. Children are the dynamic, array-bearing parts a
    transform may trace or map over; ``aux`` is static metadata compared by
    equality when a backend reassembles the structure.
    """

    __slots__ = ()

    @abstractmethod
    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Return ``(children, aux)`` for this container.

        Returns
        -------
        tuple
            ``children`` — dynamic, array-bearing members, in a stable order;
            ``aux`` — static metadata sufficient, with ``children``, to rebuild.
        """

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux: Any, children: tuple[Any, ...]) -> Self:
        """Rebuild an instance from ``aux`` and ``children``.

        Parameters
        ----------
        aux : object
            The static metadata returned by :meth:`tree_flatten`.
        children : tuple
            The dynamic members, possibly transformed or traced.
        """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register each subclass that implements ``tree_flatten``, even if abstract otherwise.

        The test is whether the class can flatten, not ``inspect.isabstract``; registering a
        class that is never instantiated is harmless.
        """
        super().__init_subclass__(**kwargs)
        # Abstract bases are skipped; concrete subclasses register on definition.
        if getattr(cls.tree_flatten, "__isabstractmethod__", False):
            return
        registry.register_class(cls)

"""
Registry wiring SpaceCore's pytree containers into each backend's tree protocol.

JAX and torch keep separate pytree registries, so every container class must be
registered with every backend. :class:`PyTreeRegistry` records both sides and wires
each new class or backend against the other, whatever the import order. It only
grows, and this module imports nothing, so it cannot form an import cycle.
"""
from __future__ import annotations

import warnings
from abc import ABC
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
        # Registration happens while modules import, which the import lock serializes.
        self._classes: dict[type, None] = {}  # ordered set
        self._backends: dict[str, TreeRegistrar] = {}
        self._done: set[tuple[str, type]] = set()

    def register_class(self, cls: type) -> None:
        """Record a container class and wire it into every backend known so far.

        Parameters
        ----------
        cls : type
            Concrete container class providing ``tree_flatten``/``tree_unflatten``.
        """
        # Remember the class once, then wire it into every backend seen so far.
        self._classes.setdefault(cls)
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
        self._backends[name] = registrar
        for cls in self._classes:
            self._wire(name, registrar, cls)

    def registered_classes(self) -> tuple[type, ...]:
        """Return the container classes recorded so far, in registration order."""
        return tuple(self._classes)

    def backends(self) -> tuple[str, ...]:
        """Return the names of the backends whose tree protocol is installed."""
        return tuple(self._backends)

    def is_wired(self, name: str, cls: type) -> bool:
        """Return whether ``cls`` has been wired into backend ``name``."""
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


def field_values(obj: Any) -> dict[str, Any]:
    """Map each constructor field ``obj`` declares in ``_fields`` to its current value."""
    values = {}
    for spec in type(obj)._fields:
        param, _, attr = spec.partition(":")
        values[param] = getattr(obj, attr or param)
    return values


def rebuild_class(cls: type) -> type:
    """Return the class a conversion rebuilds ``cls`` through: the nearest one whose ``__new__`` picks the subclass.

    Unflattening keeps the exact class, so a traced rebuild re-validates its invariants;
    converting may change them (a real space becoming complex), so it goes through the factory.
    """
    for base in cls.__mro__:
        if base is PyTreeNode or base is object:
            break
        if "__new__" in vars(base):
            return base
    return cls


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

    # A subclass may declare ``_fields``, the constructor keywords that rebuild it
    # (``"param:attr"`` reads ``attr`` when the attribute is named differently), and
    # ``_dynamic``, the subset that are children rather than aux. That gives the
    # generic flatten/unflatten here and the generic convert and equality of the
    # context-bound bases; classes with derived or cached state override instead.
    # No defaults live here, so a declaration on any base in the MRO is found.

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Return ``(children, aux)`` for this container.

        Returns
        -------
        tuple
            ``children`` — dynamic, array-bearing members, in a stable order;
            ``aux`` — static metadata sufficient, with ``children``, to rebuild.
        """
        values = field_values(self)
        children = tuple(values.pop(name) for name in getattr(self, "_dynamic", ()))
        return children, tuple(values.items())

    @classmethod
    def tree_unflatten(cls, aux: Any, children: tuple[Any, ...]) -> Self:
        """Rebuild an instance from ``aux`` and ``children``.

        Parameters
        ----------
        aux : object
            The static metadata returned by :meth:`tree_flatten`.
        children : tuple
            The dynamic members, possibly transformed or traced.
        """
        return cls(**dict(aux), **dict(zip(getattr(cls, "_dynamic", ()), children)))

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # A container that cannot flatten is abstract: refuse to build it.
        if cls.tree_flatten is PyTreeNode.tree_flatten and not getattr(cls, "_fields", ()):
            raise TypeError(
                f"Can't instantiate {cls.__name__}: it declares no _fields and does not implement tree_flatten."
            )
        return super().__new__(cls)

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # Unpickling calls ``cls.__new__`` with these; a ``__new__`` that dispatches on its
        # arguments needs them, and the instance state is restored afterwards as usual.
        return (), field_values(self) if getattr(type(self), "_fields", ()) else {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register each subclass that implements ``tree_flatten``, even if abstract otherwise.

        The test is whether the class can flatten, not ``inspect.isabstract``; registering a
        class that is never instantiated is harmless.
        """
        super().__init_subclass__(**kwargs)
        # Classes that neither declare fields nor flatten themselves are skipped.
        if cls.tree_flatten is PyTreeNode.tree_flatten and not getattr(cls, "_fields", ()):
            return
        registry.register_class(cls)

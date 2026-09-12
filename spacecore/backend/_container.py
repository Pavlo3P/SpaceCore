"""Backend-neutral container protocol and the registry that wires it to backends.

A SpaceCore ``LinOp``/``Functional``/``Space`` is a *structured container*: it
decomposes into ``(children, aux)`` — dynamic array-bearing children plus static
metadata — and rebuilds from that pair. Every backend with function-transform
machinery needs to know that structure in order to see inside our objects:
``jax.jit``/``grad``/``vmap`` and ``lax.while_loop`` consult ``jax.tree_util``'s
pytree registry, ``torch.compile``/``torch.func`` consult Torch's. There is **no
shared cross-framework registry** — registering with one does nothing for the
other — so supporting N backends means N registration calls per class.

That is the bookkeeping no pytree library hands you, and it is all this module
does. Two sets grow independently and in an order nobody controls:

* **container classes**, as ``linop``/``functional``/``space`` modules import;
* **backends with a tree protocol**, as each ``BackendOps`` is loaded.

:class:`PyTreeRegistry` keeps both and, whenever either side gains a member,
wires it against everything currently on the other side (*two-axis back-fill*).
Registration therefore does not depend on import order, which is what lets a
backend loaded late still pick up classes defined early, and vice versa.

The flatten protocol itself lives on :class:`PyTreeNode`, the capability mixin
concrete containers inherit; the per-backend translation lives in each backend's
adapter.

This module sits in :mod:`spacecore.backend` — the backend-*neutral* abstraction
layer, beside the :class:`~spacecore.backend.BackendOps` contract whose
``install_pytree_protocol`` feeds the registry — and imports **nothing**, not even
from its own package. No individual backend is named here; each one reaches in
from its own adapter. So the concept stays backend-agnostic while living next to
the layer it serves, and the module is safe to import from anywhere without
risking a cycle.

Note on global state: ``registry`` is a process-wide singleton, but a *monotonic*
one — classes and backends are only ever added, wiring is idempotent, and the
final state is a function of what got imported rather than of the order it got
imported in. That grow-only property is what makes a shared instance safe here;
do not add removal or reset to it (tests construct their own instance instead).
"""
from __future__ import annotations

import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Self

#: Registers one class with one backend's pytree system. Supplied by that
#: backend's adapter, which owns the translation from our ``tree_flatten`` /
#: ``tree_unflatten`` pair into whatever shape the backend expects.
TreeRegistrar = Callable[[type], None]


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
        key = (name, cls)
        if key in self._done:
            return
        self._done.add(key)
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
        """Register subclasses that provide the flatten protocol.

        Gates on the *protocol method* rather than on :func:`inspect.isabstract`
        because the two ask different questions. ``isabstract`` asks "is anything
        still unimplemented?"; the registry only needs "is this class flattenable
        yet?". They disagree for a class that supplies ``tree_flatten`` but stays
        abstract for an unrelated reason — say a ``LinOp`` subclass that factors
        out flattening while leaving ``apply`` to its own subclasses. Such a class
        is perfectly registrable: a registrar records the type and consults the
        two protocol methods, nothing else. Registering it is also harmless, since
        it is never instantiated.

        (``inspect.isabstract`` *does* report correctly from inside
        ``__init_subclass__``. ``__abstractmethods__`` is not populated until
        ``ABCMeta.__new__`` returns, but CPython detects that and falls back to
        scanning for abstract methods manually. The choice here is about which
        question to ask, not a workaround for a broken one.)
        """
        super().__init_subclass__(**kwargs)
        if getattr(cls.tree_flatten, "__isabstractmethod__", False):
            return
        registry.register_class(cls)

from __future__ import annotations

import importlib
import warnings
from functools import lru_cache
from importlib import metadata

from ._ops import BackendOps
from .numpy import NumpyOps

# (submodule of spacecore.backend, ops attribute, dependency import name)
_OPTIONAL_OPS: tuple[tuple[str, str, str], ...] = (
    (".jax", "JaxOps", "jax"),
    (".cupy", "CuPyOps", "cupy"),
    (".torch", "TorchOps", "torch"),
)

# Packaging entry-point group that external packages use to advertise a backend.
_ENTRY_POINT_GROUP = "spacecore.backends"


def _entry_points(group: str) -> list:
    """Return the entry points in ``group``, across ``importlib.metadata`` APIs."""
    eps = metadata.entry_points()
    select = getattr(eps, "select", None)
    if select is not None:  # Python 3.10+: EntryPoints.select(group=...)
        return list(select(group=group))
    return list(eps.get(group, []))  # Python < 3.10: dict-of-lists API


def discover_entry_point_ops() -> list[type[BackendOps]]:
    """Discover backend ops classes advertised by installed packages.

    A third-party package registers a SpaceCore backend by declaring a
    ``spacecore.backends`` entry point pointing at a :class:`BackendOps`
    subclass, e.g. in its ``pyproject.toml``::

        [project.entry-points."spacecore.backends"]
        mlx = "spacecore_mlx:MLXOps"

    This makes the backend layer *open for extension* — a new backend is added by
    installing a package, not by editing SpaceCore. Each entry point is loaded
    defensively: a load failure, or a target that is not a ``BackendOps``
    subclass, is skipped with a :class:`UserWarning` rather than aborting
    ``import spacecore`` — the same non-fatal contract as :func:`import_backend`.
    """
    discovered: list[type[BackendOps]] = []
    for ep in _entry_points(_ENTRY_POINT_GROUP):
        name = getattr(ep, "name", ep)
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not be fatal
            warnings.warn(
                f"SpaceCore backend entry point {name!r} failed to load "
                f"({type(exc).__name__}: {exc}); skipping it.",
                stacklevel=2,
            )
            continue
        if not (isinstance(obj, type) and issubclass(obj, BackendOps)):
            warnings.warn(
                f"SpaceCore backend entry point {name!r} does not point at a "
                f"BackendOps subclass (got {obj!r}); skipping it.",
                stacklevel=2,
            )
            continue
        discovered.append(obj)
    return discovered


def _backend_absent(exc: ModuleNotFoundError, dep: str) -> bool:
    """Return True iff the failure is the backend dependency itself being missing."""
    return exc.name == dep


def import_backend(module: str, dep: str):
    """Import an optional backend submodule of ``spacecore.backend``.

    Parameters
    ----------
    module : str
        Submodule to import, relative to ``spacecore.backend`` (for example
        ``".jax"``).
    dep : str
        Import name of the optional third-party dependency the submodule
        needs (for example ``"jax"``).

    Returns
    -------
    module or None
        The imported module, or ``None`` when ``dep`` is not installed.

    Notes
    -----
    An *absent* dependency (``dep`` not installed) is turned into ``None``
    silently — that is the normal "backend not present" case. Any *other*
    import failure means the backend is installed but cannot load: broken
    against a mismatched runtime, shadowed by a namespace shim, a partial
    install, or a missing transitive dependency. Those are also non-fatal
    (an optional backend must never abort ``import spacecore``) but they are
    surfaced with a :class:`UserWarning` rather than hidden, so a genuinely
    broken install is diagnosable instead of invisible.
    """
    try:
        return importlib.import_module(module, package=__package__)
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and _backend_absent(exc, dep):
            return None
        warnings.warn(
            f"Optional backend {module!r} is installed but failed to import "
            f"({type(exc).__name__}: {exc}); skipping it.",
            stacklevel=2,
        )
        return None


@lru_cache(maxsize=1)
def available_ops() -> tuple[type[BackendOps], ...]:
    """Backend ops classes whose optional dependency is importable.

    NumPy is always present. Each built-in optional backend is attempted; a
    missing dependency is skipped, while any *other* import failure warns and is
    skipped. External backends advertised via ``spacecore.backends`` entry points
    (see :func:`discover_entry_point_ops`) are appended last; a built-in family
    takes precedence, so a plugin cannot shadow (for example) ``"numpy"``.

    **Cached for the process.** Discovery is not free — it attempts a real import
    per optional backend and scans packaging metadata for entry points — and it is
    called from several places during ``import spacecore``. Without the cache a
    present-but-broken backend emits its warning once per call, so one broken
    install reads as several distinct problems.

    The result is a tuple, not a list, precisely because it is shared: a cached
    mutable sequence would let one caller's edit reach every other caller.

    Availability is a property of the environment and does not change within a
    process, so the cache never needs invalidating in normal use. Registering a
    backend at runtime goes to :class:`~spacecore.backend.OpsRegistry` and does not
    pass through here. Tests that monkeypatch the import machinery or the entry
    points must call ``available_ops.cache_clear()``.

    Returns
    -------
    tuple of type of BackendOps
        Available ops classes, always starting with :class:`NumpyOps`.
    """
    ops: list[type[BackendOps]] = [NumpyOps]
    for module, attr, dep in _OPTIONAL_OPS:
        mod = import_backend(module, dep)
        if mod is None:
            continue
        cls = getattr(mod, attr, None)
        if cls is not None:
            ops.append(cls)
    seen = {o._family for o in ops}
    for cls in discover_entry_point_ops():
        if cls._family not in seen:
            ops.append(cls)
            seen.add(cls._family)
    return tuple(ops)


def available_families() -> tuple[str, ...]:
    """Backend family names whose optional dependency is importable.

    Returns
    -------
    tuple of str
        Lowercase family names (for example ``("numpy", "jax")``), always
        starting with ``"numpy"``.
    """
    return tuple(o._family for o in available_ops())


def is_available(family: str) -> bool:
    """Whether a backend ``family`` can be imported in this environment.

    Parameters
    ----------
    family : str
        Lowercase backend family name (for example ``"jax"``).

    Returns
    -------
    bool
        ``True`` if the family's ops class is importable.
    """
    return family in available_families()


def require_backend(family: str) -> type[BackendOps]:
    """Return the ops class for ``family`` or raise an actionable error.

    Parameters
    ----------
    family : str
        Lowercase backend family name (for example ``"jax"``).

    Returns
    -------
    type of BackendOps
        The ops class for ``family``.

    Raises
    ------
    ImportError
        If the family's optional dependency is not installed, with a message
        pointing at the corresponding ``spacecore`` extra.
    """
    for o in available_ops():
        if o._family == family:
            return o
    raise ImportError(
        f"The {family!r} backend requires an optional dependency that is not "
        f"installed. Install it with: pip install spacecore[{family}]"
    )

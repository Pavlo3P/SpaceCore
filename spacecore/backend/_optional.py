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
    """
    Return the backend classes advertised through ``spacecore.backends`` entry points.

    A package adds a backend with, e.g., ``mlx = "spacecore_mlx:MLXOps"`` under
    ``[project.entry-points."spacecore.backends"]``. An entry that fails to load or is
    not a ``BackendOps`` subclass is skipped with a ``UserWarning``.
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
    """
    Import an optional backend submodule, or return ``None`` if its dependency is absent.

    A missing dependency returns ``None`` silently; any other import failure also
    returns ``None``, but warns, so a broken install is visible.

    Parameters
    ----------
    module : str
        Submodule relative to ``spacecore.backend``, e.g. ``".jax"``.
    dep : str
        Import name of the third-party dependency, e.g. ``"jax"``.

    Returns
    -------
    module or None
        The module, or ``None``.
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
    """
    Return the importable backend classes, NumPy first, then built-ins, then entry-point plugins.

    A built-in family cannot be shadowed by a plugin. The result is cached for the
    process, since discovery imports modules and would otherwise repeat its warnings;
    tests that patch imports call ``available_ops.cache_clear()``.

    Returns
    -------
    tuple of type of BackendOps
        The available classes.
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
    """
    Return the backend class for ``family``, or raise ``ImportError`` naming the extra to install.

    Parameters
    ----------
    family : str
        Lowercase family name, e.g. ``"jax"``.

    Returns
    -------
    type of BackendOps
        The backend class.

    Raises
    ------
    ImportError
        If the family's dependency is not installed.
    """
    for o in available_ops():
        if o._family == family:
            return o
    raise ImportError(
        f"The {family!r} backend requires an optional dependency that is not "
        f"installed. Install it with: pip install spacecore[{family}]"
    )

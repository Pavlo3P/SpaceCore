from __future__ import annotations

import importlib
import warnings
from functools import lru_cache

from ._ops import BackendOps
from .numpy import NumpyOps

# (submodule of spacecore.backend, ops attribute, dependency import name)
_OPTIONAL_OPS: tuple[tuple[str, str, str], ...] = (
    (".jax", "JaxOps", "jax"),
    (".cupy", "CuPyOps", "cupy"),
    (".torch", "TorchOps", "torch"),
)


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
    # Dependency absent: silently None. Installed but broken: warn, then None.
    try:
        return importlib.import_module(module, package=__package__)
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == dep:
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
    Return the importable backend classes: NumPy first, then the optional built-ins.

    The result is cached for the process, since discovery imports modules and would otherwise repeat its warnings;
    tests that patch imports call ``available_ops.cache_clear()``.

    Returns
    -------
    tuple of type of BackendOps
        The available classes.
    """
    # NumPy always, then the importable optional backends.
    ops: list[type[BackendOps]] = [NumpyOps]
    for module, attr, dep in _OPTIONAL_OPS:
        mod = import_backend(module, dep)
        if mod is None:
            continue
        cls = getattr(mod, attr, None)
        if cls is not None:
            ops.append(cls)
    return tuple(ops)


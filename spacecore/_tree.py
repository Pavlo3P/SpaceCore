from __future__ import annotations

from typing import Any


def _jax_tree_util():
    """Return JAX tree utilities, or raise a clear structure-level error."""
    try:
        from jax import tree_util
    except Exception as exc:
        raise RuntimeError(
            "Pytree product structures require JAX tree utilities. "
            "Install JAX or use the default tuple ProductSpace structure."
        ) from exc
    return tree_util


def tree_flatten(x: Any) -> tuple[tuple[Any, ...], Any]:
    """Flatten a Python pytree into leaves and a structure definition."""
    leaves, treedef = _jax_tree_util().tree_flatten(x)
    return tuple(leaves), treedef


def tree_unflatten(treedef: Any, leaves: list[Any] | tuple[Any, ...]) -> Any:
    """Rebuild a Python pytree from a structure definition and leaves."""
    return _jax_tree_util().tree_unflatten(treedef, list(leaves))

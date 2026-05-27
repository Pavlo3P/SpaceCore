from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def jax_pytree_class(klass: T) -> T:
    """
    Mark a class as a JAX PyTree node, if JAX is available.

    Safe to import without JAX installed.

    Parameters
    ----------
    klass : type
        Class implementing JAX pytree methods.

    Returns
    -------
    type
        Registered class when JAX is available, otherwise ``klass`` unchanged.
    """
    try:
        from jax import tree_util
    except Exception:
        return klass
    try:
        tree_util.register_pytree_node_class(klass)
    except Exception:
        pass
    return klass

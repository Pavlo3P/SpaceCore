"""JAX backend implementation and pytree registration helpers."""

from ._pytree import jax_pytree_class as jax_pytree_class

try:
    from ._ops import JaxOps as JaxOps
except ModuleNotFoundError as exc:
    if exc.name != "jax":
        raise

__all__ = ["jax_pytree_class"]

if "JaxOps" in globals():
    __all__.append("JaxOps")

"""JAX backend implementation.

Nothing here is imported eagerly by :mod:`spacecore.backend`; the package is
reached only through ``_optional.available_ops``, whose guarded import skips a
missing dependency and warns on a broken one. Pytree registration now lives on
:meth:`JaxOps.install_pytree_protocol`, driven by the backend-neutral registry in
:mod:`spacecore.backend._container`.
"""

try:
    from ._ops import JaxOps as JaxOps
except ModuleNotFoundError as exc:
    if exc.name != "jax":
        raise

__all__ = []

if "JaxOps" in globals():
    __all__.append("JaxOps")

"""Backend contexts and operation implementations."""

from .._check_policy import CHECK_LEVELS, CheckLevel
from ._context import Context
from ._ops import BackendOps
from ._family import BackendFamily
from .jax._pytree import jax_pytree_class
from .numpy import NumpyOps

try:
    from .jax import JaxOps as JaxOps
except ImportError:
    pass
try:
    from .cupy import CuPyOps as CuPyOps
except ModuleNotFoundError as exc:
    if exc.name != "cupy":
        raise

try:
    from .torch import TorchOps as TorchOps
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

__all__ = [
    "Context",
    "CheckLevel",
    "CHECK_LEVELS",
    "BackendFamily",
    "BackendOps",
    "jax_pytree_class",
    "NumpyOps",
]

if "JaxOps" in globals():
    __all__.append("JaxOps")
if "CuPyOps" in globals():
    __all__.append("CuPyOps")
if "TorchOps" in globals():
    __all__.append("TorchOps")

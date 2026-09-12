"""Backend contexts and operation implementations."""

from .._check_policy import CHECK_LEVELS, CheckLevel
from ._container import PyTreeNode, PyTreeRegistry, registry as pytree_registry
from ._ops import BackendOps
from ._family import BackendFamily
from .numpy import NumpyOps
from ._optional import available_ops
from ._registry import OpsRegistry, backend_key, registry as ops_registry

# Bind each optional backend whose dependency is importable (JaxOps, CuPyOps,
# TorchOps). ``_optional.available_ops`` centralizes the guarded import — a
# missing dependency is skipped, a broken install warns and is skipped — so the
# per-backend try/except blocks collapse to this loop over its single source of
# truth. ``NumpyOps`` is always first and already imported above.
#
# Each bound backend also installs its tree protocol with the container registry.
# This runs while ``spacecore.backend`` is importing, i.e. before ``linop``,
# ``functional`` and ``space`` define their container classes, so those classes
# register against an already-present protocol. A backend that arrives later
# still back-fills every class defined so far — the registry is order-free by
# construction — and a backend with no transform machinery inherits a no-op.
_ops = available_ops()
for _cls in _ops:
    globals().setdefault(_cls.__name__, _cls)
    _cls.install_pytree_protocol()

__all__ = [
    "CheckLevel",
    "CHECK_LEVELS",
    "BackendFamily",
    "BackendOps",
    "NumpyOps",
    "OpsRegistry",
    "PyTreeNode",
    "PyTreeRegistry",
    "backend_key",
    "ops_registry",
    "pytree_registry",
    "available_ops",
    *(_cls.__name__ for _cls in _ops if _cls.__name__ != "NumpyOps"),
]

"""Backend/dtype context, context-bound objects, and ambient context state.

This package is the public home of :class:`Context` (which backend ops and
dtype a value is expressed in) and :class:`ContextBound` (the mixin every
space, operator, and functional derives from). Validation policy lives on the
bound object as ``check_level``, not on the context; the module-level
``get_check_level`` / ``set_check_level`` / ``use_check_level`` helpers seed
and scope that default.
"""

from ._context import Context
from ._bound import ContextBound as ContextBound
from ._state import (
    enforce_convert_policy as enforce_convert_policy,
    get_check_level as get_check_level,
    get_context as get_context,
    normalize_context as normalize_context,
    normalize_ops as normalize_ops,
    register_ops as register_ops,
    resolve_context_priority as resolve_context_priority,
    set_check_level as set_check_level,
    set_context as set_context,
    use_check_level as use_check_level,
    use_context as use_context,
)
from .._errors import (
    ContextConflictError as ContextConflictError,
    ContextError as ContextError,
    ContextInferenceError as ContextInferenceError,
    UnknownBackendError as UnknownBackendError,
)

__all__ = [
    "Context",
    "ContextBound",
    "ContextConflictError",
    "ContextError",
    "ContextInferenceError",
    "UnknownBackendError",
    "enforce_convert_policy",
    "get_check_level",
    "get_context",
    "normalize_context",
    "normalize_ops",
    "register_ops",
    "resolve_context_priority",
    "set_check_level",
    "set_context",
    "use_check_level",
    "use_context",
]

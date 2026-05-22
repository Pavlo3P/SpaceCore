from __future__ import annotations

from enum import StrEnum, auto


class ContextPolicy(StrEnum):
    """
    Policy for backend-incompatible context conversion.

    Values
    ------
    warning:
        Allow conversion to a different backend family and issue a warning.
        This is the default.
    error:
        Reject conversion to a different backend family. Use this when
        accidental backend migration should be forbidden.
    silent:
        Allow conversion to a different backend family without warning. Use
        this when automatic conversion is expected and controlled.
    """

    warning = auto()
    error = auto()
    silent = auto()


class DtypePreservePolicy(StrEnum):
    """
    Policy for dtype handling during context conversion.

    Values
    ------
    keep_native:
        Preserve the source object's dtype where possible by converting it to an
        equivalent dtype in the target backend. This is the default.
    convert:
        Use the dtype provided by the resolved target context. This prioritizes
        dtype unification under the target context.
    """

    keep_native = auto()
    convert = auto()


class ContextError(RuntimeError):
    pass


class ContextInferenceError(ContextError):
    pass


class ContextConflictError(ContextError):
    pass


class UnknownBackendError(ContextError):
    pass


class ContextConversionError(ContextError):
    pass

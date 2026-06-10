from __future__ import annotations


class ContextError(RuntimeError):
    pass


class ContextInferenceError(ContextError):
    pass


class ContextConflictError(ContextError):
    pass


class UnknownBackendError(ContextError):
    pass

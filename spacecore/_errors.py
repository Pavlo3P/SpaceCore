"""Errors shared by the backend registry and the context layer.

These live at the top level, imported by nothing, because both
:mod:`spacecore.backend` (which owns the backend-ops registry) and
:mod:`spacecore.contextual` (which owns ambient policy and re-exports them as
public API) need to raise them. Defining them in either package would force the
other to import it, and ``contextual`` already depends on ``backend``.

The ``Context*`` names are kept for backward compatibility: they were public from
``spacecore.contextual`` before the registry was extracted, and renaming them
would break callers catching them.
"""
from __future__ import annotations


class ContextError(RuntimeError):
    """Base class for context resolution and backend registration failures."""


class ContextInferenceError(ContextError):
    """A value's backend could not be inferred unambiguously."""


class ContextConflictError(ContextError):
    """A backend family is already registered under a different implementation."""


class UnknownBackendError(ContextError):
    """No backend is registered under the requested family name."""

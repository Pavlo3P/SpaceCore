from __future__ import annotations

from typing import Literal
from warnings import warn

CheckLevel = Literal["none", "cheap", "standard", "strict"]

CHECK_LEVELS: tuple[CheckLevel, ...] = ("none", "cheap", "standard", "strict")
_CHECK_LEVEL_ORDER = {level: index for index, level in enumerate(CHECK_LEVELS)}


def normalize_check_level(
    check_level: CheckLevel | str | None = None,
    *,
    enable_checks: bool | None = None,
    default: CheckLevel = "standard",
    warn_legacy: bool = False,
) -> CheckLevel:
    """Normalize the public policy and its deprecated Boolean compatibility shim."""
    if check_level is not None and enable_checks is not None:
        raise TypeError("Use either check_level or enable_checks, not both.")

    if enable_checks is not None:
        if not isinstance(enable_checks, bool):
            raise TypeError("enable_checks must be a bool or None.")
        if warn_legacy:
            warn(
                "enable_checks is deprecated; use check_level='standard' or "
                "check_level='none' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return "standard" if enable_checks else "none"

    level = default if check_level is None else check_level
    if level not in _CHECK_LEVEL_ORDER:
        allowed = ", ".join(repr(item) for item in CHECK_LEVELS)
        raise ValueError(f"Unknown check_level {level!r}. Expected one of: {allowed}.")
    return level  # type: ignore[return-value]


def check_level_at_least(actual: CheckLevel, required: CheckLevel) -> bool:
    """Return whether ``actual`` includes checks assigned to ``required``."""
    return _CHECK_LEVEL_ORDER[actual] >= _CHECK_LEVEL_ORDER[required]


def minimum_check_level(levels: tuple[CheckLevel, ...]) -> CheckLevel:
    """Return the least expensive policy shared by all supplied levels."""
    if not levels:
        return "none"
    return min(levels, key=_CHECK_LEVEL_ORDER.__getitem__)

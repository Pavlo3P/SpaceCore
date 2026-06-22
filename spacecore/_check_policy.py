from __future__ import annotations

from typing import Literal
from warnings import warn

CheckLevel = Literal["none", "cheap", "standard", "strict"]

CHECK_LEVELS: tuple[CheckLevel, ...] = ("none", "cheap", "standard", "strict")
_CHECK_LEVEL_ORDER = {level: index for index, level in enumerate(CHECK_LEVELS)}


def require_mutually_exclusive(
    name_a: str,
    value_a: object,
    name_b: str,
    value_b: object,
    *,
    verb: str = "Use",
) -> None:
    """Raise ``TypeError`` when two mutually exclusive arguments are both supplied.

    Parameters
    ----------
    name_a, name_b : str
        Public names of the two arguments, used in the error message.
    value_a, value_b : object
        Supplied values. Either being ``None`` means "not provided".
    verb : str, optional
        Leading verb of the error message (``"Use"`` or ``"Specify"``).

    Raises
    ------
    TypeError
        If both ``value_a`` and ``value_b`` are not ``None``.
    """
    if value_a is not None and value_b is not None:
        raise TypeError(f"{verb} either {name_a} or {name_b}, not both.")


def level_to_enabled(level: CheckLevel) -> bool:
    """Return the deprecated Boolean view of ``level`` (``True`` unless ``"none"``)."""
    return level != "none"


def enabled_to_level(flag: bool) -> CheckLevel:
    """Map the deprecated Boolean check flag to a :data:`CheckLevel`."""
    return "standard" if flag else "none"


def normalize_check_level(
    check_level: CheckLevel | str | None = None,
    *,
    enable_checks: bool | None = None,
    default: CheckLevel = "standard",
    warn_legacy: bool = False,
) -> CheckLevel:
    """Normalize the public policy and its deprecated Boolean compatibility shim."""
    require_mutually_exclusive("check_level", check_level, "enable_checks", enable_checks)

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
        return enabled_to_level(enable_checks)

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

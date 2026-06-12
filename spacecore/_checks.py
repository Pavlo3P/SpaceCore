from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ._check_policy import CheckLevel, normalize_check_level


def _as_positions(
    arg_pos: int | None,
    arg_positions: int | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Normalize legacy and multi-position argument selectors."""
    if arg_pos is not None and arg_positions is not None:
        raise TypeError("Use either arg_pos or arg_positions, not both.")
    if arg_positions is None:
        return (0,) if arg_pos is None else (arg_pos,)
    if isinstance(arg_positions, int):
        return (arg_positions,)
    return tuple(arg_positions)


def _space_target(self: Any, space_name: str) -> Any:
    """Return the space object named by ``space_name``."""
    return self if space_name == "self" else getattr(self, space_name)


def _object_check_level(obj: Any) -> CheckLevel:
    """Read the new policy while supporting legacy decorator users."""
    level = getattr(obj, "check_level", None)
    if level is not None:
        return normalize_check_level(level)
    return "standard" if getattr(obj, "_enable_checks", True) else "none"


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_pos: int | None = None,
    arg_positions: int | tuple[int, ...] | None = None,
    in_batched: bool = False,
    out_batched: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Build a decorator that validates method inputs and outputs against spaces.

    Parameters
    ----------
    in_space : str or None, optional
        Name of the attribute on ``self`` containing the input
        :class:`~spacecore.space.Space`, ``"self"`` to validate against the
        receiver itself, or ``None`` to skip input validation.
    out_space : str or None, optional
        Name of the attribute on ``self`` containing the output
        :class:`~spacecore.space.Space`, ``"self"`` to validate against the
        receiver itself, or ``None`` to skip output validation.
    arg_pos : int or None, optional
        Deprecated alias for a single entry in ``arg_positions``.
    arg_positions : int, tuple of int, or None, optional
        Zero-based positions in ``*args`` of input values that should be checked
        against ``in_space``. Defaults to ``(0,)``.
    in_batched : bool, optional
        Validate inputs as leading-axis batches instead of single elements.
    out_batched : bool, optional
        Validate outputs as leading-axis batches instead of single elements.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator that wraps a method and performs checks selected by
        ``self.check_level``. Legacy objects exposing only ``_enable_checks``
        continue to map ``True`` to ``"standard"`` and ``False`` to ``"none"``.
    """
    positions = _as_positions(arg_pos, arg_positions)

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            check_level = _object_check_level(self)
            if check_level != "none" and in_space is not None:
                check_target = _space_target(self, in_space)
                for pos in positions:
                    if in_batched:
                        from ._batching import _check_batched

                        _check_batched(check_target, args[pos])
                    else:
                        check_target._check_member(args[pos])

            y = method(self, *args, **kwargs)

            if check_level != "none" and out_space is not None:
                check_target = _space_target(self, out_space)
                if out_batched:
                    from ._batching import _check_batched

                    _check_batched(check_target, y)
                else:
                    check_target._check_member(y)

            return y

        return wrapper

    return decorate

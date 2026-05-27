from __future__ import annotations

from functools import wraps
from typing import Any, Callable


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


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_pos: int | None = None,
    arg_positions: int | tuple[int, ...] | None = None,
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

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator that wraps a method, performs Python-level checks when
        ``self._enable_checks`` is true, and otherwise forwards directly to the
        wrapped method.
    """
    positions = _as_positions(arg_pos, arg_positions)

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._enable_checks and in_space is not None:
                check_target = _space_target(self, in_space)
                for pos in positions:
                    check_target._check_member(args[pos])

            y = method(self, *args, **kwargs)

            if self._enable_checks and out_space is not None:
                _space_target(self, out_space)._check_member(y)

            return y

        return wrapper

    return decorate

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_pos: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Build a decorator that validates method inputs and outputs against spaces.

    Parameters
    ----------
    in_space:
        Name of the attribute on ``self`` containing the input
        :class:`~spacecore.space.Space`, or ``None`` to skip input validation.
    out_space:
        Name of the attribute on ``self`` containing the output
        :class:`~spacecore.space.Space`, or ``None`` to skip output validation.
    arg_pos:
        Zero-based position in ``*args`` of the input value that should be
        checked against ``in_space``.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator that wraps a method, performs Python-level checks when
        ``self._enable_checks`` is true, and otherwise forwards directly to the
        wrapped method.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._enable_checks and in_space is not None:
                x = args[arg_pos]
                getattr(self, in_space)._check_member(x)

            y = method(self, *args, **kwargs)

            if self._enable_checks and out_space is not None:
                getattr(self, out_space)._check_member(y)

            return y

        return wrapper

    return decorate

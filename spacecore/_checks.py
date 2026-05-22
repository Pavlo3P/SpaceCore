from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_pos: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate methods with optional Space membership checks."""

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

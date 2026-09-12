from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ._check_policy import (
    CheckLevel,
    check_level_at_least,
    enabled_to_level,
    normalize_check_level,
    require_mutually_exclusive,
)


def _as_positions(
    arg_pos: int | None,
    arg_positions: int | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Normalize legacy and multi-position argument selectors."""
    require_mutually_exclusive("arg_pos", arg_pos, "arg_positions", arg_positions)
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
    return enabled_to_level(getattr(obj, "_enable_checks", True))


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_pos: int | None = None,
    arg_positions: int | tuple[int, ...] | None = None,
    in_batched: bool = False,
    out_batched: bool = False,
    out_batched_scalar: bool = False,
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
    out_batched_scalar : bool, optional
        Validate that the output is a vector of scalars, one per input element:
        ``shape == (N,)`` for a leading batch of size ``N``. The batch size is
        read from the *input* named by ``in_space`` at the first checked
        position, so this flag requires ``in_space``. Deliberately retained:
        batched Field membership cannot enforce exactly one leading axis.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator that wraps a method and performs checks selected by
        ``self.check_level``. Legacy objects exposing only ``_enable_checks``
        continue to map ``True`` to ``"standard"`` and ``False`` to ``"none"``.

    Notes
    -----
    The fast path is the ``check_level == "none"`` case: the wrapper does
    a single attribute read and calls the underlying method directly.
    The validated path resolves ``in_space``/``out_space`` once per call
    (rather than per argument position) and uses the per-instance
    ``_check_member`` cache populated by :class:`spacecore.space.Space`.

    The scalar-output checks run at ``standard`` and above, matching the
    hand-written ``_checks_at_least("standard")`` gating they replace. Shape is
    static under ``jax.jit``, so they are safe to run while tracing.
    """
    if out_batched_scalar and in_space is None:
        # TypeError, matching ``require_mutually_exclusive``: both are misuse of
        # the decorator's argument combination, not a bad runtime value.
        raise TypeError("out_batched_scalar requires in_space to locate the batch size.")
    positions = _as_positions(arg_pos, arg_positions)
    # Resolve positions to a single-element fast path or a tuple iteration.
    single_pos = positions[0] if len(positions) == 1 else None
    # The batch size for out_batched_scalar comes from the first checked input.
    batch_pos = positions[0]

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Fast path: ``check_level == "none"`` means no validation
            # on either side. Read the attribute directly; the
            # legacy-fallback path is only taken on objects that don't
            # expose ``check_level`` (older user code).
            level = getattr(self, "check_level", None)
            if level is None:
                level = enabled_to_level(getattr(self, "_enable_checks", True))
            if level == "none":
                return method(self, *args, **kwargs)

            if in_space is not None:
                check_target = self if in_space == "self" else getattr(self, in_space)
                if in_batched:
                    from ._batching import _check_batched

                    if single_pos is not None:
                        _check_batched(check_target, args[single_pos])
                    else:
                        for pos in positions:
                            _check_batched(check_target, args[pos])
                else:
                    check_member = check_target._check_member
                    if single_pos is not None:
                        check_member(args[single_pos])
                    else:
                        for pos in positions:
                            check_member(args[pos])

            y = method(self, *args, **kwargs)

            if out_space is not None:
                out_target = self if out_space == "self" else getattr(self, out_space)
                if out_batched:
                    from ._batching import _check_batched

                    _check_batched(out_target, y)
                else:
                    out_target._check_member(y)

            if out_batched_scalar and check_level_at_least(level, "standard"):
                from ._batching import _check_scalar_shape, _leading_batch_size

                batch_target = self if in_space == "self" else getattr(self, in_space)
                _check_scalar_shape(y, (_leading_batch_size(batch_target, args[batch_pos]),))

            return y

        return wrapper

    return decorate

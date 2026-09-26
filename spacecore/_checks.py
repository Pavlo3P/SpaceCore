from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ._check_policy import check_level_at_least


def _as_positions(arg_positions: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Normalize the checked-argument selector to a tuple of positions."""
    if arg_positions is None:
        return (0,)
    if isinstance(arg_positions, int):
        return (arg_positions,)
    return tuple(arg_positions)


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_positions: int | tuple[int, ...] | None = None,
    in_batched: bool = False,
    out_batched: bool = False,
    out_batched_scalar: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Build a decorator that checks a method's inputs and outputs against spaces.

    Checks follow ``self.check_level``; at ``"none"`` the wrapper calls the method directly.

    Parameters
    ----------
    in_space : str or None, optional
        Attribute of ``self`` holding the input space, ``"self"``, or ``None`` for no input check.
    out_space : str or None, optional
        Attribute of ``self`` holding the output space, ``"self"``, or ``None`` for no output check.
    arg_positions : int, tuple of int, or None, optional
        Positions in ``*args`` checked against ``in_space``; ``(0,)`` by default.
    in_batched : bool, optional
        Check inputs as leading-axis batches.
    out_batched : bool, optional
        Check outputs as leading-axis batches.
    out_batched_scalar : bool, optional
        Check the output has shape ``(N,)`` for an input batch of size ``N``; needs ``in_space``.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        The decorator.
    """
    if out_batched_scalar and in_space is None:
        # TypeError, matching ``require_mutually_exclusive``: both are misuse of
        # the decorator's argument combination, not a bad runtime value.
        raise TypeError("out_batched_scalar requires in_space to locate the batch size.")
    # Precompute which arguments to check, so the wrapper stays cheap.
    positions = _as_positions(arg_positions)
    # Resolve positions to a single-element fast path or a tuple iteration.
    single_pos = positions[0] if len(positions) == 1 else None
    # The batch size for out_batched_scalar comes from the first checked input.
    batch_pos = positions[0]

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Fast path: ``check_level == "none"`` means no validation on either side.
            level = getattr(self, "check_level", "standard")
            if level == "none":
                return method(self, *args, **kwargs)

            # Check the selected inputs: element membership, or batch membership.
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

            # Run the method, then check its output the same way.
            y = method(self, *args, **kwargs)

            if out_space is not None:
                out_target = self if out_space == "self" else getattr(self, out_space)
                if out_batched:
                    from ._batching import _check_batched

                    _check_batched(out_target, y)
                else:
                    out_target._check_member(y)

            # A batched scalar output must have shape (N,).
            if out_batched_scalar and check_level_at_least(level, "standard"):
                from ._batching import _check_scalar_shape, _leading_batch_size

                batch_target = self if in_space == "self" else getattr(self, in_space)
                _check_scalar_shape(y, (_leading_batch_size(batch_target, args[batch_pos]),))

            return y

        return wrapper

    return decorate

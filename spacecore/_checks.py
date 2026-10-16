from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Literal, TypeVar, cast


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


def normalize_check_level(
    check_level: CheckLevel | str | None = None,
    *,
    default: CheckLevel = "standard",
) -> CheckLevel:
    """Return ``check_level`` (or ``default`` when ``None``), validated against the known levels."""
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
    return min(levels, key=_CHECK_LEVEL_ORDER.__getitem__, default="none")


# Cap on every object's check level while a backend traces (jit, vmap, compile).
# Value checks cannot run on a tracer, so the default turns checks off under tracing.
_traced_check_level: CheckLevel = "none"


def set_traced_check_level(level: CheckLevel | None) -> None:
    """
    Set the process-wide cap on check levels while a backend is tracing.

    Inside ``jax.jit``, ``jax.vmap`` or ``torch.compile`` an object checks at the lower of
    its own level and this cap. ``"cheap"`` keeps shape and dtype checks, which run once at
    trace time and cost nothing in the compiled program; ``"standard"`` and above compare
    values, which a tracer cannot do.

    Parameters
    ----------
    level : {"none", "cheap", "standard", "strict"} or None
        New cap. ``None`` restores the default, ``"none"``.
    """
    global _traced_check_level
    _traced_check_level = normalize_check_level(level, default="none")


def get_traced_check_level() -> CheckLevel:
    """Return the process-wide cap on check levels while a backend is tracing."""
    return _traced_check_level


def effective_check_level(level: CheckLevel, ops: Any) -> CheckLevel:
    """Return ``level``, capped by :func:`get_traced_check_level` when ``ops`` is tracing."""
    if level == "none" or ops is None or not ops.is_tracing():
        return level
    return minimum_check_level((level, _traced_check_level))


_Method = TypeVar("_Method", bound=Callable[..., Any])


def checked_method(
    *,
    in_space: str | None = None,
    out_space: str | None = None,
    arg_positions: int | tuple[int, ...] | None = None,
    in_batched: bool = False,
    out_batched: bool = False,
    out_batched_scalar: bool = False,
) -> Callable[[_Method], _Method]:
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
    Callable[[_Method], _Method]
        The decorator.
    """
    if out_batched_scalar and in_space is None:
        # TypeError, matching ``require_mutually_exclusive``: both are misuse of
        # the decorator's argument combination, not a bad runtime value.
        raise TypeError("out_batched_scalar requires in_space to locate the batch size.")
    # Precompute which arguments to check, so the wrapper stays cheap.
    positions = (
        (0,) if arg_positions is None
        else (arg_positions,) if isinstance(arg_positions, int)
        else tuple(arg_positions)
    )
    # Resolve positions to a single-element fast path or a tuple iteration.
    single_pos = positions[0] if len(positions) == 1 else None
    # The batch size for out_batched_scalar comes from the first checked input.
    batch_pos = positions[0]
    last_pos = max(positions)

    def decorate(method: _Method) -> _Method:
        # Parameter names after ``self``, to find a checked argument passed by keyword.
        names = tuple(inspect.signature(method).parameters)[1:]

        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Fast path: ``check_level == "none"`` means no validation on either side.
            level: CheckLevel = getattr(self, "check_level", "standard")
            if level == "none":
                return method(self, *args, **kwargs)
            level = effective_check_level(level, getattr(self, "ops", None))
            if level == "none":
                return method(self, *args, **kwargs)
            if len(args) <= last_pos:
                # A checked argument came by keyword: pass the leading ones positionally.
                lead = [name for name in names[len(args):last_pos + 1] if name in kwargs]
                args = args + tuple(kwargs.pop(name) for name in lead)

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

                batch_target = self if in_space == "self" else getattr(self, cast(str, in_space))
                _check_scalar_shape(y, (_leading_batch_size(batch_target, args[batch_pos]),))

            return y

        wrapper._spacecore_checked = True  # type: ignore[attr-defined]
        return cast(_Method, wrapper)

    return decorate

from __future__ import annotations

from typing import Any, ClassVar

from ..._contextual import ContextBound
from ...backend import Context
from ..checks import SpaceCheck, _run_checks


class Space(ContextBound):
    """
    General space capability: context ownership and membership checks.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    """

    checks: ClassVar[tuple[SpaceCheck, ...]] = ()

    def __init__(self, ctx: Context | str | None = None) -> None:
        super().__init__(ctx)

    def __eq__(self, other: Any) -> bool:
        if type(self) is type(other):
            return self.ctx == other.ctx
        return False

    def member_checks(self) -> tuple[SpaceCheck, ...]:
        checks: list[SpaceCheck] = []
        for klass in reversed(type(self).__mro__):
            checks.extend(klass.__dict__.get("checks", ()))
            local_checks = klass.__dict__.get("_local_checks")
            if local_checks is not None:
                checks.extend(local_checks(self))
        return tuple(checks)

    def _check_member(self, x: Any) -> None:
        """Raise if ``x`` is not a valid element of this space."""
        _run_checks(self, x, allow_leading=False)

    def check_member(self, x: Any) -> None:
        if self.check_level != "none":
            self._check_member(x)

    def _convert(self, new_ctx: Context) -> Space:
        raise NotImplementedError()

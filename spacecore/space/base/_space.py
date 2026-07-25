from __future__ import annotations

from typing import Any, ClassVar, Literal

from ..._check_policy import CheckLevel, check_level_at_least, normalize_check_level
from ..._lazy_algebra import is_recognizably_nonreal
from ...contextual import ContextBound
from ..._repr import field_symbol
from ...contextual import Context
from ..checks import SpaceCheck, SpaceValidationError, _run_checks


class Space(ContextBound):
    """
    General space capability: context ownership and membership checks.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this object. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Unlike the
        backend/dtype context, the validation policy is a property of the bound
        object, not of the :class:`Context`.
    """

    checks: ClassVar[tuple[SpaceCheck, ...]] = ()

    def __init__(
        self,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        super().__init__(ctx, check_level=check_level)
        # Lazy caches. Populated on first call to member_checks() /
        # _checks_for_level() and reused for every subsequent membership
        # validation. Spaces are immutable after construction (`convert`
        # returns a fresh instance), so caching the MRO-walked check list
        # is safe.
        self._cached_member_checks: tuple[SpaceCheck, ...] | None = None
        self._cached_checks_by_level: dict[CheckLevel, tuple[SpaceCheck, ...]] = {}

    #: Scalar field this space is closed under, when that is *narrower* than the
    #: field implied by the dtype. ``None`` (the default) means "same as
    #: :attr:`field`" — the common case, where entries and scalars agree.
    #: Subclasses override it with a literal; see :attr:`scalar_field`.
    declared_scalar_field: ClassVar[Literal["real", "complex"] | None] = None

    @property
    def field(self) -> Literal["real", "complex"]:
        """Return the field the *entries* live in, derived from the context dtype.

        ``Context.dtype`` controls array representation. This property records
        only whether those entries are real or complex.

        This is a fact about representation, and it is **not** always the field
        of scalars the space is a vector space over — see :attr:`scalar_field`.
        Equality and repr use this one: a real-dtype and a complex-dtype space
        of the same shape are different spaces regardless of what they are
        closed under.
        """
        return "complex" if self.ops.is_complex_dtype(self.dtype) else "real"

    @property
    def scalar_field(self) -> Literal["real", "complex"]:
        """Return the field of scalars this space is closed under.

        Defaults to :attr:`field` — entries and scalars agree for coordinate
        spaces. A subclass whose structure survives only real scaling declares
        otherwise via :attr:`declared_scalar_field`: the complex Hermitian
        matrices have complex entries but form a **real** vector space, because
        ``i·H`` is anti-Hermitian for Hermitian ``H``.

        Declared, never inferred. The dtype cannot imply this — it is a property
        of the structure the space enforces, not of how elements are stored.
        """
        return self.declared_scalar_field or self.field

    def _check_scalar(self, a: Any) -> None:
        """Raise if ``a`` is not an admissible multiplier for this space."""
        if self.scalar_field == "real" and is_recognizably_nonreal(a):
            raise SpaceValidationError(
                f"{self._space_descriptor()} is a vector space over ℝ, so scaling by "
                f"the non-real scalar {a!r} leaves the space. Multiply the underlying "
                "array directly if you intend to exit this space."
            )

    def check_scalar(self, a: Any) -> None:
        """Raise if ``a`` is not an admissible multiplier, unless checks are off.

        Only rejects a multiplier that is *provably* inadmissible, so a traced
        scalar under ``jax.jit`` always passes (see
        :func:`spacecore._lazy_algebra.is_recognizably_nonreal`).
        """
        if self.check_level != "none":
            self._check_scalar(a)

    def __eq__(self, other: Any) -> bool:
        # Tier 1: backend compatibility (type + ops family + dtype, ignoring
        # check_level). Tier 2/3: per-subclass algebraic comparison.
        if not self.same_math(other):
            return NotImplemented
        return self._eq_algebra(other)

    def _eq_algebra(self, other: Any) -> bool:
        """Algebraic-equality comparison, run only after the backend gate passes.

        Subclasses extend this via ``super()._eq_algebra(other) and ...`` so each
        adds its own structural/numerical checks. The base contributes the scalar
        field (real vs complex): two spaces over different fields are never equal.
        """
        return self.field == other.field

    def _field_symbol(self) -> str:
        """Return the scalar-field glyph (``ℝ``/``ℂ``) for this space."""
        try:
            return field_symbol(self.field)
        except Exception:
            return "?"

    def _space_descriptor(self) -> str:
        """Return a compact math descriptor used in reprs and operator arrows.

        The base descriptor is just the scalar-field glyph; coordinate,
        Hermitian, stacked, and tree spaces refine it with shape/structure.
        """
        return self._field_symbol()

    def _repr_body(self) -> str:
        return self._space_descriptor()

    def member_checks(self) -> tuple[SpaceCheck, ...]:
        """Return every ``SpaceCheck`` this instance must satisfy.

        Walks the MRO and collects ``checks`` class attributes plus any
        instance-state-driven ``_local_checks`` factories. The result is
        cached on first access because spaces are immutable post-init.
        Subclasses that depend on mutable state (none in 0.4.0) must
        clear ``self._cached_member_checks`` themselves.
        """
        cached = self._cached_member_checks
        if cached is not None:
            return cached
        checks: list[SpaceCheck] = []
        for klass in reversed(type(self).__mro__):
            checks.extend(klass.__dict__.get("checks", ()))
            local_checks = klass.__dict__.get("_local_checks")
            if local_checks is not None:
                checks.extend(local_checks(self))
        result = tuple(checks)
        self._cached_member_checks = result
        return result

    def _checks_for_level(self, level: CheckLevel) -> tuple[SpaceCheck, ...]:
        """Return the subset of ``member_checks`` that apply at ``level``.

        Cached per level. This is the hot-path validator used by
        ``_check_member`` and ``_run_checks``; precomputing it removes
        per-call MRO walks and per-check ``minimum_level`` comparisons.
        """
        cached = self._cached_checks_by_level.get(level)
        if cached is not None:
            return cached
        result = tuple(
            check for check in self.member_checks()
            if check_level_at_least(level, check.minimum_level)
        )
        self._cached_checks_by_level[level] = result
        return result

    def _check_member(self, x: Any) -> None:
        """Raise if ``x`` is not a valid element of this space."""
        # Hot path: use the per-level cached check list and run inline,
        # avoiding the generic ``_run_checks`` MRO walk.
        level = normalize_check_level(getattr(self, "check_level", "standard"))
        for check in self._checks_for_level(level):
            if not check.validate(self, x, allow_leading=False):
                raise SpaceValidationError(
                    check.validation_message(self, x, allow_leading=False)
                )

    def check_member(self, x: Any) -> None:
        if self.check_level != "none":
            self._check_member(x)

    def _convert(self, new_ctx: Context) -> Space:
        raise NotImplementedError()

    # ``_run_checks`` is kept importable for backwards compatibility but
    # callers inside SpaceCore go through ``_check_member`` (single
    # element) or ``_check_batched_member`` (batched) above.
    _legacy_run_checks = staticmethod(_run_checks)

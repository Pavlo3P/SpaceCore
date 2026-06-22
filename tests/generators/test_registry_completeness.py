"""Guard that every concrete public object class is exercised by a generated case.

This is the section-11 registry-completeness check. It introspects
``spacecore.__all__``, collects the concrete (instantiable, non-abstract) public
subclasses of :class:`~spacecore.LinOp`, :class:`~spacecore.Space`, and
:class:`~spacecore.Functional`, and asserts that each one appears as
``type(case.obj)`` for at least one generated case drawn from the case-factory
registry.

Checklist section 11:
- Every concrete public ``LinOp`` subclass is produced by ``linop_cases()``.
- Every concrete public ``Space`` subclass is produced by one of the space case
  factories (dense, vector, inner-product, tree, jordan, mixed).
- Every concrete public ``Functional`` subclass is produced by
  ``functional_cases()``.
- Abstract bases (``inspect.isabstract`` or non-empty ``__abstractmethods__``)
  are skipped automatically; intentional non-generated concrete classes live in
  an explicit, justified allowlist that is kept minimal.
"""

from __future__ import annotations

import inspect

import pytest

import spacecore as sc
from tests.generators import (
    dense_coordinate_space_cases,
    dense_vector_space_cases,
    functional_cases,
    inner_product_space_cases,
    jordan_space_cases,
    linop_cases,
    mixed_jordan_tree_case,
    tree_space_generated_cases,
    vector_space_law_cases,
)


# === Allowlist =============================================================
# Concrete public classes that are intentionally NOT exercised by a generated
# case. Each entry must be justified. Keep this minimal.
#
# - "StackedSpace": concrete but never instantiable as this exact type through
#   the public API. ``StackedSpace.__new__`` always dispatches to a
#   capability-specific private subclass (``_StackedInnerProductSpace``,
#   ``_StackedEuclideanJordanStarSpace``, ...) based on the base space's
#   capabilities, and every public coordinate base carries an inner product, so
#   the plain ``StackedSpace`` type is unreachable. Its concrete behavior is
#   covered through the dispatched subclasses exercised by the jordan cases
#   (``jordan-stacked-real``).
ALLOWLIST: frozenset[str] = frozenset({"StackedSpace"})


# === Helpers ===============================================================
def _is_abstract(cls: type) -> bool:
    return inspect.isabstract(cls) or bool(getattr(cls, "__abstractmethods__", frozenset()))


def _concrete_public_subclasses(base: type) -> dict[str, type]:
    """Map name -> class for concrete public subclasses of ``base`` in ``__all__``."""
    found: dict[str, type] = {}
    for name in sc.__all__:
        obj = getattr(sc, name, None)
        if not isinstance(obj, type):
            continue
        if obj is base or not issubclass(obj, base):
            continue
        if _is_abstract(obj):
            continue
        if name in ALLOWLIST:
            continue
        found[name] = obj
    return found


def _generated_types(cases) -> set[type]:
    return {type(case.obj) for case in cases}


def _space_case_types() -> set[type]:
    cases = [
        *dense_coordinate_space_cases(),
        *dense_vector_space_cases(),
        *inner_product_space_cases(),
        *tree_space_generated_cases(),
        *jordan_space_cases(),
        *vector_space_law_cases(),
        mixed_jordan_tree_case(),
    ]
    return _generated_types(cases)


def _missing(base: type, present: set[type]) -> list[str]:
    return sorted(
        name
        for name, cls in _concrete_public_subclasses(base).items()
        if cls not in present
    )


# === LinOp coverage ========================================================
class TestLinOpCoverage:
    def test_allowlisted_classes_are_concrete_public_subclasses(self):
        # Defensive: an allowlist entry that is no longer a concrete public LinOp
        # should be removed. Only assert for entries that are LinOp subclasses.
        for name in ALLOWLIST:
            obj = getattr(sc, name, None)
            if isinstance(obj, type) and issubclass(obj, sc.LinOp):
                assert not _is_abstract(obj), f"{name} is abstract; drop from allowlist."

    def test_every_concrete_linop_is_generated(self):
        present = _generated_types(linop_cases())
        missing = _missing(sc.LinOp, present)
        assert not missing, f"Concrete public LinOp classes without a generated case: {missing}"


# === Space coverage ========================================================
class TestSpaceCoverage:
    def test_allowlisted_classes_are_concrete_public_subclasses(self):
        for name in ALLOWLIST:
            obj = getattr(sc, name, None)
            if isinstance(obj, type) and issubclass(obj, sc.Space):
                assert not _is_abstract(obj), f"{name} is abstract; drop from allowlist."

    def test_every_concrete_space_is_generated(self):
        present = _space_case_types()
        missing = _missing(sc.Space, present)
        assert not missing, f"Concrete public Space classes without a generated case: {missing}"


# === Functional coverage ===================================================
class TestFunctionalCoverage:
    def test_allowlisted_classes_are_concrete_public_subclasses(self):
        for name in ALLOWLIST:
            obj = getattr(sc, name, None)
            if isinstance(obj, type) and issubclass(obj, sc.Functional):
                assert not _is_abstract(obj), f"{name} is abstract; drop from allowlist."

    def test_every_concrete_functional_is_generated(self):
        present = _generated_types(functional_cases())
        missing = _missing(sc.Functional, present)
        assert not missing, (
            f"Concrete public Functional classes without a generated case: {missing}"
        )


# === Allowlist hygiene =====================================================
class TestAllowlistHygiene:
    @pytest.mark.parametrize("name", sorted(ALLOWLIST))
    def test_allowlist_entries_are_public_classes(self, name):
        obj = getattr(sc, name, None)
        assert isinstance(obj, type), f"Allowlisted name {name!r} is not a public class."
        assert name in sc.__all__, f"Allowlisted name {name!r} is not in spacecore.__all__."

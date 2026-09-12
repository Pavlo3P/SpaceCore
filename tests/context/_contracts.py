"""Reusable custom assertions for ``spacecore.contextual`` contract tests.

These are xUnit *Custom Assertion* / *Test Utility Method* helpers (Meszaros,
*xUnit Test Patterns*, 2007, ch. 21): domain-specific assertions that check one
named property and report *which* property failed, so the individual tests stay
free of *Assertion Roulette* and *Test Code Duplication* (same source, test
smells).

The relation/equality contracts encoded here are the standard value-object
contracts (reflexive / symmetric / transitive, hash-consistency) that
*Design by Contract* asks us to test as a contract rather than an
implementation (Hunt & Thomas, *The Pragmatic Programmer*, tip 37).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence


def assert_equivalence_relation(
    rel: Callable[[Any, Any], bool],
    *,
    equivalent: Sequence[Any],
    unrelated: Sequence[Any],
) -> None:
    """Assert ``rel`` is an equivalence relation over the given samples.

    Parameters
    ----------
    rel:
        Binary predicate under test (e.g. ``Context.same_math``).
    equivalent:
        Two or more items that must all be pairwise related (one equivalence
        class). Their pairwise-relatedness exercises transitivity.
    unrelated:
        Items that must *not* be related to the equivalence class, so the
        relation is shown to discriminate rather than return ``True`` always.
    """
    equivalent = list(equivalent)
    unrelated = list(unrelated)
    assert len(equivalent) >= 2, "need >= 2 equivalent samples to test transitivity"
    everyone = equivalent + unrelated

    # Reflexive.
    for x in everyone:
        assert rel(x, x), f"not reflexive: rel(x, x) is False for {x!r}"

    # Symmetric.
    for i, a in enumerate(everyone):
        for b in everyone[i + 1 :]:
            assert rel(a, b) == rel(b, a), f"not symmetric for {a!r} and {b!r}"

    # Every pair inside the class is related (=> transitivity within the class).
    for i, a in enumerate(equivalent):
        for b in equivalent[i + 1 :]:
            assert rel(a, b), f"expected equivalent but rel is False: {a!r} ~ {b!r}"

    # The class is discriminated from the unrelated items.
    ref = equivalent[0]
    for u in unrelated:
        assert not rel(ref, u), f"expected unrelated but rel is True: {ref!r} ~ {u!r}"


def assert_equality_contract(
    *,
    equal: Sequence[Any],
    distinct: Sequence[Any],
) -> None:
    """Assert the Python equality/hash contract for a value object.

    Checks reflexivity, symmetry, hash-consistency for the ``equal`` group, and
    inequality against every item in ``distinct`` and against foreign types.

    Parameters
    ----------
    equal:
        Two or more objects that must compare ``==`` and hash-equal (an
        "equal but distinct instance" set — build them independently rather
        than aliasing one object).
    distinct:
        Objects that must each compare ``!=`` to the first ``equal`` item.
    """
    equal = list(equal)
    distinct = list(distinct)
    assert len(equal) >= 2, "need >= 2 independently-built equal instances"
    a = equal[0]

    # Reflexive.
    assert a == a, f"not reflexive: {a!r} != itself"

    # Symmetric equality + hash-consistency across the equal group.
    for b in equal[1:]:
        assert a == b and b == a, f"equality not symmetric for {a!r} and {b!r}"
        assert hash(a) == hash(b), f"hash inconsistent for equal {a!r} and {b!r}"

    # Inequality against genuinely different contexts, both directions.
    for d in distinct:
        assert a != d and d != a, f"expected inequality: {a!r} vs {d!r}"

    # Foreign types compare unequal, never raise.
    assert (a == object()) is False, "equality with a foreign object should be False"
    assert (a == None) is False  # noqa: E711 - explicitly exercising __eq__(None)

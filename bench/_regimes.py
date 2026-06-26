"""Configuration regimes: the dispatch / cache toggles of ADR-023.

A *regime* is a named bundle of the SpaceCore optimization toggles the
runner activates around an otherwise-identical probe, so the *same*
operation on the *same* ``(seed, size)`` inputs is timed with dispatch
off, dispatch on, and so on. The regime is part of every result row's
coordinate (:attr:`bench._probes.ProbeResult.regime`) and a group-by /
pivot axis in the dashboard.

Regimes are driven by the real dispatch API
(:func:`spacecore.kernels.dispatch_mode`). SpaceCore has *no* separate
cache on/off flag — the ADR-022 materialized-form memo warms lazily
under dispatch and lives on the operator instance — so:

* ``dispatch_cache`` is the natural warm-cache measurement that
  :func:`bench.harness.time_op` already produces by reusing one operator
  across its repeats;
* ``dispatch`` (routing with a *cold* cache each call) needs the probe to
  expose its routed operator so the runner can reset the memo per sample,
  and is therefore a follow-up (it is not swept by default yet);
* ``verify`` routes *and* checks the routed result against the generic,
  so it is slower by construction — a correctness regime, not a headline
  speed number, hence opt-in.

``space`` and ``functional`` probes have no dispatch path and only ever
run ``baseline``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

Regime = Literal["baseline", "dispatch", "dispatch_cache", "verify"]

REGIMES: tuple[Regime, ...] = ("baseline", "dispatch", "dispatch_cache", "verify")
"""Every regime the result model and dashboard understand."""

BASELINE: Regime = "baseline"

DEFAULT_DISPATCH_REGIMES: tuple[Regime, ...] = ("baseline", "dispatch_cache")
"""Regimes the runner sweeps by default for a dispatch-eligible probe.

``dispatch`` (cold cache) is deferred until probes expose their routed
operator; ``verify`` is opt-in (slower by design). Both can still be
requested explicitly.
"""

_DISPATCH_MODE: dict[Regime, str] = {
    "baseline": "off",
    "dispatch": "on",
    "dispatch_cache": "on",
    "verify": "verify",
}


def dispatch_eligible(family: str) -> bool:
    """Whether a probe family routes through ADR-016 dispatch.

    Only ``linop`` operators have a dispatch path today; ``space`` and
    ``functional`` probes run the ``baseline`` regime only.
    """
    return family == "linop"


def regimes_for(
    family: str, requested: tuple[Regime, ...] | None
) -> tuple[Regime, ...]:
    """Resolve the regimes to sweep for one probe family.

    A non-dispatch family is always ``("baseline",)``. For a dispatch
    family, ``requested=None`` selects :data:`DEFAULT_DISPATCH_REGIMES`;
    an explicit request is filtered to known regimes and always includes
    ``baseline`` (the runner needs it as the reference for
    ``regime_speedup``).
    """
    if not dispatch_eligible(family):
        return (BASELINE,)
    if requested is None:
        return DEFAULT_DISPATCH_REGIMES
    chosen = tuple(r for r in requested if r in REGIMES)
    if BASELINE not in chosen:
        chosen = (BASELINE,) + chosen
    return chosen


@contextmanager
def benchmark_regime(regime: Regime) -> Iterator[None]:
    """Activate the dispatch/cache configuration for ``regime``.

    Wraps the timed section so every ``sc`` call runs under the regime's
    dispatch mode. ``baseline`` pins dispatch ``off`` (today's default
    path), so a wired call site is result-identical to its inline path.
    """
    from spacecore.kernels import dispatch_mode

    with dispatch_mode(_DISPATCH_MODE[regime]):
        yield

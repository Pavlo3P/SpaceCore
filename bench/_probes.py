"""Operation probe types.

A *probe* describes one benchmarkable operation: a SpaceCore call paired
with a bare reference (and optionally an optimized kernel variant). The
benchmark runner times each, records correctness against the reference,
and emits a :class:`ProbeResult` row.

A probe is *not* a closed callable. It is a factory keyed by ``(seed,
problem_size, ctx)`` so the multi-seed harness can build fresh inputs
per seed without paying probe-construction overhead inside the timing
loop. The factory returns a :class:`ProbeCase` whose ``bare`` and
``sc`` closures are the actual functions the harness times.

The ``operation_family`` field is used by plots and the verdict to roll
results up by kind (``space``, ``linop``, ``functional``, ``linalg``,
``kernel``). The ``operation_name`` is the human-readable dotted name
(``space.inner``, ``linop.dense.apply``, ``linalg.cg``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from spacecore._check_policy import CheckLevel

OperationFamily = Literal["space", "linop", "functional", "linalg", "kernel"]


@dataclass(frozen=True, slots=True)
class ProbeCase:
    """One realized benchmarkable case for a single ``(seed, size)`` pair.

    Both callables must be **closed**: calling them takes no arguments
    and returns the value to be timed.
    """

    bare_label: str
    sc_label: str
    bare: Callable[[], Any]
    sc: Callable[[], Any]
    reference: Callable[[], Any] | None = None
    optimized: Callable[[], Any] | None = None
    unchecked: Callable[[], Any] | None = None
    bare_inputs: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class Probe:
    """Declarative description of one benchmarkable operation.

    Attributes
    ----------
    name
        Dotted operation name (``space.inner``, ``linop.dense.apply``).
        Unique across the registry.
    family
        Roll-up family used by plots and verdict summary.
    factory
        Callable ``(backend: str, seed: int, size: int) -> ProbeCase``
        that builds fresh inputs and returns the case. ``backend`` is
        one of the values listed in :attr:`backends`.
    sizes
        The problem sizes the harness must run this probe on.
    backends
        Backend families this probe runs on. ``("numpy",)`` is the
        default. Adding ``"jax"`` or ``"torch"`` lets the runner
        exercise the same operation on those backends; the runner
        skips a backend at runtime if its library is not installed.
    jit_compatible
        Whether the JAX backend should JIT-compile the SpaceCore call.
        When ``True`` and the backend is ``"jax"``, the runner times
        the first call separately to record JIT compilation latency.
    notes
        One-line note shown in ``python -m bench list``.
    """

    name: str
    family: OperationFamily
    factory: Callable[..., ProbeCase]
    """Either ``(backend, seed, size)`` or ``(backend, device, seed, size)``.

    The runner inspects the callable's parameter count and supplies the
    extra ``device`` argument only when the probe declares it.
    """
    sizes: tuple[int, ...]
    backends: tuple[str, ...] = ("numpy",)
    device_aware: bool = False
    """``True`` if the factory takes ``device`` and the runner should
    iterate over every available device for the backend."""
    jit_compatible: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SeedTiming:
    """Per-seed timing record for one ``(probe, size, backend)`` triple."""

    seed: int
    bare_best_ns: float
    bare_median_ns: float
    sc_best_ns: float
    sc_median_ns: float
    optimized_best_ns: float | None
    optimized_median_ns: float | None
    error_vs_reference: float
    sc_peak_bytes: int
    bare_peak_bytes: int
    compile_ns: float | None = None
    """JAX JIT first-call latency. ``None`` for non-JIT backends."""
    unchecked_best_ns: float | None = None
    unchecked_median_ns: float | None = None
    jit_best_ns: float | None = None
    jit_median_ns: float | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Aggregate result for one ``(probe, size, backend)`` triple."""

    operation_name: str
    family: OperationFamily
    size: int
    seeds: tuple[SeedTiming, ...]
    bare_median_ns: float
    sc_median_ns: float
    speedup: float
    speedup_std: float
    error_max: float
    sc_peak_bytes_median: int
    bare_peak_bytes_median: int
    backend: str = "numpy"
    """Backend family this row was measured on."""
    device: str = "cpu"
    """Device label (``cpu`` / ``cuda`` / ``mps`` / ``gpu`` / ``tpu``)."""
    check_level: CheckLevel = "cheap"
    """Validation mode used for this result row."""
    optimized_speedup: float | None = None
    compile_ns_median: float | None = None
    """Median JAX JIT first-call compile latency, or ``None``."""
    unchecked_median_ns: float | None = None
    abstraction_overhead_ns: float | None = None
    validation_overhead_ns: float | None = None
    jit_median_ns: float | None = None
    notes: str = ""


@dataclass(slots=True)
class ProbeRegistry:
    """In-memory collection of every :class:`Probe`.

    Probes register at import time. Probes are stored in registration
    order so ``python -m bench list`` shows them in a stable sequence.
    """

    _probes: dict[str, Probe] = field(default_factory=dict)

    def register(self, probe: Probe) -> Probe:
        if probe.name in self._probes:
            raise ValueError(f"duplicate probe name {probe.name!r}")
        self._probes[probe.name] = probe
        return probe

    def all(self) -> tuple[Probe, ...]:
        return tuple(self._probes.values())

    def by_family(self, family: OperationFamily) -> tuple[Probe, ...]:
        return tuple(p for p in self._probes.values() if p.family == family)

    def names(self) -> tuple[str, ...]:
        return tuple(self._probes.keys())

    def get(self, name: str) -> Probe:
        return self._probes[name]

    def filter(self, *, families: Sequence[OperationFamily] | None = None,
               name_substring: str | None = None) -> tuple[Probe, ...]:
        """Return probes matching the filter criteria."""
        out = self._probes.values()
        if families is not None:
            fams = set(families)
            out = (p for p in out if p.family in fams)
        if name_substring is not None:
            sub = name_substring
            out = (p for p in out if sub in p.name)
        return tuple(out)


registry: ProbeRegistry = ProbeRegistry()
"""The process-wide singleton."""

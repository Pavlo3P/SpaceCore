"""Macrobenchmark registry.

Every macrobenchmark is a :class:`MacroBenchmark` registered at import
time. The runner walks the registry, runs each benchmark across the
requested backends / sizes / seeds / modes, and emits
:class:`MacroResult` rows.

A macrobenchmark is *not* a single function. It is a builder that
returns a :class:`MacroPayload` keyed by ``(backend, device, size_label,
seed)``. The payload exposes one closure per mode plus an optional
``setup`` callable timed separately and the per-mode ``iterations`` and
``size_params`` metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ._schema import ModeName


@dataclass(frozen=True, slots=True)
class MacroPayload:
    """One realized benchmark case for a ``(backend, size, seed)`` triple.

    Attributes
    ----------
    iterations
        Number of inner-loop iterations the timed callable performs.
        Used to compute ``time_per_iteration_ns``.
    size_params
        Dict of parameters that describe the problem size (``n``,
        ``m``, ``d``, ``batch``, ``epsilon``, ...). Forwarded verbatim
        into :attr:`MacroResult.size_params`.
    setup
        Optional setup callable. Timed separately from the inner loop;
        the runner reports its wall-clock as
        :attr:`MacroResult.setup_time_ns`.
    mode_callables
        Dict ``{mode: callable}`` returning the per-mode inner-loop
        result. Modes not implemented are simply absent from the dict;
        the runner skips them.
    reference_metric_extractor
        Optional callable ``(mode_result) -> dict[str, float]`` that
        extracts numerical state for ``error_vs_bare`` / ``residual`` /
        ``objective`` / ``extra`` metrics. The runner uses the
        ``bare`` mode's extraction as the canonical reference.
    extra_metric_extractor
        Optional callable ``(mode_result) -> dict[str, Any]`` whose
        output is merged into :attr:`MacroResult.extra`.
    """

    iterations: int
    size_params: dict[str, Any]
    mode_callables: dict[ModeName, Callable[[], Any]]
    setup: Callable[[], None] | None = None
    reference_metric_extractor: Callable[[Any], dict[str, float]] | None = None
    extra_metric_extractor: Callable[[Any], dict[str, Any]] | None = None
    throughput_per_iteration: float | None = None
    """Optional fixed work-per-iteration unit used to compute
    :attr:`MacroResult.throughput` (work / s). For example, a CG
    iteration that does one matvec sets this to the matvec count."""


@dataclass(frozen=True, slots=True)
class MacroBenchmark:
    """Declarative description of one macrobenchmark.

    Attributes
    ----------
    name
        Stable dotted name (``cg_poisson``, ``pdhg.l1_lsq``, ...).
    workload
        Free-form human-readable description.
    sizes
        Mapping ``{size_label: dict[str, Any]}`` enumerating the problem
        sizes. The dict values are passed to ``factory(size_params=...)``.
    backends
        Backend list the benchmark supports.
    quick_sizes
        Optional subset of size labels to run in ``--quick`` mode.
        Defaults to the first size.
    factory
        ``factory(backend, device, seed, size_params) -> MacroPayload``.
    """

    name: str
    workload: str
    sizes: dict[str, dict[str, Any]]
    backends: tuple[str, ...]
    factory: Callable[..., MacroPayload]
    quick_sizes: tuple[str, ...] = ()
    notes: str = ""

    def quick_size_labels(self) -> tuple[str, ...]:
        return self.quick_sizes or (next(iter(self.sizes)),)


@dataclass(slots=True)
class MacroRegistry:
    """In-memory collection of every :class:`MacroBenchmark`."""

    _benchmarks: dict[str, MacroBenchmark] = field(default_factory=dict)

    def register(self, benchmark: MacroBenchmark) -> MacroBenchmark:
        if benchmark.name in self._benchmarks:
            raise ValueError(f"duplicate macrobenchmark {benchmark.name!r}")
        self._benchmarks[benchmark.name] = benchmark
        return benchmark

    def all(self) -> tuple[MacroBenchmark, ...]:
        return tuple(self._benchmarks.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._benchmarks.keys())

    def get(self, name: str) -> MacroBenchmark:
        return self._benchmarks[name]

    def filter(
        self,
        *,
        names: Sequence[str] | None = None,
        name_substring: str | None = None,
    ) -> tuple[MacroBenchmark, ...]:
        out: Iterable[MacroBenchmark] = self._benchmarks.values()
        if names is not None:
            wanted = set(names)
            out = (b for b in out if b.name in wanted)
        if name_substring is not None:
            sub = name_substring
            out = (b for b in out if sub in b.name)
        return tuple(out)


registry: MacroRegistry = MacroRegistry()

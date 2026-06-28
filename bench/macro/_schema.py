"""JSON result schema for macrobenchmarks.

Every macrobenchmark emits :class:`MacroResult` rows that match the
public contract documented in ``docs/benchmarks.md``. The fields listed
here are the canonical names; the dashboard and aggregation layers key
off them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ModeName = Literal[
    "bare",
    "spacecore_public_none",
    "spacecore_public_cheap",
    "spacecore_lowered",
]
"""The four run modes every macrobenchmark must support
(``spacecore_lowered`` may equal ``spacecore_public_none`` when no
distinct lowered kernel exists)."""

RUN_MODES: tuple[ModeName, ...] = (
    "bare",
    "spacecore_public_none",
    "spacecore_public_cheap",
    "spacecore_lowered",
)


# CheckLevel for each mode. ``None`` for modes that don't go through a
# SpaceCore check pipeline.
MODE_CHECK_LEVEL: dict[ModeName, str | None] = {
    "bare": None,
    "spacecore_public_none": "none",
    "spacecore_public_cheap": "cheap",
    "spacecore_lowered": "none",
}


@dataclass(frozen=True, slots=True)
class MacroResult:
    """One row of macrobenchmark output.

    The field set matches the JSON schema documented in
    ``docs/benchmarks.md``. Required fields have no default; optional
    metrics default to ``None`` and are emitted as JSON ``null``.
    """

    benchmark_name: str
    workload: str
    backend: str
    device: str
    mode: ModeName
    check_level: str | None
    size_label: str
    size_params: dict[str, Any]
    seed: int
    iterations: int
    setup_time_ns: float
    run_time_ns: float
    family: str = "macro"
    compile_time_ns: float | None = None
    time_per_iteration_ns: float | None = None
    throughput: float | None = None
    memory_peak_bytes: int | None = None
    error_vs_bare: float | None = None
    residual: float | None = None
    objective: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form benchmark-specific metrics
    (eigenvalue estimates, orthogonality loss, primal/dual residuals,
    apply/rapply timing fractions, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "family": self.family,
            "workload": self.workload,
            "backend": self.backend,
            "device": self.device,
            "mode": self.mode,
            "check_level": self.check_level,
            "size_label": self.size_label,
            "size_params": dict(self.size_params),
            "seed": self.seed,
            "iterations": self.iterations,
            "setup_time_ns": self.setup_time_ns,
            "compile_time_ns": self.compile_time_ns,
            "run_time_ns": self.run_time_ns,
            "time_per_iteration_ns": self.time_per_iteration_ns,
            "throughput": self.throughput,
            "memory_peak_bytes": self.memory_peak_bytes,
            "error_vs_bare": self.error_vs_bare,
            "residual": self.residual,
            "objective": self.objective,
            "notes": self.notes,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MacroResult:
        """Reconstruct a :class:`MacroResult` from a dict."""
        return cls(
            benchmark_name=payload["benchmark_name"],
            workload=payload["workload"],
            backend=payload["backend"],
            device=payload["device"],
            mode=payload["mode"],
            check_level=payload.get("check_level"),
            size_label=payload["size_label"],
            size_params=dict(payload.get("size_params", {})),
            seed=int(payload["seed"]),
            iterations=int(payload["iterations"]),
            setup_time_ns=float(payload["setup_time_ns"]),
            run_time_ns=float(payload["run_time_ns"]),
            family=payload.get("family", "macro"),
            compile_time_ns=payload.get("compile_time_ns"),
            time_per_iteration_ns=payload.get("time_per_iteration_ns"),
            throughput=payload.get("throughput"),
            memory_peak_bytes=payload.get("memory_peak_bytes"),
            error_vs_bare=payload.get("error_vs_bare"),
            residual=payload.get("residual"),
            objective=payload.get("objective"),
            notes=payload.get("notes", ""),
            extra=dict(payload.get("extra", {})),
        )


REQUIRED_FIELDS: tuple[str, ...] = (
    "benchmark_name",
    "family",
    "workload",
    "backend",
    "device",
    "mode",
    "check_level",
    "size_label",
    "size_params",
    "seed",
    "iterations",
    "setup_time_ns",
    "compile_time_ns",
    "run_time_ns",
    "time_per_iteration_ns",
    "throughput",
    "memory_peak_bytes",
    "error_vs_bare",
    "residual",
    "objective",
    "notes",
)
"""Fields that must be present in every emitted dict."""


def validate(payload: dict[str, Any]) -> None:
    """Raise :class:`ValueError` if ``payload`` is missing a required field."""
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"MacroResult missing required fields: {missing}")
    if payload["family"] != "macro":
        raise ValueError(
            f"MacroResult.family must be 'macro', got {payload['family']!r}"
        )
    if payload["mode"] not in RUN_MODES:
        raise ValueError(
            f"MacroResult.mode must be one of {RUN_MODES}, got {payload['mode']!r}"
        )

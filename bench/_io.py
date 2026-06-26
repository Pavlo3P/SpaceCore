"""Persistence helpers for :class:`bench._probes.ProbeResult` records.

Results are saved as JSON: one top-level dict with ``results`` (list of
flattened records) and ``meta`` (Python / NumPy / SpaceCore versions
recorded at run time). Loading reconstructs :class:`ProbeResult` objects
so the verdict and plots can run against historical artifacts.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ._probes import ProbeResult, SeedTiming


def _metadata() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for pkg in ("spacecore", "numpy", "scipy", "jax", "torch", "cupy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "versions": versions,
    }


def to_dict(results: Iterable[ProbeResult]) -> dict[str, Any]:
    """Serialize a result set to a JSON-ready dict."""
    return {
        "meta": _metadata(),
        "results": [asdict(r) for r in results],
    }


def save(results: Iterable[ProbeResult], path: str | Path) -> Path:
    """Write ``results`` to ``path`` as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(results), indent=2, allow_nan=False))
    return p


def load(path: str | Path) -> list[ProbeResult]:
    """Reconstruct :class:`ProbeResult` objects from a saved artifact."""
    raw = json.loads(Path(path).read_text())
    out: list[ProbeResult] = []
    for row in raw["results"]:
        seeds = tuple(
            SeedTiming(
                seed=s["seed"],
                bare_best_ns=s["bare_best_ns"],
                bare_median_ns=s["bare_median_ns"],
                sc_best_ns=s["sc_best_ns"],
                sc_median_ns=s["sc_median_ns"],
                optimized_best_ns=s.get("optimized_best_ns"),
                optimized_median_ns=s.get("optimized_median_ns"),
                error_vs_reference=s["error_vs_reference"],
                sc_peak_bytes=s["sc_peak_bytes"],
                bare_peak_bytes=s["bare_peak_bytes"],
                compile_ns=s.get("compile_ns"),
                unchecked_best_ns=s.get("unchecked_best_ns"),
                unchecked_median_ns=s.get("unchecked_median_ns"),
                jit_best_ns=s.get("jit_best_ns"),
                jit_median_ns=s.get("jit_median_ns"),
            )
            for s in row["seeds"]
        )
        out.append(
            ProbeResult(
                operation_name=row["operation_name"],
                family=row["family"],
                size=row["size"],
                seeds=seeds,
                bare_median_ns=row["bare_median_ns"],
                sc_median_ns=row["sc_median_ns"],
                speedup=row["speedup"],
                speedup_std=row["speedup_std"],
                error_max=row["error_max"],
                sc_peak_bytes_median=row["sc_peak_bytes_median"],
                bare_peak_bytes_median=row["bare_peak_bytes_median"],
                optimized_speedup=row.get("optimized_speedup"),
                compile_ns_median=row.get("compile_ns_median"),
                unchecked_median_ns=row.get("unchecked_median_ns"),
                abstraction_overhead_ns=row.get("abstraction_overhead_ns"),
                validation_overhead_ns=row.get("validation_overhead_ns"),
                jit_median_ns=row.get("jit_median_ns"),
                backend=row.get("backend", "numpy"),
                device=row.get("device", "cpu"),
                check_level=row.get("check_level", "cheap"),
                notes=row.get("notes", ""),
            )
        )
    return out

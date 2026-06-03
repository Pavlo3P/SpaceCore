from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .harness import time_op


LayerFn = tuple[str, Callable[[], Any]]


def _time(fn: Callable[[], Any], repeat: int, number: int, warmup: int) -> dict[str, float]:
    return time_op(fn, repeat=repeat, number=number, warmup=warmup)


def _primitive(fn: Callable[[], Any], repeat: int, number: int, warmup: int) -> float:
    return _time(fn, repeat=repeat, number=number, warmup=warmup)["best_ns"]


def measure_ladder(
    layers: list[LayerFn],
    *,
    repeat: int,
    number: int,
    warmup: int,
    baseline_layer: str | None = None,
    full_layer: str | None = None,
    total_overhead_ns: float | None = None,
    primitives: dict[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    """Time an incremental reconstruction ladder and standalone primitives."""
    timings = []
    for name, fn in layers:
        timing = _time(fn, repeat=repeat, number=number, warmup=warmup)
        timings.append({"layer": name, "time_ns": timing["best_ns"], "median_ns": timing["median_ns"]})

    previous = None
    for entry in timings:
        if previous is None:
            entry["delta_ns"] = None
            entry["delta_status"] = "baseline"
        else:
            delta = entry["time_ns"] - previous["time_ns"]
            jitter = max(entry["median_ns"] - entry["time_ns"], 0.0) + max(previous["median_ns"] - previous["time_ns"], 0.0)
            entry["delta_ns"] = delta
            if delta < -max(jitter, 50.0):
                entry["delta_status"] = "noise_negative"
            elif abs(delta) <= max(jitter, 50.0):
                entry["delta_status"] = "below_noise"
            else:
                entry["delta_status"] = "measured"
        previous = entry

    baseline_name = baseline_layer or layers[0][0]
    full_name = full_layer or layers[-1][0]
    by_name = {entry["layer"]: entry for entry in timings}
    baseline_time = by_name[baseline_name]["time_ns"]
    full_time = by_name[full_name]["time_ns"]
    ladder_sum = full_time - baseline_time

    measured_primitives: dict[str, float | None] = {}
    if primitives:
        for name, fn in primitives.items():
            try:
                measured_primitives[name] = _primitive(fn, repeat=repeat, number=number, warmup=warmup)
            except Exception:
                measured_primitives[name] = None

    return {
        "ladder": timings,
        "baseline_layer": baseline_name,
        "full_layer": full_name,
        "measured_primitives": measured_primitives,
        "ladder_sum_ns": ladder_sum,
        "ladder_vs_total_gap_ns": None if total_overhead_ns is None else ladder_sum - total_overhead_ns,
    }


def _check_apply(domain, codomain, core: Callable[[Any], Any], x: Any) -> Any:
    domain._check_member(x)
    y = core(x)
    codomain._check_member(y)
    return y


def _check_rapply(domain, codomain, core: Callable[[Any], Any], y: Any) -> Any:
    codomain._check_member(y)
    x = core(y)
    domain._check_member(x)
    return x


def _check_batched_apply(check_in, check_out, core: Callable[[Any], Any], xs: Any) -> Any:
    check_in(xs)
    ys = core(xs)
    check_out(ys)
    return ys


def linop_breakdown(
    op: Any,
    operation: str,
    bare_fn: Callable[[], Any],
    arg: Any,
    *,
    repeat: int,
    number: int,
    warmup: int,
    total_overhead_ns: float,
) -> dict[str, Any]:
    """Measured ladder for matrix-backed LinOp operations."""
    from spacecore._batching import _check_batched

    core_name = f"_{operation}_core"
    core = getattr(op, core_name, None)
    public = getattr(op, operation)
    if core is None:
        return generic_breakdown(
            bare_fn,
            public,
            repeat=repeat,
            number=number,
            warmup=warmup,
            total_overhead_ns=total_overhead_ns,
        )

    if operation == "apply":
        sample_out = core(arg)
        layers = [
            ("bare_backend", bare_fn),
            ("core_dispatch", lambda: core(arg)),
            ("public_nocheck", lambda: public(arg)),
            ("checks", lambda: _check_apply(op.domain, op.codomain, core, arg)),
        ]
        primitives = {
            "check_member_in_ns": lambda: op.domain._check_member(arg),
            "check_member_out_ns": lambda: op.codomain._check_member(sample_out),
        }
    elif operation == "rapply":
        sample_out = core(arg)
        if not op.domain.is_euclidean or not op.codomain.is_euclidean:
            euclidean = getattr(op, "_euclidean_rapply_core", core)
            sample_dual = euclidean(op.codomain.riesz(arg))
            layers = [
                ("coordinate_adjoint", lambda: euclidean(arg)),
                ("codomain_riesz", lambda: euclidean(op.codomain.riesz(arg))),
                ("metric_bare", bare_fn),
                ("core_dispatch", lambda: core(arg)),
                ("public_nocheck", lambda: public(arg)),
                ("checks", lambda: _check_rapply(op.domain, op.codomain, core, arg)),
            ]
            primitives = {
                "check_member_in_ns": lambda: op.codomain._check_member(arg),
                "check_member_out_ns": lambda: op.domain._check_member(sample_out),
                "riesz_codomain_ns": lambda: op.codomain.riesz(arg),
                "riesz_domain_inverse_ns": lambda: op.domain.riesz_inverse(sample_dual),
            }
            return measure_ladder(
                layers,
                repeat=repeat,
                number=number,
                warmup=warmup,
                baseline_layer="metric_bare",
                full_layer="public_nocheck",
                total_overhead_ns=total_overhead_ns,
                primitives=primitives,
            )
        layers = [
            ("bare_backend", bare_fn),
            ("core_dispatch", lambda: core(arg)),
            ("public_nocheck", lambda: public(arg)),
            ("checks", lambda: _check_rapply(op.domain, op.codomain, core, arg)),
        ]
        primitives = {
            "check_member_in_ns": lambda: op.codomain._check_member(arg),
            "check_member_out_ns": lambda: op.domain._check_member(sample_out),
        }
    elif operation == "vapply":
        layers = [
            ("bare_backend", bare_fn),
            ("core_dispatch", lambda: core(arg)),
            ("public_nocheck", lambda: public(arg)),
            ("checks", lambda: _check_batched_apply(lambda v: _check_batched(op.domain, v), lambda v: _check_batched(op.codomain, v), core, arg)),
        ]
        primitives = {"check_batch_in_ns": lambda: _check_batched(op.domain, arg)}
    elif operation == "rvapply":
        layers = [
            ("bare_backend", bare_fn),
            ("core_dispatch", lambda: core(arg)),
            ("public_nocheck", lambda: public(arg)),
            ("checks", lambda: _check_batched_apply(lambda v: _check_batched(op.codomain, v), lambda v: _check_batched(op.domain, v), core, arg)),
        ]
        primitives = {"check_batch_in_ns": lambda: _check_batched(op.codomain, arg)}
    else:
        return {}

    return measure_ladder(
        layers,
        repeat=repeat,
        number=number,
        warmup=warmup,
        baseline_layer="bare_backend",
        full_layer="public_nocheck",
        total_overhead_ns=total_overhead_ns,
        primitives=primitives,
    )


def functional_breakdown(
    functional: Any,
    operation: str,
    bare_fn: Callable[[], Any],
    arg: Any,
    *,
    repeat: int,
    number: int,
    warmup: int,
    total_overhead_ns: float,
) -> dict[str, Any]:
    """Measured ladder for Functional value/grad/vvalue/vgrad operations."""
    public = getattr(functional, operation)
    layers = [("bare_backend", bare_fn), ("public", lambda: public(arg))]
    primitives = {"check_domain_ns": lambda: functional.domain._check_member(arg)}
    if operation in {"vvalue", "vgrad"}:
        from spacecore._batching import _check_batched

        primitives = {"check_batch_ns": lambda: _check_batched(functional.domain, arg)}
    return measure_ladder(
        layers,
        repeat=repeat,
        number=number,
        warmup=warmup,
        baseline_layer="bare_backend",
        full_layer="public",
        total_overhead_ns=total_overhead_ns,
        primitives=primitives,
    )


def generic_breakdown(
    bare_fn: Callable[[], Any],
    public_fn: Callable[[], Any],
    *,
    repeat: int,
    number: int,
    warmup: int,
    total_overhead_ns: float,
) -> dict[str, Any]:
    """Fallback two-rung ladder."""
    return measure_ladder(
        [("bare_backend", bare_fn), ("public", public_fn)],
        repeat=repeat,
        number=number,
        warmup=warmup,
        baseline_layer="bare_backend",
        full_layer="public",
        total_overhead_ns=total_overhead_ns,
    )

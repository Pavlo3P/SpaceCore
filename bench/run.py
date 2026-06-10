from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.cases import BenchCase, default_cases, full_sweep
from bench.cost_model import calibrate_primitives, classify_gap, predict_overhead
from bench.diagnose import add_trend_flags, make_verdict
from bench.harness import BenchResult, time_op, time_op_first_call
from bench.render import render_file


def _numbers_for(case: BenchCase) -> tuple[int, int, int]:
    if case.backend == "jax-jit":
        return 9, 3 if case.size_name == "large" else 10, 1
    if case.size_name == "large":
        return 9, 1 if case.operator_type == "DenseLinOp" and case.size >= 2048 else 5, 1
    if case.size_name == "small":
        return 13, 20, 2
    return 17, 1000, 3


def _breakdown_numbers_for(case: BenchCase) -> tuple[int, int, int]:
    if case.size_name == "large":
        return 5, 1 if case.operator_type == "DenseLinOp" and case.size >= 2048 else 3, 1
    if case.size_name == "small":
        return 7, 10, 1
    return 9, 300, 1


def _run_case(case: BenchCase, costs) -> BenchResult:
    compile_bare = compile_sc = None
    if case.backend == "jax-jit":
        compile_bare = time_op_first_call(case.bare_fn)
        compile_sc = time_op_first_call(case.sc_fn)
    case.assert_equal()
    repeat, number, warmup = _numbers_for(case)
    bare = time_op(case.bare_fn, repeat=repeat, number=number, warmup=warmup)
    sc = time_op(case.sc_fn, repeat=repeat, number=number, warmup=warmup)
    overhead = sc["median_ns"] - bare["median_ns"]
    ratio = sc["median_ns"] / bare["median_ns"] if bare["median_ns"] else float("inf")
    predicted, components = predict_overhead(case, costs)
    gap = classify_gap(overhead, predicted)
    if case.size_name == "large":
        jitter = max(sc["median_ns"] - sc["best_ns"], 0.0) + max(
            bare["median_ns"] - bare["best_ns"], 0.0
        )
        if ratio <= 1.15 or overhead <= 3.0 * jitter:
            gap = "ok"
    verdict = make_verdict(
        case=case,
        overhead_ns=overhead,
        bare_ns=bare["median_ns"],
        predicted_ns=predicted,
        components=components,
        gap=gap,
    )
    breakdown = None
    if case.breakdown_fn is not None:
        brepeat, bnumber, bwarmup = _breakdown_numbers_for(case)
        try:
            breakdown = case.breakdown_fn(
                repeat=brepeat,
                number=bnumber,
                warmup=bwarmup,
                total_overhead_ns=sc["best_ns"] - bare["best_ns"],
            )
        except Exception as err:
            breakdown = {"error": f"{type(err).__name__}: {err}"}
    return BenchResult(
        case_id=case.case_id,
        label=case.label,
        backend=case.backend,
        operator_type=case.operator_type,
        operation=case.operation,
        geometry=case.geometry,
        shape_kind=case.shape_kind,
        size_name=case.size_name,
        size=case.size,
        checks=case.checks,
        batch=case.batch,
        bare_label=case.bare_label,
        sc_label=case.sc_label,
        bare_best_ns=bare["best_ns"],
        bare_median_ns=bare["median_ns"],
        sc_best_ns=sc["best_ns"],
        sc_median_ns=sc["median_ns"],
        overhead_ns=overhead,
        ratio=ratio,
        predicted_overhead_ns=predicted,
        gap=gap,
        components=components,
        verdict=verdict,
        compile_bare_ns=compile_bare,
        compile_sc_ns=compile_sc,
        breakdown=breakdown,
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else None


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [
        r
        for r in results
        if r["gap"] != "ok"
        or (r.get("trend") in {"constant-above-1.0", "grows-with-size"} and r["ratio"] > 1.15)
    ]
    return {
        "count": len(results),
        "median_ratio": _median([r["ratio"] for r in results]),
        "median_ratio_above_1e3": _median([r["ratio"] for r in results if r["size"] > 1_000]),
        "median_ratio_jit": _median([r["ratio"] for r in results if r["backend"] == "jax-jit"]),
        "flagged_count": len(flagged),
    }


def _compare_baseline(current: list[dict[str, Any]], baseline_path: Path) -> list[str]:
    old = json.loads(baseline_path.read_text())
    old_by_id = {r["case_id"]: r for r in old["results"]}
    failures = []
    for row in current:
        old_row = old_by_id.get(row["case_id"])
        if not old_row:
            continue
        old_overhead = max(old_row["overhead_ns"], 1.0)
        if row["overhead_ns"] > old_overhead * 1.2 + 1_000.0:
            failures.append(
                f"{row['case_id']}: overhead {row['overhead_ns']:.0f} ns > baseline {old_overhead:.0f} ns"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SpaceCore overhead diagnostics.")
    parser.add_argument("--suite", choices=["default", "full"], default="default")
    parser.add_argument("--json", default="bench/out/overhead.json")
    parser.add_argument("--html", default="bench/out/overhead.html")
    parser.add_argument("--baseline", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases for smoke testing."
    )
    args = parser.parse_args(argv)

    cases = full_sweep() if args.suite == "full" else default_cases()
    if args.limit is not None:
        cases = cases[: args.limit]

    print("Calibrating primitives...", flush=True)
    costs = calibrate_primitives()
    print(f"Running {len(cases)} cases...", flush=True)
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.case_id}", flush=True)
        results.append(_run_case(case, costs).to_json())
    add_trend_flags(results)

    artifact = {
        "suite": args.suite,
        "primitive_costs": costs.to_json(),
        "summary": _summary(results),
        "results": results,
    }
    json_path = Path(args.json)
    html_path = Path(args.html)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    render_file(json_path, html_path)

    print(json.dumps(artifact["summary"], indent=2))
    print(f"Wrote {json_path} and {html_path}")

    if args.baseline:
        failures = _compare_baseline(results, Path(args.baseline))
        if failures:
            print("Baseline regressions:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

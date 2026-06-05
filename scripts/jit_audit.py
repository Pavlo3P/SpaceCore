from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "jaxpr_lanczos_smallest.txt"


def _ctx():
    import jax
    import spacecore as sc

    dtype = np.float64 if jax.config.read("jax_enable_x64") else np.float32
    return sc.Context(sc.JaxOps(), dtype=dtype, enable_checks=False)


def _spd_operator(n: int):
    import spacecore as sc

    ctx = _ctx()
    space = sc.DenseCoordinateSpace((n,), ctx)
    matrix = np.diag(np.arange(2.0, 2.0 + n))
    matrix += 0.05 * np.ones((n, n))
    return sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)


def _rect_operator():
    import spacecore as sc

    ctx = _ctx()
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    return sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)


def _same_shape_inputs(A: Any) -> tuple[Any, Any]:
    ctx = A.ctx
    x0 = ctx.asarray(np.linspace(1.0, 2.0, A.domain.shape[0]))
    x1 = ctx.asarray(np.linspace(2.0, 3.0, A.domain.shape[0]))
    return x0, x1


def _audit_solver(
    name: str,
    fn_factory: Callable[[dict[str, int]], Callable[..., Any]],
    A: Any,
    first_rhs: Any,
    second_rhs: Any,
    shape_changed_A: Any,
    shape_changed_rhs: Any,
    static_name: str,
) -> dict[str, Any]:
    import jax

    traces = {"count": 0}
    fn = fn_factory(traces)
    jitted = jax.jit(fn, static_argnames=(static_name,))

    out0 = jitted(A, first_rhs, **{static_name: 4})
    out1 = jitted(A, second_rhs, **{static_name: 4})
    same_shape_traces = traces["count"]

    out2 = jitted(A, first_rhs, **{static_name: 5})
    static_changed_traces = traces["count"]

    out3 = jitted(shape_changed_A, shape_changed_rhs, **{static_name: 4})
    shape_changed_traces = traces["count"]

    for out in (out0, out1, out2, out3):
        jax.block_until_ready(out)

    return {
        "solver": name,
        "traces_after_two_same_shape_calls": same_shape_traces,
        "traces_after_static_change": static_changed_traces,
        "traces_after_shape_change": shape_changed_traces,
        "stable_values_retraced": same_shape_traces > 1,
        "static_change_retraced": static_changed_traces > same_shape_traces,
        "shape_change_retraced": shape_changed_traces > static_changed_traces,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit JAX trace stability for SpaceCore solvers.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any solver retraces on same-shape inputs.",
    )
    parser.add_argument(
        "--log-compiles",
        action="store_true",
        help="Enable jax_log_compiles for manual inspection.",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Write the lanczos_smallest JAXPR fixture. Disabled by default in --check mode.",
    )
    return parser.parse_args()


def _audit_failed(item: dict[str, Any]) -> bool:
    if "status" in item:
        return True
    return (
        item["traces_after_two_same_shape_calls"] != 1
        or item["stable_values_retraced"]
        or not item["static_change_retraced"]
        or not item["shape_change_retraced"]
    )


def main() -> None:
    import jax
    import spacecore as sc

    args = _parse_args()
    if args.log_compiles:
        jax.config.update("jax_log_compiles", True)

    A2 = _spd_operator(2)
    A3 = _spd_operator(3)
    x2a, x2b = _same_shape_inputs(A2)
    x3a, _ = _same_shape_inputs(A3)
    R2 = _rect_operator()
    R3 = sc.DenseLinOp(
        _ctx().asarray([[1.0, 0.0, 0.5], [0.0, 2.0, -1.0], [1.0, -1.0, 0.25], [0.5, 0.0, 1.0]]),
        sc.DenseCoordinateSpace((3,), _ctx()),
        sc.DenseCoordinateSpace((4,), _ctx()),
        _ctx(),
    )
    b2a = R2.codomain.ctx.asarray([1.0, 2.0, 3.0])
    b2b = R2.codomain.ctx.asarray([3.0, 2.0, 1.0])
    b4 = R3.codomain.ctx.asarray([1.0, 2.0, 3.0, 4.0])

    audits = [
        _audit_solver(
            "cg",
            lambda traces: (
                lambda A, b, maxiter: (
                    traces.__setitem__("count", traces["count"] + 1)
                    or sc.cg(A, b, maxiter=maxiter).x
                )
            ),
            A2,
            x2a,
            x2b,
            A3,
            x3a,
            "maxiter",
        ),
        _audit_solver(
            "lsqr",
            lambda traces: (
                lambda A, b, maxiter: (
                    traces.__setitem__("count", traces["count"] + 1)
                    or sc.lsqr(A, b, maxiter=maxiter).x
                )
            ),
            R2,
            b2a,
            b2b,
            R3,
            b4,
            "maxiter",
        ),
        _audit_solver(
            "lanczos_smallest",
            lambda traces: (
                lambda A, x, max_iter: (
                    traces.__setitem__("count", traces["count"] + 1)
                    or sc.lanczos_smallest(A, x, max_iter=max_iter).eigenvalue
                )
            ),
            A2,
            x2a,
            x2b,
            A3,
            x3a,
            "max_iter",
        ),
        _audit_solver(
            "power_iteration",
            lambda traces: (
                lambda A, x, maxiter: (
                    traces.__setitem__("count", traces["count"] + 1)
                    or sc.power_iteration(A, x0=x, maxiter=maxiter).eigenvalue
                )
            ),
            A2,
            x2a,
            x2b,
            A3,
            x3a,
            "maxiter",
        ),
    ]

    if hasattr(sc, "expm_multiply"):
        audits.append(
            _audit_solver(
                "expm_multiply",
                lambda traces: (
                    lambda A, x, max_iter: (
                        traces.__setitem__("count", traces["count"] + 1)
                        or sc.expm_multiply(A, x, max_iter=max_iter).result
                    )
                ),
                A2,
                x2a,
                x2b,
                A3,
                x3a,
                "max_iter",
            )
        )
    else:
        audits.append({"solver": "expm_multiply", "status": "not available before Task 1"})

    print("JIT audit summary")
    for item in audits:
        print(item)

    if args.write_fixture or not args.check:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        jaxpr = jax.make_jaxpr(
            lambda A, x: sc.lanczos_smallest(A, x, max_iter=3, check_every=1).eigenvalue
        )(A2, x2a)
        FIXTURE.write_text(str(jaxpr))
        print(f"wrote {FIXTURE.relative_to(ROOT)}")

    if args.check:
        failures = [item for item in audits if _audit_failed(item)]
        if failures:
            print("JIT audit check failed")
            for item in failures:
                print(item)
            raise SystemExit(1)


if __name__ == "__main__":
    main()

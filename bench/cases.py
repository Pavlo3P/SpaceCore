from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse as sps

from .breakdown import functional_breakdown, linop_breakdown


@dataclass(slots=True)
class BenchCase:
    case_id: str
    label: str
    backend: str
    operator_type: str
    operation: str
    geometry: str
    shape_kind: str
    size_name: str
    size: int
    checks: bool
    batch: int | None
    mode: str
    bare_label: str
    sc_label: str
    bare_fn: Callable[[], Any]
    sc_fn: Callable[[], Any]
    assert_equal: Callable[[], None]
    breakdown_fn: Callable[..., dict[str, Any]] | None = None


def _allclose(a: Any, b: Any, *, rtol=1e-5, atol=1e-6) -> bool:
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(
            _allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b)
        )
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def _assert_close(bare_fn, sc_fn) -> None:
    bare = bare_fn()
    actual = sc_fn()
    if not _allclose(actual, bare):
        raise AssertionError("SpaceCore result does not match bare reference.")


def _ctx(sc, backend: str, checks: bool):
    if backend == "numpy-eager":
        return sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=checks)
    if backend in {"jax-eager", "jax-jit"}:
        import jax

        jax.config.update("jax_enable_x64", True)
        return sc.Context(sc.JaxOps(), dtype=np.float64, enable_checks=checks)
    if backend == "torch":
        return sc.Context(sc.TorchOps(), dtype=np.float64, enable_checks=checks)
    raise ValueError(f"Unknown backend {backend!r}.")


def _as_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


class GeneralDiagonalInnerProduct:
    """Duck-typed non-Weighted geometry used to force the general metric path."""

    def __init__(self, weights):
        self.weights = weights

    def inner(self, ops, x, y):
        return ops.vdot(x, self.weights * y)

    def riesz(self, ops, x):
        return self.weights * x

    def riesz_inverse(self, ops, x):
        return x / self.weights

    def convert(self, ctx):
        return GeneralDiagonalInnerProduct(ctx.asarray(self.weights))

    @property
    def is_euclidean(self):
        return False

    def __eq__(self, other):
        return type(other) is type(self) and np.allclose(_as_numpy(self.weights), _as_numpy(other.weights))


def _case_id(*parts: Any) -> str:
    return ".".join(str(part).replace(" ", "_") for part in parts if part is not None)


def _wrap_jit(sc, backend: str, bare_fn, sc_fn):
    if backend != "jax-jit":
        return bare_fn, sc_fn
    import jax

    return jax.jit(bare_fn), jax.jit(sc_fn)


def _dense_cases(sc, rng, backend: str, size_name: str, n: int, checks: bool) -> list[BenchCase]:
    ctx = _ctx(sc, backend, checks)
    A = ctx.asarray(rng.standard_normal((n, n)))
    AH = A.T.conj()
    x = ctx.asarray(rng.standard_normal(n))
    y = ctx.asarray(rng.standard_normal(n))
    X = sc.DenseCoordinateSpace((n,), ctx)
    op = sc.DenseLinOp(A, X, X, ctx)
    cases = []
    for operation, bare_label, sc_label, bare, call, arg in [
        ("apply", "A @ x", "op.apply(x)", lambda: A @ x, lambda: op.apply(x), x),
        ("rapply", "AH @ y", "op.rapply(y)", lambda: AH @ y, lambda: op.rapply(y), y),
    ]:
        bare, call = _wrap_jit(sc, backend, bare, call)
        breakdown = None
        if backend == "numpy-eager":
            def breakdown(
                *, repeat, number, warmup, total_overhead_ns,
                op=op, operation=operation, bare=bare, arg=arg
            ):
                return linop_breakdown(
                    op,
                    operation,
                    bare,
                    arg,
                    repeat=repeat,
                    number=number,
                    warmup=warmup,
                    total_overhead_ns=total_overhead_ns,
                )
        cases.append(
            BenchCase(
                _case_id(backend, "dense", operation, size_name, "checks" if checks else "nocheck"),
                f"{size_name} dense {operation}",
                backend,
                "DenseLinOp",
                operation,
                "euclidean",
                "flat",
                size_name,
                n,
                checks,
                None,
                "EUCLIDEAN_FLAT",
                bare_label,
                sc_label,
                bare,
                call,
                lambda bare=bare, call=call: _assert_close(bare, call),
                breakdown,
            )
        )
    return cases


def _diagonal_cases(
    sc, rng, backend: str, size_name: str, n: int, checks: bool, weighted=False, general=False
) -> list[BenchCase]:
    ctx = _ctx(sc, backend, checks)
    d = ctx.asarray(0.5 + rng.random(n))
    x = ctx.asarray(rng.standard_normal(n))
    y = ctx.asarray(rng.standard_normal(n))
    batch = 8 if size_name != "large" else 4
    xs = ctx.asarray(rng.standard_normal((batch, n)))
    ys = ctx.asarray(rng.standard_normal((batch, n)))
    if general:
        geometry = "general-metric"
        geom = GeneralDiagonalInnerProduct(ctx.asarray(1.0 + rng.random(n)))
        mode = "GENERAL_METRIC"
    elif weighted:
        geometry = "weighted"
        geom = sc.WeightedInnerProduct(ctx.asarray(1.0 + rng.random(n)))
        mode = "WEIGHTED_FUSED"
    else:
        geometry = "euclidean"
        geom = None
        mode = "EUCLIDEAN"
    X = sc.DenseCoordinateSpace((n,), ctx, geometry=geom)
    op = sc.DiagonalLinOp(d, X, ctx)
    rows = []
    for operation, bare_label, sc_label, bare, call, batch_value in [
        ("apply", "d * x", "op.apply(x)", lambda: d * x, lambda: op.apply(x), None),
        ("rapply", "d * y", "op.rapply(y)", lambda: d * y, lambda: op.rapply(y), None),
        ("vapply", "d * xs", "op.vapply(xs)", lambda: d * xs, lambda: op.vapply(xs), batch),
        ("rvapply", "d * ys", "op.rvapply(ys)", lambda: d * ys, lambda: op.rvapply(ys), batch),
    ]:
        bare, call = _wrap_jit(sc, backend, bare, call)
        arg = x if operation == "apply" else y if operation == "rapply" else xs if operation == "vapply" else ys
        breakdown = None
        if backend == "numpy-eager":
            def breakdown(
                *, repeat, number, warmup, total_overhead_ns,
                op=op, operation=operation, bare=bare, arg=arg
            ):
                return linop_breakdown(
                    op,
                    operation,
                    bare,
                    arg,
                    repeat=repeat,
                    number=number,
                    warmup=warmup,
                    total_overhead_ns=total_overhead_ns,
                )
        rows.append(
            BenchCase(
                _case_id(backend, geometry, "diagonal", operation, size_name, "checks" if checks else "nocheck"),
                f"{size_name} {geometry} diagonal {operation}",
                backend,
                "DiagonalLinOp",
                operation,
                geometry,
                "flat",
                size_name,
                n,
                checks,
                batch_value,
                mode,
                bare_label,
                sc_label,
                bare,
                call,
                lambda bare=bare, call=call: _assert_close(bare, call),
                breakdown,
            )
        )
    return rows


def _sparse_cases(sc, rng, size_name: str, n: int, checks: bool, weighted=False) -> list[BenchCase]:
    backend = "numpy-eager"
    ctx = _ctx(sc, backend, checks)
    offsets = np.array([-1, 0, 1])
    diagonals = [rng.standard_normal(n - abs(int(k))) for k in offsets]
    S = sps.diags(diagonals, offsets, shape=(n, n), format="csr")
    ST = S.T
    x = ctx.asarray(rng.standard_normal(n))
    y = ctx.asarray(rng.standard_normal(n))
    batch = 8 if size_name != "large" else 4
    xs = ctx.asarray(rng.standard_normal((batch, n)))
    ys = ctx.asarray(rng.standard_normal((batch, n)))
    if weighted:
        w = ctx.asarray(1.0 + rng.random(n))
        X = sc.DenseCoordinateSpace((n,), ctx, geometry=sc.WeightedInnerProduct(w))
        geometry = "weighted"
        def rbare():
            return (ST @ (w * y)) / w

        def rvbare():
            return (ST @ (ys * w).T).T / w

        mode = "WEIGHTED_FUSED"
    else:
        X = sc.DenseCoordinateSpace((n,), ctx)
        geometry = "euclidean"
        def rbare():
            return ST @ y

        def rvbare():
            return (ST @ ys.T).T

        mode = "EUCLIDEAN_FLAT"
    op = sc.SparseLinOp(S, X, X, ctx)
    specs = [
        ("apply", "S @ x", "op.apply(x)", lambda: S @ x, lambda: op.apply(x), None),
        ("rapply", "ST @ y", "op.rapply(y)", rbare, lambda: op.rapply(y), None),
        ("vapply", "(S @ xs.T).T", "op.vapply(xs)", lambda: (S @ xs.T).T, lambda: op.vapply(xs), batch),
        ("rvapply", "(ST @ ys.T).T", "op.rvapply(ys)", rvbare, lambda: op.rvapply(ys), batch),
    ]
    return [
        BenchCase(
            _case_id(backend, geometry, "sparse", operation, size_name, "checks" if checks else "nocheck"),
            f"{size_name} {geometry} sparse {operation}",
            backend,
            "SparseLinOp",
            operation,
            geometry,
            "flat",
            size_name,
            n,
            checks,
            batch_value,
            mode,
            bare_label,
            sc_label,
            bare,
            call,
            lambda bare=bare, call=call: _assert_close(bare, call),
        )
        for operation, bare_label, sc_label, bare, call, batch_value in specs
    ]


def _weighted_dense_cases(sc, rng, size_name: str, n: int, checks: bool, matrix_free=False) -> list[BenchCase]:
    backend = "numpy-eager"
    ctx = _ctx(sc, backend, checks)
    A = ctx.asarray(rng.standard_normal((n, n)))
    AH = A.T.conj()
    wx = ctx.asarray(1.0 + rng.random(n))
    wy = ctx.asarray(1.0 + rng.random(n))
    y = ctx.asarray(rng.standard_normal(n))
    batch = 8 if size_name != "large" else 4
    ys = ctx.asarray(rng.standard_normal((batch, n)))
    X = sc.DenseCoordinateSpace((n,), ctx, geometry=sc.WeightedInnerProduct(wx))
    Y = sc.DenseCoordinateSpace((n,), ctx, geometry=sc.WeightedInnerProduct(wy))
    if matrix_free:
        op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
            lambda z: A @ z,
            lambda dual_y: AH @ dual_y,
            X,
            Y,
            ctx,
            vapply=lambda zs: zs @ A.T,
            coordinate_rvapply=lambda dual_ys: dual_ys @ AH.T,
        )
        operator_type = "MatrixFreeLinOp"
        mode = "coordinate_adjoint"
    else:
        op = sc.DenseLinOp(A, X, Y, ctx)
        operator_type = "DenseLinOp"
        mode = "WEIGHTED_FUSED"
    def rbare():
        return (AH @ (wy * y)) / wx

    def rvbare():
        return ((ys * wy) @ AH.T) / wx

    rbreakdown = None if matrix_free else (
        lambda *, repeat, number, warmup, total_overhead_ns, op=op, bare=rbare, y=y:
        linop_breakdown(
            op,
            "rapply",
            bare,
            y,
            repeat=repeat,
            number=number,
            warmup=warmup,
            total_overhead_ns=total_overhead_ns,
        )
    )
    rvbreakdown = None if matrix_free else (
        lambda *, repeat, number, warmup, total_overhead_ns, op=op, bare=rvbare, ys=ys:
        linop_breakdown(
            op,
            "rvapply",
            bare,
            ys,
            repeat=repeat,
            number=number,
            warmup=warmup,
            total_overhead_ns=total_overhead_ns,
        )
    )
    return [
        BenchCase(
            _case_id(backend, "weighted", operator_type, "rapply", size_name, "checks" if checks else "nocheck"),
            f"{size_name} weighted {operator_type} rapply",
            backend,
            operator_type,
            "rapply",
            "weighted",
            "flat",
            size_name,
            n,
            checks,
            None,
            mode,
            "R_X^-1 AH R_Y y",
            "op.rapply(y)",
            rbare,
            lambda: op.rapply(y),
            lambda: _assert_close(rbare, lambda: op.rapply(y)),
            rbreakdown,
        ),
        BenchCase(
            _case_id(backend, "weighted", operator_type, "rvapply", size_name, "checks" if checks else "nocheck"),
            f"{size_name} weighted {operator_type} rvapply",
            backend,
            operator_type,
            "rvapply",
            "weighted",
            "flat",
            size_name,
            n,
            checks,
            batch,
            mode,
            "R_X^-1 AH R_Y ys",
            "op.rvapply(ys)",
            rvbare,
            lambda: op.rvapply(ys),
            lambda: _assert_close(rvbare, lambda: op.rvapply(ys)),
            rvbreakdown,
        ),
    ]


def _algebra_cases(sc, rng, size_name: str, n: int, checks: bool) -> list[BenchCase]:
    ctx = _ctx(sc, "numpy-eager", checks)
    A = ctx.asarray(rng.standard_normal((n, n)))
    B = ctx.asarray(rng.standard_normal((n, n)))
    AH = A.T.conj()
    BH = B.T.conj()
    x = ctx.asarray(rng.standard_normal(n))
    y = ctx.asarray(rng.standard_normal(n))
    X = sc.DenseCoordinateSpace((n,), ctx)
    Aop = sc.DenseLinOp(A, X, X, ctx)
    Bop = sc.DenseLinOp(B, X, X, ctx)
    cases = [
        ("ComposedLinOp", "apply", "B @ (A @ x)", "(Bop @ Aop).apply(x)", lambda: B @ (A @ x), lambda: (Bop @ Aop).apply(x)),
        ("SumLinOp", "apply", "A @ x + B @ x", "(Aop + Bop).apply(x)", lambda: A @ x + B @ x, lambda: (Aop + Bop).apply(x)),
        ("ScaledLinOp", "apply", "2.5 * (A @ x)", "(2.5 * Aop).apply(x)", lambda: 2.5 * (A @ x), lambda: (2.5 * Aop).apply(x)),
        ("ProductLinOp", "apply", "(A @ x, B @ y)", "block.apply((x, y))", lambda: (A @ x, B @ y), lambda: sc.BlockDiagonalLinOp.from_operators((Aop, Bop)).apply((x, y))),
        ("ProductLinOp", "rapply", "AH @ y + BH @ y", "stacked.rapply((y, y))", lambda: AH @ y + BH @ y, lambda: sc.StackedLinOp.from_operators((Aop, Bop)).rapply((y, y))),
    ]
    out = []
    for operator_type, operation, bare_label, sc_label, bare, call in cases:
        out.append(
            BenchCase(
                _case_id("numpy-eager", operator_type, operation, size_name, "checks" if checks else "nocheck"),
                f"{size_name} {operator_type} {operation}",
                "numpy-eager",
                operator_type,
                operation,
                "euclidean",
                "flat",
                size_name,
                n,
                checks,
                None,
                "composite",
                bare_label,
                sc_label,
                bare,
                call,
                lambda bare=bare, call=call: _assert_close(bare, call),
            )
        )
    return out


def _functional_cases(sc, rng, size_name: str, n: int, checks: bool) -> list[BenchCase]:
    ctx = _ctx(sc, "numpy-eager", checks)
    X = sc.DenseCoordinateSpace((n,), ctx)
    c = ctx.asarray(rng.standard_normal(n))
    x = ctx.asarray(rng.standard_normal(n))
    xs = ctx.asarray(rng.standard_normal((8 if size_name != "large" else 4, n)))
    F = sc.InnerProductFunctional(c, X, ctx)
    D = sc.DiagonalLinOp(ctx.asarray(1.0 + rng.random(n)), X, ctx)
    Q = sc.LinOpQuadraticForm(D, sc.InnerProductFunctional(c, X, ctx), 0.25, ctx)
    specs = [
        ("InnerProductFunctional", "value", "X.inner(c, x)", "F.value(x)", lambda: X.inner(c, x), lambda: F.value(x), None, F, x),
        ("InnerProductFunctional", "grad", "c", "F.grad(x)", lambda: c, lambda: F.grad(x), None, F, x),
        ("InnerProductFunctional", "vvalue", "xs @ c", "F.vvalue(xs)", lambda: xs @ c, lambda: F.vvalue(xs), xs.shape[0], F, xs),
        (
            "InnerProductFunctional",
            "vgrad",
            "broadcast_to(c)",
            "F.vgrad(xs)",
            lambda: ctx.ops.broadcast_to(c, xs.shape),
            lambda: F.vgrad(xs),
            xs.shape[0],
            F,
            xs,
        ),
        ("LinOpQuadraticForm", "value", "0.5*inner(x,Dx)+inner(c,x)+a", "Q.value(x)", lambda: 0.5 * X.inner(x, D.apply(x)) + X.inner(c, x) + 0.25, lambda: Q.value(x), None, Q, x),
        ("LinOpQuadraticForm", "grad", "D.apply(x)+c", "Q.grad(x)", lambda: D.apply(x) + c, lambda: Q.grad(x), None, Q, x),
        (
            "LinOpQuadraticForm",
            "vvalue",
            "batched quadratic expression",
            "Q.vvalue(xs)",
            lambda: 0.5 * ctx.ops.sum(xs * D.vapply(xs), axis=1) + xs @ c + 0.25,
            lambda: Q.vvalue(xs),
            xs.shape[0],
            Q,
            xs,
        ),
        (
            "LinOpQuadraticForm",
            "vgrad",
            "D.vapply(xs)+c",
            "Q.vgrad(xs)",
            lambda: D.vapply(xs) + c,
            lambda: Q.vgrad(xs),
            xs.shape[0],
            Q,
            xs,
        ),
    ]
    out = []
    for operator_type, operation, bare_label, sc_label, bare, call, batch, functional, arg in specs:
        def breakdown(
            *, repeat, number, warmup, total_overhead_ns,
            functional=functional, operation=operation, bare=bare, arg=arg
        ):
            return functional_breakdown(
                functional,
                operation,
                bare,
                arg,
                repeat=repeat,
                number=number,
                warmup=warmup,
                total_overhead_ns=total_overhead_ns,
            )
        out.append(BenchCase(
            _case_id("numpy-eager", operator_type, operation, size_name, "checks" if checks else "nocheck"),
            f"{size_name} {operator_type} {operation}",
            "numpy-eager",
            operator_type,
            operation,
            "euclidean",
            "flat",
            size_name,
            n,
            checks,
            batch,
            "functional",
            bare_label,
            sc_label,
            bare,
            call,
            lambda bare=bare, call=call: _assert_close(bare, call),
            breakdown,
        ))
    return out


def _torch_cases(sc, rng) -> list[BenchCase]:
    try:
        import torch  # noqa: F401

        getattr(sc, "TorchOps")
    except Exception:
        return []
    return _dense_cases(sc, rng, "torch", "small", 512, False)


def default_cases() -> list[BenchCase]:
    """Common, runnable subset used by the dashboard default."""
    import spacecore as sc

    rng = np.random.default_rng(20240601)
    cases: list[BenchCase] = []
    for size_name, n in (("tiny", 8), ("small", 1024), ("large", 4096)):
        cases.extend(_dense_cases(sc, rng, "numpy-eager", size_name, n, False))
    for size_name, n in (("tiny", 8), ("small", 4096), ("large", 1_000_000)):
        cases.extend(_diagonal_cases(sc, rng, "numpy-eager", size_name, n, False))
        cases.extend(_diagonal_cases(sc, rng, "numpy-eager", size_name, n, False, weighted=True))
    for size_name, n in (("tiny", 16), ("small", 4096), ("large", 100_000)):
        cases.extend(_sparse_cases(sc, rng, size_name, n, False))
        cases.extend(_sparse_cases(sc, rng, size_name, n, False, weighted=True))
    for size_name, n in (("tiny", 8), ("small", 1024), ("large", 2048)):
        cases.extend(_weighted_dense_cases(sc, rng, size_name, n, False))
        cases.extend(_weighted_dense_cases(sc, rng, size_name, n, False, matrix_free=True))
        cases.extend(_algebra_cases(sc, rng, size_name, n, False))
        cases.extend(_functional_cases(sc, rng, size_name, n, False))
    if _jax_available():
        for size_name, n in (("small", 1024), ("large", 2048)):
            cases.extend(_dense_cases(sc, rng, "jax-jit", size_name, n, False))
            cases.extend(_diagonal_cases(sc, rng, "jax-jit", size_name, n, False))
    cases.extend(_torch_cases(sc, rng))
    return cases


def full_sweep() -> list[BenchCase]:
    """Expanded anomaly scan with checks and extra backend/geometry paths."""
    import spacecore as sc

    rng = np.random.default_rng(20240602)
    cases = default_cases()
    for checks in (True,):
        for size_name, n in (("tiny", 8), ("small", 1024)):
            cases.extend(_dense_cases(sc, rng, "numpy-eager", size_name, n, checks))
            cases.extend(_diagonal_cases(sc, rng, "numpy-eager", size_name, n, checks))
            cases.extend(_weighted_dense_cases(sc, rng, size_name, n, checks))
            cases.extend(_algebra_cases(sc, rng, size_name, n, checks))
            cases.extend(_functional_cases(sc, rng, size_name, n, checks))
    if _jax_available():
        for size_name, n in (("tiny", 8), ("small", 1024)):
            cases.extend(_dense_cases(sc, rng, "jax-eager", size_name, n, False))
            cases.extend(_diagonal_cases(sc, rng, "jax-eager", size_name, n, False, weighted=True))
    for size_name, n in (("tiny", 8), ("small", 1024)):
        cases.extend(_diagonal_cases(sc, rng, "numpy-eager", size_name, n, False, general=True))
    return cases


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except Exception:
        return False

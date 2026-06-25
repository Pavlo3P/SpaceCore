from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

import spacecore as sc

from ._protocol import GeneratedCase


FunctionalCase = GeneratedCase[sc.Functional]
NUMPY_FUNCTIONAL_DTYPES = (np.float64, np.complex128)


def _context(dtype: Any, check_level: sc.CheckLevel | str) -> sc.Context:
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=check_level)


def _target_dtype(dtype: Any) -> np.dtype[Any]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.float32)
    if dtype == np.dtype(np.complex128):
        return np.dtype(np.complex64)
    raise ValueError(f"Unsupported functional generator dtype {dtype}.")


def _values(dtype: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        x = np.asarray([0.5 + 0.25j, -1.0 + 0.75j, 2.0 - 0.5j], dtype=dtype)
        c = np.asarray([1.5 - 0.5j, -0.25 + 1.0j, 0.75 + 0.25j], dtype=dtype)
    else:
        x = np.asarray([0.5, -1.0, 2.0], dtype=dtype)
        c = np.asarray([1.5, -0.25, 0.75], dtype=dtype)
    weights = np.asarray([2.0, 5.0, 11.0], dtype=dtype)
    diagonal = np.asarray([1.25, 2.5, 4.0], dtype=dtype)
    return x, c, weights, diagonal


def _space(ctx: sc.Context, weighted: bool) -> sc.DenseCoordinateSpace:
    if not weighted:
        return sc.DenseCoordinateSpace((3,), ctx)
    weights = ctx.asarray(np.asarray([2.0, 5.0, 11.0], dtype=np.dtype(ctx.dtype)))
    return sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(weights))


def _metric_matrix(weights: np.ndarray, weighted: bool) -> np.ndarray:
    if weighted:
        return np.diag(weights)
    return np.eye(weights.size, dtype=weights.dtype)


def _dense_case(
    dtype: Any,
    check_level: sc.CheckLevel | str,
    *,
    kind: str,
    weighted: bool,
) -> FunctionalCase:
    ctx = _context(dtype, check_level)
    domain = _space(ctx, weighted)
    x_np, c_np, weights, diagonal = _values(dtype)
    x = ctx.asarray(x_np)
    c = ctx.asarray(c_np)
    metric = _metric_matrix(weights, weighted)
    geometry = "weighted" if weighted else "euclidean"

    if kind == "zero":
        representer = np.zeros_like(c_np)
        functional = sc.InnerProductFunctional(ctx.asarray(representer), domain, ctx)
        value = np.vdot(representer, metric @ x_np)
        gradient = representer
    elif kind == "linear":
        functional = sc.InnerProductFunctional(c, domain, ctx)
        value = np.vdot(c_np, metric @ x_np)
        gradient = c_np
    elif kind == "quadratic":
        q = sc.DiagonalLinOp(ctx.asarray(diagonal), domain, ctx)
        linear = sc.InnerProductFunctional(c, domain, ctx)
        offset = np.asarray(0.75, dtype=dtype)
        functional = sc.LinOpQuadraticForm(q, linear, ctx.asarray(offset), ctx)
        value = 0.5 * np.vdot(x_np, metric @ (diagonal * x_np))
        value = value + np.vdot(c_np, metric @ x_np) + offset
        gradient = diagonal * x_np + c_np
    else:
        raise ValueError(f"Unknown functional case kind {kind!r}.")

    matrix = np.asarray(
        [[1.0, 0.25, 0.0], [-0.5, 1.5, 0.25], [0.0, -0.75, 1.25]],
        dtype=dtype,
    )
    operator = sc.DenseLinOp(ctx.asarray(matrix), domain, domain, ctx)
    pullback = functional.compose(operator)
    applied = matrix @ x_np
    metric_adjoint = np.linalg.solve(metric, matrix.conj().T @ metric)
    if kind == "quadratic":
        pullback_value = 0.5 * np.vdot(applied, metric @ (diagonal * applied))
        pullback_value = pullback_value + np.vdot(c_np, metric @ applied) + offset
        pullback_gradient = metric_adjoint @ (diagonal * applied + c_np)
    else:
        pullback_value = np.vdot(gradient, metric @ applied)
        pullback_gradient = metric_adjoint @ gradient

    target_ctx = _context(_target_dtype(dtype), check_level)
    capabilities = {"gradient", "pullback", "conversion", geometry, kind}
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        capabilities.add("complex")
    else:
        capabilities.add("real")

    return FunctionalCase(
        obj=functional,
        reference={
            "kind": kind,
            "domain": domain,
            "x": x,
            "value": value,
            "gradient": ctx.asarray(gradient),
            "coordinate_gradient": ctx.asarray(metric @ gradient),
            "operator": operator,
            "pullback": pullback,
            "pullback_x": x,
            "pullback_value": pullback_value,
            "pullback_gradient": ctx.asarray(pullback_gradient),
            "target_ctx": target_ctx,
            "check_level": check_level,
        },
        capabilities=frozenset(capabilities),
        id=f"{kind}-{geometry}-{np.dtype(dtype).name}-checks-{check_level}",
    )


_BATTERY_WEIGHTS = np.asarray([2.0, 5.0, 11.0])


def _battery_case(
    dtype: Any,
    check_level: sc.CheckLevel | str,
    *,
    kind: str,
    weighted: bool,
    build: Any,
    x_np: np.ndarray,
    value: Any,
    euclidean_gradient: np.ndarray,
) -> FunctionalCase:
    """Build a generated case for an ADR-019 battery functional.

    ``euclidean_gradient`` is the coordinate gradient ``d phi / d x_i``; the
    recorded Riesz gradient divides it by the weights on a weighted metric (the
    identity exercised by the directional-derivative law).
    """
    ctx = _context(dtype, check_level)
    domain = _space(ctx, weighted)
    weights = np.asarray(_BATTERY_WEIGHTS, dtype=dtype)
    gradient = euclidean_gradient / weights if weighted else euclidean_gradient
    geometry = "weighted" if weighted else "euclidean"
    functional = build(domain, ctx)
    target_ctx = _context(_target_dtype(dtype), check_level)
    return FunctionalCase(
        obj=functional,
        reference={
            "kind": kind,
            "domain": domain,
            "x": ctx.asarray(x_np),
            "value": value,
            "gradient": ctx.asarray(gradient),
            "target_ctx": target_ctx,
            "check_level": check_level,
        },
        capabilities=frozenset({"gradient", "conversion", "real", geometry, kind}),
        id=f"{kind}-{geometry}-{np.dtype(dtype).name}-checks-{check_level}",
    )


def _battery_cases(
    dtype: Any, check_level: sc.CheckLevel | str
) -> tuple[FunctionalCase, ...]:
    """Generated cases for the ADR-019 battery functionals (real coordinates)."""
    x = np.asarray([0.5, -1.0, 2.0], dtype=dtype)
    x_pos = np.asarray([0.5, 1.0, 2.0], dtype=dtype)
    x_huber = np.asarray([0.5, -1.5, 2.0], dtype=dtype)
    target = np.asarray([1.0, 2.0, 0.5], dtype=dtype)
    weights = np.asarray(_BATTERY_WEIGHTS, dtype=dtype)
    delta = 1.0

    lp_norm = float(np.sum(np.abs(x) ** 2.0) ** 0.5)
    a = np.abs(x_huber)
    huber_value = float(
        np.sum(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta)))
    )

    cases: list[FunctionalCase] = []
    for weighted in (False, True):
        metric_diag = weights if weighted else np.ones_like(weights)
        cases.append(
            _battery_case(
                dtype, check_level, kind="squared-l2-norm", weighted=weighted,
                build=lambda dom, ctx: sc.SquaredL2NormFunctional(dom, ctx),
                x_np=x,
                value=float(0.5 * np.sum(metric_diag * x * x)),
                euclidean_gradient=metric_diag * x,  # Riesz gradient is x
            )
        )
        cases.append(
            _battery_case(
                dtype, check_level, kind="lp-norm", weighted=weighted,
                build=lambda dom, ctx: sc.LpNormFunctional(dom, 2.0, ctx),
                x_np=x,
                value=lp_norm,
                euclidean_gradient=x / lp_norm,
            )
        )
        cases.append(
            _battery_case(
                dtype, check_level, kind="negative-entropy", weighted=weighted,
                build=lambda dom, ctx: sc.NegativeEntropyFunctional(dom, ctx),
                x_np=x_pos,
                value=float(np.sum(x_pos * np.log(x_pos))),
                euclidean_gradient=np.log(x_pos) + 1.0,
            )
        )
        cases.append(
            _battery_case(
                dtype, check_level, kind="kl-divergence", weighted=weighted,
                build=lambda dom, ctx: sc.KLDivergenceFunctional(ctx.asarray(target), dom, ctx),
                x_np=x_pos,
                value=float(np.sum(x_pos * np.log(x_pos / target))),
                euclidean_gradient=np.log(x_pos / target) + 1.0,
            )
        )
        cases.append(
            _battery_case(
                dtype, check_level, kind="huber", weighted=weighted,
                build=lambda dom, ctx: sc.HuberFunctional(dom, delta, ctx),
                x_np=x_huber,
                value=huber_value,
                euclidean_gradient=np.where(a <= delta, x_huber, delta * np.sign(x_huber)),
            )
        )
    return tuple(cases)


def _spectral_case(dtype: Any, check_level: sc.CheckLevel | str) -> FunctionalCase:
    """Generated case for the spectral (Schatten) p-norm on a Hermitian space.

    Uses ``p = 2`` so the value is the Frobenius norm and the gradient is
    ``X / ||X||_F`` -- both computable without an eigendecomposition, giving an
    independent reference. The ``"real"`` capability is intentionally omitted so
    the directional-derivative law (which assumes a length-3 coordinate
    direction) skips this matrix domain; that identity is covered in
    ``tests/functional/tools/test_spectral.py``.
    """
    ctx = _context(dtype, check_level)
    domain = sc.HermitianSpace(2, ctx=ctx)
    m = np.asarray([[2.0, 0.5], [0.5, 3.0]], dtype=dtype)
    frobenius = float(np.linalg.norm(m, "fro"))
    target_ctx = _context(_target_dtype(dtype), check_level)
    return FunctionalCase(
        obj=sc.SpectralLpNormFunctional(domain, 2.0, ctx),
        reference={
            "kind": "spectral-lp-norm",
            "domain": domain,
            "x": ctx.asarray(m),
            "value": frobenius,
            "gradient": ctx.asarray(m / frobenius),
            "target_ctx": target_ctx,
            "check_level": check_level,
        },
        capabilities=frozenset({"gradient", "conversion", "euclidean", "spectral"}),
        id=f"spectral-lp-norm-frobenius-{np.dtype(dtype).name}-checks-{check_level}",
    )


def _composed_case(dtype: Any, check_level: sc.CheckLevel | str) -> FunctionalCase:
    ctx = _context(dtype, check_level)
    domain = sc.DenseCoordinateSpace((3,), ctx)
    x_np, c_np, _weights, _diagonal = _values(dtype)
    matrix = np.asarray(
        [[1.0, -0.5, 0.25], [0.0, 1.5, -0.75], [0.5, 0.0, 1.25]],
        dtype=dtype,
    )
    x = ctx.asarray(x_np)
    c = ctx.asarray(c_np)
    operator = sc.DenseLinOp(ctx.asarray(matrix), domain, domain, ctx)
    source = sc.MatrixFreeLinearFunctional(lambda y: domain.inner(c, y), domain, ctx)
    functional = source.compose(operator)
    return FunctionalCase(
        obj=functional,
        reference={
            "kind": "composed",
            "domain": domain,
            "x": x,
            "value": np.vdot(c_np, matrix @ x_np),
            "gradient": None,
            "source_functional": source,
            "operator": operator,
            "target_ctx": None,
            "check_level": check_level,
        },
        capabilities=frozenset({"pullback", "euclidean", "composed"}),
        id=f"composed-euclidean-{np.dtype(dtype).name}-checks-{check_level}",
    )


def _explicit_composed_case(dtype: Any, check_level: sc.CheckLevel | str) -> FunctionalCase:
    """Build a ComposedFunctional directly, not via ``source.compose(operator)``.

    Mirrors :func:`_composed_case` schema-for-schema (kind, reference fields, and
    capabilities) so it satisfies the same generated laws. The only difference is
    that the object is constructed with ``sc.ComposedFunctional(...)`` instead of
    arising implicitly from ``.compose()``. The functional has no analytic Riesz
    representer, so ``gradient`` is recorded as ``None`` and the gradient-bearing
    laws skip it, exactly as the implicit composed case does.
    """
    ctx = _context(dtype, check_level)
    domain = sc.DenseCoordinateSpace((3,), ctx)
    x_np, c_np, _weights, _diagonal = _values(dtype)
    matrix = np.asarray(
        [[1.5, 0.25, -0.5], [0.0, 1.0, 0.75], [-0.25, 0.5, 1.0]],
        dtype=dtype,
    )
    x = ctx.asarray(x_np)
    c = ctx.asarray(c_np)
    operator = sc.DenseLinOp(ctx.asarray(matrix), domain, domain, ctx)
    source = sc.MatrixFreeLinearFunctional(lambda y: domain.inner(c, y), domain, ctx)
    functional = sc.ComposedFunctional(source, operator)
    return FunctionalCase(
        obj=functional,
        reference={
            "kind": "composed",
            "domain": domain,
            "x": x,
            "value": np.vdot(c_np, matrix @ x_np),
            "gradient": None,
            "source_functional": source,
            "operator": operator,
            "target_ctx": None,
            "check_level": check_level,
        },
        capabilities=frozenset({"pullback", "euclidean", "composed"}),
        id=f"composed-explicit-euclidean-{np.dtype(dtype).name}-checks-{check_level}",
    )


def _matrix_free_linear_case(dtype: Any, check_level: sc.CheckLevel | str) -> FunctionalCase:
    """Build a MatrixFreeLinearFunctional case exercised by the value law only.

    The functional stores a Python closure rather than an analytic Riesz
    representer, so it advertises neither ``gradient`` nor ``conversion`` (its
    ``convert`` keeps the original-dtype closure and would fail the converted
    value law). The value-bearing law still exercises it directly. ``gradient``
    is recorded as ``None`` to satisfy the required-field guard, matching how the
    composed case handles a missing analytic gradient.
    """
    ctx = _context(dtype, check_level)
    domain = sc.DenseCoordinateSpace((3,), ctx)
    x_np, c_np, _weights, _diagonal = _values(dtype)
    x = ctx.asarray(x_np)
    c = ctx.asarray(c_np)
    functional = sc.MatrixFreeLinearFunctional(lambda y: domain.inner(c, y), domain, ctx)
    capabilities = {"linear", "euclidean"}
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        capabilities.add("complex")
    else:
        capabilities.add("real")
    return FunctionalCase(
        obj=functional,
        reference={
            "kind": "matrix-free-linear",
            "domain": domain,
            "x": x,
            "value": np.vdot(c_np, x_np),
            "gradient": None,
            "target_ctx": None,
            "check_level": check_level,
        },
        capabilities=frozenset(capabilities),
        id=f"matrix-free-linear-euclidean-{np.dtype(dtype).name}-checks-{check_level}",
    )


def _tree_case(dtype: Any, check_level: sc.CheckLevel | str) -> FunctionalCase:
    ctx = _context(dtype, check_level)
    left = sc.DenseCoordinateSpace((2,), ctx)
    right = sc.DenseCoordinateSpace((1,), ctx)
    domain = sc.TreeSpace.from_leaf_spaces((left, right), ctx=ctx)
    complex_dtype = np.issubdtype(np.dtype(dtype), np.complexfloating)
    x_np = (
        np.asarray([1.0 + (0.5j if complex_dtype else 0.0), -2.0], dtype=dtype),
        np.asarray([0.75 - (0.25j if complex_dtype else 0.0)], dtype=dtype),
    )
    c_np = (
        np.asarray([0.5, 1.25 - (0.5j if complex_dtype else 0.0)], dtype=dtype),
        np.asarray([-1.5 + (0.25j if complex_dtype else 0.0)], dtype=dtype),
    )
    x = tuple(ctx.asarray(leaf) for leaf in x_np)
    c = tuple(ctx.asarray(leaf) for leaf in c_np)
    functional = sc.InnerProductFunctional(c, domain, ctx)
    value = sum(np.vdot(ci, xi) for ci, xi in zip(c_np, x_np))
    target_ctx = _context(_target_dtype(dtype), check_level)
    return FunctionalCase(
        obj=functional,
        reference={
            "kind": "tree-linear",
            "domain": domain,
            "x": x,
            "value": value,
            "gradient": c,
            "coordinate_gradient": c,
            "target_ctx": target_ctx,
            "check_level": check_level,
        },
        capabilities=frozenset({"gradient", "conversion", "tree", "euclidean", "linear"}),
        id=f"tree-linear-{np.dtype(dtype).name}-checks-{check_level}",
    )


def functional_cases(
    *,
    dtypes: Iterable[Any] = NUMPY_FUNCTIONAL_DTYPES,
    check_levels: Iterable[sc.CheckLevel | str] = ("standard",),
) -> tuple[FunctionalCase, ...]:
    """Generate deterministic scalar-functional cases with direct references."""
    cases = []
    for check_level in check_levels:
        for dtype in dtypes:
            for weighted in (False, True):
                for kind in ("zero", "linear", "quadratic"):
                    cases.append(
                        _dense_case(dtype, check_level, kind=kind, weighted=weighted)
                    )
            cases.append(_composed_case(dtype, check_level))
            cases.append(_explicit_composed_case(dtype, check_level))
            cases.append(_matrix_free_linear_case(dtype, check_level))
            cases.append(_tree_case(dtype, check_level))
        # ADR-019 battery functionals are real-coordinate objectives; generate
        # them once per check level for float64 when that dtype is requested.
        if any(np.dtype(d) == np.dtype(np.float64) for d in dtypes):
            cases.extend(_battery_cases(np.float64, check_level))
            cases.append(_spectral_case(np.float64, check_level))
    return tuple(cases)

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from time import perf_counter
from typing import Any, Literal

import numpy as np

from spacecore.backend import BackendOps, Context
from spacecore.linop import DenseLinOp
from spacecore.space import Space, VectorSpace, WeightedInnerProduct
from spacecore.types import DenseArray


@dataclass(slots=True)
class LinOpTestData:
    ctx: Context
    domain: Space
    codomain: Space
    operator: DenseArray
    x: DenseArray
    y: DenseArray
    xs: DenseArray | None = None
    ys: DenseArray | None = None
    domain_weights: DenseArray | None = None
    codomain_weights: DenseArray | None = None
    bare_time_s: dict[str, float] = field(default_factory=dict)
    spacecore_time_s: dict[str, float] = field(default_factory=dict)
    bare_outputs: dict[str, DenseArray] = field(default_factory=dict)
    spacecore_outputs: dict[str, DenseArray] = field(default_factory=dict)


def make_dense_linop_data(
    ctx: Context,
    *,
    domain_shape: tuple[int, ...],
    codomain_shape: tuple[int, ...],
    batch: int | None = None,
    weighted: bool = False,
    seed: int = 0,
) -> LinOpTestData:
    """Generate DenseLinOp reference data using NumPy first, then ``ctx``."""
    rng = np.random.default_rng(seed)
    domain_shape = _shape(domain_shape, "domain_shape")
    codomain_shape = _shape(codomain_shape, "codomain_shape")
    if batch is not None and batch <= 0:
        raise ValueError(f"batch must be positive or None, got {batch!r}.")

    A_np = rng.standard_normal(codomain_shape + domain_shape)
    x_np = rng.standard_normal(domain_shape)
    y_np = rng.standard_normal(codomain_shape)
    xs_np = None if batch is None else rng.standard_normal((batch,) + domain_shape)
    ys_np = None if batch is None else rng.standard_normal((batch,) + codomain_shape)

    if weighted:
        domain_weights_np = 0.5 + rng.random(domain_shape)
        codomain_weights_np = 0.5 + rng.random(codomain_shape)
        domain_weights = ctx.asarray(domain_weights_np)
        codomain_weights = ctx.asarray(codomain_weights_np)
        domain = VectorSpace(
            domain_shape,
            ctx,
            geometry=WeightedInnerProduct(domain_weights),
        )
        codomain = VectorSpace(
            codomain_shape,
            ctx,
            geometry=WeightedInnerProduct(codomain_weights),
        )
    else:
        domain_weights = None
        codomain_weights = None
        domain = VectorSpace(domain_shape, ctx)
        codomain = VectorSpace(codomain_shape, ctx)

    return LinOpTestData(
        ctx=ctx,
        domain=domain,
        codomain=codomain,
        operator=ctx.asarray(A_np),
        x=ctx.asarray(x_np),
        y=ctx.asarray(y_np),
        xs=None if xs_np is None else ctx.asarray(xs_np),
        ys=None if ys_np is None else ctx.asarray(ys_np),
        domain_weights=domain_weights,
        codomain_weights=codomain_weights,
    )


def bare_dense_linop(
    ops: BackendOps,
    data: LinOpTestData,
    kind: Literal["apply", "rapply", "vapply", "rvapply"],
    *,
    time: bool = False,
    jit: bool = False,
) -> DenseArray:
    """Compute a dense LinOp reference directly with backend array operations."""
    domain_shape = tuple(data.domain.shape)
    codomain_shape = tuple(data.codomain.shape)
    domain_size = prod(domain_shape)
    codomain_size = prod(codomain_shape)
    A = data.operator.reshape((codomain_size, domain_size))
    A_conj = ops.conj(A)
    AH = A_conj.T
    AT = A.T
    weighted = data.domain_weights is not None or data.codomain_weights is not None
    if weighted and (data.domain_weights is None or data.codomain_weights is None):
        raise ValueError("Weighted dense reference requires both domain and codomain weights.")
    wx = None if data.domain_weights is None else data.domain_weights.reshape((domain_size,))
    wy = None if data.codomain_weights is None else data.codomain_weights.reshape((codomain_size,))

    if kind == "apply":
        x_flat = data.x.reshape((domain_size,))

        def kernel(A2: DenseArray, x2: DenseArray) -> DenseArray:
            return A2 @ x2

        args = (A, x_flat)
        out_shape = codomain_shape
    elif kind == "rapply":
        y_flat = data.y.reshape((codomain_size,))
        if weighted:

            def kernel(
                A2H: DenseArray,
                y2: DenseArray,
                wy2: DenseArray,
                wx2: DenseArray,
            ) -> DenseArray:
                return (A2H @ (wy2 * y2)) / wx2

            args = (AH, y_flat, wy, wx)
        else:

            def kernel(A2H: DenseArray, y2: DenseArray) -> DenseArray:
                return A2H @ y2

            args = (AH, y_flat)
        out_shape = domain_shape
    elif kind == "vapply":
        if data.xs is None:
            raise ValueError("vapply reference requires batched xs data.")
        xs_flat = data.xs.reshape((data.xs.shape[0], domain_size))

        def kernel(xs2: DenseArray, A2T: DenseArray) -> DenseArray:
            return xs2 @ A2T

        args = (xs_flat, AT)
        out_shape = (data.xs.shape[0],) + codomain_shape
    elif kind == "rvapply":
        if data.ys is None:
            raise ValueError("rvapply reference requires batched ys data.")
        ys_flat = data.ys.reshape((data.ys.shape[0], codomain_size))
        if weighted:

            def kernel(
                ys2: DenseArray,
                A2c: DenseArray,
                wy2: DenseArray,
                wx2: DenseArray,
            ) -> DenseArray:
                return ((wy2 * ys2) @ A2c) / wx2

            args = (ys_flat, A_conj, wy, wx)
        else:

            def kernel(ys2: DenseArray, A2c: DenseArray) -> DenseArray:
                return ys2 @ A2c

            args = (ys_flat, A_conj)
        out_shape = (data.ys.shape[0],) + domain_shape
    else:
        raise ValueError(f"Unknown dense reference kind: {kind!r}.")

    selected_kernel = _jit_kernel(ops, kernel) if jit else kernel
    if time:
        result_flat, elapsed = _time_kernel(selected_kernel, *args)
        data.bare_time_s[f"{kind}:jit" if jit else kind] = elapsed
    else:
        result_flat = selected_kernel(*args)
        _sync(result_flat)
    return result_flat.reshape(out_shape)


def check_dense_linop(
    data: LinOpTestData,
    kind: Literal["apply", "rapply", "vapply", "rvapply"],
    *,
    time: bool = False,
    jit: bool = False,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> bool:
    """Run DenseLinOp and backend-only references, storing outputs and timings."""
    op = DenseLinOp(data.operator, data.domain, data.codomain, data.ctx)
    key = kind if not jit else f"{kind}:jit"
    bare = bare_dense_linop(data.ctx.ops, data, kind, time=time, jit=jit)
    kernel, arg = _spacecore_kernel(op, data, kind, jit=jit)

    if time:
        actual, elapsed = _time_kernel(kernel, arg)
        data.spacecore_time_s[key] = elapsed
    else:
        actual = kernel(arg)
        _sync(actual)

    data.bare_outputs[key] = bare
    data.spacecore_outputs[key] = actual
    return bool(data.ctx.ops.allclose(actual, bare, rtol=rtol, atol=atol))


def _shape(shape: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise TypeError(f"{name} must be a nonempty tuple[int, ...], got {shape!r}.")
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError(f"{name} must contain positive dimensions, got {shape!r}.")
    return tuple(int(dim) for dim in shape)


def _sync(result: Any) -> None:
    block = getattr(result, "block_until_ready", None)
    if block is not None:
        block()


def _time_kernel(kernel: Any, *args: Any) -> tuple[Any, float]:
    start = perf_counter()
    result = kernel(*args)
    _sync(result)
    return result, perf_counter() - start


def _spacecore_kernel(
    op: DenseLinOp,
    data: LinOpTestData,
    kind: Literal["apply", "rapply", "vapply", "rvapply"],
    *,
    jit: bool,
) -> tuple[Any, DenseArray]:
    if kind == "apply":
        method = op._apply_core if jit else op.apply
        arg = data.x
    elif kind == "rapply":
        method = op._rapply_core if jit else op.rapply
        arg = data.y
    elif kind == "vapply":
        if data.xs is None:
            raise ValueError("check_dense_linop(kind='vapply') requires data.xs.")
        method = op._vapply_core if jit else op.vapply
        arg = data.xs
    elif kind == "rvapply":
        if data.ys is None:
            raise ValueError("check_dense_linop(kind='rvapply') requires data.ys.")
        method = op._rvapply_core if jit else op.rvapply
        arg = data.ys
    else:
        raise ValueError(f"Unknown dense reference kind: {kind!r}.")

    if jit:
        jax = getattr(data.ctx.ops, "jax", None)
        if jax is None:
            raise TypeError("jit=True is supported only for JAX contexts.")
        core_method = method
        method = jax.jit(lambda z: core_method(z))
    return method, arg


def _jit_kernel(ops: BackendOps, kernel: Any) -> Any:
    jax = getattr(ops, "jax", None)
    if jax is None:
        raise TypeError("jit=True is supported only for JAX contexts.")
    return jax.jit(kernel)


__all__ = [
    "LinOpTestData",
    "make_dense_linop_data",
    "bare_dense_linop",
    "check_dense_linop",
]

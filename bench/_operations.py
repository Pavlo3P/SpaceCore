"""Operation probes that cover the same surface as the test suite.

Every probe builds a fresh problem on a per-seed, per-backend basis and
returns a :class:`ProbeCase` whose ``bare`` and ``sc`` callables are
timed by the multi-seed harness. The set mirrors what
``tests/spaces/``, ``tests/linops/``, ``tests/functional/``, and
``tests/linalg/`` cover.

Backend support per probe is declared by ``Probe.backends``. The runner
skips a backend at runtime if its library is not installed.

JAX-specific notes
~~~~~~~~~~~~~~~~~~

JAX-compatible probes report eager execution and a separate ``jax.jit``
steady-state result. The runner warms the compiled callable before timing
steady state and records the first compiled call separately as ``compile_ns``.

Sparse paths are NumPy-only because JAX sparse support is partial and
the SpaceCore sparse path goes through SciPy.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np
import scipy.sparse as sps

import spacecore as sc
from ._probes import Probe, ProbeCase, registry


# ---------------------------------------------------------------------------
# Backend context helpers


_REAL_DTYPE_PER_BACKEND = {
    "numpy": np.float64,
    "jax": None,   # resolved at call time via jax_real_dtype()
    "torch": None,  # resolved via torch.get_default_dtype()
}

_ACTIVE_CHECK_LEVEL: ContextVar[str] = ContextVar(
    "bench_check_level", default="cheap"
)


@contextmanager
def benchmark_check_level(check_level: str):
    """Build probe contexts with one explicit benchmark check level."""
    token = _ACTIVE_CHECK_LEVEL.set(check_level)
    try:
        yield
    finally:
        _ACTIVE_CHECK_LEVEL.reset(token)


def _backend_ctx(backend: str, *, check_level: str | None = None) -> sc.Context:
    """Build a SpaceCore ``Context`` for the requested backend."""
    check_level = check_level or _ACTIVE_CHECK_LEVEL.get()
    if backend == "numpy":
        return sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)
    if backend == "jax":
        from tests._helpers import jax_real_dtype

        return sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level=check_level)
    if backend == "torch":
        from tests._helpers import torch_real_dtype

        td = torch_real_dtype()
        return sc.Context(sc.TorchOps(), dtype=td, check_level=check_level)
    raise ValueError(f"unknown backend {backend!r}")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _np_dtype(ctx: sc.Context) -> np.dtype:
    """Return the NumPy dtype matching this context's dtype."""
    try:
        return np.dtype(ctx.dtype)
    except TypeError:
        # Torch dtypes do not convert directly; route through str.
        name = str(ctx.dtype).rsplit(".", 1)[-1]
        return np.dtype(name)


def _dense_vector(ctx: sc.Context, n: int, seed: int) -> tuple[Any, Any, np.ndarray]:
    """Return ``(space, x_array, x_np)`` for a fresh dense vector."""
    from tests.generators import dense_array_case

    space = sc.DenseCoordinateSpace((n,), ctx)
    generated = dense_array_case(ctx, (n,), seed=seed)
    x_np = np.asarray(generated.reference["array"], dtype=_np_dtype(ctx))
    return space, generated.obj, x_np


def _dense_matrix(ctx: sc.Context, n: int, seed: int) -> tuple[Any, np.ndarray]:
    """Return ``(matrix_array, matrix_np)`` for a square dense matrix."""
    from tests.generators import dense_array_case

    generated = dense_array_case(ctx, (n, n), seed=seed)
    a_np = np.asarray(generated.reference["array"], dtype=_np_dtype(ctx))
    return generated.obj, a_np


# ---------------------------------------------------------------------------
# Space probes
#
# ``space`` probes test the cost of SpaceCore's ``DenseCoordinateSpace``
# arithmetic and validation wrappers vs raw NumPy / JAX / Torch on the
# underlying arrays. The "bare" callable is the un-wrapped operation.


def _make_space_add(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    _, y, y_np = _dense_vector(ctx, size, seed + 100)
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ux, uy = unchecked_ctx.asarray(x_np), unchecked_ctx.asarray(y_np)
    return ProbeCase(
        bare_label=f"{backend}: x + y",
        sc_label="DenseCoordinateSpace.add",
        bare=lambda: x + y,
        sc=lambda: space.add(x, y),
        unchecked=lambda: unchecked_space.add(ux, uy),
        reference=lambda: x_np + y_np,
        bare_inputs=(x, y),
    )


def _make_space_scale(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    alpha = float(_rng(seed + 1).standard_normal())
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ux = unchecked_ctx.asarray(x_np)
    return ProbeCase(
        bare_label=f"{backend}: alpha * x",
        sc_label="DenseCoordinateSpace.scale",
        bare=lambda: alpha * x,
        sc=lambda: space.scale(alpha, x),
        unchecked=lambda: unchecked_space.scale(alpha, ux),
        reference=lambda: alpha * x_np,
        bare_inputs=(x,),
    )


def _make_space_inner(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    _, y, y_np = _dense_vector(ctx, size, seed + 100)
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ux, uy = unchecked_ctx.asarray(x_np), unchecked_ctx.asarray(y_np)
    return ProbeCase(
        bare_label=f"{backend}: vdot(x, y)",
        sc_label="DenseCoordinateSpace.inner",
        bare=lambda: ctx.ops.vdot(x, y),
        sc=lambda: space.inner(x, y),
        unchecked=lambda: unchecked_space.inner(ux, uy),
        reference=lambda: np.vdot(x_np, y_np),
        bare_inputs=(x, y),
    )


def _make_space_norm(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ux = unchecked_ctx.asarray(x_np)
    return ProbeCase(
        bare_label=f"{backend}: linalg.norm(x)",
        sc_label="DenseCoordinateSpace.norm",
        bare=lambda: ctx.ops.sqrt(ctx.ops.real(ctx.ops.vdot(x, x))),
        sc=lambda: space.norm(x),
        unchecked=lambda: unchecked_space.norm(ux),
        reference=lambda: np.linalg.norm(x_np),
        bare_inputs=(x,),
    )


def _make_space_zeros(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    np_dtype = _np_dtype(ctx)
    return ProbeCase(
        bare_label=f"{backend}: zeros(n)",
        sc_label="DenseCoordinateSpace.zeros",
        bare=lambda: ctx.ops.zeros((size,), dtype=ctx.dtype),
        sc=lambda: space.zeros(),
        reference=lambda: np.zeros(size, dtype=np_dtype),
    )


for _name, _factory in [
    ("space.add", _make_space_add),
    ("space.scale", _make_space_scale),
    ("space.inner", _make_space_inner),
    ("space.norm", _make_space_norm),
    ("space.zeros", _make_space_zeros),
]:
    registry.register(
        Probe(
            name=_name,
            family="space",
            factory=_factory,
            sizes=(256, 4096, 65536),
            backends=("numpy", "jax", "torch"),
            jit_compatible=_name in {"space.add", "space.scale", "space.inner", "space.norm"},
        )
    )


# ---------------------------------------------------------------------------
# LinOp probes


def _make_dense_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    _, x, x_np = _dense_vector(ctx, size, seed + 7)
    op = sc.DenseLinOp(a, space, space, ctx)
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ua, ux = unchecked_ctx.asarray(a_np), unchecked_ctx.asarray(x_np)
    unchecked_op = sc.DenseLinOp(ua, unchecked_space, unchecked_space, unchecked_ctx)
    return ProbeCase(
        bare_label=f"{backend}: A @ x",
        sc_label="DenseLinOp.apply",
        bare=lambda: a @ x,
        sc=lambda: op.apply(x),
        unchecked=lambda: unchecked_op.apply(ux),
        reference=lambda: a_np @ x_np,
        bare_inputs=(a, x),
    )


def _make_dense_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    _, y, y_np = _dense_vector(ctx, size, seed + 8)
    op = sc.DenseLinOp(a, space, space, ctx)
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ua, uy = unchecked_ctx.asarray(a_np), unchecked_ctx.asarray(y_np)
    unchecked_op = sc.DenseLinOp(ua, unchecked_space, unchecked_space, unchecked_ctx)
    return ProbeCase(
        bare_label=f"{backend}: A.T.conj() @ y",
        sc_label="DenseLinOp.rapply",
        bare=lambda: a.conj().T @ y,
        sc=lambda: op.rapply(y),
        unchecked=lambda: unchecked_op.rapply(uy),
        reference=lambda: a_np.conj().T @ y_np,
        bare_inputs=(a, y),
    )


def _make_dense_vapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    rng = _rng(seed + 9)
    xs_np = np.asarray(rng.standard_normal((8, size)), dtype=_np_dtype(ctx))
    xs = ctx.asarray(xs_np)
    op = sc.DenseLinOp(a, space, space, ctx)
    unchecked_ctx = _backend_ctx(backend, check_level="none")
    unchecked_space = sc.DenseCoordinateSpace((size,), unchecked_ctx)
    ua, uxs = unchecked_ctx.asarray(a_np), unchecked_ctx.asarray(xs_np)
    unchecked_op = sc.DenseLinOp(ua, unchecked_space, unchecked_space, unchecked_ctx)
    return ProbeCase(
        bare_label=f"{backend}: xs @ A.T",
        sc_label="DenseLinOp.vapply",
        bare=lambda: xs @ a.T,
        sc=lambda: op.vapply(xs),
        unchecked=lambda: unchecked_op.vapply(uxs),
        reference=lambda: xs_np @ a_np.T,
        bare_inputs=(xs, a),
    )


def _make_diagonal_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=_np_dtype(ctx))
    _, x, x_np = _dense_vector(ctx, size, seed + 10)
    op = sc.DiagonalLinOp(ctx.asarray(d_np), space, ctx)
    d = op.diagonal
    return ProbeCase(
        bare_label=f"{backend}: d * x",
        sc_label="DiagonalLinOp.apply",
        bare=lambda: d * x,
        sc=lambda: op.apply(x),
        reference=lambda: d_np * x_np,
        bare_inputs=(d, x),
    )


def _make_sparse_apply(backend: str, seed: int, size: int) -> ProbeCase:
    # NumPy-only: SpaceCore sparse goes through SciPy.
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    density = min(0.05, 32.0 / size)
    a_np = sps.random(
        size, size, density=density, format="csr", random_state=rng, dtype=_np_dtype(ctx)
    )
    _, x, x_np = _dense_vector(ctx, size, seed + 11)
    op = sc.SparseLinOp(ctx.assparse(a_np), space, space, ctx)
    return ProbeCase(
        bare_label="scipy: A @ x",
        sc_label="SparseLinOp.apply",
        bare=lambda: a_np @ x_np,
        sc=lambda: op.apply(x),
        reference=lambda: np.asarray(a_np @ x_np),
    )


def _make_identity_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    op = sc.IdentityLinOp(space, ctx)
    return ProbeCase(
        bare_label="x",
        sc_label="IdentityLinOp.apply",
        bare=lambda: x_np,
        sc=lambda: op.apply(x),
        reference=lambda: x_np,
    )


def _make_composed_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, x, x_np = _dense_vector(ctx, size, seed + 12)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    composed = op_a @ op_b
    return ProbeCase(
        bare_label=f"{backend}: A @ (B @ x)",
        sc_label="(A @ B).apply",
        bare=lambda: a_np @ (b_np @ x_np),
        sc=lambda: composed.apply(x),
        reference=lambda: a_np @ (b_np @ x_np),
    )


def _make_summed_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, x, x_np = _dense_vector(ctx, size, seed + 13)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    summed = op_a + op_b
    return ProbeCase(
        bare_label=f"{backend}: A @ x + B @ x",
        sc_label="(A + B).apply",
        bare=lambda: a_np @ x_np + b_np @ x_np,
        sc=lambda: summed.apply(x),
        reference=lambda: a_np @ x_np + b_np @ x_np,
    )


def _make_scaled_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    _, x, x_np = _dense_vector(ctx, size, seed + 14)
    alpha = float(_rng(seed + 2).standard_normal())
    op_a = sc.DenseLinOp(a, space, space, ctx)
    scaled = alpha * op_a
    return ProbeCase(
        bare_label=f"{backend}: alpha * (A @ x)",
        sc_label="(alpha * A).apply",
        bare=lambda: alpha * (a_np @ x_np),
        sc=lambda: scaled.apply(x),
        reference=lambda: alpha * (a_np @ x_np),
    )


_LINOP_PROBES = [
    ("linop.dense.apply", _make_dense_apply, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.dense.rapply", _make_dense_rapply, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.dense.vapply", _make_dense_vapply, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.diagonal.apply", _make_diagonal_apply, (256, 4096, 65536), ("numpy", "jax", "torch")),
    ("linop.sparse.apply", _make_sparse_apply, (256, 4096, 65536), ("numpy",)),
    ("linop.identity.apply", _make_identity_apply, (256, 4096), ("numpy", "jax", "torch")),
    ("linop.composed.apply", _make_composed_apply, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.summed.apply", _make_summed_apply, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.scaled.apply", _make_scaled_apply, (64, 256, 1024), ("numpy", "jax", "torch")),
]
for _name, _factory, _sizes, _backends in _LINOP_PROBES:
    registry.register(
        Probe(
            name=_name,
            family="linop",
            factory=_factory,
            sizes=_sizes,
            backends=_backends,
            jit_compatible=_name.startswith("linop.dense."),
        )
    )


# ---------------------------------------------------------------------------
# Functional probes


def _make_inner_product_functional_value(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    _, w, w_np = _dense_vector(ctx, size, seed + 1)
    func = sc.InnerProductFunctional(w, space, ctx)
    return ProbeCase(
        bare_label=f"{backend}: vdot(w, x)",
        sc_label="InnerProductFunctional.value",
        bare=lambda: float(np.vdot(w_np, x_np)),
        sc=lambda: float(func.value(x)),
        reference=lambda: float(np.vdot(w_np, x_np)),
    )


def _make_inner_product_functional_grad(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    _, w, w_np = _dense_vector(ctx, size, seed + 1)
    func = sc.InnerProductFunctional(w, space, ctx)
    return ProbeCase(
        bare_label=f"{backend}: w",
        sc_label="InnerProductFunctional.grad",
        bare=lambda: w_np,
        sc=lambda: func.grad(x),
        reference=lambda: w_np,
    )


def _make_linop_quadratic_form_value(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    a_np = 0.5 * (a_np + a_np.T)
    a = ctx.asarray(a_np)
    _, x, x_np = _dense_vector(ctx, size, seed + 1)
    op = sc.DenseLinOp(a, space, space, ctx)
    func = sc.LinOpQuadraticForm(op, ctx=ctx)
    return ProbeCase(
        bare_label=f"{backend}: 0.5 * x @ A @ x",
        sc_label="LinOpQuadraticForm.value",
        bare=lambda: float(0.5 * x_np @ (a_np @ x_np)),
        sc=lambda: float(func.value(x)),
        reference=lambda: float(0.5 * x_np @ (a_np @ x_np)),
    )


for _name, _factory in [
    ("functional.inner_product.value", _make_inner_product_functional_value),
    ("functional.inner_product.grad", _make_inner_product_functional_grad),
    ("functional.quadratic.value", _make_linop_quadratic_form_value),
]:
    registry.register(
        Probe(
            name=_name,
            family="functional",
            factory=_factory,
            sizes=(256, 4096),
            backends=("numpy", "jax", "torch"),
        )
    )


# ---------------------------------------------------------------------------
# Generator-backed probes
#
# These reuse the same public-object generators as the law tests. Most
# benchmarks above scale synthetic square problems to expose amortization;
# the generated probes complement that with small, current-API instances that
# track the test suite's object construction paths.


def _make_generated_functional_value(backend: str, seed: int, size: int) -> ProbeCase:
    from tests._helpers import to_numpy
    from tests.generators import functional_cases

    # Functional generators are NumPy-backed by design.
    check_level = _ACTIVE_CHECK_LEVEL.get()
    generated = next(
        case for case in functional_cases(
            dtypes=(np.float64,), check_levels=(check_level,)
        )
        if {"linear", "euclidean"}.issubset(case.capabilities)
    )
    func = generated.obj
    ref = generated.reference
    x = ref["x"]
    gradient_np = np.asarray(to_numpy(ref["gradient"]))
    x_np = np.asarray(to_numpy(x))
    expected = ref["value"]
    return ProbeCase(
        bare_label="numpy: generated vdot(c, x)",
        sc_label="generated InnerProductFunctional.value",
        bare=lambda: np.vdot(gradient_np, x_np),
        sc=lambda: func.value(x),
        reference=lambda: expected,
    )


for _name, _factory, _backends in [
    (
        "functional.generated_linear.value",
        _make_generated_functional_value,
        ("numpy",),
    ),
]:
    registry.register(
        Probe(
            name=_name,
            family="functional" if _name.startswith("functional.") else "linop",
            factory=_factory,
            sizes=(3,),
            backends=_backends,
            notes="built with tests.generators",
        )
    )


# ---------------------------------------------------------------------------
# Extended Space probes — DenseCoordinateSpace flatten/unflatten/convert
#
# These exercise the validation wrappers around reshape / no-op paths so
# the probe surface tracks the spec coverage in tests/spaces/.


def _make_space_flatten(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    n = int(size)
    space = sc.DenseCoordinateSpace((n, n), ctx)
    x_np = np.asarray(_rng(seed).standard_normal((n, n)), dtype=_np_dtype(ctx))
    x = ctx.asarray(x_np)
    flat_np = x_np.reshape((-1,))
    return ProbeCase(
        bare_label=f"{backend}: x.reshape(-1)",
        sc_label="DenseCoordinateSpace.flatten",
        bare=lambda: x_np.reshape((-1,)),
        sc=lambda: space.flatten(x),
        reference=lambda: flat_np,
    )


def _make_space_unflatten(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    n = int(size)
    space = sc.DenseCoordinateSpace((n, n), ctx)
    v_np = np.asarray(_rng(seed).standard_normal(n * n), dtype=_np_dtype(ctx))
    v = ctx.asarray(v_np)
    mat_np = v_np.reshape((n, n))
    return ProbeCase(
        bare_label=f"{backend}: v.reshape(n, n)",
        sc_label="DenseCoordinateSpace.unflatten",
        bare=lambda: v_np.reshape((n, n)),
        sc=lambda: space.unflatten(v),
        reference=lambda: mat_np,
    )


for _name, _factory in [
    ("space.flatten", _make_space_flatten),
    ("space.unflatten", _make_space_unflatten),
]:
    registry.register(
        Probe(
            name=_name,
            family="space",
            factory=_factory,
            sizes=(32, 64, 128),
            backends=("numpy", "jax", "torch"),
        )
    )


def _make_space_convert_self(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    return ProbeCase(
        bare_label="space",
        sc_label="DenseCoordinateSpace.convert(self.ctx)",
        bare=lambda: space,
        sc=lambda: space.convert(ctx),
        reference=lambda: space,
    )


registry.register(
    Probe(
        name="space.convert.self",
        family="space",
        factory=_make_space_convert_self,
        sizes=(256, 4096),
        backends=("numpy", "jax", "torch"),
    )
)


# ---------------------------------------------------------------------------
# HermitianSpace probes


def _hermitian_matrix(ctx: sc.Context, n: int, seed: int) -> tuple[Any, np.ndarray]:
    """Return ``(matrix_array, matrix_np)`` for a random Hermitian (symmetric)."""
    a_np = np.asarray(_rng(seed).standard_normal((n, n)), dtype=_np_dtype(ctx))
    sym_np = 0.5 * (a_np + a_np.T)
    return ctx.asarray(sym_np), sym_np


def _make_hermitian_symmetrize(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.HermitianSpace(int(size), ctx=ctx)
    a_np = np.asarray(_rng(seed).standard_normal((size, size)), dtype=_np_dtype(ctx))
    a = ctx.asarray(a_np)
    ref = 0.5 * (a_np + a_np.conj().T)
    return ProbeCase(
        bare_label=f"{backend}: 0.5 * (A + A.conj().T)",
        sc_label="HermitianSpace.symmetrize",
        bare=lambda: 0.5 * (a_np + a_np.conj().T),
        sc=lambda: space.symmetrize(a),
        reference=lambda: ref,
    )


def _make_hermitian_inner(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.HermitianSpace(int(size), ctx=ctx)
    a, a_np = _hermitian_matrix(ctx, size, seed)
    b, b_np = _hermitian_matrix(ctx, size, seed + 1)
    ref = np.vdot(a_np.ravel(), b_np.ravel())
    return ProbeCase(
        bare_label=f"{backend}: vdot(A.ravel(), B.ravel())",
        sc_label="HermitianSpace.inner",
        bare=lambda: np.vdot(a_np.ravel(), b_np.ravel()),
        sc=lambda: space.inner(a, b),
        reference=lambda: ref,
    )


def _make_hermitian_spectral_decompose(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.HermitianSpace(int(size), ctx=ctx)
    a, a_np = _hermitian_matrix(ctx, size, seed)
    return ProbeCase(
        bare_label=f"{backend}: linalg.eigh(A)",
        sc_label="HermitianSpace.spectral_decompose",
        bare=lambda: np.linalg.eigh(a_np),
        sc=lambda: space.spectral_decompose(a),
        reference=lambda: np.linalg.eigh(a_np),
    )


def _make_hermitian_spectrum(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.HermitianSpace(int(size), ctx=ctx)
    a, a_np = _hermitian_matrix(ctx, size, seed)
    return ProbeCase(
        bare_label=f"{backend}: linalg.eigvalsh(A)",
        sc_label="HermitianSpace.spectrum",
        bare=lambda: np.linalg.eigvalsh(a_np),
        sc=lambda: space.spectrum(a),
        reference=lambda: np.linalg.eigvalsh(a_np),
    )


def _make_hermitian_from_spectrum(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.HermitianSpace(int(size), ctx=ctx)
    _, a_np = _hermitian_matrix(ctx, size, seed)
    vals_np, vecs_np = np.linalg.eigh(a_np)
    vals = ctx.asarray(vals_np)
    vecs = ctx.asarray(vecs_np)
    ref = vecs_np @ np.diag(vals_np) @ vecs_np.conj().T
    return ProbeCase(
        bare_label=f"{backend}: V @ diag(lambda) @ V.conj().T",
        sc_label="HermitianSpace.from_spectrum",
        bare=lambda: vecs_np @ np.diag(vals_np) @ vecs_np.conj().T,
        sc=lambda: space.from_spectrum(vals, vecs),
        reference=lambda: ref,
    )


for _name, _factory in [
    ("space.hermitian.symmetrize", _make_hermitian_symmetrize),
    ("space.hermitian.inner", _make_hermitian_inner),
    ("space.hermitian.spectral_decompose", _make_hermitian_spectral_decompose),
    ("space.hermitian.spectrum", _make_hermitian_spectrum),
    ("space.hermitian.from_spectrum", _make_hermitian_from_spectrum),
]:
    registry.register(
        Probe(
            name=_name,
            family="space",
            factory=_factory,
            sizes=(8, 16, 32, 64),
            backends=("numpy", "jax", "torch"),
        )
    )


# ---------------------------------------------------------------------------
# StackedSpace probes


def _make_stacked_add(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    base = sc.DenseCoordinateSpace((size,), ctx)
    stacked = sc.StackedSpace(base, 4, ctx)
    np_dtype = _np_dtype(ctx)
    x_np = np.asarray(_rng(seed).standard_normal((4, size)), dtype=np_dtype)
    y_np = np.asarray(_rng(seed + 1).standard_normal((4, size)), dtype=np_dtype)
    x = ctx.asarray(x_np)
    y = ctx.asarray(y_np)
    return ProbeCase(
        bare_label=f"{backend}: x + y",
        sc_label="StackedSpace.add",
        bare=lambda: x_np + y_np,
        sc=lambda: stacked.add(x, y),
        reference=lambda: x_np + y_np,
    )


def _make_stacked_inner(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    base = sc.DenseCoordinateSpace((size,), ctx)
    stacked = sc.StackedSpace(base, 4, ctx)
    np_dtype = _np_dtype(ctx)
    x_np = np.asarray(_rng(seed).standard_normal((4, size)), dtype=np_dtype)
    y_np = np.asarray(_rng(seed + 1).standard_normal((4, size)), dtype=np_dtype)
    x = ctx.asarray(x_np)
    y = ctx.asarray(y_np)
    ref = np.vdot(x_np, y_np)
    return ProbeCase(
        bare_label=f"{backend}: vdot(x, y)",
        sc_label="StackedSpace.inner",
        bare=lambda: np.vdot(x_np, y_np),
        sc=lambda: stacked.inner(x, y),
        reference=lambda: ref,
    )


def _make_stacked_zeros(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    base = sc.DenseCoordinateSpace((size,), ctx)
    stacked = sc.StackedSpace(base, 4, ctx)
    np_dtype = _np_dtype(ctx)
    return ProbeCase(
        bare_label=f"{backend}: zeros((4, n))",
        sc_label="StackedSpace.zeros",
        bare=lambda: np.zeros((4, size), dtype=np_dtype),
        sc=lambda: stacked.zeros(),
        reference=lambda: np.zeros((4, size), dtype=np_dtype),
    )


for _name, _factory in [
    ("space.stacked.add", _make_stacked_add),
    ("space.stacked.inner", _make_stacked_inner),
    ("space.stacked.zeros", _make_stacked_zeros),
]:
    registry.register(
        Probe(
            name=_name,
            family="space",
            factory=_factory,
            sizes=(256, 4096),
            backends=("numpy", "jax", "torch"),
        )
    )


# ---------------------------------------------------------------------------
# ElementwiseJordanSpace probes


def _make_elementwise_jordan_spectrum(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.ElementwiseJordanSpace((size,), ctx)
    x_np = np.asarray(_rng(seed).standard_normal(size), dtype=_np_dtype(ctx))
    x = ctx.asarray(x_np)
    return ProbeCase(
        bare_label=f"{backend}: x",
        sc_label="ElementwiseJordanSpace.spectrum",
        bare=lambda: x_np,
        sc=lambda: space.spectrum(x),
        reference=lambda: x_np,
    )


def _make_elementwise_jordan_spectral_decompose(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.ElementwiseJordanSpace((size,), ctx)
    x_np = np.asarray(_rng(seed).standard_normal(size), dtype=_np_dtype(ctx))
    x = ctx.asarray(x_np)
    return ProbeCase(
        bare_label=f"{backend}: (x, None)",
        sc_label="ElementwiseJordanSpace.spectral_decompose",
        bare=lambda: (x_np, None),
        sc=lambda: space.spectral_decompose(x),
        reference=lambda: (x_np, None),
    )


def _make_elementwise_jordan_from_spectrum(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.ElementwiseJordanSpace((size,), ctx)
    vals_np = np.asarray(_rng(seed).standard_normal(size), dtype=_np_dtype(ctx))
    vals = ctx.asarray(vals_np)
    return ProbeCase(
        bare_label=f"{backend}: eigvals",
        sc_label="ElementwiseJordanSpace.from_spectrum",
        bare=lambda: vals_np,
        sc=lambda: space.from_spectrum(vals, None),
        reference=lambda: vals_np,
    )


def _make_elementwise_jordan_star(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.ElementwiseJordanSpace((size,), ctx)
    x_np = np.asarray(_rng(seed).standard_normal(size), dtype=_np_dtype(ctx))
    x = ctx.asarray(x_np)
    ref = x_np.conj()
    return ProbeCase(
        bare_label=f"{backend}: conj(x)",
        sc_label="ElementwiseJordanSpace.star",
        bare=lambda: x_np.conj(),
        sc=lambda: space.star(x),
        reference=lambda: ref,
    )


for _name, _factory in [
    ("space.elementwise_jordan.spectrum", _make_elementwise_jordan_spectrum),
    ("space.elementwise_jordan.spectral_decompose", _make_elementwise_jordan_spectral_decompose),
    ("space.elementwise_jordan.from_spectrum", _make_elementwise_jordan_from_spectrum),
    ("space.elementwise_jordan.star", _make_elementwise_jordan_star),
]:
    registry.register(
        Probe(
            name=_name,
            family="space",
            factory=_factory,
            sizes=(256, 4096),
            backends=("numpy", "jax", "torch"),
        )
    )


# ---------------------------------------------------------------------------
# Extended LinOp probes — fill missing apply/rapply/vapply/rvapply combos


def _make_dense_H_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    _, y, y_np = _dense_vector(ctx, size, seed + 21)
    op = sc.DenseLinOp(a, space, space, ctx)
    op_H = op.H
    ref = a_np.conj().T @ y_np
    return ProbeCase(
        bare_label=f"{backend}: A.conj().T @ y",
        sc_label="DenseLinOp.H.apply",
        bare=lambda: a_np.conj().T @ y_np,
        sc=lambda: op_H.apply(y),
        reference=lambda: ref,
    )


def _make_dense_to_dense(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    op = sc.DenseLinOp(a, space, space, ctx)
    return ProbeCase(
        bare_label=f"{backend}: A",
        sc_label="DenseLinOp.to_dense",
        bare=lambda: a_np,
        sc=lambda: op.to_dense(),
        reference=lambda: a_np,
    )


def _make_diagonal_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=_np_dtype(ctx))
    _, y, y_np = _dense_vector(ctx, size, seed + 22)
    op = sc.DiagonalLinOp(ctx.asarray(d_np), space, ctx)
    ref = d_np.conj() * y_np
    return ProbeCase(
        bare_label=f"{backend}: conj(d) * y",
        sc_label="DiagonalLinOp.rapply",
        bare=lambda: d_np.conj() * y_np,
        sc=lambda: op.rapply(y),
        reference=lambda: ref,
    )


def _make_diagonal_vapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=_np_dtype(ctx))
    xs_np = np.asarray(_rng(seed + 23).standard_normal((8, size)), dtype=_np_dtype(ctx))
    xs = ctx.asarray(xs_np)
    op = sc.DiagonalLinOp(ctx.asarray(d_np), space, ctx)
    ref = xs_np * d_np
    return ProbeCase(
        bare_label=f"{backend}: xs * d",
        sc_label="DiagonalLinOp.vapply",
        bare=lambda: xs_np * d_np,
        sc=lambda: op.vapply(xs),
        reference=lambda: ref,
    )


def _make_diagonal_rvapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=_np_dtype(ctx))
    ys_np = np.asarray(_rng(seed + 24).standard_normal((8, size)), dtype=_np_dtype(ctx))
    ys = ctx.asarray(ys_np)
    op = sc.DiagonalLinOp(ctx.asarray(d_np), space, ctx)
    ref = ys_np * d_np.conj()
    return ProbeCase(
        bare_label=f"{backend}: ys * conj(d)",
        sc_label="DiagonalLinOp.rvapply",
        bare=lambda: ys_np * d_np.conj(),
        sc=lambda: op.rvapply(ys),
        reference=lambda: ref,
    )


def _make_sparse_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    # NumPy-only.
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    density = min(0.05, 32.0 / size)
    a_np = sps.random(
        size, size, density=density, format="csr", random_state=rng, dtype=_np_dtype(ctx)
    )
    _, y, y_np = _dense_vector(ctx, size, seed + 25)
    op = sc.SparseLinOp(ctx.assparse(a_np), space, space, ctx)
    ref_np = np.asarray(a_np.conj().T @ y_np)
    return ProbeCase(
        bare_label="scipy: A.conj().T @ y",
        sc_label="SparseLinOp.rapply",
        bare=lambda: a_np.conj().T @ y_np,
        sc=lambda: op.rapply(y),
        reference=lambda: ref_np,
    )


def _make_composed_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, y, y_np = _dense_vector(ctx, size, seed + 26)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    composed = op_a @ op_b
    ref = b_np.conj().T @ (a_np.conj().T @ y_np)
    return ProbeCase(
        bare_label=f"{backend}: B.conj().T @ (A.conj().T @ y)",
        sc_label="(A @ B).rapply",
        bare=lambda: b_np.conj().T @ (a_np.conj().T @ y_np),
        sc=lambda: composed.rapply(y),
        reference=lambda: ref,
    )


_EXTRA_LINOP_PROBES_BASIC = [
    ("linop.dense.H_apply", _make_dense_H_apply, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.dense.to_dense", _make_dense_to_dense, (64, 256, 1024), ("numpy", "jax", "torch")),
    ("linop.diagonal.rapply", _make_diagonal_rapply, (256, 4096, 65536), ("numpy", "jax", "torch")),
    ("linop.diagonal.vapply", _make_diagonal_vapply, (256, 4096, 65536), ("numpy", "jax", "torch")),
    ("linop.diagonal.rvapply", _make_diagonal_rvapply, (256, 4096, 65536), ("numpy", "jax", "torch")),
    ("linop.sparse.rapply", _make_sparse_rapply, (256, 4096, 65536), ("numpy",)),
    ("linop.composed.rapply", _make_composed_rapply, (64, 256, 1024), ("numpy", "jax", "torch")),
]
for _name, _factory, _sizes, _backends in _EXTRA_LINOP_PROBES_BASIC:
    registry.register(
        Probe(name=_name, family="linop", factory=_factory, sizes=_sizes, backends=_backends)
    )


# ---------------------------------------------------------------------------
# ZeroLinOp probes


def _make_zero_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    _, x, _ = _dense_vector(ctx, size, seed)
    op = sc.ZeroLinOp(space, space, ctx)
    np_dtype = _np_dtype(ctx)
    return ProbeCase(
        bare_label=f"{backend}: zeros(n)",
        sc_label="ZeroLinOp.apply",
        bare=lambda: np.zeros(size, dtype=np_dtype),
        sc=lambda: op.apply(x),
        reference=lambda: np.zeros(size, dtype=np_dtype),
    )


def _make_zero_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    _, y, _ = _dense_vector(ctx, size, seed)
    op = sc.ZeroLinOp(space, space, ctx)
    np_dtype = _np_dtype(ctx)
    return ProbeCase(
        bare_label=f"{backend}: zeros(n)",
        sc_label="ZeroLinOp.rapply",
        bare=lambda: np.zeros(size, dtype=np_dtype),
        sc=lambda: op.rapply(y),
        reference=lambda: np.zeros(size, dtype=np_dtype),
    )


for _name, _factory in [
    ("linop.zero.apply", _make_zero_apply),
    ("linop.zero.rapply", _make_zero_rapply),
]:
    registry.register(
        Probe(
            name=_name,
            family="linop",
            factory=_factory,
            sizes=(256, 4096),
            backends=("numpy", "jax", "torch"),
        )
    )


# ---------------------------------------------------------------------------
# MatrixFreeLinOp probes — caller-defined apply/rapply functions


def _make_matrix_free_diagonal_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=_np_dtype(ctx))
    d = ctx.asarray(d_np)
    _, x, x_np = _dense_vector(ctx, size, seed + 31)
    apply_fn = lambda v: d * v  # noqa: E731
    rapply_fn = lambda v: ctx.ops.conj(d) * v  # noqa: E731
    op = sc.MatrixFreeLinOp(apply_fn, rapply_fn, space, space, ctx)
    ref = d_np * x_np
    return ProbeCase(
        bare_label=f"{backend}: d * x",
        sc_label="MatrixFreeLinOp(diagonal).apply",
        bare=lambda: d_np * x_np,
        sc=lambda: op.apply(x),
        reference=lambda: ref,
    )


def _make_matrix_free_diagonal_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=_np_dtype(ctx))
    d = ctx.asarray(d_np)
    _, y, y_np = _dense_vector(ctx, size, seed + 32)
    apply_fn = lambda v: d * v  # noqa: E731
    rapply_fn = lambda v: ctx.ops.conj(d) * v  # noqa: E731
    op = sc.MatrixFreeLinOp(apply_fn, rapply_fn, space, space, ctx)
    ref = d_np.conj() * y_np
    return ProbeCase(
        bare_label=f"{backend}: conj(d) * y",
        sc_label="MatrixFreeLinOp(diagonal).rapply",
        bare=lambda: d_np.conj() * y_np,
        sc=lambda: op.rapply(y),
        reference=lambda: ref,
    )


def _make_matrix_free_fft_apply(backend: str, seed: int, size: int) -> ProbeCase:
    # FFT operators here use a complex context so apply/rapply round-trip works.
    ctx = sc.Context(
        sc.NumpyOps(), dtype=np.complex128, check_level=_ACTIVE_CHECK_LEVEL.get()
    )
    space = sc.DenseCoordinateSpace((size,), ctx)
    rng = _rng(seed)
    x_np = np.asarray(rng.standard_normal(size), dtype=np.complex128)
    x = ctx.asarray(x_np)
    n = int(size)

    def apply_fn(v):
        v_np = np.asarray(v)
        return ctx.asarray(np.fft.fft(v_np))

    def rapply_fn(v):
        v_np = np.asarray(v)
        return ctx.asarray(np.fft.ifft(v_np) * n)

    op = sc.MatrixFreeLinOp(apply_fn, rapply_fn, space, space, ctx)
    ref = np.fft.fft(x_np)
    return ProbeCase(
        bare_label="numpy: fft(x)",
        sc_label="MatrixFreeLinOp(fft).apply",
        bare=lambda: np.fft.fft(x_np),
        sc=lambda: op.apply(x),
        reference=lambda: ref,
    )


def _make_matrix_free_shift_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    _, x, x_np = _dense_vector(ctx, size, seed + 41)

    def apply_fn(v):
        v_np = np.asarray(v)
        return ctx.asarray(np.roll(v_np, 1))

    def rapply_fn(v):
        v_np = np.asarray(v)
        return ctx.asarray(np.roll(v_np, -1))

    op = sc.MatrixFreeLinOp(apply_fn, rapply_fn, space, space, ctx)
    ref = np.roll(x_np, 1)
    return ProbeCase(
        bare_label="numpy: roll(x, 1)",
        sc_label="MatrixFreeLinOp(shift).apply",
        bare=lambda: np.roll(x_np, 1),
        sc=lambda: op.apply(x),
        reference=lambda: ref,
    )


_MATRIX_FREE_LINOP_PROBES = [
    ("linop.matrix_free.diagonal_action.apply", _make_matrix_free_diagonal_apply, (256, 4096, 65536), ("numpy", "jax", "torch")),
    ("linop.matrix_free.diagonal_action.rapply", _make_matrix_free_diagonal_rapply, (256, 4096, 65536), ("numpy", "jax", "torch")),
    ("linop.matrix_free.fft_action.apply", _make_matrix_free_fft_apply, (256, 4096, 65536), ("numpy",)),
    ("linop.matrix_free.shift_action.apply", _make_matrix_free_shift_apply, (256, 4096, 65536), ("numpy",)),
]
for _name, _factory, _sizes, _backends in _MATRIX_FREE_LINOP_PROBES:
    registry.register(
        Probe(name=_name, family="linop", factory=_factory, sizes=_sizes, backends=_backends)
    )


# ---------------------------------------------------------------------------
# Tree-structured LinOp probes


def _make_block_diagonal_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, x1, x1_np = _dense_vector(ctx, size, seed + 51)
    _, x2, x2_np = _dense_vector(ctx, size, seed + 52)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    block = sc.BlockDiagonalLinOp((op_a, op_b), ctx=ctx)
    x = (x1, x2)
    ref = (a_np @ x1_np, b_np @ x2_np)
    return ProbeCase(
        bare_label=f"{backend}: (m1 @ x1, m2 @ x2)",
        sc_label="BlockDiagonalLinOp.apply",
        bare=lambda: (a_np @ x1_np, b_np @ x2_np),
        sc=lambda: block.apply(x),
        reference=lambda: ref,
    )


def _make_block_diagonal_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, y1, y1_np = _dense_vector(ctx, size, seed + 53)
    _, y2, y2_np = _dense_vector(ctx, size, seed + 54)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    block = sc.BlockDiagonalLinOp((op_a, op_b), ctx=ctx)
    y = (y1, y2)
    ref = (a_np.conj().T @ y1_np, b_np.conj().T @ y2_np)
    return ProbeCase(
        bare_label=f"{backend}: (m1.conj().T @ y1, m2.conj().T @ y2)",
        sc_label="BlockDiagonalLinOp.rapply",
        bare=lambda: (a_np.conj().T @ y1_np, b_np.conj().T @ y2_np),
        sc=lambda: block.rapply(y),
        reference=lambda: ref,
    )


def _make_stacked_linop_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, x, x_np = _dense_vector(ctx, size, seed + 55)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    cod = sc.TreeSpace.from_leaf_spaces((op_a.codomain, op_b.codomain), ctx)
    stacked = sc.StackedLinOp(space, cod, (op_a, op_b), ctx)
    ref = (a_np @ x_np, b_np @ x_np)
    return ProbeCase(
        bare_label=f"{backend}: (m1 @ x, m2 @ x)",
        sc_label="StackedLinOp.apply",
        bare=lambda: (a_np @ x_np, b_np @ x_np),
        sc=lambda: stacked.apply(x),
        reference=lambda: ref,
    )


def _make_sum_to_single_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, size, seed + 1)
    _, x1, x1_np = _dense_vector(ctx, size, seed + 56)
    _, x2, x2_np = _dense_vector(ctx, size, seed + 57)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space, ctx)
    dom = sc.TreeSpace.from_leaf_spaces((op_a.domain, op_b.domain), ctx)
    sum_op = sc.SumToSingleLinOp(dom, space, (op_a, op_b), ctx)
    x = (x1, x2)
    ref = a_np @ x1_np + b_np @ x2_np
    return ProbeCase(
        bare_label=f"{backend}: m1 @ x1 + m2 @ x2",
        sc_label="SumToSingleLinOp.apply",
        bare=lambda: a_np @ x1_np + b_np @ x2_np,
        sc=lambda: sum_op.apply(x),
        reference=lambda: ref,
    )


# Ragged-block variants. The uniform probes above stack same-shape blocks,
# which the ADR-016 ``block-diagonal-uniform-dense-batched`` fold collapses
# into one batched ``matmul`` under dispatch. The ragged variants below mix
# block shapes, so no uniform fold applies: both the bare and the SpaceCore
# path stay a per-block loop, isolating the abstraction's own cost when there
# is no fast path to route to. The bare is the idiomatic per-block tuple.


def _dense_rect(ctx: sc.Context, rows: int, cols: int, seed: int) -> tuple[Any, np.ndarray]:
    """Return ``(matrix_array, matrix_np)`` for a rectangular dense matrix."""
    from tests.generators import dense_array_case

    generated = dense_array_case(ctx, (rows, cols), seed=seed)
    a_np = np.asarray(generated.reference["array"], dtype=_np_dtype(ctx))
    return generated.obj, a_np


def _make_block_diagonal_apply_ragged(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    h = max(2, size // 2)
    space_a = sc.DenseCoordinateSpace((size,), ctx)
    space_b = sc.DenseCoordinateSpace((h,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, h, seed + 1)
    _, x1, x1_np = _dense_vector(ctx, size, seed + 51)
    _, x2, x2_np = _dense_vector(ctx, h, seed + 52)
    op_a = sc.DenseLinOp(a, space_a, space_a, ctx)
    op_b = sc.DenseLinOp(b, space_b, space_b, ctx)
    block = sc.BlockDiagonalLinOp((op_a, op_b), ctx=ctx)
    x = (x1, x2)
    ref = (a_np @ x1_np, b_np @ x2_np)
    return ProbeCase(
        bare_label=f"{backend}: (m1 @ x1, m2 @ x2) [ragged]",
        sc_label="BlockDiagonalLinOp.apply [ragged]",
        bare=lambda: (a_np @ x1_np, b_np @ x2_np),
        sc=lambda: block.apply(x),
        reference=lambda: ref,
    )


def _make_block_diagonal_rapply_ragged(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    h = max(2, size // 2)
    space_a = sc.DenseCoordinateSpace((size,), ctx)
    space_b = sc.DenseCoordinateSpace((h,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_matrix(ctx, h, seed + 1)
    _, y1, y1_np = _dense_vector(ctx, size, seed + 53)
    _, y2, y2_np = _dense_vector(ctx, h, seed + 54)
    op_a = sc.DenseLinOp(a, space_a, space_a, ctx)
    op_b = sc.DenseLinOp(b, space_b, space_b, ctx)
    block = sc.BlockDiagonalLinOp((op_a, op_b), ctx=ctx)
    y = (y1, y2)
    ref = (a_np.conj().T @ y1_np, b_np.conj().T @ y2_np)
    return ProbeCase(
        bare_label=f"{backend}: (m1.conj().T @ y1, m2.conj().T @ y2) [ragged]",
        sc_label="BlockDiagonalLinOp.rapply [ragged]",
        bare=lambda: (a_np.conj().T @ y1_np, b_np.conj().T @ y2_np),
        sc=lambda: block.rapply(y),
        reference=lambda: ref,
    )


def _make_stacked_linop_apply_ragged(backend: str, seed: int, size: int) -> ProbeCase:
    # Same domain, differently-shaped codomains: op_b maps R^size -> R^h.
    ctx = _backend_ctx(backend)
    h = max(2, size // 2)
    space = sc.DenseCoordinateSpace((size,), ctx)
    space_h = sc.DenseCoordinateSpace((h,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_rect(ctx, h, size, seed + 1)
    _, x, x_np = _dense_vector(ctx, size, seed + 55)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space, space_h, ctx)
    cod = sc.TreeSpace.from_leaf_spaces((op_a.codomain, op_b.codomain), ctx)
    stacked = sc.StackedLinOp(space, cod, (op_a, op_b), ctx)
    ref = (a_np @ x_np, b_np @ x_np)
    return ProbeCase(
        bare_label=f"{backend}: (m1 @ x, m2 @ x) [ragged]",
        sc_label="StackedLinOp.apply [ragged]",
        bare=lambda: (a_np @ x_np, b_np @ x_np),
        sc=lambda: stacked.apply(x),
        reference=lambda: ref,
    )


def _make_sum_to_single_apply_ragged(backend: str, seed: int, size: int) -> ProbeCase:
    # Differently-shaped domains, same codomain: op_b maps R^h -> R^size.
    ctx = _backend_ctx(backend)
    h = max(2, size // 2)
    space = sc.DenseCoordinateSpace((size,), ctx)
    space_h = sc.DenseCoordinateSpace((h,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    b, b_np = _dense_rect(ctx, size, h, seed + 1)
    _, x1, x1_np = _dense_vector(ctx, size, seed + 56)
    _, x2, x2_np = _dense_vector(ctx, h, seed + 57)
    op_a = sc.DenseLinOp(a, space, space, ctx)
    op_b = sc.DenseLinOp(b, space_h, space, ctx)
    dom = sc.TreeSpace.from_leaf_spaces((op_a.domain, op_b.domain), ctx)
    sum_op = sc.SumToSingleLinOp(dom, space, (op_a, op_b), ctx)
    x = (x1, x2)
    ref = a_np @ x1_np + b_np @ x2_np
    return ProbeCase(
        bare_label=f"{backend}: m1 @ x1 + m2 @ x2 [ragged]",
        sc_label="SumToSingleLinOp.apply [ragged]",
        bare=lambda: a_np @ x1_np + b_np @ x2_np,
        sc=lambda: sum_op.apply(x),
        reference=lambda: ref,
    )


_TREE_LINOP_PROBES = [
    ("linop.block_diagonal.apply", _make_block_diagonal_apply, (32, 128, 512), ("numpy", "jax", "torch"), "uniform blocks; dispatch fold target"),
    ("linop.block_diagonal.rapply", _make_block_diagonal_rapply, (32, 128, 512), ("numpy", "jax", "torch"), "uniform blocks; dispatch fold target"),
    ("linop.stacked.apply", _make_stacked_linop_apply, (32, 128, 512), ("numpy", "jax", "torch"), "uniform blocks; dispatch fold target"),
    ("linop.sum_to_single.apply", _make_sum_to_single_apply, (32, 128, 512), ("numpy", "jax", "torch"), "uniform blocks; dispatch fold target"),
    ("linop.block_diagonal.apply.ragged", _make_block_diagonal_apply_ragged, (32, 128, 512), ("numpy", "jax", "torch"), "ragged blocks; no uniform fold"),
    ("linop.block_diagonal.rapply.ragged", _make_block_diagonal_rapply_ragged, (32, 128, 512), ("numpy", "jax", "torch"), "ragged blocks; no uniform fold"),
    ("linop.stacked.apply.ragged", _make_stacked_linop_apply_ragged, (32, 128, 512), ("numpy", "jax", "torch"), "ragged blocks; no uniform fold"),
    ("linop.sum_to_single.apply.ragged", _make_sum_to_single_apply_ragged, (32, 128, 512), ("numpy", "jax", "torch"), "ragged blocks; no uniform fold"),
]
for _name, _factory, _sizes, _backends, _notes in _TREE_LINOP_PROBES:
    registry.register(
        Probe(
            name=_name, family="linop", factory=_factory, sizes=_sizes,
            backends=_backends, notes=_notes,
        )
    )


# ---------------------------------------------------------------------------
# Functional — matrix-free


def _make_matrix_free_linear_functional_value(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space, x, x_np = _dense_vector(ctx, size, seed)
    _, _, w_np = _dense_vector(ctx, size, seed + 1)
    w = ctx.asarray(w_np)

    def value_fn(v):
        return ctx.ops.vdot(w, v)

    func = sc.MatrixFreeLinearFunctional(value_fn, space, ctx)
    ref = float(np.vdot(w_np, x_np))
    return ProbeCase(
        bare_label=f"{backend}: vdot(w, x)",
        sc_label="MatrixFreeLinearFunctional.value",
        bare=lambda: float(np.vdot(w_np, x_np)),
        sc=lambda: float(func.value(x)),
        reference=lambda: ref,
    )


registry.register(
    Probe(
        name="functional.matrix_free.value",
        family="functional",
        factory=_make_matrix_free_linear_functional_value,
        sizes=(256, 4096),
        backends=("numpy", "jax", "torch"),
    )
)


# ---------------------------------------------------------------------------
# Device-aware sample probe — same op as linop.dense.apply but routes the
# input array onto the explicitly-requested device.


def _make_dense_apply_device(backend: str, device: str, seed: int, size: int) -> ProbeCase:
    from ._devices import device_object

    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    _, x_default, x_np = _dense_vector(ctx, size, seed + 7)

    # Route both the operator matrix and the input onto the requested device
    # when the backend supports it. Keeping them on the same device is
    # required for ``apply`` to succeed.
    dev = device_object(backend, device)
    x = x_default
    a_dev = a
    if backend == "jax" and dev is not None:
        import jax

        x = jax.device_put(x_default, dev)
        a_dev = jax.device_put(a, dev)
    elif backend == "torch" and dev is not None:
        x = x_default.to(dev)
        a_dev = a.to(dev)

    op = sc.DenseLinOp(a_dev, space, space, ctx)

    ref = a_np @ x_np
    return ProbeCase(
        bare_label=f"{backend}/{device}: A @ x",
        sc_label="DenseLinOp.apply",
        bare=lambda: a_np @ x_np,
        sc=lambda: op.apply(x),
        reference=lambda: ref,
    )


registry.register(
    Probe(
        name="linop.dense.apply.device",
        family="linop",
        factory=_make_dense_apply_device,
        sizes=(64, 256, 1024),
        backends=("numpy", "jax", "torch"),
        device_aware=True,
    )
)

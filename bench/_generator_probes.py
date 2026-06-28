"""Generator-driven probes that scale test-suite cases to bench sizes.

The :mod:`tests.generators` modules build small, hand-crafted SpaceCore
objects (LinOps, Functionals) at fixed shapes (typically 2x3 or
length-3). Those generators pin the SpaceCore operations the public
tests assert against — ``apply`` / ``rapply`` for LinOps, ``value`` /
``grad`` for Functionals.

This module mirrors each generator at *bench-relevant* problem sizes:
the same operator family, the same operation, but with random inputs
drawn at ``size`` (and ``size // 2`` for non-square shapes) so the
timing loop has enough work to escape measurement noise.

Each probe pairs:

* ``sc`` — the SpaceCore call that the corresponding generator pins
  (``op.apply``, ``op.rapply``, ``func.value``, ``func.grad``).
* ``bare`` and ``reference`` — a hand-coded NumPy expression equivalent
  to the generator's reference, scaled to the same inputs.

Backend routing reuses :func:`bench._operations._backend_ctx`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps

import spacecore as sc
from ._operations import _backend_ctx, _dense_matrix, _dense_vector, _np_dtype, _rng
from ._probes import Probe, ProbeCase, registry


# ---------------------------------------------------------------------------
# LinOp generator probes
#
# ``dense_linop_case`` builds a 3x2 dense operator (codomain x domain).
# The scaled probe keeps the same rectangular layout: codomain ``size``,
# domain ``max(size // 2, 2)``. ``sparse_linop_case`` follows the same
# layout. ``diagonal_linop_case`` and ``matrix_free_linop_case`` are
# square in the generator and stay square here.


def _rect_shape(size: int) -> tuple[int, int]:
    """Return ``(rows, cols)`` matching the generator's 3x2 layout, scaled."""
    return size, max(size // 2, 2)


def _make_generator_dense_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    rows, cols = _rect_shape(size)
    np_dtype = _np_dtype(ctx)
    rng = _rng(seed)
    a_np = np.asarray(rng.standard_normal((rows, cols)), dtype=np_dtype)
    x_np = np.asarray(_rng(seed + 1).standard_normal(cols), dtype=np_dtype)
    domain = sc.DenseCoordinateSpace((cols,), ctx)
    codomain = sc.DenseCoordinateSpace((rows,), ctx)
    a = ctx.asarray(a_np)
    x = ctx.asarray(x_np)
    op = sc.DenseLinOp(a, domain, codomain, ctx)
    return ProbeCase(
        bare_label=f"{backend}: A @ x",
        sc_label="DenseLinOp.apply (generator)",
        bare=lambda: a_np @ x_np,
        sc=lambda: op.apply(x),
        reference=lambda: a_np @ x_np,
    )


def _make_generator_dense_rapply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    rows, cols = _rect_shape(size)
    np_dtype = _np_dtype(ctx)
    rng = _rng(seed)
    a_np = np.asarray(rng.standard_normal((rows, cols)), dtype=np_dtype)
    y_np = np.asarray(_rng(seed + 2).standard_normal(rows), dtype=np_dtype)
    domain = sc.DenseCoordinateSpace((cols,), ctx)
    codomain = sc.DenseCoordinateSpace((rows,), ctx)
    a = ctx.asarray(a_np)
    y = ctx.asarray(y_np)
    op = sc.DenseLinOp(a, domain, codomain, ctx)
    return ProbeCase(
        bare_label=f"{backend}: A.conj().T @ y",
        sc_label="DenseLinOp.rapply (generator)",
        bare=lambda: a_np.conj().T @ y_np,
        sc=lambda: op.rapply(y),
        reference=lambda: a_np.conj().T @ y_np,
    )


def _make_generator_sparse_apply(backend: str, seed: int, size: int) -> ProbeCase:
    # SpaceCore sparse routes through SciPy — NumPy only.
    ctx = _backend_ctx(backend)
    rows, cols = _rect_shape(size)
    np_dtype = _np_dtype(ctx)
    rng = _rng(seed)
    density = min(0.05, 32.0 / max(rows, cols))
    a_np = sps.random(
        rows, cols, density=density, format="csr", random_state=rng, dtype=np_dtype
    )
    x_np = np.asarray(_rng(seed + 3).standard_normal(cols), dtype=np_dtype)
    domain = sc.DenseCoordinateSpace((cols,), ctx)
    codomain = sc.DenseCoordinateSpace((rows,), ctx)
    x = ctx.asarray(x_np)
    op = sc.SparseLinOp(ctx.assparse(a_np), domain, codomain, ctx)
    return ProbeCase(
        bare_label="scipy: A @ x",
        sc_label="SparseLinOp.apply (generator)",
        bare=lambda: a_np @ x_np,
        sc=lambda: op.apply(x),
        reference=lambda: np.asarray(a_np @ x_np),
    )


def _make_generator_diagonal_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    np_dtype = _np_dtype(ctx)
    rng = _rng(seed)
    d_np = np.asarray(rng.standard_normal(size), dtype=np_dtype)
    x_np = np.asarray(_rng(seed + 4).standard_normal(size), dtype=np_dtype)
    space = sc.DenseCoordinateSpace((size,), ctx)
    d = ctx.asarray(d_np)
    x = ctx.asarray(x_np)
    op = sc.DiagonalLinOp(d, space, ctx)
    return ProbeCase(
        bare_label=f"{backend}: d * x",
        sc_label="DiagonalLinOp.apply (generator)",
        bare=lambda: d_np * x_np,
        sc=lambda: op.apply(x),
        reference=lambda: d_np * x_np,
    )


def _make_generator_matrix_free_apply(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    np_dtype = _np_dtype(ctx)
    x_np = np.asarray(_rng(seed + 5).standard_normal(size), dtype=np_dtype)
    factor = 2.0
    adjoint_factor = factor
    space = sc.DenseCoordinateSpace((size,), ctx)
    x = ctx.asarray(x_np)
    op = sc.MatrixFreeLinOp(
        lambda value: factor * value,
        lambda value: adjoint_factor * value,
        space,
        space,
        ctx,
    )
    return ProbeCase(
        bare_label=f"{backend}: factor * x",
        sc_label="MatrixFreeLinOp.apply (generator)",
        bare=lambda: factor * x_np,
        sc=lambda: op.apply(x),
        reference=lambda: factor * x_np,
    )


# ---------------------------------------------------------------------------
# Functional generator probes
#
# The generator's linear functional is ``InnerProductFunctional(c)`` —
# ``value(x) = <c, x>`` and ``grad(x) = c``. The quadratic functional is
# ``LinOpQuadraticForm(Q)`` over a symmetrized dense ``Q``.


def _make_generator_inner_product_value(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    np_dtype = _np_dtype(ctx)
    rng = _rng(seed)
    x_np = np.asarray(rng.standard_normal(size), dtype=np_dtype)
    c_np = np.asarray(_rng(seed + 6).standard_normal(size), dtype=np_dtype)
    space = sc.DenseCoordinateSpace((size,), ctx)
    c = ctx.asarray(c_np)
    x = ctx.asarray(x_np)
    func = sc.InnerProductFunctional(c, space, ctx)
    return ProbeCase(
        bare_label=f"{backend}: vdot(c, x)",
        sc_label="InnerProductFunctional.value (generator)",
        bare=lambda: float(np.vdot(c_np, x_np)),
        sc=lambda: float(func.value(x)),
        reference=lambda: float(np.vdot(c_np, x_np)),
    )


def _make_generator_inner_product_grad(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    np_dtype = _np_dtype(ctx)
    rng = _rng(seed)
    x_np = np.asarray(rng.standard_normal(size), dtype=np_dtype)
    c_np = np.asarray(_rng(seed + 7).standard_normal(size), dtype=np_dtype)
    space = sc.DenseCoordinateSpace((size,), ctx)
    c = ctx.asarray(c_np)
    x = ctx.asarray(x_np)
    func = sc.InnerProductFunctional(c, space, ctx)
    return ProbeCase(
        bare_label=f"{backend}: c",
        sc_label="InnerProductFunctional.grad (generator)",
        bare=lambda: c_np,
        sc=lambda: func.grad(x),
        reference=lambda: c_np,
    )


def _make_generator_quadratic_form_value(backend: str, seed: int, size: int) -> ProbeCase:
    ctx = _backend_ctx(backend)
    space = sc.DenseCoordinateSpace((size,), ctx)
    a, a_np = _dense_matrix(ctx, size, seed)
    a_np = 0.5 * (a_np + a_np.T)
    a = ctx.asarray(a_np)
    _, x, x_np = _dense_vector(ctx, size, seed + 8)
    op = sc.DenseLinOp(a, space, space, ctx)
    func = sc.LinOpQuadraticForm(op, ctx=ctx)
    return ProbeCase(
        bare_label=f"{backend}: 0.5 * x @ A @ x",
        sc_label="LinOpQuadraticForm.value (generator)",
        bare=lambda: float(0.5 * x_np @ (a_np @ x_np)),
        sc=lambda: float(func.value(x)),
        reference=lambda: float(0.5 * x_np @ (a_np @ x_np)),
    )


# ---------------------------------------------------------------------------
# Registration


_FULL_BACKENDS = ("numpy", "jax", "torch")
_NUMPY_ONLY = ("numpy",)


_GENERATOR_PROBES: tuple[tuple[str, Any, tuple[int, ...], tuple[str, ...]], ...] = (
    ("generator.dense_linop.apply", _make_generator_dense_apply, (32, 256, 2048), _FULL_BACKENDS),
    ("generator.dense_linop.rapply", _make_generator_dense_rapply, (32, 256, 2048), _FULL_BACKENDS),
    ("generator.sparse_linop.apply", _make_generator_sparse_apply, (32, 1024, 16384), _NUMPY_ONLY),
    ("generator.diagonal_linop.apply", _make_generator_diagonal_apply, (32, 1024, 16384), _FULL_BACKENDS),
    (
        "generator.matrix_free_linop.apply",
        _make_generator_matrix_free_apply,
        (32, 1024, 16384),
        _FULL_BACKENDS,
    ),
)
for _name, _factory, _sizes, _backends in _GENERATOR_PROBES:
    registry.register(
        Probe(
            name=_name,
            family="linop",
            factory=_factory,
            sizes=_sizes,
            backends=_backends,
        )
    )


_GENERATOR_FUNCTIONAL_PROBES: tuple[tuple[str, Any], ...] = (
    ("generator.inner_product_functional.value", _make_generator_inner_product_value),
    ("generator.inner_product_functional.grad", _make_generator_inner_product_grad),
    ("generator.linop_quadratic_form.value", _make_generator_quadratic_form_value),
)
for _name, _factory in _GENERATOR_FUNCTIONAL_PROBES:
    registry.register(
        Probe(
            name=_name,
            family="functional",
            factory=_factory,
            sizes=(256, 4096),
            backends=_FULL_BACKENDS,
        )
    )

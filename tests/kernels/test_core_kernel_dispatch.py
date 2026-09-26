"""Core-kernel binding: operators bind the check-free cores as class attributes.

The concrete cores live in :mod:`spacecore.kernels.core` and each operator class
binds them directly (``_apply_core = dense_apply_core``). These tests pin that
the bound cores are the submodule functions (not duplicated inline) and that
the fused paths compute and validate correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.kernels.core import algebra as kalg


# ---------------------------------------------------------------------------
# Operator bindings: the cores come from the kernels submodule, not inline
# ---------------------------------------------------------------------------
@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _endo(ctx, X, seed=0):
    rng = np.random.default_rng(seed)
    return sc.DenseLinOp(ctx.asarray(rng.standard_normal((3, 3))), X, X, ctx)


@pytest.mark.parametrize(
    "cls_name, kernel_module, has_rvapply",
    [
        ("ComposedLinOp", "spacecore.kernels.core.algebra", False),
        ("ScaledLinOp", "spacecore.kernels.core.algebra", False),
        ("SumLinOp", "spacecore.kernels.core.algebra", False),
        ("ZeroLinOp", "spacecore.kernels.core.algebra", False),
        ("IdentityLinOp", "spacecore.kernels.core.algebra", False),
        ("MatrixFreeLinOp", "spacecore.kernels.core.algebra", False),
        ("DenseLinOp", "spacecore.kernels.core.dense", True),
        ("DiagonalLinOp", "spacecore.kernels.core.diagonal", True),
        ("SparseLinOp", "spacecore.kernels.core.sparse", True),
    ],
)
def test_operator_cores_come_from_kernels_submodule(cls_name, kernel_module, has_rvapply):
    """Every operator's cores are submodule kernels, not inline methods.

    Guards against drift back to inline cores: the bound ``_apply_core`` (and the
    leaf ``_rvapply_core``) on each operator must be a function defined in the
    kernels subpackage, not in ``spacecore.linop``.
    """
    cls = getattr(sc, cls_name)
    core = cls.__dict__.get("_apply_core")
    assert core is not None, f"{cls_name} has no bound _apply_core"
    assert core.__module__ == kernel_module, (
        f"{cls_name}._apply_core is {core.__module__}, expected {kernel_module}"
    )
    if has_rvapply:
        rvcore = cls.__dict__.get("_rvapply_core")
        assert rvcore is not None and rvcore.__module__ == kernel_module, (
            f"{cls_name}._rvapply_core is not a {kernel_module} kernel"
        )


# ---------------------------------------------------------------------------
# Behavior is unchanged by the relocation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls_name, core_attr",
    [
        ("LinearFunctional", "_grad_core"),
        ("InnerProductFunctional", "_value_core"),
        ("MatrixFreeLinearFunctional", "_value_core"),
        ("LinOpQuadraticForm", "_value_core"),
        ("ComposedFunctional", "_value_core"),
    ],
)
def test_functional_cores_come_from_kernels_submodule(cls_name, core_attr):
    """Functional cores are submodule kernels, not inline methods."""
    import spacecore.functional as scf

    cls = getattr(scf, cls_name)
    core = cls.__dict__.get(core_attr)
    assert core is not None, f"{cls_name} has no bound {core_attr}"
    assert core.__module__ == "spacecore.kernels.core.functional", (
        f"{cls_name}.{core_attr} is {core.__module__}, expected spacecore.kernels.core.functional"
    )


def test_relocated_kernels_compute_correctly(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    A = _endo(ctx, X, seed=1)
    B = _endo(ctx, X, seed=2)
    op = 2.0 * (A.H @ B) + sc.IdentityLinOp(X, ctx)
    x = ctx.asarray(np.asarray([1.0, -2.0, 3.0]))
    expected = 2.0 * A.rapply(B.apply(x)) + x
    np.testing.assert_allclose(np.asarray(op.apply(x)), np.asarray(expected))


def test_quadratic_value_validates_input_once(ctx):
    """``LinOpQuadraticForm.value`` checks its input once, not per sub-term.

    Pre-kernel, ``value`` called ``Q.apply(x)`` and ``linear.value(x)``, each
    re-validating ``x``. The fused cores reach ``Q._apply_core`` and
    ``linear._value_core`` instead, so only the public boundary validates.
    """
    X = sc.DenseCoordinateSpace((3,), ctx)
    Q = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0, 4.0]), X, ctx)
    c = sc.InnerProductFunctional(ctx.asarray([1.0, 1.0, 1.0]), X, ctx)
    qf = sc.LinOpQuadraticForm(Q, c, 0.5, ctx)
    x = ctx.asarray([1.0, 2.0, 3.0])

    counter = [0]
    original = qf.domain._check_member

    def counting(xx):
        counter[0] += 1
        return original(xx)

    qf.domain._check_member = counting
    value = qf.value(x)
    assert counter[0] == 1, "quadratic value re-validated an intermediate"

    expected = 0.5 * float(np.asarray([1.0, 2.0, 3.0]) @ (np.asarray([2.0, 3.0, 4.0]) * [1.0, 2.0, 3.0]))
    expected += float(np.asarray([1.0, 1.0, 1.0]) @ [1.0, 2.0, 3.0]) + 0.5
    np.testing.assert_allclose(float(np.asarray(value)), expected)


def test_compose_chain_rule_flattens(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    a, b, c = (_endo(ctx, X, s) for s in range(3))
    chain = a @ b @ c
    # The fusion rule lives in the kernels submodule and is what populates
    # the operator's cached _apply_chain.
    assert kalg.compose_chain(chain) == chain._apply_chain == (c, b, a)

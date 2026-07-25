"""Behavioral tests for what each ``check_level`` actually enforces.

Drives real spaces, linear operators, functionals, and solvers to pin which
invariants each level (``none``/``cheap``/``standard``/``strict``) checks or
skips end-to-end. The level is a property of the context-bound object, so every
case sets it through the object's ``check_level=`` keyword rather than through
the :class:`spacecore.Context`.

Book justification: ``check_level`` is a *contract* about which invariants are
enforced, so it is tested as a contract rather than by inspecting internals
(Hunt & Thomas, *The Pragmatic Programmer*, tip 37). Strict levels must assert
states believed impossible while lenient levels must not — assertive
programming / fail-fast (same source, tips 38-39; Ousterhout, *A Philosophy of
Software Design*, "when to crash"). Numerical precision is treated as part of
correctness across levels (Irving et al., *Research Software Engineering with
Python*).
"""
import numpy as np
import pytest

import spacecore as sc
from spacecore._check_policy import normalize_check_level


def _ctx(dtype=np.float64) -> sc.Context:
    return sc.Context(sc.NumpyOps(), dtype=dtype)


def test_check_level_public_api_and_legacy_mapping():
    ctx = _ctx()

    assert sc.CHECK_LEVELS == ("none", "cheap", "standard", "strict")
    assert sc.DenseCoordinateSpace((2,), ctx).check_level == "standard"
    assert sc.DenseCoordinateSpace((2,), ctx, check_level="cheap").check_level == "cheap"
    assert sc.DenseCoordinateSpace((2,), ctx, check_level="strict").check_level == "strict"

    assert normalize_check_level(enable_checks=True) == "standard"
    assert normalize_check_level(enable_checks=False) == "none"
    with pytest.warns(DeprecationWarning, match="enable_checks"):
        assert normalize_check_level(enable_checks=True, warn_legacy=True) == "standard"

    with pytest.raises(TypeError, match="either check_level or enable_checks"):
        normalize_check_level("strict", enable_checks=True)
    with pytest.raises(ValueError, match="Unknown check_level"):
        normalize_check_level("fast")


def test_derived_object_uses_the_least_expensive_source_level():
    ctx = _ctx()
    strict_space = sc.DenseCoordinateSpace((1,), ctx, check_level="strict")
    cheap_space = sc.DenseCoordinateSpace((1,), ctx, check_level="cheap")

    product = sc.TreeSpace.from_leaf_spaces((strict_space, cheap_space))

    assert product.check_level == "cheap"


def test_none_skips_optional_space_linop_and_batched_checks():
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx, check_level="none")
    identity = sc.IdentityLinOp(space, ctx, check_level="none")
    invalid = ctx.asarray([1.0, 2.0, 3.0])
    invalid_batch = ctx.asarray([[1.0, 2.0, 3.0]])

    space.check_member(invalid)
    assert identity.apply(invalid) is invalid
    assert identity.vapply(invalid_batch) is invalid_batch


def test_cheap_checks_shape_dtype_backend_and_tree_structure_only():
    ctx = _ctx(np.float32)
    vector = sc.DenseCoordinateSpace((2,), ctx, check_level="cheap")

    with pytest.raises(sc.SpaceValidationError, match="Expected shape"):
        vector.check_member(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(sc.SpaceValidationError, match="Expected dtype"):
        vector.check_member(np.asarray([1.0, 2.0], dtype=np.float64))

    product = sc.TreeSpace.from_leaf_spaces((vector, vector), ctx)
    with pytest.raises(sc.SpaceValidationError, match="structure mismatch"):
        product.check_member([ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0])])

    with pytest.raises(sc.SpaceValidationError, match=r"\$\[0\]"):
        product.check_member((ctx.asarray([1.0]), ctx.asarray([2.0, 3.0, 4.0])))


def test_standard_adds_recursive_and_hermitian_membership():
    ctx = _ctx()
    vector = sc.DenseCoordinateSpace((2,), ctx, check_level="standard")
    product = sc.TreeSpace.from_leaf_spaces((vector, vector), ctx)

    with pytest.raises(sc.SpaceValidationError, match=r"\$\[0\]"):
        product.check_member((ctx.asarray([1.0]), ctx.asarray([2.0, 3.0])))

    hermitian = sc.HermitianSpace(2, ctx=ctx, check_level="standard")
    with pytest.raises(sc.SpaceValidationError, match="not Hermitian"):
        hermitian.check_member(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))

    cheap_hermitian = sc.HermitianSpace(2, ctx=ctx, check_level="cheap")
    cheap_product = sc.TreeSpace.from_leaf_spaces((cheap_hermitian,), ctx)
    cheap_product.check_member((ctx.asarray([[1.0, 2.0], [0.0, 1.0]]),))

    standard_product = sc.TreeSpace.from_leaf_spaces((hermitian,), ctx)
    with pytest.raises(sc.SpaceValidationError, match=r"\$\[0\].*not Hermitian"):
        standard_product.check_member((ctx.asarray([[1.0, 2.0], [0.0, 1.0]]),))


def test_checked_method_and_batched_validation_follow_cheap_policy():
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx, check_level="cheap")
    identity = sc.IdentityLinOp(space, ctx, check_level="cheap")

    with pytest.raises(sc.SpaceValidationError, match="Expected shape"):
        identity.apply(ctx.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(sc.SpaceValidationError, match="trailing shape"):
        identity.vapply(ctx.asarray([[1.0, 2.0, 3.0]]))


def test_functional_scalar_output_shape_is_standard():
    ctx = _ctx()
    cheap_space = sc.DenseCoordinateSpace((2,), ctx, check_level="cheap")
    cheap_functional = sc.MatrixFreeLinearFunctional(
        lambda _x: ctx.asarray([1.0]), cheap_space, ctx, check_level="cheap"
    )
    assert cheap_functional.value(ctx.asarray([1.0, 2.0])).shape == (1,)

    standard_space = sc.DenseCoordinateSpace((2,), ctx, check_level="standard")
    standard_functional = sc.MatrixFreeLinearFunctional(
        lambda _x: ctx.asarray([1.0]), standard_space, ctx, check_level="standard"
    )
    with pytest.raises(ValueError, match="scalar batch output"):
        standard_functional.value(ctx.asarray([1.0, 2.0]))


def test_strict_matrix_free_adjoint_probe_is_strict_only():
    ctx = _ctx()
    standard_space = sc.DenseCoordinateSpace((2,), ctx, check_level="standard")
    sc.MatrixFreeLinOp(
        lambda x: x,
        lambda y: ctx.asarray([0.0, 0.0]),
        standard_space,
        standard_space,
        ctx,
        check_level="standard",
    )

    strict_space = sc.DenseCoordinateSpace((2,), ctx, check_level="strict")
    with pytest.raises(ValueError, match="adjoint consistency check failed"):
        sc.MatrixFreeLinOp(
            lambda x: x,
            lambda y: ctx.asarray([0.0, 0.0]),
            strict_space,
            strict_space,
            ctx,
            check_level="strict",
        )


def test_strict_matrix_free_coordinate_adjoint_preserves_non_euclidean_metric():
    ctx = _ctx()
    domain = sc.DenseCoordinateSpace(
        (2,),
        ctx,
        geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 3.0])),
        check_level="strict",
    )
    codomain = sc.DenseCoordinateSpace(
        (2,),
        ctx,
        geometry=sc.WeightedInnerProduct(ctx.asarray([5.0, 7.0])),
        check_level="strict",
    )
    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda x: x,
        lambda y: y,
        domain,
        codomain,
        ctx,
    )

    y = ctx.asarray([3.0, 4.0])
    np.testing.assert_allclose(op.rapply(y), ctx.asarray([7.5, 28.0 / 3.0]))


def test_linalg_keeps_square_invariant_and_adds_strict_cg_probe():
    ctx = _ctx()
    domain = sc.DenseCoordinateSpace((2,), ctx, check_level="none")
    codomain = sc.DenseCoordinateSpace((3,), ctx, check_level="none")
    rectangular = sc.ZeroLinOp(domain, codomain, ctx, check_level="none")
    with pytest.raises(ValueError, match="square LinOp"):
        sc.cg(rectangular, ctx.asarray([1.0, 1.0, 1.0]), maxiter=0)

    standard_space = sc.DenseCoordinateSpace((2,), ctx, check_level="standard")
    standard_negative = sc.DiagonalLinOp(
        ctx.asarray([-1.0, -1.0]), standard_space, ctx, check_level="standard"
    )
    sc.cg(standard_negative, ctx.asarray([1.0, 1.0]), maxiter=0)

    strict_space = sc.DenseCoordinateSpace((2,), ctx, check_level="strict")
    strict_negative = sc.DiagonalLinOp(
        ctx.asarray([-1.0, -1.0]), strict_space, ctx, check_level="strict"
    )
    with pytest.raises(ValueError, match="positive curvature"):
        sc.cg(strict_negative, ctx.asarray([1.0, 1.0]), maxiter=0)

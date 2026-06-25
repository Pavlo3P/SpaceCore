import numpy as np
import pytest

import spacecore as sc


def _ctx(level: sc.CheckLevel, dtype=np.float64) -> sc.Context:
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=level)


def test_check_level_public_api_and_legacy_mapping():
    assert sc.CHECK_LEVELS == ("none", "cheap", "standard", "strict")
    assert sc.Context(sc.NumpyOps()).check_level == "standard"
    assert sc.Context(sc.NumpyOps(), check_level="cheap").check_level == "cheap"
    assert sc.normalize_context("numpy", check_level="strict").check_level == "strict"

    with pytest.warns(DeprecationWarning, match="enable_checks"):
        checked = sc.Context(sc.NumpyOps(), enable_checks=True)
    with pytest.warns(DeprecationWarning, match="enable_checks"):
        unchecked = sc.Context(sc.NumpyOps(), enable_checks=False)

    assert checked.check_level == "standard"
    assert checked.enable_checks is True
    assert unchecked.check_level == "none"
    assert unchecked.enable_checks is False

    with pytest.raises(TypeError, match="either check_level or enable_checks"):
        sc.Context(sc.NumpyOps(), enable_checks=True, check_level="strict")
    with pytest.raises(ValueError, match="Unknown check_level"):
        sc.Context(sc.NumpyOps(), check_level="fast")


def test_inferred_context_uses_the_least_expensive_source_level():
    strict_ctx = _ctx("strict")
    cheap_ctx = _ctx("cheap")
    strict_space = sc.DenseCoordinateSpace((1,), strict_ctx)
    cheap_space = sc.DenseCoordinateSpace((1,), cheap_ctx)

    product = sc.TreeSpace.from_leaf_spaces((strict_space, cheap_space))

    assert product.check_level == "cheap"


def test_none_skips_optional_space_linop_and_batched_checks():
    ctx = _ctx("none")
    space = sc.DenseCoordinateSpace((2,), ctx)
    identity = sc.IdentityLinOp(space, ctx)
    invalid = ctx.asarray([1.0, 2.0, 3.0])
    invalid_batch = ctx.asarray([[1.0, 2.0, 3.0]])

    space.check_member(invalid)
    assert identity.apply(invalid) is invalid
    assert identity.vapply(invalid_batch) is invalid_batch


def test_cheap_checks_shape_dtype_backend_and_tree_structure_only():
    ctx = _ctx("cheap", np.float32)
    vector = sc.DenseCoordinateSpace((2,), ctx)

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
    ctx = _ctx("standard")
    vector = sc.DenseCoordinateSpace((2,), ctx)
    product = sc.TreeSpace.from_leaf_spaces((vector, vector), ctx)

    with pytest.raises(sc.SpaceValidationError, match=r"\$\[0\]"):
        product.check_member((ctx.asarray([1.0]), ctx.asarray([2.0, 3.0])))

    hermitian = sc.HermitianSpace(2, ctx=ctx)
    with pytest.raises(sc.SpaceValidationError, match="not Hermitian"):
        hermitian.check_member(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))

    cheap_ctx = _ctx("cheap")
    cheap_hermitian = sc.HermitianSpace(2, ctx=cheap_ctx)
    cheap_product = sc.TreeSpace.from_leaf_spaces((cheap_hermitian,), cheap_ctx)
    cheap_product.check_member((cheap_ctx.asarray([[1.0, 2.0], [0.0, 1.0]]),))

    standard_product = sc.TreeSpace.from_leaf_spaces((hermitian,), ctx)
    with pytest.raises(sc.SpaceValidationError, match=r"\$\[0\].*not Hermitian"):
        standard_product.check_member((ctx.asarray([[1.0, 2.0], [0.0, 1.0]]),))


def test_checked_method_and_batched_validation_follow_cheap_policy():
    ctx = _ctx("cheap")
    space = sc.DenseCoordinateSpace((2,), ctx)
    identity = sc.IdentityLinOp(space, ctx)

    with pytest.raises(sc.SpaceValidationError, match="Expected shape"):
        identity.apply(ctx.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(sc.SpaceValidationError, match="trailing shape"):
        identity.vapply(ctx.asarray([[1.0, 2.0, 3.0]]))


def test_functional_scalar_output_shape_is_standard():
    cheap_ctx = _ctx("cheap")
    cheap_space = sc.DenseCoordinateSpace((2,), cheap_ctx)
    cheap_functional = sc.MatrixFreeLinearFunctional(
        lambda _x: cheap_ctx.asarray([1.0]), cheap_space, cheap_ctx
    )
    assert cheap_functional.value(cheap_ctx.asarray([1.0, 2.0])).shape == (1,)

    standard_ctx = _ctx("standard")
    standard_space = sc.DenseCoordinateSpace((2,), standard_ctx)
    standard_functional = sc.MatrixFreeLinearFunctional(
        lambda _x: standard_ctx.asarray([1.0]), standard_space, standard_ctx
    )
    with pytest.raises(ValueError, match="scalar batch output"):
        standard_functional.value(standard_ctx.asarray([1.0, 2.0]))


def test_strict_matrix_free_adjoint_probe_is_strict_only():
    standard_ctx = _ctx("standard")
    standard_space = sc.DenseCoordinateSpace((2,), standard_ctx)
    sc.MatrixFreeLinOp(
        lambda x: x,
        lambda y: standard_ctx.asarray([0.0, 0.0]),
        standard_space,
        standard_space,
        standard_ctx,
    )

    strict_ctx = _ctx("strict")
    strict_space = sc.DenseCoordinateSpace((2,), strict_ctx)
    with pytest.raises(ValueError, match="adjoint consistency check failed"):
        sc.MatrixFreeLinOp(
            lambda x: x,
            lambda y: strict_ctx.asarray([0.0, 0.0]),
            strict_space,
            strict_space,
            strict_ctx,
        )


def test_strict_matrix_free_coordinate_adjoint_preserves_non_euclidean_metric():
    ctx = _ctx("strict")
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 3.0]))
    )
    codomain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([5.0, 7.0]))
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
    none_ctx = _ctx("none")
    domain = sc.DenseCoordinateSpace((2,), none_ctx)
    codomain = sc.DenseCoordinateSpace((3,), none_ctx)
    rectangular = sc.ZeroLinOp(domain, codomain, none_ctx)
    with pytest.raises(ValueError, match="square LinOp"):
        sc.cg(rectangular, none_ctx.asarray([1.0, 1.0, 1.0]), maxiter=0)

    standard_ctx = _ctx("standard")
    standard_space = sc.DenseCoordinateSpace((2,), standard_ctx)
    standard_negative = sc.DiagonalLinOp(
        standard_ctx.asarray([-1.0, -1.0]), standard_space, standard_ctx
    )
    sc.cg(standard_negative, standard_ctx.asarray([1.0, 1.0]), maxiter=0)

    strict_ctx = _ctx("strict")
    strict_space = sc.DenseCoordinateSpace((2,), strict_ctx)
    strict_negative = sc.DiagonalLinOp(
        strict_ctx.asarray([-1.0, -1.0]), strict_space, strict_ctx
    )
    with pytest.raises(ValueError, match="positive curvature"):
        sc.cg(strict_negative, strict_ctx.asarray([1.0, 1.0]), maxiter=0)

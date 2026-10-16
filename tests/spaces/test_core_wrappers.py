"""Spaces: families implement ``_*_core``; every public operation is the checked wrapper."""
import numpy as np
import pytest

import spacecore as sc


def _ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def test_an_undecorated_override_keeps_the_base_checks():
    class _Space(sc.DenseVectorSpace):
        def axpy(self, a, x, y):  # older style: overrides the public operation
            return a * x + y

    X = _Space((2,), _ctx())
    np.testing.assert_allclose(X.axpy(2.0, np.ones(2), np.ones(2)), [3.0, 3.0])
    with pytest.raises(sc.SpaceValidationError):
        X.axpy(2.0, np.ones(3), np.ones(3))


def test_a_real_space_rejects_a_non_real_multiplier():
    X = sc.DenseVectorSpace((2,), _ctx())
    for operation in (lambda: X.scale(1j, np.ones(2)), lambda: X.axpy(1j, np.ones(2), np.ones(2))):
        with pytest.raises(sc.SpaceValidationError, match="non-real scalar"):
            operation()


def test_riesz_maps_accept_an_element_or_a_batch_and_check_its_shape():
    ctx = _ctx()
    X = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0])))
    np.testing.assert_allclose(X.riesz(np.ones(2)), [2.0, 5.0])
    np.testing.assert_allclose(X.riesz_inverse(np.ones((3, 2))), np.tile([0.5, 0.2], (3, 1)))
    with pytest.raises(sc.SpaceValidationError):
        X.riesz(np.ones(3))
    with pytest.raises(sc.SpaceValidationError):
        X.norm(np.ones(3))


def test_coordinate_operations_check_elements_and_batches():
    X = sc.DenseCoordinateSpace((2,), _ctx())
    with pytest.raises(sc.SpaceValidationError):
        X.flatten(np.ones(3))
    with pytest.raises(sc.SpaceValidationError):
        X.flatten_batch(np.ones((4, 3)))
    with pytest.raises(sc.SpaceValidationError):
        X.add_batch(np.ones((4, 2)), np.ones((4, 3)))
    with pytest.raises(sc.SpaceValidationError, match="non-real scalar"):
        X.scale_batch(1j, np.ones((4, 2)))
    np.testing.assert_allclose(X.vpair(np.ones(2), np.ones((3, 2))), [2.0, 2.0, 2.0])


def test_internal_paths_call_cores_not_public_operations(monkeypatch):
    X = sc.DenseCoordinateSpace((2,), _ctx())
    A = sc.DenseLinOp(X.ctx.asarray(np.eye(2)), X, X, X.ctx)
    for name in ("flatten", "unflatten"):
        monkeypatch.setattr(type(X), name, lambda *a, **k: pytest.fail("public op called internally"))
    np.testing.assert_allclose(A.apply(X.ctx.asarray([1.0, 2.0])), [1.0, 2.0])

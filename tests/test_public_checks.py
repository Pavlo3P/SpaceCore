"""One mechanism for every object type: an undecorated public override keeps the base checks."""
import numpy as np
import pytest

import spacecore as sc


def _space():
    return sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), dtype=np.float64))


def test_linop_public_override_is_checked():
    X = _space()

    class _Doubling(sc.LinOp):
        _fields = ("dom", "cod", "ctx")

        def apply(self, x):  # undecorated public override
            return 2.0 * x

    A = _Doubling(X, X, X.ctx)
    np.testing.assert_allclose(A.apply(X.ctx.asarray([1.0, 2.0])), [2.0, 4.0])
    with pytest.raises(sc.SpaceValidationError):
        A.apply(np.ones(3))


@pytest.mark.parametrize("owner, name", [(sc.VectorSpace, "add"), (sc.LinOp, "apply"), (sc.Functional, "grad")])
def test_checked_public_methods_record_their_core(owner, name):
    method = vars(owner)[name]
    assert method._spacecore_checked and method._spacecore_core == f"_{name}_core"


def test_an_undecorated_override_becomes_the_core_so_internal_paths_use_it():
    calls = []

    class _Counting(sc.DenseCoordinateSpace):
        def flatten(self, x):  # older style: public, undecorated
            calls.append(1)
            return x.reshape((-1,))

    X = _Counting((2,), sc.Context(sc.NumpyOps(), dtype=np.float64))
    assert vars(_Counting)["_flatten_core"] is not sc.DenseCoordinateSpace._flatten_core
    assert vars(_Counting)["flatten"] is vars(sc.CoordinateSpace)["flatten"]
    A = sc.DenseLinOp(X.ctx.asarray(np.eye(2)), X, X, X.ctx)
    A.apply(X.ctx.asarray([1.0, 2.0]))  # DenseLinOp's core flattens through the space core
    assert calls
    with pytest.raises(sc.SpaceValidationError):
        X.flatten(np.ones(3))


def test_an_override_calling_super_on_its_own_name_stays_public():
    class _Logged(sc.DenseCoordinateSpace):
        def flatten(self, x):
            return super().flatten(x)

    assert "_flatten_core" not in vars(_Logged)
    X = _Logged((2,), sc.Context(sc.NumpyOps(), dtype=np.float64))
    np.testing.assert_allclose(X.flatten(X.ctx.asarray([1.0, 2.0])), [1.0, 2.0])
    with pytest.raises(sc.SpaceValidationError):
        X.flatten(np.ones(3))


def test_assigning_a_base_method_restores_the_base_core():
    class _Native(sc.Functional):
        _fields = ("dom", "ctx")

        def value(self, x):
            return self.ops.sum(x)

        def derivative_at(self, x):
            return sc.InnerProductFunctional(self.ops.ones_like(x), self.domain, self.ctx)

    class _Default(_Native):
        derivative_at = sc.Functional.derivative_at

        def grad(self, x):
            return 3.0 * self.ops.ones_like(x)

    X = _space()
    np.testing.assert_allclose(_Default(X).derivative_at(X.ctx.asarray([1.0, 2.0])).representer, [3.0, 3.0])

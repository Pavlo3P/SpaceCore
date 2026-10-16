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

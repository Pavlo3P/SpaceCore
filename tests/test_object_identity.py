"""Spaces, operators and functionals are hashable value objects, consistent with ``==``."""
import numpy as np

import spacecore as sc
from tests.context._contracts import assert_equality_contract


def _ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def test_spaces_follow_the_value_object_contract():
    ctx = _ctx()
    assert_equality_contract(
        equal=[sc.HermitianSpace(3, ctx=ctx), sc.HermitianSpace(3, ctx=ctx)],
        distinct=[sc.HermitianSpace(2, ctx=ctx), sc.DenseCoordinateSpace((3, 3), ctx)],
    )


def test_operators_follow_the_value_object_contract():
    ctx = _ctx()
    X = sc.DenseVectorSpace((2,), ctx)
    make = lambda m: sc.DenseLinOp(ctx.asarray(m), X, X, ctx)  # noqa: E731
    assert_equality_contract(equal=[make(np.eye(2)), make(np.eye(2))], distinct=[make(2 * np.eye(2))])
    assert {2.0 * make(np.eye(2)): "scaled"}[2.0 * make(np.eye(2))] == "scaled"


def test_functionals_with_their_own_eq_stay_hashable():
    ctx = _ctx()
    X = sc.DenseVectorSpace((2,), ctx)
    f, g = (sc.InnerProductFunctional(ctx.asarray([1.0, 2.0]), X, ctx) for _ in range(2))
    assert f == g and hash(f) == hash(g)
    assert hash(sc.SquaredL2NormFunctional(X)) == hash(sc.SquaredL2NormFunctional(X))


def test_notebook_repr_escapes_the_text_repr():
    X = sc.DenseVectorSpace((2,), _ctx())
    import html

    assert X._repr_html_() == f"<code>{html.escape(repr(X))}</code>"

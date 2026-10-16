"""Everything compiles under ``jax.jit`` and ``torch.compile(fullgraph=True)``, at any check level.

Checks are not a compilation target: while a backend traces, each object checks at the
lower of its own level and ``get_traced_check_level()``, which defaults to ``"none"``.
Each case is compiled and compared with its eager result, so a graph break or a wrong
trace fails.
"""
import numpy as np
import pytest

import spacecore as sc
from spacecore._checks import CHECK_LEVELS


def _backend(name):
    if name == "jax":
        jax = pytest.importorskip("jax")
        return sc.Context(sc.JaxOps(), dtype=jax.numpy.float32), jax.jit
    torch = pytest.importorskip("torch")
    torch._dynamo.reset()
    return sc.Context(sc.TorchOps(), dtype=torch.float32), lambda f: torch.compile(f, fullgraph=True)


def _objects(ctx, level="none"):
    with sc.use_check_level(level):
        H = sc.HermitianSpace(3, ctx=ctx)
        A = sc.DenseLinOp(ctx.asarray(np.arange(81.0).reshape(3, 3, 3, 3) / 81.0), H, H, ctx)
        return H, A, sc.SquaredL2NormFunctional(H)


@pytest.fixture
def traced_level():
    """Restore the process-wide traced cap after a test changes it."""
    yield sc.set_traced_check_level
    sc.set_traced_check_level(None)


CASES = {
    "space operation": lambda H, A, F: lambda a: H.add(a, 2.0 * a),
    "element arithmetic": lambda H, A, F: lambda a: (2.0 * sc.DenseElement(H, a) - sc.DenseElement(H, a)).array,
    "lazy operator algebra built in the trace": lambda H, A, F: lambda a: (A + 2.0 * A).apply(a),
    "lazy functional algebra, value and gradient": lambda H, A, F: lambda a: ((F + 2.0 * F)(a), F.grad(a)),
    "operator applied to an element": lambda H, A, F: lambda a: (A @ sc.DenseElement(H, a)).array,
}

TORCH_MATMUL = pytest.mark.xfail(
    strict=True,
    reason="torch 2.11 Dynamo cannot trace the builtin `@` between two user-defined objects; "
    "`A.__matmul__(x)` compiles. Strict, so this fails once torch supports it.",
)


@pytest.mark.parametrize("level", CHECK_LEVELS)
@pytest.mark.parametrize(
    "backend, case",
    [
        pytest.param(b, c, marks=TORCH_MATMUL if (b, c) == ("torch", "operator applied to an element") else ())
        for b in ("jax", "torch")
        for c in CASES
    ],
)
def test_compiles_and_matches_eager(backend, case, level):
    ctx, compile_ = _backend(backend)
    run = CASES[case](*_objects(ctx, level))
    a = ctx.asarray(np.eye(3))
    got, want = compile_(run)(a), run(a)
    for g, w in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,)):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5)


def test_traced_level_defaults_to_none_and_resets(traced_level):
    assert sc.get_traced_check_level() == "none"
    traced_level("cheap")
    assert sc.get_traced_check_level() == "cheap"
    traced_level(None)
    assert sc.get_traced_check_level() == "none"
    with pytest.raises(ValueError, match="Unknown check_level"):
        traced_level("loud")


def test_eager_checks_ignore_the_traced_level(traced_level):
    ctx, _ = _backend("jax")
    H, _, _ = _objects(ctx, "standard")
    traced_level("none")
    with pytest.raises(sc.SpaceValidationError):
        H.add(ctx.asarray(np.triu(np.ones((3, 3)))), ctx.asarray(np.eye(3)))


@pytest.mark.parametrize(
    "backend, case",
    [
        pytest.param(b, c, marks=TORCH_MATMUL if (b, c) == ("torch", "operator applied to an element") else ())
        for b in ("jax", "torch")
        for c in CASES
    ],
)
def test_traced_cheap_compiles_and_matches_eager(backend, case, traced_level):
    ctx, compile_ = _backend(backend)
    run = CASES[case](*_objects(ctx, "standard"))
    traced_level("cheap")
    a = ctx.asarray(np.eye(3))
    got, want = compile_(run)(a), run(a)
    for g, w in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,)):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_traced_cheap_keeps_shape_checks_and_skips_value_checks(backend, traced_level):
    ctx, compile_ = _backend(backend)
    H, _, _ = _objects(ctx, "standard")
    traced_level("cheap")
    add = compile_(lambda a, b: H.add(a, b))
    # Value check skipped under the trace: a non-Hermitian input passes.
    add(ctx.asarray(np.triu(np.ones((3, 3)))), ctx.asarray(np.eye(3)))
    # Shape check kept, raised once at trace time. Under fullgraph=True, Dynamo cannot keep a
    # `raise` in the graph, so it reports the SpaceValidationError as Unsupported.
    if backend == "torch":
        import torch._dynamo.exc

        expected = pytest.raises(torch._dynamo.exc.Unsupported, match="SpaceValidationError")
    else:
        expected = pytest.raises(sc.SpaceValidationError)
    with expected:
        add(ctx.asarray(np.ones((2, 2))), ctx.asarray(np.eye(2)))


def test_traced_none_skips_shape_checks_too(traced_level):
    ctx, jit = _backend("jax")
    H, _, _ = _objects(ctx, "standard")
    jit(lambda a, b: H.add(a, b))(ctx.asarray(np.ones((2, 2))), ctx.asarray(np.eye(2)))

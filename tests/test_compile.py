"""Everything compiles at ``check_level="none"``: ``jax.jit`` and ``torch.compile(fullgraph=True)``.

Checks are not a compilation target; compiled code runs with them off. Each case is
compiled and compared with its eager result, so a graph break or a wrong trace fails.
"""
import numpy as np
import pytest

import spacecore as sc


def _backend(name):
    if name == "jax":
        jax = pytest.importorskip("jax")
        return sc.Context(sc.JaxOps(), dtype=jax.numpy.float32), jax.jit
    torch = pytest.importorskip("torch")
    torch._dynamo.reset()
    return sc.Context(sc.TorchOps(), dtype=torch.float32), lambda f: torch.compile(f, fullgraph=True)


def _objects(ctx):
    with sc.use_check_level("none"):
        H = sc.HermitianSpace(3, ctx=ctx)
        A = sc.DenseLinOp(ctx.asarray(np.arange(81.0).reshape(3, 3, 3, 3) / 81.0), H, H, ctx)
        return H, A, sc.SquaredL2NormFunctional(H)


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


@pytest.mark.parametrize(
    "backend, case",
    [
        pytest.param(b, c, marks=TORCH_MATMUL if (b, c) == ("torch", "operator applied to an element") else ())
        for b in ("jax", "torch")
        for c in CASES
    ],
)
def test_compiles_and_matches_eager(backend, case):
    ctx, compile_ = _backend(backend)
    run = CASES[case](*_objects(ctx))
    a = ctx.asarray(np.eye(3))
    got, want = compile_(run)(a), run(a)
    for g, w in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,)):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5)

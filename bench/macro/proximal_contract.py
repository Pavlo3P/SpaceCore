"""Whole-loop public prox/Moreau benchmark against handwritten array updates."""
import numpy as np

from examples.proximal_contract import SplitValue, build_problem, pdhg, proximal_gradient
from ._registry import MacroBenchmark, MacroPayload, registry


def factory(*, backend, device, seed, size_params):
    """Build a deterministic consumer case; JAX compiles both compared loops."""
    if device != "cpu":
        raise ValueError("The proximal-contract acceptance benchmark currently targets CPU.")
    n, iterations = size_params["n"], size_params["iterations"]
    native, weighted = size_params["native"], size_params["weighted"]
    K, f, g, optimum = build_problem(native=native, weighted=weighted, backend=backend, count=n)
    ctx, ops = K.ctx, K.ops
    k, b = ctx.asarray(np.linspace(1, 2, n)), ctx.asarray(np.linspace(-2, 2, n))
    w = ctx.asarray(np.linspace(1, 3, n) if weighted else np.ones(n))
    method = pdhg if size_params["method"] == "pdhg" else proximal_gradient
    step = ctx.asarray(.4 if method is pdhg else .2)

    def bare(step):
        def shrink(x):
            radius = ops.xp.abs(x)
            return ops.xp.sign(x) * ops.xp.maximum(radius - step * .15 / w, ops.xp.zeros_like(radius))

        zeros = ops.xp.zeros_like(b)
        if method is proximal_gradient:
            def update(i, x):
                return shrink(x - step * k * (k*x - b) / w)
            return ops.fori_loop(0, iterations, update, zeros)

        def update(i, carry):
            x, p, extrapolated = carry
            p = (p + step * (k*extrapolated - b)) / (1 + step)
            next_x = shrink(x - step * k * p / w)
            return next_x, p, 2*next_x - x
        return ops.fori_loop(0, iterations, update, (zeros, zeros, zeros))[0]

    def public(step):
        return method(K, f, g, iterations=iterations, step=step)

    if backend == "jax":
        import jax
        bare, public = jax.jit(bare), jax.jit(public)

    def output(fn):
        result = fn(step)
        # The harness synchronizes ordinary tuples, including native leaves.
        return (result.head, result.tail) if isinstance(result, SplitValue) else result

    def metrics(result):
        x = (np.concatenate((np.asarray(result[0]).reshape(1), np.asarray(result[1])))
             if isinstance(result, tuple) else np.asarray(result))
        # Compare every coordinate, not just a scalar objective or norm.
        return {**{f"x{i}": float(v) for i, v in enumerate(x)},
                "residual": float(np.max(np.abs(x - optimum)))}

    return MacroPayload(
        iterations=iterations, size_params=dict(size_params),
        mode_callables={"bare": lambda: output(bare), "spacecore_public_none": lambda: output(public)},
        reference_metric_extractor=metrics,
    )


registry.register(MacroBenchmark(
    name="proximal_contract",
    workload="Public metric prox and Hilbert conjugate prox in unchanged consumer algorithms",
    sizes={
        f"{method}_{representation}_{metric}_{n}": dict(
            n=n, iterations=200, method=method, native=representation == "native",
            weighted=metric == "weighted",
        )
        for method in ("pg", "pdhg")
        for representation in ("dense", "native")
        for metric in ("euclidean", "weighted")
        for n in (32, 1024)
    },
    quick_sizes=("pg_dense_weighted_32", "pdhg_native_weighted_32"),
    backends=("numpy", "jax"), factory=factory,
    notes="CPU; deterministic problem; both JAX paths compiled; preparation excluded.",
))

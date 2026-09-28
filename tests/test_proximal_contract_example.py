"""Consumer acceptance: unchanged algorithms across geometry and storage."""
import numpy as np
import pytest

import spacecore as sc
from examples.proximal_contract import (
    SplitValue, as_numpy, build_problem, pdhg, proximal_gradient,
)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_identical_loops_reach_metric_independent_optimum(backend, native, weighted):
    if backend == "jax":
        jax = pytest.importorskip("jax")
    K, f, g, optimum = build_problem(native=native, weighted=weighted, backend=backend)
    for method, step in ((proximal_gradient, .2), (pdhg, .4)):
        results = [method(K, f, g, step=step)]
        if backend == "jax":
            results.append(jax.jit(lambda t: method(K, f, g, step=t))(K.ctx.asarray(step)))
        for result in results:
            np.testing.assert_allclose(as_numpy(result), optimum, atol=2e-5)
            # Independent coordinate KKT residual, without reusing a prox.
            x = as_numpy(result)
            k, b = np.linspace(1, 2, len(x)), np.linspace(-2, 2, len(x))
            derivative = k * (k * x - b)
            active = np.abs(x) > 1e-6
            np.testing.assert_allclose(derivative[active] + .15*np.sign(x[active]), 0, atol=2e-5)
            assert np.all(np.abs(derivative[~active]) <= .15 + 2e-5)
            if native:
                assert isinstance(result, SplitValue)
                assert not hasattr(K.domain, "shape") and not hasattr(K.domain, "flatten")
                with pytest.raises(TypeError, match="stay native"):
                    np.asarray(result)


@pytest.mark.parametrize("weighted", [False, True])
def test_same_loops_accept_indicator_in_place_of_penalty(weighted):
    K, f, _, _ = build_problem(weighted=weighted)
    g = sc.IndicatorFunctional(sc.NonnegativeOrthant(K.domain))
    optimum = np.maximum(np.linspace(-2, 2, 16) / np.linspace(1, 2, 16), 0)
    for method in (proximal_gradient, pdhg):
        result = method(K, f, g)
        np.testing.assert_allclose(as_numpy(result), optimum, atol=2e-5)
        assert float(g(result)) == 0


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("method", ["pg", "pdhg"])
def test_benchmark_reference_matches_every_coordinate(backend, method):
    if backend == "jax":
        pytest.importorskip("jax")
    from bench.macro.proximal_contract import factory
    for native in (False, True):
        payload = factory(backend=backend, device="cpu", seed=0,
                          size_params=dict(n=16, iterations=150, native=native,
                                           weighted=True, method=method))
        metrics = [payload.reference_metric_extractor(fn()) for fn in payload.mode_callables.values()]
        assert metrics[0]["residual"] < 2e-5
        assert metrics[1]["residual"] < 2e-5
        np.testing.assert_allclose(list(metrics[0].values()), list(metrics[1].values()), atol=2e-5)

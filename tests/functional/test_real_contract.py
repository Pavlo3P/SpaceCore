"""Real output is a mathematical invariant, independent of validation policy."""
import numpy as np
import pytest

import spacecore as sc


@pytest.mark.parametrize("level", ["none", "cheap", "standard", "strict"])
@pytest.mark.parametrize("value", [1j, 1 + 0j, np.asarray(2j), np.complex64(1), np.array(1j, dtype=object)])
def test_complex_callbacks_and_codomain_rejected(level, value):
    X = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), np.float64),
                                check_level=level)
    with pytest.raises(TypeError, match="real Field"):
        sc.MatrixFreeLinearFunctional(lambda x: 1., X, cod=sc.ComplexField(),
                                      check_level=level)
    f = sc.MatrixFreeLinearFunctional(lambda x: value, X, check_level=level,
                                      vvalue=lambda xs: np.full(len(xs), value))
    for evaluate in (lambda: f(X.zeros()), lambda: f.value(X.zeros()),
                     lambda: f.vvalue(np.zeros((3, 2))),
                     lambda: (f * f)(X.zeros())):
        with pytest.raises(TypeError, match="real-valued"):
            evaluate()


@pytest.mark.parametrize("method", ["value", "vvalue", "value_and_grad",
                                     "value_and_derivative", "_value_core", "_vvalue_core"])
def test_subclass_overrides_cannot_bypass_real_output(method):
    def invalid(self, x):
        return (1j, x) if method.startswith("value_and_") else np.asarray(1j)

    cls = type("Invalid", (sc.Functional,), {"_fields": ("dom",), method: invalid})
    X = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), np.float64))
    f = cls(X, check_level="none")
    with pytest.raises(TypeError, match="real-valued"):
        getattr(f, method)(X.zeros())


@pytest.mark.parametrize("build", [
    lambda X, f, a: sc.ScaledFunctional(a, f),
    lambda X, f, a: a * f,
    lambda X, f, a: a * sc.ZeroFunctional(X),
    lambda X, f, a: f + a,
    lambda X, f, a: a + f,
    lambda X, f, a: f - a,
    lambda X, f, a: a - f,
    lambda X, f, a: sc.ConstantFunctional(X, a),
    lambda X, f, a: sc.make_constant_functional(X, a),
    lambda X, f, a: sc.LinOpQuadraticForm(sc.IdentityLinOp(X), a=a),
])
@pytest.mark.parametrize("coefficient", [1j, np.asarray(0j), np.asarray(1 + 0j)])
def test_complex_coefficients_rejected_before_evaluation(build, coefficient):
    X = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), np.complex128),
                                check_level="none")
    f = sc.InnerProductFunctional(X.ctx.asarray([1., 1.]), X, check_level="none")
    with pytest.raises(TypeError, match="real-valued"):
        build(X, f, coefficient)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_real_outputs_complex_domain_and_backend_compilation(backend):
    if backend == "jax":
        jax = pytest.importorskip("jax")
        ops, dtype = sc.JaxOps(), "complex64"
        compile_ = jax.jit
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        ops, dtype = sc.TorchOps(), "complex64"
        def compile_(f):
            torch.compiler.reset()
            return torch.compile(f, backend="eager", fullgraph=True)
    else:
        ops, dtype = sc.NumpyOps(), np.complex128
        def compile_(f):
            return f
    ctx = sc.Context(ops, dtype)
    X = sc.DenseCoordinateSpace((2,), ctx, check_level="none")
    x, c = ctx.asarray([1 + 2j, 3 - 1j]), ctx.asarray([2 + 1j, -1 + 3j])
    linear = sc.InnerProductFunctional(c, X, check_level="none")
    q = sc.LinOpQuadraticForm(sc.IdentityLinOp(X), linear, a=2., check_level="none")
    for f in (linear, q, linear * linear + 2, sc.ZeroFunctional(X),
              sc.ConstantFunctional(X, 3.), sc.ScaledFunctional(2., linear)):
        assert f.codomain.scalar_field == "real"
        assert not ops.is_complex_dtype(compile_(f.value)(x).dtype)
        assert not ops.is_complex_dtype(f.vvalue(ops.stack([x, x])).dtype)
        value, grad = compile_(f.value_and_grad)(x)
        assert not ops.is_complex_dtype(value.dtype)
        assert ops.is_complex_dtype(grad.dtype)
        children, aux = f.tree_flatten()
        restored = type(f).tree_unflatten(aux, children)
        assert restored.codomain.scalar_field == "real"
        converted = f.convert(sc.Context(sc.NumpyOps(), np.complex64))
        assert not np.iscomplexobj(converted(converted.ctx.asarray([1 + 2j, 3 - 1j])))
    bad = sc.MatrixFreeLinearFunctional(lambda v: v[0], X, check_level="none")
    if backend != "torch":  # Torch wraps user exceptions in its own tracing error.
        with pytest.raises(TypeError, match="real-valued"):
            compile_(bad.value)(x)
    with pytest.raises(TypeError, match="real-valued"):
        bad(x)


def test_extended_real_values_remain_supported():
    X = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), np.float64))
    assert np.isposinf(sc.ConstantFunctional(X, np.inf)(X.zeros()))

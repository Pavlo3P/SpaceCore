"""Functional outputs and coefficients are checked for realness whenever checks are on."""
import numpy as np
import pytest

import spacecore as sc

CHECKED = ["cheap", "standard", "strict"]


def _space(level, dtype=np.float64):
    return sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), dtype), check_level=level)


@pytest.mark.parametrize("level", CHECKED)
@pytest.mark.parametrize("value", [1j, 1 + 0j, np.asarray(2j), np.complex64(1)])
def test_complex_output_is_caught(level, value):
    X = _space(level)
    f = sc.MatrixFreeLinearFunctional(lambda x: value, X, check_level=level,
                                      vvalue=lambda xs: np.full(len(xs), value))
    for evaluate in (lambda: f(X.zeros()), lambda: f.value(X.zeros()),
                     lambda: f.vvalue(np.zeros((3, 2))),
                     lambda: (f * f)(X.zeros())):
        with pytest.raises(TypeError, match="real value"):
            evaluate()


def test_unchecked_output_passes_through():
    X = _space("none")
    f = sc.MatrixFreeLinearFunctional(lambda x: np.asarray(1j), X, check_level="none")
    assert f(X.zeros()) == 1j


def test_codomain_is_the_real_field():
    f = sc.MatrixFreeLinearFunctional(lambda x: 1., _space("standard"))
    assert isinstance(f.codomain, sc.RealField)
    with pytest.raises(TypeError):
        sc.MatrixFreeLinearFunctional(lambda x: 1., _space("standard"), cod=sc.RealField())


@pytest.mark.parametrize("method", ["value", "vvalue", "value_and_grad", "value_and_derivative"])
def test_overriding_evaluators_are_checked(method):
    def invalid(self, x):
        return (np.asarray(1j), x) if method.startswith("value_and_") else np.asarray(1j)

    cls = type("Invalid", (sc.Functional,), {"_fields": ("dom",), method: invalid})
    X = _space("standard")
    with pytest.raises(TypeError, match="real value"):
        getattr(cls(X, check_level="cheap"), method)(X.zeros())
    getattr(cls(X, check_level="none"), method)(X.zeros())


@pytest.mark.parametrize("build", [
    lambda X, f, a: sc.ScaledFunctional(a, f),
    lambda X, f, a: a * f,
    lambda X, f, a: f + a,
    lambda X, f, a: a - f,
    lambda X, f, a: sc.ConstantFunctional(X, a, check_level="cheap"),
    lambda X, f, a: sc.LinOpQuadraticForm(sc.IdentityLinOp(X), a=a, check_level="cheap"),
])
@pytest.mark.parametrize("coefficient", [1j, np.asarray(1 + 1j)])
def test_complex_coefficients_are_caught(build, coefficient):
    X = _space("cheap", np.complex128)
    f = sc.InnerProductFunctional(X.ctx.asarray([1., 1.]), X, check_level="cheap")
    with pytest.raises(TypeError, match="real value"):
        build(X, f, coefficient)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_real_outputs_on_complex_domains_compile(backend):
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
    for f in (linear, linear * linear + 2, sc.ZeroFunctional(X),
              sc.ConstantFunctional(X, 3.), sc.ScaledFunctional(2., linear)):
        assert isinstance(f.codomain, sc.RealField)
        assert not ops.is_complex_dtype(compile_(f.value)(x).dtype)
        assert not ops.is_complex_dtype(f.vvalue(ops.stack([x, x])).dtype)
        value, grad = compile_(f.value_and_grad)(x)
        assert not ops.is_complex_dtype(value.dtype)
        assert ops.is_complex_dtype(grad.dtype)
        children, aux = f.tree_flatten()
        assert isinstance(type(f).tree_unflatten(aux, children).codomain, sc.RealField)
        converted = f.convert(sc.Context(sc.NumpyOps(), np.complex64))
        assert not np.iscomplexobj(converted(converted.ctx.asarray([1 + 2j, 3 - 1j])))


def test_extended_real_values_remain_supported():
    X = _space("standard")
    assert np.isposinf(sc.ConstantFunctional(X, np.inf)(X.zeros()))

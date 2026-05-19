import importlib
import numpy as np
import pytest

from tests._helpers import has_torch


def test_builtin_backends_are_usable():
    sc = importlib.import_module("spacecore")
    assert isinstance(sc.NumpyOps(), sc.BackendOps)


def test_register_ops_adds_backend():
    sc = importlib.import_module("spacecore")

    class DummyOps(sc.BackendOps):
        import array_api_compat.numpy as xp

        _family = "dummy"
        _allow_sparse = False

        @property
        def dense_array(self):
            return np.ndarray

        @property
        def sparse_array(self):
            return None

        def sanitize_dtype(self, dtype): return np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
        def assparse(self, x, dtype=None): raise NotImplementedError
        def sparse_matmul(self, a, b): raise NotImplementedError
        def logsumexp(self, *args, **kwargs): raise NotImplementedError

        def index_set(self, x, index, values, copy=True):
            y = np.array(x, copy=True)
            y[index] = values
            return y

        def ix_(self, *args): return np.ix_(*args)
        def fori_loop(self, lower, upper, body_fun, init_val, **kwargs):
            val = init_val
            for i in range(lower, upper): val = body_fun(i, val)
            return val
        def while_loop(self, cond_fun, body_fun, init_val):
            val=init_val
            while cond_fun(val): val=body_fun(val)
            return val
        def scan(self, f, init, xs, **kwargs):
            carry=init
            ys=[]
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
            return carry, np.array(ys)
        def cond(self, pred, true_fun, false_fun, *operands): return true_fun(*operands) if pred else false_fun(*operands)
        def index_add(self, x, index, values, copy=True):
            y=np.array(x, copy=True)
            y[index]+=values
            return y
        def allclose_sparse(self, a, b, **kwargs): return False
    sc.register_ops(DummyOps)
    assert "dummy" in sc._contextual.manager.ctx_manager.available_ops
    ops = DummyOps()
    x = ops.reshape(ops.arange(6), (2, 3))
    assert np.allclose(ops.sum(x, axis=0), [3, 5, 7])


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_backend_aliases_resolve_when_available():
    sc = importlib.import_module("spacecore")

    assert isinstance(sc.TorchOps(), sc.BackendOps)
    assert sc.VectorSpace((1,), "torch").ctx.ops.family == "torch"
    assert sc.VectorSpace((1,), "pytorch").ctx.ops.family == "torch"

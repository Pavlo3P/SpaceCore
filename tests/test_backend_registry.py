import importlib
import numpy as np


def test_builtin_backends_are_usable():
    sc = importlib.import_module("spacecore")
    assert isinstance(sc.NumpyOps(), sc.BackendOps)


def test_register_ops_adds_backend():
    sc = importlib.import_module("spacecore")

    class DummyOps(sc.BackendOps):
        _family = "dummy"
        _allow_sparse = False
        _dense_array = np.ndarray
        _sparse_array = object
        def sanitize_dtype(self, dtype): return np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
        def asarray(self, x, dtype=None): return np.asarray(x, dtype=dtype)
        def assparse(self, x, dtype=None): raise NotImplementedError
        def is_dense(self, x): return isinstance(x, np.ndarray)
        def is_sparse(self, x): return False
        def get_dtype(self, x): return x.dtype
        def zeros(self, shape, dtype=None): return np.zeros(shape, dtype=dtype)
        def ones(self, shape, dtype=None): return np.ones(shape, dtype=dtype)
        def full(self, shape, fill_value, dtype=None): return np.full(shape, fill_value, dtype=dtype)
        def empty(self, shape, dtype=None): return np.empty(shape, dtype=dtype)
        def eye(self, n, dtype=None): return np.eye(n, dtype=dtype)
        def arange(self, *args, **kwargs): return np.arange(*args, **kwargs)
        def reshape(self, x, shape): return np.reshape(x, shape)
        def ravel(self, x): return np.ravel(x)
        def transpose(self, x, axes=None): return np.transpose(x, axes=axes)
        def conj(self, x): return np.conj(x)
        def sum(self, x, axis=None, **kwargs): return np.sum(x, axis=axis)
        def prod(self, x, axis=None, **kwargs): return np.prod(x, axis=axis)
        def trace(self, x, **kwargs): return np.trace(x)
        def argsort(self, x, **kwargs): return np.argsort(x)
        def sort(self, x, **kwargs): return np.sort(x)
        def argmin(self, x, **kwargs): return np.argmin(x)
        def argmax(self, x, **kwargs): return np.argmax(x)
        def vdot(self, a, b, **kwargs): return np.vdot(a, b)
        def matmul(self, a, b, **kwargs): return a @ b
        def sparse_matmul(self, a, b): raise NotImplementedError
        def kron(self, a, b): return np.kron(a, b)
        def einsum(self, subscripts, *operands, **kwargs): return np.einsum(subscripts, *operands)
        def eigh(self, x, **kwargs): return np.linalg.eigh(x)
        def logsumexp(self, *args, **kwargs): raise NotImplementedError
        def exp(self, x): return np.exp(x)
        def log(self, x): return np.log(x)
        def maximum(self, x, y): return np.maximum(x, y)
        def minimum(self, x, y): return np.minimum(x, y)
        def where(self, condition, x=None, y=None, **kwargs): return np.where(condition, x, y)
        def concatenate(self, arrays, axis=0, dtype=None):
            out = np.concatenate(arrays, axis=axis)
            return out.astype(dtype) if dtype is not None else out
        def stack(self, arrays, axis=0): return np.stack(arrays, axis=axis)
        def sqrt(self, x): return np.sqrt(x)
        def abs(self, x): return np.abs(x)
        def real(self, x): return np.real(x)
        def imag(self, x): return np.imag(x)
        def sign(self, x): return np.sign(x)

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
        def allclose(self, a, b, **kwargs): return np.allclose(a, b, **kwargs)
        def allclose_sparse(self, a, b, **kwargs): return False
    sc.register_ops(DummyOps)
    assert "dummy" in sc._contextual.manager.ctx_manager.available_ops

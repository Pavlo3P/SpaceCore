from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Type

from .._family import BackendFamily
from .._ops import BackendOps
from ...types import ArrayLike, Carry, DenseArray, DType, Index, R, SparseArray, T, X, Y


class CuPyOps(BackendOps):
    """
    BackendOps implementation for CuPy GPU arrays.

    This backend uses CuPy for dense array operations and ``cupyx.scipy.sparse``
    for sparse arrays. Most operations follow CuPy's NumPy-compatible API and
    execute on the active CUDA device.

    Dense arrays
        ``cupy.ndarray``

    Sparse arrays
        ``cupyx.scipy.sparse`` matrix types such as CSR, CSC, and COO.
    """

    import cupy as cp
    import cupyx.scipy as cpx_scipy
    import cupyx.scipy.sparse as cpx_sparse

    xp = cp

    _family = BackendFamily.cupy.value.lower()
    _allow_sparse = True

    @property
    def dense_array(self) -> Type[Any]:
        """Dense CuPy array type."""
        return self.cp.ndarray

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """Sparse CuPy array type tuple."""
        sparse = self.cpx_sparse
        types: list[type[Any]] = []
        for name in ("spmatrix", "csr_matrix", "csc_matrix", "coo_matrix"):
            typ = getattr(sparse, name, None)
            if typ is not None:
                types.append(typ)
        return tuple(types)

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize a dtype specifier using CuPy.

        ``None`` follows NumPy/CuPy's float64 default.
        """
        if dtype is None:
            return self.cp.float64
        return self.cp.dtype(dtype)

    def assparse(
        self,
        x: Any,
        *,
        format: Literal["csr", "csc", "coo"] = "csr",
        dtype: DType | None = None,
    ) -> SparseArray:
        """
        Convert input to a CuPy sparse matrix.

        Dense inputs must be two-dimensional. Existing sparse inputs are
        converted to the requested sparse format.
        """
        sparse = self.cpx_sparse

        if self.is_sparse(x):
            if format == "csr":
                return x.tocsr()
            if format == "csc":
                return x.tocsc()
            if format == "coo":
                return x.tocoo()
            raise ValueError(f"Unknown sparse format: {format!r}")

        x_arr = self.asarray(x, dtype=dtype)
        if x_arr.ndim != 2:
            raise ValueError("CuPy sparse conversion currently expects a 2D array.")

        if format == "csr":
            return sparse.csr_matrix(x_arr)
        if format == "csc":
            return sparse.csc_matrix(x_arr)
        if format == "coo":
            return sparse.coo_matrix(x_arr)
        raise ValueError(f"Unknown sparse format: {format!r}")

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """Multiply a CuPy sparse matrix by a CuPy dense array."""
        if not self.is_sparse(a):
            raise TypeError("sparse_matmul expects a CuPy sparse matrix.")
        if not self.is_dense(b):
            raise TypeError("sparse_matmul expects a CuPy dense array.")
        return a @ b

    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
    ) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """Compute log-sum-exp using ``cupyx.scipy.special``."""
        return self.cpx_scipy.special.logsumexp(
            a,
            axis=axis,
            b=b,
            keepdims=keepdims,
            return_sign=return_sign,
        )

    def index_set(
        self,
        x: DenseArray,
        index: Index,
        values: ArrayLike,
        *,
        copy: bool = True,
    ) -> DenseArray:
        """Set indexed values in a CuPy array."""
        y = x.copy() if copy else x
        y[index] = values
        return y

    def index_add(
        self,
        x: DenseArray,
        index: Index,
        values: DenseArray,
        *,
        copy: bool = True,
    ) -> DenseArray:
        """Add values into indexed entries of a CuPy array."""
        y = x.copy() if copy else x
        self.cp.add.at(y, index, values)
        return y

    def ix_(self, *args: Any) -> Any:
        """Build open-mesh indices using CuPy."""
        return self.cp.ix_(*args)

    def fori_loop(
        self,
        lower: int,
        upper: int,
        body_fun: Callable[[int, T], T],
        init_val: T,
    ) -> T:
        """Run a counted loop eagerly in Python for CuPy."""
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    def while_loop(
        self,
        cond_fun: Callable[[T], bool],
        body_fun: Callable[[T], T],
        init_val: T,
    ) -> T:
        """Run a while loop eagerly in Python for CuPy."""
        val = init_val
        while bool(cond_fun(val)):
            val = body_fun(val)
        return val

    def _tree_map(self, f: Callable[[Any], Any], tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: self._tree_map(f, v) for k, v in tree.items()}
        if isinstance(tree, tuple):
            return tuple(self._tree_map(f, v) for v in tree)
        if isinstance(tree, list):
            return [self._tree_map(f, v) for v in tree]
        return f(tree)

    def _tree_multimap(self, f: Callable[..., Any], *trees: Any) -> Any:
        t0 = trees[0]
        if isinstance(t0, dict):
            return {k: self._tree_multimap(f, *(t[k] for t in trees)) for k in t0.keys()}
        if isinstance(t0, tuple):
            return tuple(self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0)))
        if isinstance(t0, list):
            return [self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0))]
        return f(*trees)

    def _tree_take0(self, xs: Any) -> Any:
        if isinstance(xs, dict):
            return self._tree_take0(next(iter(xs.values())))
        if isinstance(xs, (tuple, list)):
            return self._tree_take0(xs[0])
        return xs

    def _tree_index(self, xs: Any, i: int) -> Any:
        def _idx(a: Any) -> Any:
            try:
                return a[i]
            except Exception:
                return a

        return self._tree_map(_idx, xs)

    def _tree_stack(self, ys_list: Sequence[Any]) -> Any:
        if not ys_list:
            return ()

        def _stack_leaves(*leaves: Any) -> Any:
            try:
                return self.cp.stack(leaves, axis=0)
            except Exception:
                return self.cp.asarray(leaves)

        return self._tree_multimap(_stack_leaves, *ys_list)

    def scan(
        self,
        f: Callable[[Carry, X], Tuple[Carry, Y]],
        init: Carry,
        xs: X,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
    ) -> Tuple[Carry, Y]:
        """Run a scan loop eagerly in Python for CuPy."""
        carry = init
        if xs is None:
            if length is None:
                raise ValueError("scan(xs=None) requires an explicit `length`.")
            n = int(length)
            indices = range(n - 1, -1, -1) if reverse else range(n)
            ys_steps: list[Any] = []
            for _i in indices:
                carry, y = f(carry, None)  # type: ignore[arg-type]
                ys_steps.append(y)
            if reverse:
                ys_steps.reverse()
            return carry, self._tree_stack(ys_steps)

        if length is None:
            leaf0 = self._tree_take0(xs)
            try:
                n = int(leaf0.shape[0])
            except Exception as e:
                raise ValueError(
                    "Could not infer scan length from `xs`; pass `length=` explicitly."
                ) from e
        else:
            n = int(length)

        indices = range(n - 1, -1, -1) if reverse else range(n)
        ys_steps = []
        for i in indices:
            x_i = self._tree_index(xs, i)
            carry, y = f(carry, x_i)
            ys_steps.append(y)
        if reverse:
            ys_steps.reverse()
        return carry, self._tree_stack(ys_steps)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[T], R],
        false_fun: Callable[[T], R],
        *operands: Any,
    ) -> R:
        """Run conditional branch selection eagerly in Python for CuPy."""
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Compare two CuPy sparse matrices by dense values."""
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError("allclose_sparse expects two CuPy sparse matrices.")
        return bool(self.cp.asnumpy(self.cp.allclose(a.toarray(), b.toarray(), rtol=rtol, atol=atol)))

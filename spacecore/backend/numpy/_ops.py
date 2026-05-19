from __future__ import annotations

from typing import Any, Sequence, Tuple, Literal, Callable, Optional, Type

from .._family import BackendFamily
from .._ops import BackendOps
from ...types import DenseArray, SparseArray, DType, Index, X, T, Y, R, Carry


class NumpyOps(BackendOps):
    """
    BackendOps implementation for the NumPy/SciPy ecosystem.

    This backend uses NumPy for dense array operations and SciPy for sparse
    array operations.

    Dense arrays
        numpy.ndarray

    Sparse arrays
        scipy.sparse.spmatrix
        scipy.sparse.sparray

    Methods
        Most methods mirror the corresponding NumPy or SciPy signatures and
        delegate directly to NumPy/SciPy implementations. Backend-specific
        behavior, dtype promotion, broadcasting, memory layout, and error modes
        therefore follow NumPy/SciPy semantics.

    Backend handles
      - np : module
        NumPy module stored on the class and available through instances as
        `ops.np`. Advanced users may use it when SpaceCore's portable API
        does not expose a required NumPy feature.

      - sp : module
        SciPy module stored on the class and available through instances as
        `ops.sp`. Advanced users may use it for SciPy-specific functionality.

    Notes
        Code intended to remain backend-portable should prefer `BackendOps`
        methods. Direct use of `ops.np` or `ops.sp` is an explicit
        NumPy/SciPy-specific escape hatch.

        NumPy's `device` keyword is present for Array API interoperability.
        When supplied, it must be `"cpu"` or `None`; see the corresponding NumPy
        documentation for each method.
    """
    import numpy as np
    import scipy as sp
    import array_api_compat.numpy as xp

    _family = BackendFamily.numpy.value.lower()
    _allow_sparse = True

    def __init__(self) -> None:
        super().__init__()

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using NumPy.

        Returns:
            Concrete dense array class accepted by this backend.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
        """
        return self.np.ndarray

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using SciPy.

        Returns:
            Concrete sparse array classes accepted by this backend, or None.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html
        """
        sparse = self.sp.sparse
        types: list[type[Any]] = []
        if hasattr(sparse, "spmatrix"):
            types.append(sparse.spmatrix)
        if hasattr(sparse, "sparray"):
            types.append(sparse.sparray)
        return tuple(types)


    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize a dtype specifier using NumPy.

        Input:
            dtype: Optional dtype requested by SpaceCore or the caller.

        Output:
            Backend dtype object accepted by array constructors.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.dtype.html
        """

        if dtype is None:
            return self.np.float64
        return self.np.dtype(dtype)

    def assparse(self, x: Any, *, format: Literal["csr", "csc", "coo"] = "csr", dtype: DType | None = None) -> SparseArray:
        """
        Convert input to a sparse array using SciPy.

        Input:
            x: Dense, sparse, or array-like input plus sparse-format options.

        Output:
            Sparse backend array.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

        Backend-specific notes:
            SpaceCore currently converts dense inputs to 2-D SciPy sparse matrices in the requested format.
        """
        sparse = self.sp.sparse

        if self.is_sparse(x):
            if format == "csr":
                return x.tocsr()
            if format == "csc":
                return x.tocsc()
            if format == "coo":
                return x.tocoo()
            raise ValueError(f"Unknown sparse format: {format!r}")

        x_arr = self.asarray(x)

        if x_arr.ndim != 2:
            raise ValueError("NumPy/SciPy sparse conversion currently expects a 2D array.")

        if format == "csr":
            return sparse.csr_matrix(x_arr)
        if format == "csc":
            return sparse.csc_matrix(x_arr)
        if format == "coo":
            return sparse.coo_matrix(x_arr)

        raise ValueError(f"Unknown sparse format: {format!r}")

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """
        Multiply sparse and dense arrays using SciPy.

        Input:
            a: Sparse backend array; b: Dense backend array.

        Output:
            Dense backend array containing the product.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

        Backend-specific notes:
            Uses SciPy sparse multiplication before returning a dense NumPy result when applicable.
        """
        if not self.is_sparse(a):
            raise TypeError("sparse_matmul expects `a` to be a SciPy sparse matrix/array.")
        if not self.is_dense(b):
            raise TypeError("sparse_matmul expects `b` to be a Numpy dense object.")
        return a @ b

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """
        Compute a stable log-sum-exp reduction using SciPy.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        See:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        return self.sp.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign)

    def index_set(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Set indexed values using NumPy.

        Input:
            x: Dense backend array; index: Selection; values: Replacement values; copy controls mutation policy.

        Output:
            Dense backend array with indexed values set.

        See:
            https://numpy.org/doc/stable/user/basics.indexing.html

        Backend-specific notes:
            With copy=True this copies before assignment; with copy=False it mutates the input array.
        """
        if copy:
            y = x.copy()
            y[index] = values
            return y
        else:
            x[index] = values
            return x

    def ix_(self, *args: Any) -> Any:
        """
        Build open mesh index arrays using NumPy.

        Input:
            args: One-dimensional index arrays or sequences.

        Output:
            Tuple of dense backend arrays usable for open-mesh indexing.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ix\\_.html
        """
        return self.np.ix_(*args)

    def fori_loop(
            self,
            lower: int,
            upper: int,
            body_fun: Callable[[int, T], T],
            init_val: T,
    ) -> T:
        """
        Run a counted loop primitive using NumPy.

        Input:
            lower, upper: Loop bounds; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html

        Backend-specific notes:
            NumPy executes this eagerly as a Python for-loop, without JAX tracing semantics.
        """
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
        """
        Run a while-loop primitive using NumPy.

        Input:
            cond_fun: Loop condition; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html

        Backend-specific notes:
            NumPy executes this eagerly as a Python while-loop, without JAX tracing semantics.
        """
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
        # assumes matching structure
        t0 = trees[0]
        if isinstance(t0, dict):
            return {k: self._tree_multimap(f, *(t[k] for t in trees)) for k in t0.keys()}
        if isinstance(t0, tuple):
            return tuple(self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0)))
        if isinstance(t0, list):
            return [self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0))]
        return f(*trees)

    def _tree_take0(self, xs: Any) -> Any:
        """
        Grab a representative leaf to infer leading length.
        """
        if isinstance(xs, dict):
            return self._tree_take0(next(iter(xs.values())))
        if isinstance(xs, (tuple, list)):
            return self._tree_take0(xs[0])
        return xs

    def _tree_index(self, xs: Any, i: int) -> Any:
        """
        Take per-step slice xs[i] along axis=0 for each leaf.
        """

        def _idx(a: Any) -> Any:
            # If it's an ndarray-like with leading axis, slice it; else treat as scalar leaf.
            try:
                return a[i]
            except Exception:
                return a

        return self._tree_map(_idx, xs)

    def _tree_stack(self, ys_list: Sequence[Any]) -> Any:
        """
        Stack a list of per-step outputs into a single pytree of arrays
        by stacking leaves along axis=0.
        """
        if not ys_list:
            # JAX would return empty stacked outputs when length == 0
            # Here we return an empty tuple (could also raise).
            return ()

        def _stack_leaves(*leaves: Any) -> Any:
            # Prefer self.np.stack when possible; fallback to array() if stacking fails.
            try:
                return self.np.stack(leaves, axis=0)
            except Exception:
                return self.np.array(leaves)

        # Reduce by structure: stack corresponding leaves across time
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
        """
        Run a scan primitive using NumPy.

        Input:
            f: Scan body; init: Initial carry; xs: Per-step inputs plus scan options.

        Output:
            Tuple of final carry and stacked outputs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

        Backend-specific notes:
            NumPy executes this eagerly and accepts unroll only for API parity.
        """
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

        # infer length from xs if not provided
        if length is None:
            leaf0 = self._tree_take0(xs)
            try:
                n = int(leaf0.shape[0])  # ndarray-like
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
        # Eager branch selection (NumPy has no tracing semantics)
        """
        Run conditional branch selection using NumPy.

        Input:
            pred: Predicate; true_fun and false_fun: Branch functions; operands: Branch inputs.

        Output:
            Result returned by the selected branch.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html

        Backend-specific notes:
            NumPy chooses the branch eagerly with Python truth-value conversion.
        """
        return true_fun(*operands) if bool(pred) else false_fun(*operands)

    def index_add(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
        """
        Add into indexed values using NumPy.

        Input:
            x: Dense backend array; index: Selection; values: Values to add; copy controls mutation policy.

        Output:
            Dense backend array with indexed values incremented.

        See:
            https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html

        Backend-specific notes:
            Uses numpy.add.at so repeated indices accumulate in NumPy order.
        """
        y = x.copy() if copy else x
        self.np.add.at(y, index, values)
        return y

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """
        Compare sparse arrays elementwise within tolerances using SciPy.

        Input:
            a, b: Sparse backend arrays; rtol and atol configure comparison.

        Output:
            Boolean indicating whether sparse arrays are close.

        See:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

        Backend-specific notes:
            Sparse inputs are converted to CSR and compared by logical sparse difference.
        """
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError("allclose_sparse expects two sparse arrays.")

        a = a.tocsr()
        b = b.tocsr()

        if a.shape != b.shape:
            return False

        diff = (a - b).tocsr()

        if diff.nnz == 0:
            return True

        a_abs = abs(a).tocsr()
        b_abs = abs(b).tocsr()
        scale = self.sp.sparse.csr_matrix.maximum(a_abs, b_abs)

        # tolerance_ij = atol + rtol * max(|a_ij|, |b_ij|)
        tol = scale.multiply(rtol)
        tol.data += atol

        bad = abs(diff) > tol
        return bad.nnz == 0

from __future__ import annotations

from typing import Any, Sequence, Literal, Tuple, Callable, Optional, Type
from warnings import warn

from .._family import BackendFamily
from .._ops import BackendOps
from ..numpy import NumpyOps
from ...types import DenseArray, ArrayLike, SparseArray, DType, Index, X, T, Y, R, Carry


class JaxOps(BackendOps):
    """
    BackendOps implementation for the JAX ecosystem.

    This backend uses JAX for dense array operations and JAX experimental
    sparse arrays for sparse operations.

    Dense arrays
        jax.Array

    Sparse arrays
        jax.experimental.sparse.BCOO
        jax.experimental.sparse.BCSR

    Methods
        Most methods mirror the corresponding JAX public API signatures and
        delegate to `jax.numpy`, `jax.numpy.linalg`, `jax.scipy`, or
        `jax.experimental.sparse`. Backend-specific behavior, tracing rules,
        dtype canonicalization, device placement, sharding, and error modes
        therefore follow JAX semantics.

    Backend handles
      - jax : module
            JAX module stored on the class and available through instances as
            `ops.jax`. Advanced users may use it when SpaceCore's portable API
            does not expose a required JAX feature.

      - jnp : module
            `jax.numpy` module stored on the class and available through
            instances as `ops.jnp`.

      - jsparse : module
            `jax.experimental.sparse` module stored on the class and available
            through instances as `ops.jsparse`.

    Notes
        Code intended to remain backend-portable should prefer `BackendOps`
        methods. Direct use of `ops.jax`, `ops.jnp`, or `ops.jsparse` is an
        explicit JAX-specific escape hatch.

        Some parameters are accepted for JAX signature compatibility even when
        JAX ignores them. Array-creation routines may expose `device` and
        `out_sharding` for explicit placement or sharding.
    """
    import jax
    import jax.numpy as jnp
    import jax.experimental.sparse as jsparse
    xp = jnp

    _family = BackendFamily.jax.value.lower()
    _allow_sparse = True

    def __init__(self) -> None:
        super().__init__()

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize a dtype specifier using JAX.

        Input:
            dtype: Optional dtype requested by SpaceCore or the caller.

        Output:
            Backend dtype object accepted by array constructors.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dtype.html

        Backend-specific notes:
            SpaceCore rejects dtypes that JAX would silently canonicalize under the active x64 setting.
        """
        x64_enabled = bool(self.jax.config.read("jax_enable_x64"))
        if dtype is None:
            if not x64_enabled:
                warn(
                    "jax_enable_x64 is set to False, so default JAX dtype is set to float32. "
                    "If you need float64, run `jax.config.update('jax_enable_x64', True)`.",
                    UserWarning
                )
                return self.jnp.float32
            return self.jnp.float64

        try:
            dt = self.jnp.dtype(dtype)
        except Exception as e:
            raise TypeError(f"Invalid dtype specifier for JAX: {dtype!r}.") from e

        # Ensure dtype is actually usable on this backend/device
        try:
            self.jnp.empty((), dtype=dt)
        except Exception as e:
            raise TypeError(
                f"Dtype {dt!r} is not supported by the active JAX backend/device."
            ) from e

        # Forbid implicit coercion under current JAX configuration
        dt_canon = self.jax.dtypes.canonicalize_dtype(dt)
        if dt_canon != dt:
            raise TypeError(
                f"Dtype {dt} is not permitted under current JAX configuration: "
                f"it would be canonicalized to {dt_canon}. "
                f"(jax_enable_x64={x64_enabled!r})"
            )

        return dt

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using JAX.

        Returns:
            Concrete dense array class accepted by this backend.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html
        """
        return self.jax.Array

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using JAX.

        Returns:
            Concrete sparse array classes accepted by this backend, or None.

        See:
            https://docs.jax.dev/en/latest/jax.experimental.sparse.html
        """
        return (self.jsparse.BCOO, self.jsparse.BCSR)

    def assparse(
            self,
            x: Any,
            *,
            format: Literal["bcoo", "bcsr"] = "bcoo",
            index_dtype: DType | None = None,
            nse: int | None = None,
            dtype: DType | None = None,
    ) -> SparseArray:
        """
        Convert input to a sparse array using JAX.

        Input:
            x: Dense, sparse, or array-like input plus sparse-format options.

        Output:
            Sparse backend array.

        See:
            https://docs.jax.dev/en/latest/jax.experimental.sparse.html

        Backend-specific notes:
            Dense inputs are converted with JAX sparse BCOO/BCSR constructors; SciPy sparse inputs use from_scipy_sparse.
        """
        import scipy.sparse as sps

        if self.is_sparse(x):
            return x

        if sps.issparse(x):
            if format == "bcoo":
                kwargs = {}
                if index_dtype is not None:
                    kwargs["index_dtype"] = index_dtype
                if nse is not None:
                    kwargs["nse"] = nse
                return self.jsparse.BCOO.from_scipy_sparse(x, **kwargs)

            if format == "bcsr":
                if self.jsparse.BCSR is None:
                    raise TypeError("BCSR is not available in this JAX version.")
                kwargs = {}
                if index_dtype is not None:
                    kwargs["index_dtype"] = index_dtype
                if nse is not None:
                    kwargs["nse"] = nse
                return self.jsparse.BCSR.from_scipy_sparse(x, **kwargs)

            raise ValueError(f"Unknown sparse format: {format!r}")

        x_arr = self.asarray(x)

        if format == "bcoo":
            kwargs = {}
            if index_dtype is not None:
                kwargs["index_dtype"] = index_dtype
            if nse is not None:
                kwargs["nse"] = nse
            return self.jsparse.BCOO.fromdense(x_arr, **kwargs)

        if format == "bcsr":
            if self.jsparse.BCSR is None:
                raise TypeError("BCSR is not available in this JAX version.")
            kwargs = {}
            if index_dtype is not None:
                kwargs["index_dtype"] = index_dtype
            if nse is not None:
                kwargs["nse"] = nse
            return self.jsparse.BCSR.fromdense(x_arr, **kwargs)

        raise ValueError(f"Unknown sparse format: {format!r}")

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """
        Multiply sparse and dense arrays using JAX.

        Input:
            a: Sparse backend array; b: Dense backend array.

        Output:
            Dense backend array containing the product.

        See:
            https://docs.jax.dev/en/latest/jax.experimental.sparse.html

        Backend-specific notes:
            Uses JAX sparse matmul and returns a JAX array; sparse support remains experimental in JAX.
        """
        return a @ b

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None, keepdims: bool = False,
                  return_sign: bool = False, where: DenseArray | None = None) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """
        Compute a stable log-sum-exp reduction using JAX.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.logsumexp.html
        """
        return self.jax.scipy.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign, where=where)

    def index_set(self, x: DenseArray, index: Index, values: ArrayLike, *, copy: bool = True):
        """
        Set indexed values using JAX.

        Input:
            x: Dense backend array; index: Selection; values: Replacement values; copy controls mutation policy.

        Output:
            Dense backend array with indexed values set.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html

        Backend-specific notes:
            JAX arrays are immutable; copy=False raises NotImplementedError.
        """
        if not copy:
            raise NotImplementedError(
                "JAX arrays are immutable; copy=False is not supported."
            )
        return x.at[index].set(values)

    def ix_(self, *args: Any) -> Any:
        """
        Build open mesh index arrays using JAX.

        Input:
            args: One-dimensional index arrays or sequences.

        Output:
            Tuple of dense backend arrays usable for open-mesh indexing.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ix\\_.html
        """
        return self.jnp.ix_(*args)

    def fori_loop(
        self,
        lower: int,
        upper: int,
        body_fun: Callable[[int, T], T],
        init_val: T,
        *,
        unroll: int | bool | None = None,
    ) -> T:
        """
        Run a counted loop primitive using JAX.

        Input:
            lower, upper: Loop bounds; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html

        Backend-specific notes:
            Loop bounds and unroll behavior follow JAX tracing and compilation rules.
        """
        return self.jax.lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)

    def while_loop(
        self,
        cond_fun: Callable[[T], bool],
        body_fun: Callable[[T], T],
        init_val: T,
    ) -> T:
        """
        Run a while-loop primitive using JAX.

        Input:
            cond_fun: Loop condition; body_fun: Loop body; init_val: Initial carry value.

        Output:
            Final carry value after loop execution.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html

        Backend-specific notes:
            Condition and body are staged according to JAX lax control-flow semantics.
        """
        return self.jax.lax.while_loop(cond_fun, body_fun, init_val)

    def scan(
        self,
        f: Callable[[Carry, X], Tuple[Carry, Y]],
        init: Carry,
        xs: X,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
        _split_transpose: bool = False,
    ) -> Tuple[Carry, Y]:
        """
        Run a scan primitive using JAX.

        Input:
            f: Scan body; init: Initial carry; xs: Per-step inputs plus scan options.

        Output:
            Tuple of final carry and stacked outputs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

        Backend-specific notes:
            Inputs and outputs may be pytrees and are staged according to JAX lax.scan semantics.
        """
        return self.jax.lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll, _split_transpose=_split_transpose)

    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        """
        Run conditional branch selection using JAX.

        Input:
            pred: Predicate; true_fun and false_fun: Branch functions; operands: Branch inputs.

        Output:
            Result returned by the selected branch.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html

        Backend-specific notes:
            Branches are staged according to JAX lax.cond semantics rather than Python eager branching.
        """
        return self.jax.lax.cond(pred, true_fun, false_fun, *operands)

    def index_add(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Add into indexed values using JAX.

        Input:
            x: Dense backend array; index: Selection; values: Values to add; copy controls mutation policy.

        Output:
            Dense backend array with indexed values incremented.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html

        Backend-specific notes:
            JAX arrays are immutable; copy=False raises NotImplementedError and repeated indices follow JAX scatter-add semantics.
        """
        if not copy:
            raise NotImplementedError(
                "JAX arrays are immutable; copy=False is not supported."
            )
        return x.at[index].add(values)

    def allclose_sparse(
            self,
            a: SparseArray,
            b: SparseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
    ) -> bool:
        """
        Compare sparse arrays elementwise within tolerances using JAX.

        Input:
            a, b: Sparse backend arrays; rtol and atol configure comparison.

        Output:
            Boolean indicating whether sparse arrays are close.

        See:
            https://docs.jax.dev/en/latest/jax.experimental.sparse.html

        Backend-specific notes:
            SpaceCore converts JAX sparse arrays through SciPy sparse arrays for comparison.
        """
        if not self.is_sparse(a) or not self.is_sparse(b):
            raise TypeError("allclose_sparse expects two sparse arrays.")

        np_ops = NumpyOps()
        a_sp = self._to_scipy_sparse(np_ops, a)
        b_sp = self._to_scipy_sparse(np_ops, b)

        return np_ops.allclose_sparse(a_sp, b_sp, rtol=rtol, atol=atol)

    def _to_scipy_sparse(self, np_ops: NumpyOps, x: SparseArray):
        if isinstance(x, self.jsparse.BCSR):
            x = x.to_bcoo()

        if isinstance(x, self.jsparse.BCOO):
            x = x.sum_duplicates(remove_zeros=False)

            if x.n_batch != 0 or x.n_dense != 0 or x.n_sparse != 2:
                raise NotImplementedError(
                    "_to_scipy_sparse supports only 2D unbatched sparse arrays."
                )

            row = x.indices[:, 0]
            col = x.indices[:, 1]
            data = x.data

            return np_ops.sp.coo_array((data, (row, col)), shape=x.shape)

        raise TypeError(f"Unsupported sparse type: {type(x)!r}")

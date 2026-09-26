from __future__ import annotations

from typing import Any, Sequence, Literal, Tuple, Callable, Optional, Type, cast
from warnings import warn

from .._family import BackendFamily
from .._ops import BackendOps
from ..numpy import NumpyOps
from ...types import DenseArray, ArrayLike, SparseArray, DType, Index, X, T, Y, R, Carry


class JaxOps(BackendOps):
    """
    Backend operations on JAX arrays (``jax.Array``; sparse ``BCOO`` and ``BCSR``).

    Methods delegate to ``jax.numpy`` and related modules, so dtypes, tracing, devices
    and sharding follow JAX. ``ops.jax``, ``ops.jnp`` and ``ops.jsparse`` expose the
    modules as an explicit JAX-only escape hatch.
    """

    import jax as _jax
    import jax.numpy as _jnp
    import jax.experimental.sparse as _jsparse

    # Concrete library handles exposed as ``Any`` so the portable protocols can
    # flow into typed JAX calls without per-boundary casts; mirrors the base
    # ``xp: ClassVar[Any]`` design.
    jax: Any = _jax
    jnp: Any = _jnp
    jsparse: Any = _jsparse

    xp = jnp

    _family = BackendFamily.jax.value.lower()
    _allow_sparse = True

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def install_pytree_protocol(cls) -> None:
        """Register SpaceCore containers as JAX pytree nodes.

        JAX's ``register_pytree_node_class`` consumes exactly the protocol
        :class:`~spacecore.backend._container.PyTreeNode` defines — ``tree_flatten``
        returning ``(children, aux)`` and a ``tree_unflatten(aux, children)``
        classmethod — so it *is* the registrar and no translation is needed.
        """
        import jax

        from .._container import registry

        registry.register_backend("jax", jax.tree_util.register_pytree_node_class)

    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """Normalize a dtype for JAX, rejecting dtypes JAX would silently canonicalize under the active x64 setting."""
        x64_enabled = bool(self.jax.config.read("jax_enable_x64"))
        if dtype is None:
            if not x64_enabled:
                warn(
                    "jax_enable_x64 is set to False, so default JAX dtype is set to float32. "
                    "If you need float64, run `jax.config.update('jax_enable_x64', True)`.",
                    UserWarning,
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

        Returns
        -------
            Concrete dense array class accepted by this backend.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html
        """
        return self.jax.Array

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using JAX.

        Returns
        -------
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
        """Convert to a JAX sparse array (BCOO/BCSR); SciPy inputs go through ``from_scipy_sparse``."""
        import scipy.sparse as sps
        self._reject_complex_to_real(x, dtype, operation="assparse")

        if self.is_sparse(x):
            if dtype is not None and self.get_dtype(x) != self.sanitize_dtype(dtype):
                return x.astype(self.sanitize_dtype(dtype))
            return x

        if sps.issparse(x):
            if format == "bcoo":
                kwargs = {}
                if index_dtype is not None:
                    kwargs["index_dtype"] = index_dtype
                if nse is not None:
                    kwargs["nse"] = nse
                out = self.jsparse.BCOO.from_scipy_sparse(x, **kwargs)
                return out.astype(self.sanitize_dtype(dtype)) if dtype is not None else out

            if format == "bcsr":
                if self.jsparse.BCSR is None:
                    raise TypeError("BCSR is not available in this JAX version.")
                kwargs = {}
                if index_dtype is not None:
                    kwargs["index_dtype"] = index_dtype
                if nse is not None:
                    kwargs["nse"] = nse
                out = self.jsparse.BCSR.from_scipy_sparse(x, **kwargs)
                return out.astype(self.sanitize_dtype(dtype)) if dtype is not None else out

            raise ValueError(f"Unknown sparse format: {format!r}")

        x_arr = self.asarray(x, dtype=dtype)

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
        """Multiply a JAX sparse array by a dense one, returning a dense ``jax.Array``."""
        return a @ b

    def vmap(
        self,
        fn: Callable,
        in_axes: int | Sequence[int | None] | None = 0,
        out_axes: int | Sequence[int | None] | None = 0,
    ) -> Callable:
        """Vectorize a function using ``jax.vmap``."""
        return self.jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)

    @property
    def has_native_vmap(self) -> bool:
        """Return ``True`` because JAX provides native ``vmap``."""
        return True

    def logsumexp(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        b: DenseArray | None = None,
        keepdims: bool = False,
        return_sign: bool = False,
        where: DenseArray | None = None,
    ) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """
        Compute a stable log-sum-exp reduction using JAX.

        Input:
            a: Dense backend array; axis, weights, and sign options control the reduction.

        Output:
            Dense backend array or tuple containing log-sum-exp results.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.logsumexp.html
        """
        return self.jax.scipy.special.logsumexp(
            a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign, where=where
        )

    def index_set(self, x: DenseArray, index: Index, values: ArrayLike, *, copy: bool = True):
        """Return ``x`` with ``x[index] = values`` via ``x.at[index].set``; ``copy=False`` raises, as JAX arrays are immutable."""
        if not copy:
            raise NotImplementedError("JAX arrays are immutable; copy=False is not supported.")
        return cast(Any, x).at[index].set(values)

    def ix_(self, *args: Any) -> Any:
        r"""
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
        """Run ``jax.lax.fori_loop``."""
        return self.jax.lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)

    def while_loop(
        self,
        cond_fun: Callable[[T], bool],
        body_fun: Callable[[T], T],
        init_val: T,
    ) -> T:
        """Run ``jax.lax.while_loop``."""
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
        """Run ``jax.lax.scan``; carries and outputs may be pytrees."""
        return self.jax.lax.scan(
            f,
            init,
            xs,
            length=length,
            reverse=reverse,
            unroll=unroll,
            _split_transpose=_split_transpose,
        )

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[T], R],
        false_fun: Callable[[T], R],
        *operands: Any,
    ) -> R:
        """Run ``jax.lax.cond``: both branches are traced, not chosen eagerly in Python."""
        return self.jax.lax.cond(pred, true_fun, false_fun, *operands)

    def index_add(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """Return ``x`` with ``values`` added at ``index`` via ``x.at[index].add``; repeated indices accumulate, and ``copy=False`` raises."""
        if not copy:
            raise NotImplementedError("JAX arrays are immutable; copy=False is not supported.")
        return cast(Any, x).at[index].add(values)

    def allclose_sparse(
        self,
        a: SparseArray,
        b: SparseArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Compare two JAX sparse arrays within tolerances, via SciPy sparse."""
        self._require_two_sparse(a, b)

        np_ops = NumpyOps()
        a_sp = self._to_scipy_sparse(np_ops, a)
        b_sp = self._to_scipy_sparse(np_ops, b)

        return np_ops.allclose_sparse(a_sp, b_sp, rtol=rtol, atol=atol)

    def _to_scipy_sparse(self, np_ops: NumpyOps, x: SparseArray):
        bcoo: Any = x
        if isinstance(x, self.jsparse.BCSR):
            bcoo = bcoo.to_bcoo()

        if isinstance(bcoo, self.jsparse.BCOO):
            bcoo = bcoo.sum_duplicates(remove_zeros=False)

            if bcoo.n_batch != 0 or bcoo.n_dense != 0 or bcoo.n_sparse != 2:
                raise NotImplementedError(
                    "_to_scipy_sparse supports only 2D unbatched sparse arrays."
                )

            row = bcoo.indices[:, 0]
            col = bcoo.indices[:, 1]
            data = bcoo.data

            return np_ops.sp.sparse.coo_array((data, (row, col)), shape=bcoo.shape)

        raise TypeError(f"Unsupported sparse type: {type(x)!r}")

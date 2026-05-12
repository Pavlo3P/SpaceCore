from __future__ import annotations

from typing import Any, Sequence, Literal, Tuple, Callable, Optional, Type
import inspect
from warnings import warn

from .._family import BackendFamily
from .._ops import BackendOps
from ..numpy import NumpyOps
from ...types import DenseArray, ArrayLike, SparseArray, DType, Index, X, T, Y, R, Carry


class JaxOps(BackendOps):
    """
    BackendOps implementation for the JAX ecosystem.

    Dense arrays:
      - jax.Array

    Sparse arrays:
      - jax.experimental.sparse.BCOO / BCSR

    Each method mirrors the corresponding JAX public API signature and delegates
    to `jax.numpy` / `jax.numpy.linalg` / `jax.experimental.sparse`.

    Notes:
      - Some parameters are documented as "unused by JAX" (e.g. `out` for argmin/argmax);
        these are still accepted to match the JAX signature and keep call sites uniform.
      - Array-creation routines often accept `device` and/or `out_sharding` for explicit
        placement/sharding.
    """

    import jax
    import jax.numpy as jnp
    import jax.experimental.sparse as jsparse

    _family = BackendFamily.jax.value.lower()
    _allow_sparse = True

    def __init__(self) -> None:
        self._reshape_supports_copy = "copy" in inspect.signature(self.jnp.reshape).parameters
        self._reshape_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.reshape).parameters
        self._ravel_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.ravel).parameters
        self._zeros_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.zeros).parameters
        self._empty_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.empty).parameters
        self._zeros_like_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.zeros_like).parameters
        self._ones_like_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.ones_like).parameters
        self._broadcast_to_supports_out_sharding = "out_sharding" in inspect.signature(self.jnp.broadcast_to).parameters


    def sanitize_dtype(self, dtype: DType | None) -> DType:
        """
        Normalize and validate dtype for JAX.

        Policy:
          - `dtype=None` is allowed and returned as-is.
          - Otherwise, dtype must be a valid JAX dtype specifier.
          - dtype must be supported by the active backend/device.
          - dtype must NOT be implicitly canonicalized under current JAX config
            (e.g., float64 -> float32 when jax_enable_x64=False).
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

    def get_dtype(self, x: Any) -> DType:
        if self.is_dense(x):
            return x.dtype
        elif self.is_sparse(x):
            return x.dtype
        else:
            raise TypeError(f'Expected Jax ndarray or BCOO/BCSR, got {type(x)}.')

    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return `x.shape` as a tuple.

        JAX shape metadata is available without transferring data. Under JIT or
        shape-polymorphic tracing, dimensions may need to remain static for Python use.
        """
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        """
        Return the number of dimensions of `x`.

        This is abstract-array metadata in JAX and does not materialize array values.
        """
        return int(x.ndim)

    def size(self, x: Any) -> int:
        """
        Return the logical dense element count of `x`.

        Sparse JAX arrays report logical shape here. Shape-polymorphic dimensions may
        not be usable as concrete Python integers inside traced code.
        """
        result = 1
        for dim in self.shape(x):
            result *= dim
        return result

    @property
    def dense_array(self) -> Type[Any]:
        return self.jax.Array

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        return (self.jsparse.BCOO, self.jsparse.BCSR)

    @property
    def inf(self):
        return self.jnp.array(self.jnp.inf)

    @property
    def nan(self):
        return self.jnp.array(self.jnp.nan)

    @property
    def pi(self):
        return self.jnp.array(self.jnp.pi)

    @property
    def e(self):
        return self.jnp.array(self.jnp.e)

    @property
    def eps(self):
        return self.jnp.array(self.jnp.finfo(self.jnp.float64).eps)

    def asarray(
        self,
        a: Any,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] | None = None,
        *,
        copy: bool | None = None,
        device: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.asarray`."""
        return self.jnp.asarray(a, dtype=dtype, order=order, copy=copy, device=device)

    def astype(self, x: DenseArray, dtype: DType, copy: bool = True) -> DenseArray:
        """
        Copy `x` cast to `dtype`. See: `jax.Array.astype`.

        JAX arrays are immutable; `copy` is accepted for API parity, but JAX may return
        the original value when no conversion is required. Dtype canonicalization follows
        the active JAX configuration and device support.
        """
        return x.astype(dtype, copy=copy)

    def assparse(
            self,
            x: Any,
            *,
            format: Literal["bcoo", "bcsr"] = "bcoo",
            index_dtype: DType | None = None,
            nse: int | None = None,
            dtype: DType | None = None,
    ) -> SparseArray:
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

    def empty(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.empty`."""
        if self._empty_supports_out_sharding:
            return self.jnp.empty(shape, dtype=dtype, device=device, out_sharding=out_sharding)
        return self.jnp.empty(shape, dtype=dtype, device=device)

    def zeros(
            self,
            shape: int | Tuple[int, ...],
            dtype: DType | None = None,
            *,
            device: Any | None = None,
            out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.zeros`."""
        if self._zeros_supports_out_sharding:
            return self.jnp.zeros(shape, dtype=dtype, device=device, out_sharding=out_sharding)
        return self.jnp.zeros(shape, dtype=dtype, device=device)

    def ones(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.ones`."""
        return self.jnp.ones(shape, dtype=dtype, device=device, out_sharding=out_sharding)

    def zeros_like(
        self,
        x: DenseArray,
        dtype: DType | None = None,
        shape: Any = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        Return zeros with shape and dtype like `x`. See: `jax.numpy.zeros_like`.

        JAX arrays are immutable and placement follows `device`/`out_sharding` when
        supported by the installed JAX version.
        """
        kwargs: dict[str, Any] = {"dtype": dtype, "shape": shape, "device": device}
        if self._zeros_like_supports_out_sharding:
            kwargs["out_sharding"] = out_sharding
        return self.jnp.zeros_like(x, **kwargs)

    def ones_like(
        self,
        x: DenseArray,
        dtype: DType | None = None,
        shape: Any = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        Return ones with shape and dtype like `x`. See: `jax.numpy.ones_like`.

        JAX dtype canonicalization and placement follow the active configuration and
        optional `device`/`out_sharding` support.
        """
        kwargs: dict[str, Any] = {"dtype": dtype, "shape": shape, "device": device}
        if self._ones_like_supports_out_sharding:
            kwargs["out_sharding"] = out_sharding
        return self.jnp.ones_like(x, **kwargs)

    def full_like(
        self,
        x: DenseArray,
        value: Any,
        dtype: DType | None = None,
        shape: Any = None,
        *,
        device: Any | None = None,
    ) -> DenseArray:
        """
        Return values filled with `value` and shaped like `x`. See: `jax.numpy.full_like`.

        Scalar promotion and dtype canonicalization follow JAX rules. JAX arrays are
        immutable; the result may be placed with `device` when supported.
        """
        return self.jnp.full_like(x, value, dtype=dtype, shape=shape, device=device)

    def arange(self,
               start: int,
               stop: int | None = None,
               step: int | None = None,
               dtype: DType | None = None,
               *,
               device: Any | None = None,
               out_sharding: Any | None = None,
               ) -> DenseArray:
        """See: `jax.numpy.arange`."""
        return self.jnp.arange(start, stop, step, dtype=dtype, device=device, out_sharding=out_sharding)

    def full(
        self,
        shape: int | Tuple[int, ...],
        fill_value: Any,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
    ) -> DenseArray:
        """
        See: `jax.numpy.full`.

        Note:
          - JAX exposes `device` for placement; `out_sharding` is not part of the public signature.
        """
        return self.jnp.full(shape, fill_value, dtype=dtype, device=device)

    def eye(
        self,
        N: int,
        M: int | None = None,
        k: int = 0,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.eye`."""
        return self.jnp.eye(N=N, M=M, k=k, dtype=dtype, device=device)

    def ravel(
        self,
        a: DenseArray,
        order: Literal["C", "F", "A", "K"] = "C",
        *,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        See: `jax.numpy.ravel`.
        """
        if self._ravel_supports_out_sharding:
            return self.jnp.ravel(a, order=order, out_sharding=out_sharding)
        return self.jnp.ravel(a, order=order)

    def reshape(
        self,
        a: DenseArray,
        shape: int | Tuple[int, ...],
        order: Literal["C", "F", "A"] = "C",
        *,
        copy: bool | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.reshape`."""
        kwargs: dict[str, Any] = {"order": order}
        if self._reshape_supports_copy:
            kwargs["copy"] = copy
        if self._reshape_supports_out_sharding:
            kwargs["out_sharding"] = out_sharding
        return self.jnp.reshape(a, shape, **kwargs)

    def transpose(
        self,
        x: DenseArray,
        axes: Sequence[int] | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.transpose`."""
        return self.jnp.transpose(x, axes=axes)

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """See: `jax.numpy.swapaxes`."""
        return self.jnp.swapaxes(x, axis1, axis2)

    def broadcast_to(
        self,
        x: DenseArray,
        shape: int | Tuple[int, ...],
        *,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        Broadcast `x` to `shape`. See: `jax.numpy.broadcast_to`.

        The result is immutable and shape must be trace-static under JIT. `out_sharding`
        is passed only on JAX versions that support it.
        """
        if self._broadcast_to_supports_out_sharding:
            return self.jnp.broadcast_to(x, shape, out_sharding=out_sharding)
        return self.jnp.broadcast_to(x, shape)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """
        Insert new axes into `x`. See: `jax.numpy.expand_dims`.

        Axis values must be static when used under JIT.
        """
        return self.jnp.expand_dims(x, axis=axis)

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Remove length-one axes from `x`. See: `jax.numpy.squeeze`.

        Axis values must be static when used under JIT. The result is immutable.
        """
        return self.jnp.squeeze(x, axis=axis)

    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """
        Move axes to new positions. See: `jax.numpy.moveaxis`.

        Axis values must be static when used under JIT. The result is immutable.
        """
        return self.jnp.moveaxis(x, source=source, destination=destination)

    def stack(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        out: Any | None = None,
        dtype: DType | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.stack`."""
        return self.jnp.stack(arrays, axis=axis, out=out, dtype=dtype)

    def conj(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.conj`."""
        return self.jnp.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.real`."""
        return self.jnp.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.imag`."""
        return self.jnp.imag(x)

    def abs(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.abs`."""
        return self.jnp.abs(x)

    def sign(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.sign`."""
        return self.jnp.sign(x)

    def sqrt(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.sqrt`."""
        return self.jnp.sqrt(x)

    def sum(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: Any | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | None = None,
        promote_integers: bool = True,
    ) -> DenseArray:
        """See: `jax.numpy.sum`."""
        return self.jnp.sum(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def mean(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: None = None,
        keepdims: bool = False,
        *,
        where: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute the arithmetic mean over an axis. See: `jax.numpy.mean`.

        Axis values are static under JIT. Dtype promotion and `where` semantics follow
        JAX and may differ from NumPy when x64 is disabled.
        """
        return self.jnp.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    def min(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        out: None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute minimum values over an axis. See: `jax.numpy.min`.

        Axis values are static under JIT. Empty reductions, `initial`, `where`, and NaN
        behavior follow JAX semantics.
        """
        return self.jnp.min(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def max(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        out: None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | None = None,
    ) -> DenseArray:
        """
        Compute maximum values over an axis. See: `jax.numpy.max`.

        Axis values are static under JIT. Empty reductions, `initial`, `where`, and NaN
        behavior follow JAX semantics.
        """
        return self.jnp.max(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def prod(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: Any | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | None = None,
        promote_integers: bool = True,
    ) -> DenseArray:
        """See: `jax.numpy.prod`."""
        return self.jnp.prod(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def trace(
        self,
        a: DenseArray,
        offset: int | Any = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DType | None = None,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.trace`."""
        return self.jnp.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)

    def argsort(
        self,
        a: DenseArray,
        axis: int | None = -1,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False
    ) -> DenseArray:
        """
        See: `jax.numpy.argsort`.

        Notes:
          - `kind` is deprecated in JAX; kept for signature compatibility.
          - `order` is not supported by JAX; kept for signature compatibility.
        """
        return self.jnp.argsort(a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)

    def sort(
        self,
        a: DenseArray,
        axis: int | None = -1,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False
    ) -> DenseArray:
        """
        See: `jax.numpy.sort`.

        Notes:
          - `kind` is deprecated in JAX; kept for signature compatibility.
          - `order` is not supported by JAX; kept for signature compatibility.
        """
        return self.jnp.sort(a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)

    def argmin(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: Any | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """See: `jax.numpy.argmin` (note: `out` is unused by JAX)."""
        return self.jnp.argmin(a, axis=axis, out=out, keepdims=keepdims)

    def argmax(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: Any | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """See: `jax.numpy.argmax` (note: `out` is unused by JAX)."""
        return self.jnp.argmax(a, axis=axis, out=out, keepdims=keepdims)

    def vdot(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.vdot`."""
        return self.jnp.vdot(a, b, precision=precision, preferred_element_type=preferred_element_type)

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
        out_sharding: Any | None = None
    ) -> DenseArray:
        """See: `jax.numpy.matmul`."""
        return self.jnp.matmul(
            a,
            b,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """Sparse @ dense. See: `jax.experimental.sparse` (BCOO/BCSR matmul via `@`)."""
        return a @ b

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """See: `jax.numpy.kron`."""
        return self.jnp.kron(a, b)

    def einsum(
        self,
        subscripts: str,
        /,
        *operands: DenseArray,
        out: Any | None = None,
        optimize: str | bool | list[Tuple[int, ...]] = "auto",
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
        _dot_general: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.einsum`."""
        return self.jnp.einsum(
            subscripts,
            *operands,
            out=out,
            optimize=optimize,
            precision=precision,
            preferred_element_type=preferred_element_type,
            # _dot_general=_dot_general,
            out_sharding=out_sharding,
        )

    def eigh(
        self,
        x: DenseArray,
        UPLO: Literal["L", "U"] = "L",
        symmetrize_input: bool = True
    ) -> Tuple[DenseArray, DenseArray]:
        """See: `jax.numpy.linalg.eigh`."""
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        return self.jnp.linalg.eigh(x, UPLO=UPLO, symmetrize_input=symmetrize_input)

    def norm(
        self,
        x: DenseArray,
        ord: int | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Compute a vector or matrix norm. See: `jax.numpy.linalg.norm`.

        `ord` and `axis` must be static under JIT. Precision and supported norms follow
        JAX's linear algebra implementation.
        """
        return self.jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        """
        Solve a dense linear system. See: `jax.numpy.linalg.solve`.

        JAX arrays are immutable and execution may be asynchronous/device-backed.
        Singular-input behavior and precision follow JAX.
        """
        return self.jnp.linalg.solve(A, b)

    def eigvalsh(
        self,
        A: DenseArray,
        UPLO: Literal["L", "U"] = "L",
        *,
        symmetrize_input: bool = True,
    ) -> DenseArray:
        """
        Return Hermitian/symmetric eigenvalues. See: `jax.numpy.linalg.eigvalsh`.

        JAX can symmetrize input for stable autodiff; set `symmetrize_input=False` only
        when relying on JAX-specific behavior.
        """
        return self.jnp.linalg.eigvalsh(A, UPLO=UPLO, symmetrize_input=symmetrize_input)

    def svd(
        self,
        A: DenseArray,
        full_matrices: bool = True,
        compute_uv: bool = True,
        hermitian: bool = False,
        subset_by_index: tuple[int, int] | None = None,
    ) -> DenseArray | Tuple[DenseArray, DenseArray, DenseArray]:
        """
        Compute singular value decomposition. See: `jax.numpy.linalg.svd`.

        The portable path uses `compute_uv=True`. Sign choices, precision, static-shape
        requirements, and device execution follow JAX.
        """
        return self.jnp.linalg.svd(
            A,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
            subset_by_index=subset_by_index,
        )

    def cholesky(
        self,
        A: DenseArray,
        *,
        upper: bool = False,
        symmetrize_input: bool = True,
    ) -> DenseArray:
        """
        Compute a Cholesky factor. See: `jax.numpy.linalg.cholesky`.

        The portable path returns the lower factor (`upper=False`). JAX may symmetrize
        input for autodiff stability and returns immutable device-backed arrays.
        """
        return self.jnp.linalg.cholesky(A, upper=upper, symmetrize_input=symmetrize_input)

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray | None = None, keepdims: bool = False,
                  return_sign: bool = False, where: DenseArray | None = None) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """ See: jax.scipy.special.logsumexp. """
        return self.jax.scipy.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign, where=where)

    def exp(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.exp`."""
        return self.jnp.exp(x)

    def log(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.log`."""
        return self.jnp.log(x)

    def maximum(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """See: `jax.numpy.maximum`."""
        return self.jnp.maximum(x, y)

    def minimum(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """See: `jax.numpy.minimum`."""
        return self.jnp.minimum(x, y)

    def clip(self, x: DenseArray, a_min: DenseArray, a_max: DenseArray) -> DenseArray:
        """
        Clip values to an interval. See: `jax.numpy.clip`.

        Broadcasting and dtype promotion follow JAX. Deprecated NumPy-compatible names
        are intentionally not exposed in the portable signature.
        """
        return self.jnp.clip(x, a_min, a_max)

    def isfinite(self, x: DenseArray) -> DenseArray:
        """
        Test elementwise finiteness. See: `jax.numpy.isfinite`.

        The result is an immutable boolean JAX array and stays on the active device.
        """
        return self.jnp.isfinite(x)

    def isnan(self, x: DenseArray) -> DenseArray:
        """
        Test elementwise NaN values. See: `jax.numpy.isnan`.

        The result is an immutable boolean JAX array and stays on the active device.
        """
        return self.jnp.isnan(x)

    def where(self, condition: DenseArray | bool, x: DenseArray | None = None, y: DenseArray | None = None, *,
              size: int | None = None, fill_value: DenseArray | None = None) -> DenseArray:
        """See: `jax.numpy.where`."""
        return self.jnp.where(condition, x, y, size=size, fill_value=fill_value)

    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType | None = None) -> DenseArray:
        """See: `jax.numpy.concatenate`."""
        return self.jnp.concatenate(arrays, axis=axis, dtype=dtype)

    def take(
        self,
        x: DenseArray,
        indices: DenseArray,
        axis: int | None = None,
        out: None = None,
        mode: str | None = None,
        unique_indices: bool = False,
        indices_are_sorted: bool = False,
        fill_value: Any | None = None,
    ) -> DenseArray:
        """
        Take elements by integer index. See: `jax.numpy.take`.

        Portable code should pass valid indices because JAX out-of-bounds defaults can
        differ from NumPy. `axis` and mode flags must be static under JIT.
        """
        return self.jnp.take(
            x,
            indices,
            axis=axis,
            out=out,
            mode=mode,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            fill_value=fill_value,
        )

    def diag(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Extract or construct a diagonal. See: `jax.numpy.diag`.

        The result is immutable; `k` must be static under JIT.
        """
        return self.jnp.diag(x, k=k)

    def diagonal(
        self,
        x: DenseArray,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> DenseArray:
        """
        Return selected diagonals. See: `jax.numpy.diagonal`.

        Axis and offset values must be static under JIT. The result is immutable.
        """
        return self.jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    def tril(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return the lower triangle of `x`. See: `jax.numpy.tril`.

        `k` must be static under JIT and the result is immutable.
        """
        return self.jnp.tril(x, k=k)

    def triu(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return the upper triangle of `x`. See: `jax.numpy.triu`.

        `k` must be static under JIT and the result is immutable.
        """
        return self.jnp.triu(x, k=k)

    def index_set(self, x: DenseArray, index: Index, values: ArrayLike, *, copy: bool = True):
        if not copy:
            raise NotImplementedError(
                "JAX arrays are immutable; copy=False is not supported."
            )
        return x.at[index].set(values)

    def ix_(self, *args: Any) -> Any:
        """ See: jax.numpy.ix_. """
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
        """See: `jax.lax.fori_loop`."""
        return self.jax.lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)

    def while_loop(
        self,
        cond_fun: Callable[[T], bool],
        body_fun: Callable[[T], T],
        init_val: T,
    ) -> T:
        """See: `jax.lax.while_loop`."""
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
        """See: `jax.lax.scan`."""
        return self.jax.lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll, _split_transpose=_split_transpose)

    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        return self.jax.lax.cond(pred, true_fun, false_fun, *operands)

    def index_add(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Scatter-add (accumulating) update.

        Semantics:
          y = x
          y[index] += values   (with accumulation for repeated indices)

        Notes:
          - JAX arrays are immutable; copy=False is not supported (same as index_set).
          - Uses JAX's scatter-add: x.at[index].add(values)
        """
        if not copy:
            raise NotImplementedError(
                "JAX arrays are immutable; copy=False is not supported."
            )
        return x.at[index].add(values)

    def allclose(
            self,
            a: DenseArray,
            b: DenseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            equal_nan: bool = False,
    ) -> bool:
        return self.jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose_sparse(
            self,
            a: SparseArray,
            b: SparseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
    ) -> bool:
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

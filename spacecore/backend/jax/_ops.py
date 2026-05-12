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

    def get_dtype(self, x: Any) -> DType:
        """
        Return an array dtype using JAX.

        Input:
            x: Dense or sparse backend array.

        Output:
            Backend dtype associated with x.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html
        """
        if self.is_dense(x):
            return x.dtype
        elif self.is_sparse(x):
            return x.dtype
        else:
            raise TypeError(f'Expected Jax ndarray or BCOO/BCSR, got {type(x)}.')

    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return array shape metadata using JAX.

        Input:
            x: Dense or sparse backend array.

        Output:
            Tuple describing the logical shape of x.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html
        """
        return tuple(x.shape)

    def ndim(self, x: Any) -> int:
        """
        Return array rank metadata using JAX.

        Input:
            x: Dense or sparse backend array.

        Output:
            Number of dimensions in x.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html
        """
        return int(x.ndim)

    def size(self, x: Any) -> int:
        """
        Return logical element count using JAX.

        Input:
            x: Dense or sparse backend array.

        Output:
            Total number of logical dense elements.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html

        Backend-specific notes:
            Shape-polymorphic dimensions may not be concrete Python integers inside traced code.
        """
        result = 1
        for dim in self.shape(x):
            result *= dim
        return result

    @property
    def dense_array(self) -> Type[Any]:
        """
        Dense array type using JAX.

        Input:
            None.

        Output:
            Concrete dense array class accepted by this backend.

        See:
            https://docs.jax.dev/en/latest/jax.Array.html
        """
        return self.jax.Array

    @property
    def sparse_array(self) -> Tuple[Type[Any], ...]:
        """
        Sparse array type tuple using JAX.

        Input:
            None.

        Output:
            Concrete sparse array classes accepted by this backend, or None.

        See:
            https://docs.jax.dev/en/latest/jax.experimental.sparse.html
        """
        return (self.jsparse.BCOO, self.jsparse.BCSR)

    @property
    def inf(self):
        """
        Positive infinity scalar using JAX.

        Input:
            None.

        Output:
            Backend scalar representing positive infinity.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.inf.html
        """
        return self.jnp.array(self.jnp.inf)

    @property
    def nan(self):
        """
        NaN scalar using JAX.

        Input:
            None.

        Output:
            Backend scalar representing NaN.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nan.html
        """
        return self.jnp.array(self.jnp.nan)

    @property
    def pi(self):
        """
        Pi scalar using JAX.

        Input:
            None.

        Output:
            Backend scalar representing pi.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pi.html
        """
        return self.jnp.array(self.jnp.pi)

    @property
    def e(self):
        """
        Euler number scalar using JAX.

        Input:
            None.

        Output:
            Backend scalar representing Euler's number.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.e.html
        """
        return self.jnp.array(self.jnp.e)

    @property
    def eps(self):
        """
        Machine epsilon scalar using JAX.

        Input:
            None.

        Output:
            Backend scalar for float64 machine epsilon.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.finfo.html
        """
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
        """
        Convert input to a dense array using JAX.

        Input:
            x/a: Array-like input and optional dtype or backend conversion parameters.

        Output:
            Dense backend array.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asarray.html
        """
        return self.jnp.asarray(a, dtype=dtype, order=order, copy=copy, device=device)

    def astype(self, x: DenseArray, dtype: DType, copy: bool = True) -> DenseArray:
        """
        Cast an array to a dtype using JAX.

        Input:
            x: Dense backend array; dtype: target dtype and optional casting controls.

        Output:
            Dense backend array with the requested dtype.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.Array.astype.html
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

    def empty(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        Create an uninitialized dense array using JAX.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array with uninitialized values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html

        Backend-specific notes:
            out_sharding is forwarded only when supported by the installed JAX version.
        """
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
        """
        Create a zero-filled dense array using JAX.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array filled with zeros.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.zeros.html

        Backend-specific notes:
            out_sharding is forwarded only when supported by the installed JAX version.
        """
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
        """
        Create a one-filled dense array using JAX.

        Input:
            shape: Output shape; dtype and placement options are backend-specific.

        Output:
            Dense backend array filled with ones.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ones.html
        """
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
        Create zeros shaped like another array using JAX.

        Input:
            x: Prototype dense array; dtype, shape, and placement options are backend-specific.

        Output:
            Dense backend array of zeros.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.zeros_like.html

        Backend-specific notes:
            out_sharding is forwarded only when supported by the installed JAX version.
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
        Create ones shaped like another array using JAX.

        Input:
            x: Prototype dense array; dtype, shape, and placement options are backend-specific.

        Output:
            Dense backend array of ones.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ones_like.html

        Backend-specific notes:
            out_sharding is forwarded only when supported by the installed JAX version.
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
        Create filled values shaped like another array using JAX.

        Input:
            x: Prototype dense array; value/fill_value and dtype options are backend-specific.

        Output:
            Dense backend array filled with the requested value.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.full_like.html
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
        """
        Create evenly spaced integer-range values using JAX.

        Input:
            start, stop, step: Range parameters; dtype and placement options are backend-specific.

        Output:
            One-dimensional dense backend array.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arange.html
        """
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
        Create a filled dense array using JAX.

        Input:
            shape: Output shape; fill_value and dtype options are backend-specific.

        Output:
            Dense backend array filled with fill_value.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.full.html
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
        """
        Create a dense identity-like matrix using JAX.

        Input:
            n and optional m: Matrix dimensions; dtype and placement options are backend-specific.

        Output:
            Two-dimensional dense backend array.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.eye.html
        """
        return self.jnp.eye(N=N, M=M, k=k, dtype=dtype, device=device)

    def ravel(
        self,
        a: DenseArray,
        order: Literal["C", "F", "A", "K"] = "C",
        *,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        Flatten an array using JAX.

        Input:
            x: Dense backend array plus optional order parameters.

        Output:
            One-dimensional dense backend array view or copy.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ravel.html
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
        """
        Reshape an array using JAX.

        Input:
            x: Dense backend array; shape: New shape plus backend-specific options.

        Output:
            Dense backend array with the requested shape.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html
        """
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
        """
        Permute array axes using JAX.

        Input:
            x: Dense backend array; axes: Optional axis order.

        Output:
            Dense backend array with permuted axes.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html
        """
        return self.jnp.transpose(x, axes=axes)

    def swapaxes(self, x: DenseArray, axis1: int, axis2: int) -> DenseArray:
        """
        Interchange two axes using JAX.

        Input:
            x: Dense backend array; axis1 and axis2: Axes to swap.

        Output:
            Dense backend array with the two axes exchanged.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.swapaxes.html
        """
        return self.jnp.swapaxes(x, axis1, axis2)

    def broadcast_to(
        self,
        x: DenseArray,
        shape: int | Tuple[int, ...],
        *,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        Broadcast an array to a shape using JAX.

        Input:
            x: Dense backend array; shape: Target broadcast shape.

        Output:
            Dense backend array with broadcast shape.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html
        """
        if self._broadcast_to_supports_out_sharding:
            return self.jnp.broadcast_to(x, shape, out_sharding=out_sharding)
        return self.jnp.broadcast_to(x, shape)

    def expand_dims(self, x: DenseArray, axis: int | Sequence[int]) -> DenseArray:
        """
        Insert length-one axes using JAX.

        Input:
            x: Dense backend array; axis: Position or positions to insert.

        Output:
            Dense backend array with expanded rank.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.expand_dims.html
        """
        return self.jnp.expand_dims(x, axis=axis)

    def squeeze(self, x: DenseArray, axis: int | Sequence[int] | None = None) -> DenseArray:
        """
        Remove length-one axes using JAX.

        Input:
            x: Dense backend array; axis: Optional axes to squeeze.

        Output:
            Dense backend array with selected singleton dimensions removed.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html
        """
        return self.jnp.squeeze(x, axis=axis)

    def moveaxis(
        self,
        x: DenseArray,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> DenseArray:
        """
        Move axes to new positions using JAX.

        Input:
            x: Dense backend array; source and destination: Axis positions.

        Output:
            Dense backend array with moved axes.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.moveaxis.html
        """
        return self.jnp.moveaxis(x, source=source, destination=destination)

    def stack(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        out: Any | None = None,
        dtype: DType | None = None,
    ) -> DenseArray:
        """
        Stack arrays along a new axis using JAX.

        Input:
            arrays: Sequence of dense backend arrays; axis: New axis position.

        Output:
            Dense backend array containing stacked inputs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.stack.html
        """
        return self.jnp.stack(arrays, axis=axis, out=out, dtype=dtype)

    def conj(self, x: DenseArray) -> DenseArray:
        """
        Compute complex conjugates using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array with conjugated values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.conj.html
        """
        return self.jnp.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        """
        Extract real components using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array containing real components.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.real.html
        """
        return self.jnp.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """
        Extract imaginary components using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array containing imaginary components.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.imag.html
        """
        return self.jnp.imag(x)

    def abs(self, x: DenseArray) -> DenseArray:
        """
        Compute absolute values using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of absolute values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.abs.html
        """
        return self.jnp.abs(x)

    def sign(self, x: DenseArray) -> DenseArray:
        """
        Compute signs elementwise using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of signs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html
        """
        return self.jnp.sign(x)

    def sqrt(self, x: DenseArray) -> DenseArray:
        """
        Compute square roots elementwise using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of square roots.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sqrt.html
        """
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
        """
        Reduce by summation using JAX.

        Input:
            x: Dense backend array; axis, keepdims, and dtype control the reduction.

        Output:
            Dense backend array or scalar containing sums.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html
        """
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
        Reduce by arithmetic mean using JAX.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing means.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mean.html
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
        Reduce by minimum using JAX.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing minima.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html
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
        Reduce by maximum using JAX.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense backend array or scalar containing maxima.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html
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
        """
        Reduce by product using JAX.

        Input:
            x: Dense backend array; axis, keepdims, and dtype control the reduction.

        Output:
            Dense backend array or scalar containing products.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html
        """
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
        """
        Sum diagonal entries using JAX.

        Input:
            x: Dense backend array plus optional diagonal and axis controls.

        Output:
            Dense backend array or scalar containing trace values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trace.html
        """
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
        Return sorting indices using JAX.

        Input:
            x: Dense backend array; axis and ordering options are backend-specific.

        Output:
            Dense integer backend array of indices.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html
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
        Sort values using JAX.

        Input:
            x: Dense backend array; axis and ordering options are backend-specific.

        Output:
            Dense backend array with sorted values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html
        """
        return self.jnp.sort(a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)

    def argmin(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: Any | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Return indices of minimum values using JAX.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense integer backend array or scalar of indices.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html
        """
        return self.jnp.argmin(a, axis=axis, out=out, keepdims=keepdims)

    def argmax(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: Any | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """
        Return indices of maximum values using JAX.

        Input:
            x: Dense backend array; axis and keepdims control the reduction.

        Output:
            Dense integer backend array or scalar of indices.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html
        """
        return self.jnp.argmax(a, axis=axis, out=out, keepdims=keepdims)

    def vdot(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
    ) -> DenseArray:
        """
        Compute a conjugating vector dot product using JAX.

        Input:
            x, y: Dense backend arrays accepted by the backend vdot operation.

        Output:
            Backend scalar or dense array containing the dot product.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vdot.html
        """
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
        """
        Compute matrix products using JAX.

        Input:
            a, b: Dense backend arrays with matrix-multiplication-compatible shapes.

        Output:
            Dense backend array containing the product.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html
        """
        return self.jnp.matmul(
            a,
            b,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )

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

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """
        Compute a Kronecker product using JAX.

        Input:
            a, b: Dense backend arrays.

        Output:
            Dense backend array containing the Kronecker product.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.kron.html
        """
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
        """
        Evaluate an Einstein summation expression using JAX.

        Input:
            subscripts: Einstein summation string; operands: Dense backend arrays.

        Output:
            Dense backend array containing the contraction result.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html
        """
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
        """
        Compute Hermitian eigenpairs using JAX.

        Input:
            x: Dense Hermitian or symmetric backend array.

        Output:
            Tuple of dense backend arrays containing eigenvalues and eigenvectors.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigh.html

        Backend-specific notes:
            SpaceCore rejects sparse input before delegating to JAX dense linear algebra.
        """
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
        Compute vector or matrix norms using JAX.

        Input:
            x: Dense backend array; ord, axis, and keepdims select the norm.

        Output:
            Dense backend array or scalar containing norm values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.norm.html
        """
        return self.jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(self, A: DenseArray, b: DenseArray) -> DenseArray:
        """
        Solve dense linear systems using JAX.

        Input:
            A: Dense coefficient array; b: Dense right-hand side array.

        Output:
            Dense backend array solving A @ x = b.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.solve.html
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
        Compute Hermitian eigenvalues using JAX.

        Input:
            A: Dense Hermitian or symmetric backend array.

        Output:
            Dense backend array containing eigenvalues.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigvalsh.html
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
        Compute singular value decompositions using JAX.

        Input:
            A: Dense backend array plus SVD options.

        Output:
            Dense backend arrays containing singular vectors and/or singular values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.svd.html
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
        Compute Cholesky factors using JAX.

        Input:
            A: Dense Hermitian positive-definite backend array.

        Output:
            Dense backend array containing a triangular factor.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.cholesky.html
        """
        return self.jnp.linalg.cholesky(A, upper=upper, symmetrize_input=symmetrize_input)

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

    def exp(self, x: DenseArray) -> DenseArray:
        """
        Compute exponentials elementwise using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of exponentials.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp.html
        """
        return self.jnp.exp(x)

    def log(self, x: DenseArray) -> DenseArray:
        """
        Compute natural logarithms elementwise using JAX.

        Input:
            x: Dense backend array.

        Output:
            Dense backend array of logarithms.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log.html
        """
        return self.jnp.log(x)

    def maximum(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """
        Compute elementwise maxima using JAX.

        Input:
            x, y: Arrays or scalars accepted by backend broadcasting.

        Output:
            Dense backend array containing maxima.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.maximum.html
        """
        return self.jnp.maximum(x, y)

    def minimum(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """
        Compute elementwise minima using JAX.

        Input:
            x, y: Arrays or scalars accepted by backend broadcasting.

        Output:
            Dense backend array containing minima.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.minimum.html
        """
        return self.jnp.minimum(x, y)

    def clip(self, x: DenseArray, a_min: DenseArray, a_max: DenseArray) -> DenseArray:
        """
        Clip values into an interval using JAX.

        Input:
            x: Dense backend array; a_min and a_max: Broadcastable bounds.

        Output:
            Dense backend array with clipped values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.clip.html
        """
        return self.jnp.clip(x, a_min, a_max)

    def isfinite(self, x: DenseArray) -> DenseArray:
        """
        Test finiteness elementwise using JAX.

        Input:
            x: Dense backend array.

        Output:
            Boolean dense backend array.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isfinite.html
        """
        return self.jnp.isfinite(x)

    def isnan(self, x: DenseArray) -> DenseArray:
        """
        Test NaN values elementwise using JAX.

        Input:
            x: Dense backend array.

        Output:
            Boolean dense backend array.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isnan.html
        """
        return self.jnp.isnan(x)

    def where(self, condition: DenseArray | bool, x: DenseArray | None = None, y: DenseArray | None = None, *,
              size: int | None = None, fill_value: DenseArray | None = None) -> DenseArray:
        """
        Select values by condition using JAX.

        Input:
            condition: Boolean array or scalar; x and y: Values to choose between.

        Output:
            Dense backend array containing selected values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html
        """
        return self.jnp.where(condition, x, y, size=size, fill_value=fill_value)

    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType | None = None) -> DenseArray:
        """
        Join arrays along an existing axis using JAX.

        Input:
            arrays: Sequence of dense backend arrays; axis and dtype options are backend-specific.

        Output:
            Dense backend array containing concatenated inputs.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html
        """
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
        Take values by integer indices using JAX.

        Input:
            x: Dense backend array; indices: Integer indices; axis and mode options are backend-specific.

        Output:
            Dense backend array containing selected values.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html

        Backend-specific notes:
            Out-of-bounds and mode behavior follow JAX, which can differ from NumPy.
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
        Extract or build a diagonal using JAX.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array containing a diagonal view/copy or matrix.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diag.html
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
        Return selected diagonals using JAX.

        Input:
            x: Dense backend array plus offset and axis controls.

        Output:
            Dense backend array containing selected diagonals.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diagonal.html
        """
        return self.jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    def tril(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return lower-triangular values using JAX.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array with upper entries zeroed.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tril.html
        """
        return self.jnp.tril(x, k=k)

    def triu(self, x: DenseArray, k: int = 0) -> DenseArray:
        """
        Return upper-triangular values using JAX.

        Input:
            x: Dense backend array and optional diagonal offset.

        Output:
            Dense backend array with lower entries zeroed.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.triu.html
        """
        return self.jnp.triu(x, k=k)

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
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ix_.html
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

    def allclose(
            self,
            a: DenseArray,
            b: DenseArray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            equal_nan: bool = False,
    ) -> bool:
        """
        Compare dense arrays elementwise within tolerances using JAX.

        Input:
            a, b: Dense backend arrays; rtol, atol, and equal_nan configure comparison.

        Output:
            Boolean indicating whether arrays are close.

        See:
            https://docs.jax.dev/en/latest/_autosummary/jax.numpy.allclose.html
        """
        return self.jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

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

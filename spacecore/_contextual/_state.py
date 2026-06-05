from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Iterable, Tuple
from warnings import warn

from ..types import DType
from ..backend._family import BackendFamily
from ..backend._ops import BackendOps
from ..backend.numpy import NumpyOps
from ._policies import (
    ContextConflictError,
    ContextInferenceError,
    UnknownBackendError,
)
try:
    from ..backend.jax import JaxOps
except ImportError:
    pass
try:
    from ..backend.cupy import CuPyOps
except ImportError:
    pass
try:
    from ..backend.torch import TorchOps
except ImportError:
    pass

if TYPE_CHECKING:
    from ..backend._context import Context


def _context_type() -> type[Context]:
    from ..backend._context import Context

    return Context


class Contextual:
    """Resolve contexts and backend registrations."""

    _default_ctx: Context
    _available_ops: Dict[str, type[BackendOps]]
    _default_dtype: DType | None = None
    _default_enable_checks: bool = False

    def __init__(self) -> None:
        Context = _context_type()
        ops = NumpyOps()
        self.default_ctx = Context(
            ops=ops,
            dtype=ops.sanitize_dtype(self._default_dtype),
            enable_checks=self._default_enable_checks
        )

        self._available_ops = {
            self._backend_key(NumpyOps): NumpyOps,
        }
        if "JaxOps" in globals():
            self._available_ops[self._backend_key(JaxOps)] = JaxOps
        if "CuPyOps" in globals():
            self._available_ops[self._backend_key(CuPyOps)] = CuPyOps
        if "TorchOps" in globals():
            self._available_ops[self._backend_key(TorchOps)] = TorchOps

    def normalize_context(self,
                          ctx: Context | BackendFamily | str | None = None,
                          dtype: Any = None,
                          enable_checks: bool | None = None
                          ) -> Context:
        Context = _context_type()
        if ctx is None:
            if dtype is not None or enable_checks is not None:
                warn(
                    'Provided context is None, dtype and enable_checks parameters are ignored.',
                    UserWarning,
                )
            return self.default_ctx
        if isinstance(ctx, Context):
            if dtype is not None or enable_checks is not None:
                warn(
                    'Provided concrete context, dtype and enable_checks parameters are ignored.',
                    UserWarning,
                )
            return Context(
                ops=ctx.ops,
                dtype=ctx.ops.sanitize_dtype(ctx.dtype),
                enable_checks=ctx.enable_checks
            )
        if isinstance(ctx, (str, BackendFamily)):
            ctx = self._backend_key(ctx)
            ops = self.get_ops(ctx)
            return self.ctx_from_ops(ops, dtype=dtype, enable_checks=enable_checks)
        else:
            raise TypeError(f'Expected Context, BackendFamily, str, or None, got {type(ctx)}.')

    def ctx_from_ops(self, ops: BackendOps, dtype: DType | None = None, enable_checks: bool | None = None) -> Context:
        Context = _context_type()
        dtype = ops.sanitize_dtype(dtype)
        if enable_checks is None:
            enable_checks = self._default_enable_checks
        return Context(ops=ops,
                       dtype=dtype,
                       enable_checks=enable_checks)

    @property
    def default_ctx(self) -> Context:
        return self._default_ctx

    @default_ctx.setter
    def default_ctx(self, ctx: Context | BackendFamily | str | None = None) -> None:
        ctx = self.normalize_context(ctx)
        self._default_ctx = ctx

    def get_ops(self, name: str | BackendFamily | BackendOps | type[BackendOps] | Context) -> BackendOps:
        name = self._backend_key(name)
        if name not in self.available_ops:
            allowed = ", ".join(k for k in self.available_ops.keys())
            raise UnknownBackendError(
                f"Unknown backend: {name!r}. "
                f"Expected one of: {allowed}"
            )
        return self.available_ops[name]()

    @property
    def available_ops(self) -> Dict[str, type[BackendOps]]:
        return self._available_ops

    def register_ops(self, ops: type[BackendOps]) -> type[BackendOps]:
        if not isinstance(ops, type) or not issubclass(ops, BackendOps):
            raise TypeError(f"Expected type[BackendOps], got {type(ops)!r}")
        else:
            family = self._backend_key(ops)
            if family in self.available_ops.keys():
                raise ContextConflictError(f'BackendOps {family} is already registered.')
            self._available_ops[family] = ops
            return ops

    def infer_context(self, x: Any, enable_checks: bool | None = None) -> Context | None:
        """Infer context from `.ctx` first, then registered backend arrays."""
        Context = _context_type()
        if isinstance(x, Context):
            return x

        ctx = getattr(x, "ctx", None)
        if isinstance(ctx, Context):
            return ctx

        matched: list[BackendOps] = []
        for name, ops in self.available_ops.items():
            try:
                ops = ops()
                if ops.is_array(x):
                    matched.append(ops)
            except Exception:
                # Keep inference conservative.
                continue

        if not matched:
            return None
        if len(matched) > 1:
            raise ContextInferenceError(
                f"Ambiguous backend inference for object of type {type(x)!r}: {matched!r}."
            )

        ops = matched[0]
        try:
            dtype = ops.get_dtype(x)
        except Exception:
            dtype = getattr(x, "dtype", self.default_ctx.dtype)

        return self.ctx_from_ops(ops, dtype, enable_checks)

    def infer_contexts(self, values: Iterable[Any]) -> Tuple[Context, ...]:
        out: list[Context] = []
        for x in values:
            ctx = self.infer_context(x)
            if ctx is not None:
                out.append(ctx)
        return tuple(out)

    def are_compatible_contexts(self, *ctxs: Context) -> bool:
        if len(ctxs) < 2:
            return True
        first = ctxs[0]
        return all(ctx.ops.family == first.ops.family for ctx in ctxs[1:])

    def are_compatible_values(self, *values: Any) -> bool:
        return self.are_compatible_contexts(*self.infer_contexts(values))

    def are_compatible_ops(self, *ops: BackendOps) -> bool:
        if not ops:
            return True
        first = ops[0]
        return all(op.family == first.family for op in ops)

    def enforce_convert_policy(self, x: Any, to: Context | BackendFamily | str | None = None) -> Tuple[Any, Context]:
        """Resolve the target context for ``x``."""
        self.infer_context(x)
        ctx = self.normalize_context(to)
        return x, ctx

    def _backend_key(self, x: str | BackendFamily | BackendOps | type[BackendOps] | Context) -> str:
        Context = _context_type()
        if isinstance(x, Context):
            return self._backend_key(x.ops)
        if isinstance(x, BackendOps):
            return self._backend_key(x.family)
        if isinstance(x, type) and issubclass(x, BackendOps):
            return self._backend_key(x._family)
        if isinstance(x, BackendFamily):
            return x.value.lower()
        if isinstance(x, str):
            key = x.lower()
            return "torch" if key == "pytorch" else key
        raise TypeError(f"Unsupported backend key source: {type(x)!r}")

    def resolve_context_priority(
            self,
            priority_ctx: Context | BackendFamily | str | None = None,
            *other_ctx: object,
    ) -> Context:
        """Resolve explicit context first, then compatible inferred contexts."""
        if priority_ctx is not None:
            return self.normalize_context(priority_ctx)

        inferred = self.infer_contexts(other_ctx)
        if not inferred:
            return self.default_ctx

        if not self.are_compatible_contexts(*inferred):
            fams = tuple(ctx.ops.family for ctx in inferred)
            raise ValueError(f"Incompatible inferred contexts: {fams!r}")

        first = inferred[0]
        ops = type(first.ops)()
        dtype = self._join_dtypes(ops, *(ctx.dtype for ctx in inferred))

        return self.ctx_from_ops(
            ops=ops,
            dtype=dtype,
            enable_checks=all(ctx.enable_checks for ctx in inferred),
        )

    def _join_dtypes(self, ops: BackendOps, *dtypes: DType | None) -> DType | None:
        clean = [ops.sanitize_dtype(dt) for dt in dtypes if dt is not None]
        if not clean:
            return ops.sanitize_dtype(None)

        np_ops = NumpyOps()
        joined = np_ops.np.result_type(*clean)
        return ops.sanitize_dtype(joined)


_contextual: Contextual | None = None


def _state() -> Contextual:
    """Return the process-wide contextual singleton."""
    global _contextual
    if _contextual is None:
        _contextual = Contextual()
    return _contextual


def set_context(
        ctx: Context | BackendFamily | str | None = None,
        dtype: Any = None,
        enable_checks: bool | None = None
) -> None:
    """
    Set the process-wide default SpaceCore context.

    Parameters
    ----------
    ctx : Context, BackendFamily, str, or None, optional
        Context or backend specification.
    dtype : Any, optional
        Default dtype override.
    enable_checks : bool or None, optional
        Validation-check policy override.
    """
    state = _state()
    state.default_ctx = state.normalize_context(ctx, dtype=dtype, enable_checks=enable_checks)


def get_context() -> Context:
    """
    Return the current process-wide default SpaceCore context.

    Returns
    -------
    Context
        Active process-wide default context.
    """
    return _state().default_ctx


def resolve_context_priority(
        priority_ctx: Context | BackendFamily | str | None = None,
        *other_ctx: object,
) -> Context:
    """
    Resolve the context assigned to a newly created object.

    Parameters
    ----------
    priority_ctx : Context, BackendFamily, str, or None, optional
        Explicit context that takes precedence when provided.
    *other_ctx : object
        Objects or contexts used as fallback context sources.

    Returns
    -------
    Context
        Resolved context.
    """
    return _state().resolve_context_priority(priority_ctx, *other_ctx)


def register_ops(ops: type[BackendOps]) -> type[BackendOps]:
    """
    Register a backend operations implementation.

    Parameters
    ----------
    ops : type of BackendOps
        Backend operations class to register.

    Returns
    -------
    type of BackendOps
        Registered backend operations class.
    """
    return _state().register_ops(ops)


def normalize_context(
    ctx: Context | BackendFamily | str | None = None,
    dtype: Any = None,
    enable_checks: bool | None = None,
) -> Context:
    """
    Normalize a context specification through the process-wide state.

    Parameters
    ----------
    ctx : Context, BackendFamily, str, or None, optional
        Context or backend specification.
    dtype : Any, optional
        Default dtype override.
    enable_checks : bool or None, optional
        Validation-check policy override.

    Returns
    -------
    Context
        Normalized context.
    """
    return _state().normalize_context(ctx, dtype=dtype, enable_checks=enable_checks)


def normalize_ops(
    ops: str | BackendFamily | BackendOps | type[BackendOps] | Context
) -> BackendOps:
    """
    Normalize backend operations through the process-wide state.

    Parameters
    ----------
    ops : str, BackendFamily, BackendOps, type of BackendOps, or Context
        Backend operations specification.

    Returns
    -------
    BackendOps
        Normalized backend operations singleton.
    """
    if isinstance(ops, BackendOps):
        return ops
    return _state().get_ops(ops)


def enforce_convert_policy(
    x: Any,
    to: Context | BackendFamily | str | None = None,
) -> tuple[Any, Context]:
    """Resolve a conversion target context."""
    return _state().enforce_convert_policy(x, to)

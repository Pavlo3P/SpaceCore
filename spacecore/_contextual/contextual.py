from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
from enum import StrEnum, auto
from warnings import warn

from ..types import DType
from ..backend import Context, NumpyOps, BackendFamily, BackendOps


class ContextPolicy(StrEnum):
    warning = auto()
    error = auto()
    silent = auto()


class ContextError(RuntimeError):
    pass


class ContextInferenceError(ContextError):
    pass


class ContextConflictError(ContextError):
    pass


class UnknownBackendError(ContextError):
    pass

class ContextConversionError(ContextError):
    pass


class Contextual:
    """
    Backend resolver.
    """
    _default_ctx: Context
    _available_ops: Dict[str, type(BackendOps)]
    _resolution_policy: ContextPolicy

    _default_policy: ContextPolicy = ContextPolicy.warning
    _default_dtype: float
    _default_sparse: bool = True
    _default_enable_checks: bool = False

    def __init__(self, resolution_policy: str = None) -> None:
        self.default_ctx = Context(NumpyOps(), self._default_sparse, self._default_dtype, self._default_enable_checks)

        self._available_ops = {
            BackendFamily.NUMPY.name: NumpyOps,
        }

        self.resolution_policy = resolution_policy

    def normalize_context(self, ctx: Context | str | None = None) -> Context:
        if ctx is None:
            return self.default_ctx
        if isinstance(ctx, Context):
            return ctx
        if isinstance(ctx, str):
            ops = self.get_ops(ctx)
            return self.ctx_from_ops(ops)
        else:
            raise TypeError(f'Expected Context or str, got {type(ctx)}.')

    def ctx_from_ops(self, ops: BackendOps, dtype: DType = None, allow_sparse: bool | None = None, enable_checks: bool | None = None) -> Context:
        if dtype is None:
            dtype = self._default_dtype
        if allow_sparse is None:
            allow_sparse = self._default_sparse
        if enable_checks is None:
            enable_checks = self._default_enable_checks
        return Context(ops, allow_sparse, dtype, enable_checks)

    @property
    def default_ctx(self) -> Context:
        return self._default_ctx

    @default_ctx.setter
    def default_ctx(self, ctx: Context | str | None = None) -> None:
        ctx = self.normalize_context(ctx)
        if isinstance(ctx, Context):
            self._default_ctx = ctx
        else:
            raise TypeError(f"Expected Context, got {type(ctx)}")

    @property
    def resolution_policy(self) -> ContextPolicy:
        return self._resolution_policy

    @resolution_policy.setter
    def resolution_policy(self, policy: str) -> None:
        if policy is None:
            self._resolution_policy = self._default_policy

        try:
            self._resolution_policy = (
                policy
                if isinstance(policy, ContextPolicy)
                else ContextPolicy(policy)
            )
        except ValueError as e:
            allowed = ", ".join(p.value for p in ContextPolicy)
            raise ValueError(
                f"Unknown resolution_policy={policy!r}. "
                f"Expected one of: {allowed}"
            ) from e

    def get_ops(self, name: str) -> BackendOps:
        if name not in self._available_ops:
            raise UnknownBackendError(f"Unknown backend: {name!r}.")
        return self._available_ops[name]()

    @property
    def available_ops(self) -> Dict[str, type(BackendOps)]:
        return self._available_ops

    def register_ops(self, ops: BackendOps) -> BackendOps:
        if isinstance(ops, BackendOps):
            if ops.family in self.available_ops.keys():
                raise ContextConflictError(f'BackendOps {ops.family} is already registered.')
            self._available_ops[ops.family] = ops
            return ops
        else:
            raise TypeError(f"Expected BackendOps, got {type(ops)}")

    def infer_context(self, x: Any, allow_sparse: bool | None = None, enable_checks: bool | None = None) -> Context | None:
        """
        Infer a context from a single value.

        Intended precedence:
          1. objects carrying `.ctx`
          2. backend-native arrays recognized by registered backends
        """
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
        dtype = ops.get_dtype(x)

        return self.ctx_from_ops(ops, dtype, allow_sparse, enable_checks)

    def infer_contexts(self, values: Iterable[Any]) -> tuple[Context | None, ...]:
        out: list[Context] = []
        for x in values:
            ctx = self.infer_context(x)
            if ctx is not None:
                out.append(ctx)
        return tuple(out)

    def are_compatible_dtypes(self, *dtypes: Iterable[DType]) -> bool:
        return True

    def are_compatible_ops(self, *ops: Iterable[BackendOps]) -> bool:
        first = ops[0]
        return all(op.family == first.family for op in ops)

    def are_compatible(self, *values: Any) -> bool:
        ctxs = self.infer_contexts(values)
        if not ctxs:
            return True
        dtypes = [ctx.dtype for ctx in ctxs]
        ops = [ctx.ops for ctx in ctxs]
        return self.are_compatible_dtypes(*dtypes) and self.are_compatible_ops(*ops)

    def enforce_convert_policy(self, x: Any, to: Context | str | None = None) -> Tuple[Any, Context]:
        ctx = self.normalize_context(to)
        if self.resolution_policy is not ContextPolicy.silent:
            native_ctx = self.infer_context(x)
            if not self.are_compatible(ctx, native_ctx):
                if self.resolution_policy is ContextPolicy.warning:
                    warn(f'Conversion from context {type(native_ctx)!r} to context {type(x)!r}.', UserWarning)
                else:
                    raise ContextConversionError('Attempted to convert object Context.')
        return x, ctx

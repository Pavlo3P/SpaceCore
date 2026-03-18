from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
from enum import StrEnum, auto
from warnings import warn

from ..types import DType
from ..backend import Context, NumpyOps, JaxOps, BackendFamily, BackendOps


class ContextPolicy(StrEnum):
    warning = auto()
    error = auto()
    silent = auto()

class DtypePreservePolicy(StrEnum):
    keep_native = auto()
    use_default = auto()


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
    _available_ops: Dict[str, type[BackendOps]]
    _resolution_policy: ContextPolicy

    _default_policy: ContextPolicy = ContextPolicy.warning
    _default_dtype_resolution_policy: DtypePreservePolicy = DtypePreservePolicy.use_default
    _default_dtype: DType | None = None
    _default_enable_checks: bool = False

    def __init__(self,
                 resolution_policy: str | ContextPolicy | None = None,
                 dtype_resolution_policy: str | DtypePreservePolicy | None = None
                 ) -> None:
        self.default_ctx = Context(
            ops=NumpyOps(),
            dtype=self._default_dtype,
            enable_checks=self._default_enable_checks
        )

        self._available_ops = {
            self._backend_key(NumpyOps): NumpyOps,
            self._backend_key(JaxOps): JaxOps,
        }

        self.resolution_policy = resolution_policy
        self.dtype_resolution_policy = dtype_resolution_policy

    def normalize_context(self, ctx: Context | BackendFamily | str | None = None) -> Context:
        if ctx is None:
            return self.default_ctx
        if isinstance(ctx, Context):
            return ctx
        if isinstance(ctx, (str, BackendFamily)):
            ctx = self._backend_key(ctx)
            ops = self.get_ops(ctx)
            return self.ctx_from_ops(ops)
        else:
            raise TypeError(f'Expected Context, BackendFamily, str, or None, got {type(ctx)}.')

    def ctx_like(self, base: Context | None, ctx: Context) -> Context:
        if isinstance(base, Context):
            return Context(
                ops=ctx.ops,
                dtype=base.dtype,
                enable_checks=ctx.enable_checks,
            )
        return self.default_ctx

    def normalize_context_like(self, base: Context | None, ctx: Context | BackendFamily | str | None = None) -> Context:
        if self.dtype_resolution_policy is DtypePreservePolicy.keep_native and isinstance(base, Context):
            if ctx is None:
                return self.ctx_like(base, self.default_ctx)
            if isinstance(ctx, Context):
                return self.ctx_like(base, ctx)
            if isinstance(ctx, (str, BackendFamily)):
                ctx = self._backend_key(ctx)
                ops = self.get_ops(ctx)
                return self.ctx_from_ops(
                    ops=ops,
                    dtype=base.dtype,
                    enable_checks=base.enable_checks,
                )
            else:
                raise TypeError(f'Expected Context, BackendFamily, str, or None, got {type(ctx)}.')
        else:
            return self.normalize_context(ctx)

    def ctx_from_ops(self, ops: BackendOps, dtype: DType | None = None, enable_checks: bool | None = None) -> Context:
        if dtype is None:
            dtype = self._default_dtype
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

    @property
    def resolution_policy(self) -> ContextPolicy:
        return self._resolution_policy

    @resolution_policy.setter
    def resolution_policy(self, policy: str | None = ContextPolicy.warning.value) -> None:
        if policy is None:
            self._resolution_policy = self._default_policy
            return

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

    @property
    def dtype_resolution_policy(self) -> DtypePreservePolicy:
        return self._dtype_resolution_policy

    @dtype_resolution_policy.setter
    def dtype_resolution_policy(self, policy: str | None = DtypePreservePolicy.use_default.value) -> None:
        if policy is None:
            self._dtype_resolution_policy = self._default_dtype_resolution_policy
            return

        try:
            self._dtype_resolution_policy = (
                policy
                if isinstance(policy, DtypePreservePolicy)
                else DtypePreservePolicy(policy)
            )
        except ValueError as e:
            allowed = ", ".join(p.value for p in DtypePreservePolicy)
            raise ValueError(
                f"Unknown dtype_resolution_policy={policy!r}. "
                f"Expected one of: {allowed}"
            ) from e

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
        native_ctx = self.infer_context(x)
        ctx = self.normalize_context_like(native_ctx, to)
        if self.resolution_policy is not ContextPolicy.silent:
            if native_ctx is not None and not self.are_compatible_contexts(native_ctx, ctx):
                if self.resolution_policy is ContextPolicy.warning:
                    warn(
                        f"Converting from {native_ctx!r} to {ctx!r}.",
                        UserWarning,
                    )
                else:
                    raise ContextConversionError(
                        f"Conversion from {native_ctx!r} to {ctx!r} is forbidden by policy {self.resolution_policy.value!r}."
                    )
        return x, ctx

    def _backend_key(self, x: str | BackendFamily | BackendOps | type[BackendOps] | Context) -> str:
        if isinstance(x, Context):
            return self._backend_key(x.ops)
        if isinstance(x, BackendOps):
            return self._backend_key(x.family)
        if isinstance(x, type) and issubclass(x, BackendOps):
            return self._backend_key(x._family)
        if isinstance(x, BackendFamily):
            return x.value.lower()
        if isinstance(x, str):
            return x.lower()
        raise TypeError(f"Unsupported backend key source: {type(x)!r}")

    def resolve_context_priority(
            self,
            priority_ctx: Context | BackendFamily | str | None = None,
            *other_ctx: object,
    ) -> Context:
        """
        Resolve context with priority:

        1. If priority_ctx is not None, return its normalized form.
        2. Otherwise, infer contexts from other_ctx and return the best inferred one.
        3. If inference fails, return default_ctx.

        Policy:
        - explicit context always wins
        - inferred contexts must be family-compatible
        - if inferred dtypes differ, resulting dtype is None
        """
        if priority_ctx is not None:
            return self.normalize_context(priority_ctx)

        inferred = self.infer_contexts(other_ctx)
        if not inferred:
            return self.default_ctx

        if not self.are_compatible_contexts(*inferred):
            fams = tuple(ctx.ops.family for ctx in inferred)
            raise ValueError(f"Incompatible inferred contexts: {fams!r}")

        first = inferred[0]
        dtypes = {ctx.dtype for ctx in inferred}
        dtype = next(iter(dtypes)) if len(dtypes) == 1 else None

        return Context(
            ops=type(first.ops)(),
            dtype=dtype,
            enable_checks=all(ctx.enable_checks for ctx in inferred),
        )

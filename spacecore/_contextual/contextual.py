from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
from enum import StrEnum, auto
from warnings import warn

from ..types import DType
from ..backend import Context, NumpyOps, JaxOps, BackendFamily, BackendOps
try:
    from ..backend import CuPyOps
except ImportError:
    pass
try:
    from ..backend import TorchOps
except ImportError:
    pass


class ContextPolicy(StrEnum):
    """
    Policy for backend-incompatible context conversion.

    Values
    ------
    warning:
        Allow conversion to a different backend family and issue a warning.
        This is the default.
    error:
        Reject conversion to a different backend family. Use this when
        accidental backend migration should be forbidden.
    silent:
        Allow conversion to a different backend family without warning. Use
        this when automatic conversion is expected and controlled.
    """

    warning = auto()
    error = auto()
    silent = auto()

class DtypePreservePolicy(StrEnum):
    """
    Policy for dtype handling during context conversion.

    Values
    ------
    keep_native:
        Preserve the source object's dtype where possible by converting it to an
        equivalent dtype in the target backend. This is the default.
    convert:
        Use the dtype provided by the resolved target context. This prioritizes
        dtype unification under the target context.
    """

    keep_native = auto()
    convert = auto()


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
    _default_dtype_resolution_policy: DtypePreservePolicy = DtypePreservePolicy.keep_native
    _default_dtype: DType | None = None
    _default_enable_checks: bool = False

    def __init__(self,
                 resolution_policy: str | ContextPolicy | None = None,
                 dtype_resolution_policy: str | DtypePreservePolicy | None = None
                 ) -> None:
        ops = NumpyOps()
        self.default_ctx = Context(
            ops=ops,
            dtype=ops.sanitize_dtype(self._default_dtype),
            enable_checks=self._default_enable_checks
        )

        self._available_ops = {
            self._backend_key(NumpyOps): NumpyOps,
            self._backend_key(JaxOps): JaxOps,
        }
        if "CuPyOps" in globals():
            self._available_ops[self._backend_key(CuPyOps)] = CuPyOps
        if "TorchOps" in globals():
            self._available_ops[self._backend_key(TorchOps)] = TorchOps

        self.resolution_policy = resolution_policy
        self.dtype_resolution_policy = dtype_resolution_policy

    def normalize_context(self,
                          ctx: Context | BackendFamily | str | None = None,
                          dtype: Any = None,
                          enable_checks: bool | None = None
                          ) -> Context:
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

    def ctx_like(self, base: Context | None, ctx: Context) -> Context:
        if isinstance(base, Context):
            return Context(
                ops=ctx.ops,
                dtype=ctx.ops.sanitize_dtype(base.dtype),
                enable_checks=ctx.enable_checks,
            )
        return self.default_ctx

    def normalize_context_like(self, base: Context | None, ctx: Context | BackendFamily | str | None = None) -> Context:
        """
        Normalize a target context while applying the dtype resolution policy.

        Parameters
        ----------
        base:
            Native context inferred from the object being converted, or
            ``None`` when no native context is known.
        ctx:
            Requested target context. This may be a concrete ``Context``,
            backend family string, backend enum value, or ``None``.

        Returns
        -------
        Context
            Normalized target context.

        Notes
        -----
        This method implements ``dtype_resolution_policy``:

        * ``keep_native`` preserves ``base.dtype`` when ``base`` exists, while
          adapting it to the requested backend.
        * ``convert`` ignores ``base.dtype`` and uses normal target-context
          dtype resolution.
        """
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
                    enable_checks=self._default_enable_checks,
                )
            else:
                raise TypeError(f'Expected Context, BackendFamily, str, or None, got {type(ctx)}.')
        else:
            return self.normalize_context(ctx)

    def ctx_from_ops(self, ops: BackendOps, dtype: DType | None = None, enable_checks: bool | None = None) -> Context:
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

    @property
    def resolution_policy(self) -> ContextPolicy:
        """
        Backend conversion warning/error policy.

        Returns
        -------
        ContextPolicy
            Active policy for conversion between backend families.

        Notes
        -----
        ``warning`` allows backend-incompatible conversion and emits a warning.
        ``error`` rejects backend-incompatible conversion.
        ``silent`` allows backend-incompatible conversion without warning.
        """
        return self._resolution_policy

    @resolution_policy.setter
    def resolution_policy(self, policy: str | None = ContextPolicy.warning.value) -> None:
        """
        Set the backend conversion warning/error policy.

        Parameters
        ----------
        policy:
            One of ``"warning"``, ``"error"``, ``"silent"``, a
            :class:`ContextPolicy` value, or ``None`` to restore the default.

        Raises
        ------
        ValueError
            If ``policy`` is not one of the supported policy values.

        Notes
        -----
        This policy controls conversion between different backend families:

        * ``warning``: convert and warn.
        * ``error``: reject conversion.
        * ``silent``: convert without warning.
        """
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
        """
        Dtype handling policy for context conversion.

        Returns
        -------
        DtypePreservePolicy
            Active dtype policy.

        Notes
        -----
        ``keep_native`` preserves the source dtype when converting between
        contexts. ``convert`` uses the dtype supplied by the resolved target
        context.
        """
        return self._dtype_resolution_policy

    @dtype_resolution_policy.setter
    def dtype_resolution_policy(self, policy: str | None = DtypePreservePolicy.keep_native.value) -> None:
        """
        Set the dtype handling policy for context conversion.

        Parameters
        ----------
        policy:
            One of ``"keep_native"``, ``"convert"``, a
            :class:`DtypePreservePolicy` value, or ``None`` to restore the
            default.

        Raises
        ------
        ValueError
            If ``policy`` is not one of the supported policy values.

        Notes
        -----
        This policy controls dtype choice after context resolution:

        * ``keep_native``: preserve the source object's dtype in the target
          backend when possible.
        * ``convert``: use the dtype provided by the resolved target context.
        """
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
        """
        Resolve the target context for ``x`` and enforce conversion policies.

        Parameters
        ----------
        x:
            Object being converted. If it carries a native context, that context
            is used to detect backend-family changes and native dtype.
        to:
            Requested target context. This may be a concrete ``Context``,
            backend family string, backend enum value, or ``None``.

        Returns
        -------
        tuple[Any, Context]
            The original object and the normalized target context.

        Raises
        ------
        ContextConversionError
            If the native and target contexts use different backend families and
            ``resolution_policy`` is ``"error"``.

        Warns
        -----
        UserWarning
            If the native and target contexts use different backend families and
            ``resolution_policy`` is ``"warning"``.

        Notes
        -----
        Backend-family compatibility is governed by ``resolution_policy``:

        * ``warning``: allow backend conversion and warn.
        * ``error``: reject backend conversion.
        * ``silent``: allow backend conversion without warning.

        Dtype choice is handled independently by ``dtype_resolution_policy`` via
        :meth:`normalize_context_like`.
        """
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
            key = x.lower()
            return "torch" if key == "pytorch" else key
        raise TypeError(f"Unsupported backend key source: {type(x)!r}")

    def resolve_context_priority(
            self,
            priority_ctx: Context | BackendFamily | str | None = None,
            *other_ctx: object,
    ) -> Context:
        """
        Resolve the context assigned to a newly created object.

        Parameters
        ----------
        priority_ctx:
            Explicit context supplied by the caller. If this is not ``None``,
            it wins over every inferred context.
        *other_ctx:
            Objects that may carry a ``ctx`` attribute or be backend-native
            arrays. These are used for context inference when no explicit
            context is supplied.

        Returns
        -------
        Context
            The resolved context.

        Raises
        ------
        ValueError
            If contexts can be inferred but their backend families are
            incompatible.

        Notes
        -----
        The resolution order follows the conversion-policy tutorial:

        1. Use the explicit ``priority_ctx`` if provided.
        2. Otherwise, infer contexts from input objects that carry context.
        3. Inference is possible only when all inferred contexts use the same
           backend family.
        4. If inferred dtypes differ, choose the most general dtype among them.
        5. The inferred ``enable_checks`` flag is the conjunction of source
           flags, so checks remain enabled only when all source contexts enable
           checks.
        6. If inference finds no context, use ``default_ctx``.
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

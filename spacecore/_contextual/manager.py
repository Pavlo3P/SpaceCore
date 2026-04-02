from typing import Any

from ..backend import Context, BackendOps
from .contextual import Contextual, ContextPolicy, DtypePreservePolicy
from ..backend import BackendFamily


ctx_manager = Contextual()


def set_context(
        ctx: Context | BackendFamily | str | None = None,
        dtype: Any = None,
        enable_checks: bool | None = None
) -> None:
    ctx = ctx_manager.normalize_context(ctx, dtype=dtype, enable_checks=enable_checks)
    ctx_manager.default_ctx = ctx


def get_context():
    return ctx_manager.default_ctx


def register_ops(ops: type[BackendOps]) -> type[BackendOps]:
    return ctx_manager.register_ops(ops)


def set_resolution_policy(policy: ContextPolicy | str | None = None) -> None:
    ctx_manager.resolution_policy = policy


def get_resolution_policy() -> str:
    return ctx_manager.resolution_policy.value


def set_dtype_resolution_policy(
    policy: DtypePreservePolicy | str | None = None,
) -> None:
    ctx_manager.dtype_resolution_policy = policy


def get_dtype_resolution_policy() -> str:
    return ctx_manager.dtype_resolution_policy.value

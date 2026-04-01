from ..backend import Context, BackendOps
from .contextual import Contextual, ContextPolicy, DtypePreservePolicy
from ..backend import BackendFamily


ctx_manager = Contextual()


def set_context(ctx: Context | BackendFamily | str | None = None) -> None:
    ctx_manager.default_ctx = ctx


def get_context():
    return ctx_manager.default_ctx


def register_ops(ops: type[BackendOps]) -> type[BackendOps]:
    return ctx_manager.register_ops(ops)


def set_resolution_policy(policy: ContextPolicy | str | None = None) -> None:
    ctx_manager.resolution_policy = policy


def get_resolution_policy() -> ContextPolicy:
    return ctx_manager.resolution_policy


def set_dtype_resolution_policy(
    policy: DtypePreservePolicy | str | None = None,
) -> None:
    ctx_manager.dtype_resolution_policy = policy


def get_dtype_resolution_policy() -> DtypePreservePolicy:
    return ctx_manager.dtype_resolution_policy

from ..backend import Context, BackendOps
from .contextual import Contextual
from ..backend import BackendFamily


ctx_manager = Contextual()

def set_context(ctx: Context | BackendFamily | str | None = None) -> None:
    ctx_manager.default_ctx = ctx

def get_context():
    return ctx_manager.default_ctx

def register_ops(ops: type[BackendOps]) -> type[BackendOps]:
    return ctx_manager.register_ops(ops)

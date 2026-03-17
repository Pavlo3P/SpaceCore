from ..backend import Context, BackendOps
from .contextual import Contextual


ctx_manager = Contextual()

def set_context(ctx: Context | str | None = None) -> None:
    ctx_manager.default_ctx = ctx

def register_ops(ops: BackendOps) -> BackendOps:
    return ctx_manager.register_ops(ops)

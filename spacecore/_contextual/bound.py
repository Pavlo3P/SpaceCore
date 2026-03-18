from __future__ import annotations

from abc import ABC
from typing import Self

from ..backend import Context, BackendOps
from ..types import DType
from .manager import ctx_manager

class ContextBound(ABC):
    def __init__(self, ctx: Context | str | None = None):
        ctx = ctx_manager.normalize_context(ctx)
        self._ctx = ctx

    @property
    def ops(self) -> BackendOps:
        return self.ctx.ops

    @property
    def dtype(self) -> DType:
        return self.ctx.dtype

    @property
    def ctx(self) -> Context:
        return self._ctx

    def _convert(self, new_ctx: Context) -> Self:
        raise NotImplementedError()

    def convert(self, new_ctx: Context | str | None = None) -> Self:
        _, new_ctx = ctx_manager.enforce_convert_policy(self, new_ctx)
        return self._convert(new_ctx)

from __future__ import annotations

from abc import ABC
from typing import Self

from ..backend import Context, BackendOps, BackendFamily
from ..types import DType
from .manager import ctx_manager


def _same_effective_context(left: Context, right: Context) -> bool:
    return (
        left.ops == right.ops
        and left.dtype == right.dtype
        and left.enable_checks == right.enable_checks
    )


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

    def convert(self, new_ctx: Context | BackendFamily | str | None = None) -> Self:
        _, new_ctx = ctx_manager.enforce_convert_policy(self, new_ctx)
        if _same_effective_context(self.ctx, new_ctx):
            return self
        return self._convert(new_ctx)

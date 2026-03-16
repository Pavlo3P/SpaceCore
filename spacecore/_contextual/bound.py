from abc import ABC
from ..backend import BackendContext

class ContextBound(ABC):
    ctx: BackendContext

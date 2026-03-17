from enum import StrEnum, auto


class BackendFamily(StrEnum):
    NUMPY = auto()
    JAX = auto()

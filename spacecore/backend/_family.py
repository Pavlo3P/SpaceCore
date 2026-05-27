from enum import StrEnum, auto


class BackendFamily(StrEnum):
    numpy = auto()
    jax = auto()
    torch = auto()
    cupy = auto()

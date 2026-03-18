from enum import StrEnum, auto


class BackendFamily(StrEnum):
    numpy = auto()
    jax = auto()

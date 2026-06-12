from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from spacecore.backend import Context, NumpyOps

from ._protocol import GeneratedCase


ContextCase = GeneratedCase[Context | None]


def _available_case(
    backend: str,
    dtype: Any,
    ops_factory: Callable[[], Any],
    *,
    check_level: str,
) -> ContextCase:
    ctx = Context(ops_factory(), dtype=dtype, check_level=check_level)
    field = "complex" if ctx.ops.is_complex_dtype(ctx.dtype) else "real"
    return GeneratedCase(
        obj=ctx,
        reference={"backend": backend, "dtype": ctx.dtype, "field": field, "available": True},
        capabilities=frozenset({"context", backend, field}),
        id=f"{backend}-{_dtype_name(ctx.dtype)}",
    )


def _unavailable_cases(backend: str, reason: str) -> tuple[ContextCase, ContextCase]:
    mark = pytest.mark.skip(reason=reason)
    return tuple(
        GeneratedCase(
            obj=None,
            reference={"backend": backend, "dtype": dtype, "field": field, "available": False},
            capabilities=frozenset({"context", backend, field, "unavailable"}),
            marks=(mark,),
            id=f"{backend}-{np.dtype(dtype).name}-unavailable",
        )
        for dtype, field in ((np.float64, "real"), (np.complex128, "complex"))
    )  # type: ignore[return-value]


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    for name in ("complex128", "complex64", "float64", "float32"):
        if name in text:
            return name
    return text.replace(".", "-")


def _optional_backend_cases(backend: str, check_level: str) -> tuple[ContextCase, ...]:
    import spacecore.backend as backend_module

    class_name = {"jax": "JaxOps", "torch": "TorchOps", "cupy": "CuPyOps"}[backend]
    ops_type = getattr(backend_module, class_name, None)
    if ops_type is None:
        return _unavailable_cases(backend, f"{backend} is not installed")

    if backend == "jax":
        import jax

        x64 = bool(jax.config.read("jax_enable_x64"))
        dtypes = (np.float64, np.complex128) if x64 else (np.float32, np.complex64)
    else:
        dtypes = (np.float64, np.complex128)

    cases: list[ContextCase] = []
    try:
        for dtype in dtypes:
            case = _available_case(backend, dtype, ops_type, check_level=check_level)
            assert case.obj is not None
            case.obj.asarray(np.zeros((1,), dtype=dtype))
            cases.append(case)
    except Exception as exc:
        return _unavailable_cases(backend, f"{backend} is unavailable: {exc}")
    return tuple(cases)


def context_cases(
    *,
    include_optional: bool = True,
    include_unavailable: bool = True,
    check_level: str = "standard",
) -> tuple[ContextCase, ...]:
    """Generate supported backend/dtype contexts with explicit optional-backend skips."""
    cases: list[ContextCase] = [
        _available_case("numpy", np.float64, NumpyOps, check_level=check_level),
        _available_case("numpy", np.complex128, NumpyOps, check_level=check_level),
    ]
    if include_optional:
        for backend in ("jax", "torch", "cupy"):
            optional = _optional_backend_cases(backend, check_level)
            if include_unavailable:
                cases.extend(optional)
            else:
                cases.extend(case for case in optional if case.obj is not None)
    return tuple(cases)

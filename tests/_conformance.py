"""Shared conformance harness for the whole test suite.

This module is the canonical home for cross-backend numerical comparison
infrastructure: the per-op tolerance table, the ``assert_matches_reference``
helper, and the backend-capability probes. It lives at ``tests/_conformance.py``
(not under ``tests/backend/``) so that any suite — backend, linops, linalg,
spaces, functional — can compare a backend result against an independent
reference with one consistent tolerance policy.

``tests/backend/_conformance.py`` re-exports everything here for backward
compatibility with the backend conformance tests.

A test typically:

1. Builds inputs with ``np.asarray(...)`` for the reference.
2. Builds the same inputs in the backend with ``ops.asarray(...)``.
3. Computes the operation on both sides and asserts equivalence via
   :func:`assert_matches_reference`.

Per-op tolerance lives in :data:`TOLERANCE_TABLE`. The default is tighter
than ``numpy.allclose``'s default and is intentionally per-op so that a
flaky operation (complex sqrt, eigh sorting, float32 logsumexp) can be
relaxed in one place without loosening the rest of the suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from tests._helpers import has_cupy, has_jax, has_torch, to_numpy


__all__ = [
    "Tolerance",
    "TOLERANCE_TABLE",
    "dtype_kind",
    "tolerance_for",
    "assert_matches_reference",
    "assert_eigh_identity",
    "numpy_reference",
    "available_backend_families",
    "backend_supports_dtype",
    "iter_dtypes",
]


@dataclass(frozen=True)
class Tolerance:
    """Per-op tolerance keyed by (op, dtype-kind, backend).

    ``rtol`` and ``atol`` follow ``numpy.allclose`` semantics. The optional
    ``skip_backends`` set declares backends that are not expected to match
    within tolerance — they should be tested via a different identity, or
    skipped with a documented reason.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    skip_backends: frozenset[str] = field(default_factory=frozenset)


_DEFAULT = Tolerance()
_LOOSER = Tolerance(rtol=1e-5, atol=1e-7)
_LOOSER_FLOAT32 = Tolerance(rtol=1e-4, atol=1e-5)
_LOOSE_COMPLEX = Tolerance(rtol=1e-5, atol=1e-6)


TOLERANCE_TABLE: Mapping[tuple[str, str], Tolerance] = {
    # (op_name, dtype_kind) -> Tolerance.
    # dtype_kind is one of: "real32", "real64", "complex64", "complex128".
    # When a (op, dtype) is not listed, _DEFAULT applies.
    ("eigh", "real32"): _LOOSER_FLOAT32,
    ("eigh", "complex64"): _LOOSER_FLOAT32,
    ("eigh", "real64"): _LOOSER,
    ("eigh", "complex128"): _LOOSER,
    ("svd", "real32"): _LOOSER_FLOAT32,
    ("svd", "complex64"): _LOOSER_FLOAT32,
    ("svd", "real64"): _LOOSER,
    ("svd", "complex128"): _LOOSER,
    ("sqrt", "complex64"): _LOOSE_COMPLEX,
    ("sqrt", "complex128"): _LOOSE_COMPLEX,
    ("logsumexp", "real32"): _LOOSER_FLOAT32,
    ("cholesky", "real32"): _LOOSER_FLOAT32,
    ("cholesky", "complex64"): _LOOSER_FLOAT32,
    ("solve", "real32"): _LOOSER_FLOAT32,
    ("solve", "complex64"): _LOOSER_FLOAT32,
    ("expm_multiply", "real32"): _LOOSER_FLOAT32,
}


def _canonicalize_dtype(dtype: Any) -> np.dtype:
    """Return a NumPy dtype for any framework's dtype object.

    Backend dtype objects (``torch.float32``, ``jnp.float64``, ...) are not
    directly accepted by ``np.dtype``. This helper normalizes them via the
    framework's own ``numpy()`` adapter where needed.
    """
    try:
        return np.dtype(dtype)
    except TypeError:
        pass
    name = str(dtype)
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    if name in ("float16", "float32", "float64", "complex64", "complex128",
                "int32", "int64", "bool"):
        return np.dtype(name)
    raise TypeError(f"cannot canonicalize dtype {dtype!r}")


def dtype_kind(dtype: Any) -> str:
    """Classify ``dtype`` as one of real32/real64/complex64/complex128."""
    dt = _canonicalize_dtype(dtype)
    if dt == np.float32:
        return "real32"
    if dt == np.float64:
        return "real64"
    if dt == np.complex64:
        return "complex64"
    if dt == np.complex128:
        return "complex128"
    raise ValueError(f"unsupported conformance dtype {dt}")


def tolerance_for(op_name: str, dtype: Any) -> Tolerance:
    """Return the per-op tolerance, falling back to the default if absent."""
    return TOLERANCE_TABLE.get((op_name, dtype_kind(dtype)), _DEFAULT)


def assert_matches_reference(
    op_name: str,
    backend_result: Any,
    reference: Any,
    *,
    dtype: Any,
    equal_nan: bool = False,
) -> None:
    """Assert ``backend_result`` matches ``reference`` within per-op tolerance.

    ``backend_result`` is the value returned by a backend op (NumPy, JAX,
    Torch, or CuPy). It is normalized to a NumPy array via
    :func:`tests._helpers.to_numpy` before comparison.

    For boolean / integer outputs, exact equality is required regardless of
    the tolerance table.
    """
    actual = to_numpy(backend_result)
    expected = np.asarray(to_numpy(reference)) if not isinstance(reference, np.ndarray) else reference
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{op_name}: shape mismatch — backend {actual.shape} vs reference {expected.shape}"
        )
    if expected.dtype.kind in ("b", "i", "u"):
        if not np.array_equal(actual, expected):
            raise AssertionError(
                f"{op_name}: integer/bool mismatch\n"
                f"backend={actual!r}\nreference={expected!r}"
            )
        return
    tol = tolerance_for(op_name, dtype)
    if not np.allclose(actual, expected, rtol=tol.rtol, atol=tol.atol, equal_nan=equal_nan):
        diff = np.abs(actual - expected)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = diff / np.maximum(np.abs(expected), 1e-30)
        raise AssertionError(
            f"{op_name}: tolerance miss for dtype={np.dtype(dtype)}\n"
            f"rtol={tol.rtol}, atol={tol.atol}\n"
            f"max|diff|={float(diff.max()):.3e}, max|rel|={float(np.nanmax(rel)):.3e}\n"
            f"backend={actual!r}\nreference={expected!r}"
        )


def assert_eigh_identity(
    backend_ops: Any,
    A: Any,
    *,
    dtype: Any,
) -> None:
    """Assert ``eigh`` satisfies ``A v_i = lambda_i v_i`` within tolerance.

    ``eigh`` is compared via the eigenvalue identity instead of by direct
    eigenvector equality because eigenvector signs (and complex phases) are
    ambiguous, and backends differ in eigenvalue sort order conventions.
    """
    eigvals, eigvecs = backend_ops.eigh(A)
    Av = backend_ops.matmul(A, eigvecs)
    lhs = to_numpy(Av)
    rhs = to_numpy(eigvecs) * to_numpy(eigvals)[None, :]
    tol = tolerance_for("eigh", dtype)
    if not np.allclose(lhs, rhs, rtol=tol.rtol, atol=tol.atol):
        raise AssertionError(
            f"eigh identity A v = λ v failed for dtype={np.dtype(dtype)}\n"
            f"max|diff|={float(np.abs(lhs - rhs).max()):.3e}"
        )


def numpy_reference(op_name: str) -> Callable[..., Any]:
    """Return the NumPy reference callable for ``op_name``.

    Only ops that have a direct NumPy spelling appear here. Linear-algebra
    ops live under ``numpy.linalg`` and are aliased for convenience.
    """
    if op_name in _NUMPY_REF:
        return _NUMPY_REF[op_name]
    raise KeyError(f"no NumPy reference registered for {op_name}")


_NUMPY_REF: dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "argmax": np.argmax,
    "argmin": np.argmin,
    "argsort": np.argsort,
    "broadcast_to": np.broadcast_to,
    "cholesky": np.linalg.cholesky,
    "clip": np.clip,
    "concatenate": np.concatenate,
    "conj": np.conj,
    "diag": np.diag,
    "diagonal": np.diagonal,
    "eigvalsh": np.linalg.eigvalsh,
    "einsum": np.einsum,
    "exp": np.exp,
    "expand_dims": np.expand_dims,
    "eye": np.eye,
    "imag": np.imag,
    "isfinite": np.isfinite,
    "isnan": np.isnan,
    "kron": np.kron,
    "log": np.log,
    "matmul": np.matmul,
    "max": np.max,
    "maximum": np.maximum,
    "mean": np.mean,
    "min": np.min,
    "minimum": np.minimum,
    "moveaxis": np.moveaxis,
    "norm": np.linalg.norm,
    "ones": np.ones,
    "ones_like": np.ones_like,
    "prod": np.prod,
    "ravel": np.ravel,
    "real": np.real,
    "reshape": np.reshape,
    "sign": np.sign,
    "solve": np.linalg.solve,
    "sort": np.sort,
    "sqrt": np.sqrt,
    "squeeze": np.squeeze,
    "stack": np.stack,
    "sum": np.sum,
    "swapaxes": np.swapaxes,
    "take": np.take,
    "trace": np.trace,
    "transpose": np.transpose,
    "tril": np.tril,
    "triu": np.triu,
    "vdot": np.vdot,
    "where": np.where,
    "zeros": np.zeros,
    "zeros_like": np.zeros_like,
}


def available_backend_families() -> tuple[str, ...]:
    """Return the families that should be exercised in the current run."""
    families = ["numpy"]
    if has_jax():
        families.append("jax")
    if has_torch():
        families.append("torch")
    if has_cupy():
        families.append("cupy")
    return tuple(families)


def backend_supports_dtype(family: str, dtype: Any) -> bool:
    """Return whether ``family`` natively supports ``dtype`` end-to-end.

    JAX uses 32-bit by default unless ``jax_enable_x64`` is set; Torch
    refuses complex matmul/eigh on some platforms; CuPy varies by GPU.
    Conservative; callers should also consult the per-op tolerance table.
    """
    dt = np.dtype(dtype)
    if family == "jax":
        from tests._helpers import jax_real_dtype

        wanted_real = jax_real_dtype()
        if dt == np.float64 and wanted_real != np.float64:
            return False
        if dt == np.complex128 and wanted_real != np.float64:
            return False
        return True
    if family == "torch":
        # Torch defaults to float32; honour the configured default for the
        # 64-bit cases so we don't ship spurious failures.
        from tests._helpers import torch_real_dtype

        default = torch_real_dtype()
        try:
            import torch

            if dt == np.float64:
                return default == torch.float64
            if dt == np.complex128:
                return default == torch.float64
        except ImportError:
            return False
        return True
    return True


def iter_dtypes(family: str, dtypes: Iterable[Any]) -> Iterable[Any]:
    """Yield only the dtypes ``family`` claims to support end-to-end."""
    for dt in dtypes:
        if backend_supports_dtype(family, dt):
            yield dt

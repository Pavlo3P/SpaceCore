"""Tests for the :class:`spacecore.Context` object.

Organized to match the per-object Context checklist:

1. Construction — from ``BackendOps`` instance, family string, and
   ``(family, dtype)`` pair.
2. Equality, hash, frozen identity.
3. ``Context.dtype`` defaulting per family.
4. ``ctx.asarray`` — pass-through for matching ops/dtype, cross-family
   conversion, complex→real refusal (ADR-015 Stage 1), real→complex
   broadening, dtype passthrough.
5. ``ctx.assparse`` — supported on sparse-capable backends; refused when
   ``ops.allow_sparse`` is ``False``.
6. ``__repr__`` stability.

Plus additional Context-API surface area not in the original checklist:

7. ``assert_dense`` / ``assert_sparse`` gates.
8. ``convert`` dispatch (dense → ``asarray``, sparse → ``assparse``).
9. ``check_level`` normalization and the deprecated ``enable_checks`` alias.

Generic per-op behavior lives in :mod:`tests.backend.test_operations`;
this module pins the ``Context`` API only.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import spacecore as sc

from tests._helpers import has_cupy, has_jax, has_torch, to_numpy
from tests.backend._conformance import (
    assert_matches_reference,
    backend_supports_dtype,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
_OPTIONAL_BACKEND_PROBES = {"jax": has_jax, "torch": has_torch, "cupy": has_cupy}


def _all_available_families() -> list[str]:
    families = ["numpy"]
    for family, probe in _OPTIONAL_BACKEND_PROBES.items():
        if probe():
            families.append(family)
    return families


def _supported_real_dtype(family: str):
    for dt in (np.float64, np.float32):
        if backend_supports_dtype(family, dt):
            return dt
    return None


def _supported_complex_dtype(family: str):
    for dt in (np.complex128, np.complex64):
        if backend_supports_dtype(family, dt):
            return dt
    return None


def _densify(out):
    if hasattr(out, "toarray"):
        return np.asarray(to_numpy(out.toarray()))
    if hasattr(out, "to_dense"):
        return np.asarray(to_numpy(out.to_dense()))
    if hasattr(out, "todense"):
        return np.asarray(out.todense())
    return None


def _assert_dtype(out: object, expected_np_dtype) -> None:
    """Backend-agnostic dtype equality: compare via the NumPy view.

    ``backend_ops.get_dtype(x)`` returns the backend-native dtype object
    (``numpy.dtype``, ``jax.numpy.dtype``, ``torch.dtype``, ...). ``np.dtype``
    cannot parse ``torch.complex64``, so compare via ``to_numpy(out).dtype``
    which is always a NumPy dtype.
    """
    actual = to_numpy(out).dtype
    assert actual == np.dtype(expected_np_dtype), (
        f"dtype mismatch: actual={actual!r} vs expected={np.dtype(expected_np_dtype)!r}"
    )


class _SparseDisallowedOps(sc.NumpyOps):
    """``NumpyOps`` with ``allow_sparse`` forced to ``False``.

    Used to exercise ``Context.assert_sparse``'s ``allow_sparse=False``
    refusal branch. None of the production backends advertise
    ``allow_sparse=False``, so a test-local subclass is the only way to
    pin that documented behavior.
    """

    _allow_sparse = False


# ===========================================================================
# 1. Construction
# ===========================================================================
class TestConstruction:
    def test_from_backend_ops_instance(self):
        ctx = sc.Context(sc.NumpyOps())
        assert isinstance(ctx.ops, sc.NumpyOps)
        assert ctx.ops.family == "numpy"

    def test_from_family_string_numpy(self):
        """``Context("numpy")`` resolves via ``normalize_ops``."""
        ctx = sc.Context("numpy")
        assert ctx.ops.family == "numpy"

    @pytest.mark.parametrize("family", ["jax", "torch", "cupy"])
    def test_from_family_string_optional(self, family):
        if not _OPTIONAL_BACKEND_PROBES[family]():
            pytest.skip(f"{family} is not installed")
        ctx = sc.Context(family)
        assert ctx.ops.family == family

    def test_from_family_dtype_pair_numpy(self):
        """``Context("numpy", dtype=...)`` accepts family + explicit dtype."""
        ctx = sc.Context("numpy", dtype=np.float32)
        assert ctx.ops.family == "numpy"
        assert ctx.dtype == np.dtype(np.float32)

    @pytest.mark.parametrize("family", ["jax", "torch", "cupy"])
    def test_from_family_dtype_pair_optional(self, family):
        if not _OPTIONAL_BACKEND_PROBES[family]():
            pytest.skip(f"{family} is not installed")
        dt = _supported_real_dtype(family)
        if dt is None:
            pytest.skip(f"{family} has no honored real dtype this session")
        ctx = sc.Context(family, dtype=dt)
        assert ctx.ops.family == family
        # The dtype is normalized through ops.sanitize_dtype, so compare via
        # ops.sanitize_dtype rather than direct numpy-dtype equality.
        assert ctx.dtype == sc.normalize_ops(family).sanitize_dtype(dt)

    def test_rejects_unknown_ops(self):
        with pytest.raises(TypeError):
            sc.Context(object())


# ===========================================================================
# 2. Equality and hash
# ===========================================================================
class TestEqualityAndHash:
    def test_equality_same_family_same_dtype(self):
        a = sc.Context(sc.NumpyOps(), dtype=np.float64)
        b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        assert a == b

    def test_inequality_different_dtype(self):
        a = sc.Context(sc.NumpyOps(), dtype=np.float32)
        b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        assert a != b

    def test_inequality_different_family(self):
        if not has_jax():
            pytest.skip("jax is not installed")
        # Use a dtype both backends honor this session.
        dt = np.float64 if backend_supports_dtype("jax", np.float64) else np.float32
        a = sc.Context(sc.NumpyOps(), dtype=dt)
        b = sc.Context(sc.JaxOps(), dtype=dt)
        assert a != b

    def test_inequality_different_check_level(self):
        a = sc.Context(sc.NumpyOps(), check_level="standard")
        b = sc.Context(sc.NumpyOps(), check_level="none")
        assert a != b

    def test_inequality_against_non_context(self):
        ctx = sc.Context(sc.NumpyOps())
        assert ctx != "Context"
        assert ctx != sc.NumpyOps()
        assert (ctx == 42) is False

    def test_hashable_and_dict_keyable(self):
        a = sc.Context(sc.NumpyOps(), dtype=np.float64)
        b = sc.Context(sc.NumpyOps(), dtype=np.float64)
        table = {a: "first"}
        table[b] = "second"
        assert table[a] == "second"
        assert len(table) == 1

    def test_is_frozen(self):
        ctx = sc.Context(sc.NumpyOps())
        with pytest.raises((AttributeError, Exception)):
            ctx.dtype = np.float32  # type: ignore[misc]


# ===========================================================================
# 3. Context.dtype defaulting per family
# ===========================================================================
class TestDtypeDefaulting:
    def test_default_dtype_matches_sanitize_dtype_none(self, backend_ops):
        """``Context(ops).dtype == ops.sanitize_dtype(None)`` on every backend."""
        expected = backend_ops.sanitize_dtype(None)
        ctx = sc.Context(backend_ops)
        assert ctx.dtype == expected

    def test_explicit_dtype_passes_through_sanitize(self, backend_ops):
        dt = _supported_real_dtype(backend_ops.family)
        if dt is None:
            pytest.skip(f"{backend_ops.family} has no honored real dtype this session")
        ctx = sc.Context(backend_ops, dtype=dt)
        assert ctx.dtype == backend_ops.sanitize_dtype(dt)


# ===========================================================================
# 4. ctx.asarray
# ===========================================================================
class TestAsarray:
    def test_creates_dense_array(self, backend_ops):
        dt = _supported_real_dtype(backend_ops.family)
        if dt is None:
            pytest.skip(f"{backend_ops.family} has no honored real dtype this session")
        ctx = sc.Context(backend_ops, dtype=dt)
        out = ctx.asarray(np.asarray([1.0, 2.0, 3.0], dtype=dt))
        assert backend_ops.is_dense(out)
        _assert_dtype(out, dt)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_passthrough_same_family_preserves_values(self, backend_ops, dtype):
        """``ctx.asarray(np_array)`` round-trips values losslessly."""
        if not backend_supports_dtype(backend_ops.family, dtype):
            pytest.skip(f"{backend_ops.family} does not honor {np.dtype(dtype)}")
        ctx = sc.Context(backend_ops, dtype=dtype)
        src = np.asarray([1.0, -2.0, 3.5], dtype=dtype)
        out = ctx.asarray(src)
        assert_matches_reference("asarray", out, src, dtype=dtype)

    def test_refuses_complex_to_real(self, backend_ops):
        """ADR-015 Stage 1: implicit complex→real narrowing raises on every backend."""
        real_dt = _supported_real_dtype(backend_ops.family)
        complex_dt = _supported_complex_dtype(backend_ops.family)
        if real_dt is None or complex_dt is None:
            pytest.skip(f"{backend_ops.family} has no honored real+complex pair this session")
        ctx = sc.Context(backend_ops, dtype=real_dt)
        z = np.asarray([1 + 2j, 3 - 4j], dtype=complex_dt)
        with pytest.raises(TypeError, match="complex"):
            ctx.asarray(z)

    @pytest.mark.parametrize("real_dt,complex_dt", [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
    ])
    def test_real_to_complex_broadening(self, backend_ops, real_dt, complex_dt):
        """Real → complex broadening is allowed and preserves values."""
        if not backend_supports_dtype(backend_ops.family, real_dt):
            pytest.skip(f"{backend_ops.family} does not honor {np.dtype(real_dt)}")
        if not backend_supports_dtype(backend_ops.family, complex_dt):
            pytest.skip(f"{backend_ops.family} does not honor {np.dtype(complex_dt)}")
        ctx = sc.Context(backend_ops, dtype=complex_dt)
        src = np.asarray([1.0, -2.0, 3.5], dtype=real_dt)
        out = ctx.asarray(src)
        _assert_dtype(out, complex_dt)
        assert_matches_reference("asarray", out, src.astype(complex_dt), dtype=complex_dt)

    @pytest.mark.parametrize("dtype",
                             [np.float32, np.float64, np.complex64, np.complex128])
    def test_dtype_passthrough(self, backend_ops, dtype):
        """``Context.asarray`` returns the configured dtype unchanged."""
        if not backend_supports_dtype(backend_ops.family, dtype):
            pytest.skip(f"{backend_ops.family} does not honor {np.dtype(dtype)}")
        if np.dtype(dtype).kind == "c":
            src = np.asarray([1 + 1j, 2 + 0j, 3 - 1j], dtype=dtype)
        else:
            src = np.asarray([1.0, 2.0, 3.0], dtype=dtype)
        ctx = sc.Context(backend_ops, dtype=dtype)
        out = ctx.asarray(src)
        _assert_dtype(out, dtype)


# ===========================================================================
# 4a. Cross-family conversion — parametrized over (src, dst) pairs
# ===========================================================================
def _family_pairs() -> list[tuple[str, str]]:
    families = _all_available_families()
    return [(s, d) for s in families for d in families if s != d]


class TestCrossFamilyConversion:
    @pytest.mark.parametrize("src,dst", _family_pairs())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_round_trip_via_numpy_intermediate(self, src, dst, dtype):
        """Each Context owns its own backend; cross-family transfer is
        explicit through ``to_numpy(...)`` and a destination ``asarray``.
        """
        if not backend_supports_dtype(src, dtype):
            pytest.skip(f"{src} does not honor {np.dtype(dtype)}")
        if not backend_supports_dtype(dst, dtype):
            pytest.skip(f"{dst} does not honor {np.dtype(dtype)}")
        ctx_src = sc.Context(src, dtype=dtype)
        ctx_dst = sc.Context(dst, dtype=dtype)
        src_np = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        arr_src = ctx_src.asarray(src_np)
        arr_dst = ctx_dst.asarray(to_numpy(arr_src))
        assert ctx_dst.ops.is_dense(arr_dst)
        arr_back = ctx_src.asarray(to_numpy(arr_dst))
        assert ctx_src.ops.is_dense(arr_back)
        assert_matches_reference("asarray", arr_back, src_np, dtype=dtype)


# ===========================================================================
# 5. ctx.assparse
# ===========================================================================
class TestAssparse:
    def test_supported_backend_round_trips_scipy_csr(self, backend_ops):
        """``ctx.assparse(scipy_csr)`` works on every sparse-capable backend."""
        if not backend_ops.allow_sparse:
            pytest.skip(f"{backend_ops.family} does not advertise allow_sparse")
        dt = _supported_real_dtype(backend_ops.family)
        if dt is None:
            pytest.skip(f"{backend_ops.family} has no honored real dtype this session")
        dense = np.eye(3, dtype=dt)
        csr = sps.csr_matrix(dense)
        ctx = sc.Context(backend_ops, dtype=dt)
        out = ctx.assparse(csr)
        assert backend_ops.is_sparse(out)
        dense_out = _densify(out)
        if dense_out is None:  # pragma: no cover - defensive guard
            pytest.skip(f"no densification path for {type(out).__name__}")
        np.testing.assert_allclose(dense_out, dense.astype(np.float64))

    def test_refused_when_allow_sparse_false(self):
        """``Context.assert_sparse`` raises when ``ops.allow_sparse`` is False.

        Exercised through a test-local ``NumpyOps`` subclass that flips
        ``_allow_sparse`` to ``False``. None of the production backends
        currently disallow sparse, but the gate exists and must stay
        wired.
        """
        ctx = sc.Context(_SparseDisallowedOps())
        sparse = sps.csr_matrix(np.eye(2))
        with pytest.raises(TypeError, match="disallowed"):
            ctx.assert_sparse(sparse)


# ===========================================================================
# 6. Repr stability
# ===========================================================================
class TestRepr:
    def test_repr_includes_family_dtype_check_level(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
        r = repr(ctx)
        assert "Context(" in r
        assert "NumpyOps" in r
        assert "check_level='standard'" in r

    def test_repr_is_deterministic(self):
        a = repr(sc.Context(sc.NumpyOps(), dtype=np.float64))
        b = repr(sc.Context(sc.NumpyOps(), dtype=np.float64))
        assert a == b


# ===========================================================================
# Additional Context-API coverage (not in the original checklist but part of
# the public ``Context`` surface)
# ===========================================================================


# ---- 7. assert_dense / assert_sparse gates --------------------------------
class TestAssertDenseAndSparse:
    def test_assert_dense_accepts_dense(self):
        ctx = sc.Context(sc.NumpyOps())
        x = ctx.asarray([1.0, 2.0])
        assert ctx.assert_dense(x) is x

    def test_assert_dense_rejects_non_array(self):
        ctx = sc.Context(sc.NumpyOps())
        with pytest.raises(TypeError):
            ctx.assert_dense([1.0, 2.0])

    def test_assert_sparse_accepts_sparse(self):
        ctx = sc.Context(sc.NumpyOps())
        sparse = ctx.assparse(sps.csr_matrix(np.eye(3)))
        assert ctx.assert_sparse(sparse) is sparse

    def test_assert_sparse_rejects_dense(self):
        ctx = sc.Context(sc.NumpyOps())
        x = ctx.asarray([1.0, 2.0])
        with pytest.raises(TypeError):
            ctx.assert_sparse(x)


# ---- 8. convert dispatch --------------------------------------------------
class TestConvertDispatch:
    def test_dense_invokes_asarray(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        x = ctx.asarray([1.0, 2.0])
        out = ctx.convert(x)
        assert ctx.ops.is_dense(out)

    def test_sparse_invokes_assparse(self):
        ctx = sc.Context(sc.NumpyOps())
        sparse = ctx.assparse(sps.csr_matrix(np.eye(3)))
        out = ctx.convert(sparse)
        assert ctx.ops.is_sparse(out)

    def test_rejects_unknown_type(self):
        ctx = sc.Context(sc.NumpyOps())
        with pytest.raises(NotImplementedError):
            ctx.convert([1.0, 2.0])


# ---- 9. check_level normalization + enable_checks deprecation alias -------
class TestCheckLevel:
    def test_default_check_level_is_standard(self):
        ctx = sc.Context(sc.NumpyOps())
        assert ctx.check_level == "standard"

    @pytest.mark.parametrize("level", ["none", "cheap", "standard", "strict"])
    def test_explicit_check_level(self, level):
        ctx = sc.Context(sc.NumpyOps(), check_level=level)
        assert ctx.check_level == level

    def test_enable_checks_true_maps_to_standard(self):
        with pytest.warns(DeprecationWarning):
            ctx = sc.Context(sc.NumpyOps(), enable_checks=True)
        assert ctx.check_level == "standard"

    def test_enable_checks_false_maps_to_none(self):
        with pytest.warns(DeprecationWarning):
            ctx = sc.Context(sc.NumpyOps(), enable_checks=False)
        assert ctx.check_level == "none"

    def test_enable_checks_property_is_legacy_view(self):
        assert sc.Context(sc.NumpyOps(), check_level="none").enable_checks is False
        assert sc.Context(sc.NumpyOps(), check_level="standard").enable_checks is True

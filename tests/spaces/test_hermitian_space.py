import importlib
import numpy as np
from tests._helpers import has_jax, jax_complex_dtype, prod


def test_hermitian_space_construction():
    sc = importlib.import_module("spacecore")
    H = sc.HermitianSpace(3, ctx=sc.Context(sc.NumpyOps(), dtype=np.complex128))
    assert H.shape == (3,3)
    assert prod(H.shape) == 9


def test_hermitian_check_and_symmetrize():
    sc = importlib.import_module("spacecore")
    import pytest
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128, enable_checks=True)
    H = sc.HermitianSpace(2, ctx=ctx)
    good = ctx.asarray([[1+0j,2-1j],[2+1j,3+0j]])
    bad = ctx.asarray([[1+0j,2+1j],[2+1j,3+0j]])
    H.check_member(good)
    with pytest.raises(Exception):
        H.check_member(bad)
    sym = H.symmetrize(bad)
    H.check_member(sym)


def test_hermitian_spectral_decompose_and_psd_proj():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    H = sc.HermitianSpace(2, ctx=ctx)
    x = ctx.asarray([[1+0j,0],[0,-2+0j]])
    evals, evecs = H.spectral_decompose(x)
    assert np.allclose(np.sort(np.asarray(evals)), [-2.,1.])
    y = H.psd_proj(H.symmetrize(x))
    ev = np.linalg.eigvalsh(np.asarray(y))
    assert np.min(ev) >= -1e-8


def test_hermitian_convert_uses_target_dtype():
    sc = importlib.import_module("spacecore")
    H = sc.HermitianSpace(2, ctx=sc.Context(sc.NumpyOps(), dtype=np.complex64))
    K = H.convert(sc.Context(sc.NumpyOps(), dtype=np.complex128))
    assert K.dtype == np.dtype(np.complex128)


def test_hermitian_convert_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_complex_dtype()
    H = sc.HermitianSpace(2, ctx=sc.Context(sc.NumpyOps(), dtype=dt))
    K = H.convert(sc.Context(sc.JaxOps(), dtype=dt))
    assert K.ctx.ops.family == "jax"

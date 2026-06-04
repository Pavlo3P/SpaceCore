import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, jax_complex_dtype, jax_real_dtype, to_numpy


def _real_contexts():
    sc = importlib.import_module("spacecore")
    yield sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    if has_jax():
        yield pytest.param(
            sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False),
            id="jax",
        )


def _complex_contexts():
    sc = importlib.import_module("spacecore")
    yield sc.Context(sc.NumpyOps(), dtype=np.complex128, enable_checks=False)
    if has_jax():
        yield pytest.param(
            sc.Context(sc.JaxOps(), dtype=jax_complex_dtype(), enable_checks=False),
            id="jax",
        )


def test_space_has_no_default_spectral_api():
    sc = importlib.import_module("spacecore")

    space = sc.Space(sc.Context(sc.NumpyOps(), dtype=np.float64))

    assert not hasattr(space, "spectrum")
    assert not hasattr(space, "spectral_decompose")
    assert not hasattr(space, "from_spectrum")


@pytest.mark.parametrize("ctx", list(_real_contexts()))
def test_vector_space_spectrum_and_roundtrip(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.ElementwiseJordanSpace((3,), ctx)
    x = ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    spectrum = space.spectrum(x)
    eigvals, frame = space.spectral_decompose(x)
    rebuilt = space.from_spectrum(eigvals, frame)

    assert tuple(spectrum.shape) == (2, 3)
    assert np.allclose(to_numpy(spectrum), to_numpy(x))
    assert np.allclose(to_numpy(eigvals), to_numpy(x))
    assert frame is None
    assert np.allclose(to_numpy(rebuilt), to_numpy(x))


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_hermitian_spectrum_roundtrip_and_batching(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.HermitianSpace(2, ctx=ctx)
    raw = ctx.asarray(
        [
            [
                [[1.0 + 0.0j, 2.0 - 1.0j], [0.5 + 3.0j, -1.0 + 0.0j]],
                [[3.0 + 0.0j, -1.0 + 0.25j], [4.0 - 2.0j, 2.0 + 0.0j]],
                [[2.0 + 0.0j, 1.0 + 1.5j], [-3.0 + 2.0j, 4.0 + 0.0j]],
            ],
            [
                [[-2.0 + 0.0j, 0.25 + 0.5j], [2.0 - 1.0j, 1.0 + 0.0j]],
                [[4.0 + 0.0j, -2.0 - 3.0j], [1.0 + 0.75j, 3.0 + 0.0j]],
                [[0.0 + 0.0j, 5.0 - 1.0j], [-1.0 + 3.0j, 6.0 + 0.0j]],
            ],
        ]
    )
    x = space.symmetrize(raw)

    spectrum = space.spectrum(x)
    expected = space.ops.eigh(x)[0]
    looped = space.ops.stack(
        [space.ops.stack([space.ops.eigh(x[i, j])[0] for j in range(3)], axis=0) for i in range(2)],
        axis=0,
    )
    eigvals, evecs = space.spectral_decompose(x)
    rebuilt = space.from_spectrum(eigvals, evecs)

    assert tuple(spectrum.shape) == (2, 3, 2)
    assert np.allclose(to_numpy(spectrum), to_numpy(expected), atol=1e-6)
    assert np.allclose(to_numpy(spectrum), to_numpy(looped), atol=1e-6)
    assert np.allclose(to_numpy(rebuilt), to_numpy(x), atol=1e-6)


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_hermitian_symmetrize_and_eig_to_dense_are_batch_safe(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.HermitianSpace(2, ctx=ctx)
    raw = ctx.asarray(
        [
            [[0.0 + 0.0j, 1.0 + 2.0j], [3.0 - 4.0j, 2.0 + 0.0j]],
            [[5.0 + 0.0j, -2.0 + 0.5j], [1.0 + 7.0j, -3.0 + 0.0j]],
        ]
    )
    x = space.symmetrize(raw)
    evals, evecs = space.spectral_decompose(x)
    rebuilt = space.eig_to_dense(evals, evecs)

    assert tuple(x.shape) == (2, 2, 2)
    assert np.allclose(to_numpy(x), np.conj(np.swapaxes(to_numpy(x), -1, -2)), atol=1e-6)
    assert np.allclose(to_numpy(rebuilt), to_numpy(x), atol=1e-6)


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_hermitian_eig_to_dense_matches_dense_reference(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.HermitianSpace(2, ctx=ctx)
    x = ctx.asarray(
        [
            [[2.0 + 0.0j, 1.0 - 0.5j], [1.0 + 0.5j, -1.0 + 0.0j]],
            [[3.0 + 0.0j, -2.0 + 1.0j], [-2.0 - 1.0j, 4.0 + 0.0j]],
        ]
    )
    evals, evecs = space.spectral_decompose(x)
    rebuilt = space.eig_to_dense(evals, evecs)
    ref = np.matmul(
        to_numpy(evecs) * to_numpy(evals)[..., None, :],
        np.conj(np.swapaxes(to_numpy(evecs), -1, -2)),
    )

    assert np.allclose(to_numpy(rebuilt), ref, atol=1e-6)


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_product_space_spectrum_concatenates_mixed_components(ctx):
    sc = importlib.import_module("spacecore")
    vector = sc.ElementwiseJordanSpace((2,), ctx)
    hermitian = sc.HermitianSpace(2, ctx=ctx)
    product = sc.ProductSpace((vector, hermitian), ctx)
    v = ctx.asarray([10.0 + 0.0j, 20.0 + 0.0j])
    h = ctx.asarray([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -2.0 + 0.0j]])
    x = (v, h)

    spectrum = product.spectrum(x)
    expected = np.concatenate([to_numpy(v).reshape(-1), np.linalg.eigvalsh(to_numpy(h))])
    decompositions = product.spectral_decompose(x)
    rebuilt = product.from_spectrum(decompositions)

    assert np.allclose(to_numpy(spectrum), expected, atol=1e-6)
    assert len(decompositions) == 2
    assert np.allclose(to_numpy(decompositions[0][0]), to_numpy(v), atol=1e-6)
    assert np.allclose(to_numpy(rebuilt[0]), to_numpy(v), atol=1e-6)
    assert np.allclose(to_numpy(rebuilt[1]), to_numpy(h), atol=1e-6)


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_product_space_batched_spectrum_concatenates_last_axis(ctx):
    sc = importlib.import_module("spacecore")
    vector = sc.ElementwiseJordanSpace((2,), ctx)
    hermitian = sc.HermitianSpace(2, ctx=ctx)
    product = sc.ProductSpace((vector, hermitian), ctx)
    v = ctx.asarray([[10.0 + 0.0j, 20.0 + 0.0j], [30.0 + 0.0j, 40.0 + 0.0j]])
    h = ctx.asarray(
        [
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -2.0 + 0.0j]],
            [[3.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 4.0 + 0.0j]],
        ]
    )

    spectrum = product.spectrum((v, h))
    expected = np.concatenate([to_numpy(v), np.linalg.eigvalsh(to_numpy(h))], axis=-1)

    assert tuple(spectrum.shape) == (2, 4)
    assert np.allclose(to_numpy(spectrum), expected, atol=1e-6)


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_spectrum_of_direct_sum_is_union_of_spectra(ctx):
    sc = importlib.import_module("spacecore")
    left = sc.HermitianSpace(2, ctx=ctx)
    right = sc.HermitianSpace(1, ctx=ctx)
    product = sc.ProductSpace((left, right), ctx)
    x = ctx.asarray([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 5.0 + 0.0j]])
    y = ctx.asarray([[7.0 + 0.0j]])

    expected = np.concatenate([to_numpy(left.spectrum(x)), to_numpy(right.spectrum(y))])
    assert np.allclose(to_numpy(product.spectrum((x, y))), expected, atol=1e-6)


@pytest.mark.parametrize("ctx", list(_complex_contexts()))
def test_spectral_scalar_uses_spectrum_and_element_op_uses_decomposition(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.HermitianSpace(2, ctx=ctx)
    x = ctx.asarray([[4.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 9.0 + 0.0j]])

    logdet = space.ops.sum(space.ops.log(space.spectrum(x)))
    eigvals, frame = space.spectral_decompose(x)
    clipped = space.from_spectrum(space.ops.maximum(eigvals, 5.0), frame)

    assert np.allclose(to_numpy(logdet), np.log(36.0), atol=1e-6)
    assert np.allclose(np.linalg.eigvalsh(to_numpy(clipped)), [5.0, 9.0], atol=1e-6)

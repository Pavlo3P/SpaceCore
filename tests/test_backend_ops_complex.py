import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, has_torch, jax_complex_dtype, to_numpy, torch_complex_dtype


def _check_vdot_conjugates_first_argument(ops, dtype):
    x = ops.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=dtype)
    y = ops.asarray([5.0 + 6.0j, 7.0 + 8.0j], dtype=dtype)

    np.testing.assert_allclose(to_numpy(ops.vdot(x, y)), 70.0 - 8.0j)


def test_numpy_vdot_conjugates_first_argument():
    sc = importlib.import_module("spacecore")

    _check_vdot_conjugates_first_argument(sc.NumpyOps(), np.complex128)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_vdot_conjugates_first_argument():
    sc = importlib.import_module("spacecore")

    _check_vdot_conjugates_first_argument(sc.JaxOps(), jax_complex_dtype())


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_vdot_conjugates_first_argument():
    sc = importlib.import_module("spacecore")

    _check_vdot_conjugates_first_argument(sc.TorchOps(), torch_complex_dtype())

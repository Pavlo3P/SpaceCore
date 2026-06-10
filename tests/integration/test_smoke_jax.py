import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, jax_real_dtype, to_numpy


pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


def test_smoke_jax_conversion_workflow():
    sc = importlib.import_module("spacecore")
    dtype = jax_real_dtype()
    np_ctx = sc.Context(sc.NumpyOps(), dtype=dtype)
    jx_ctx = sc.Context(sc.JaxOps(), dtype=dtype)
    x_space = sc.DenseCoordinateSpace((2,), np_ctx)
    y_space = sc.DenseCoordinateSpace((3,), np_ctx)
    op = sc.DenseLinOp(
        np_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        x_space,
        y_space,
        np_ctx,
    )

    converted = op.convert(jx_ctx)
    y = converted.apply(jx_ctx.asarray([7.0, 8.0]))

    assert np.allclose(to_numpy(y), [23.0, 53.0, 83.0])

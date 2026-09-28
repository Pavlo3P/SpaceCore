"""Native optional-backend types and portable interoperability."""
from typing import assert_type

import jax
import jax.numpy as jnp
import torch

import spacecore as sc
from spacecore.backend import JaxOps, TorchOps, CuPyOps
from spacecore.types import DenseArray


def portable(x: DenseArray) -> DenseArray:
    return x.reshape((3,))


portable(jnp.ones(3))
portable(torch.ones(3))
assert_type(sc.JaxOps(), JaxOps)
assert_type(sc.TorchOps(), TorchOps)
assert_type(sc.CuPyOps(), CuPyOps)

jops = JaxOps()
j = jops.asarray([1., 2., 3.])
assert_type(j, jax.Array)
assert_type(jops.sqrt(j), jax.Array)
assert_type(jops.index_set(j, 0, j), jax.Array)
assert_type(jops.index_add(j, 0, j), jax.Array)
assert_type(jops.stack([j, j]), jax.Array)
assert_type(sc.Context(jops).asarray([1.]), jax.Array)
j.block_until_ready()

tops = TorchOps()
t = tops.asarray([1., 2., 3.])
assert_type(t, torch.Tensor)
assert_type(tops.zeros((3,)), torch.Tensor)
assert_type(tops.ones((3,)), torch.Tensor)
assert_type(tops.stack([t, t]), torch.Tensor)
assert_type(tops.sqrt(t), torch.Tensor)
assert_type(tops.eigh(tops.eye(3)), tuple[torch.Tensor, torch.Tensor])
assert_type(sc.Context(tops).asarray([1.]), torch.Tensor)
t.detach().cpu().numpy()

backends: list[sc.BackendOps] = [sc.NumpyOps(), jops, tops, CuPyOps()]
for backend in backends:
    assert_type(backend.asarray([1.]), DenseArray)

j.detach()  # type: ignore
t.block_until_ready()  # type: ignore

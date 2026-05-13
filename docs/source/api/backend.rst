Backend API
===========

Backend classes define the execution layer used by spaces and linear
operators.

.. autosummary::
   :nosignatures:

   spacecore.backend.BackendOps
   spacecore.backend.NumpyOps
   spacecore.backend.JaxOps
   spacecore.backend.TorchOps

BackendOps
----------

.. autoclass:: spacecore.backend.BackendOps
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

NumpyOps
--------

.. autoclass:: spacecore.backend.NumpyOps
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: np, sp

JaxOps
------

.. autoclass:: spacecore.backend.JaxOps
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: jax, jnp, jsparse

TorchOps
--------

.. autoclass:: spacecore.backend.TorchOps
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: torch

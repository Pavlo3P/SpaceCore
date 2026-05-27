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

CuPyOps
-------

``CuPyOps`` is the optional CuPy backend implementation for GPU arrays and
``cupyx.scipy.sparse`` matrices. It is exported as ``spacecore.backend.CuPyOps``
only when CuPy is installed in the environment.

Install the optional backend before using it:

.. code-block:: bash

   pip install spacecore[cupy]

Use it through a normal SpaceCore context:

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.CuPyOps(), dtype=np.float64)
   x = ctx.asarray([1.0, 2.0, 3.0])

TorchOps
--------

.. autoclass:: spacecore.backend.TorchOps
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: torch

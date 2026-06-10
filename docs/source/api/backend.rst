Backend API
===========

Backend objects implement the numerical operations used by spaces, operators,
functionals, and solvers.

Backend families
----------------

.. autosummary::
   :nosignatures:

   spacecore.backend.BackendOps
   spacecore.backend.NumpyOps
   spacecore.backend.JaxOps
   spacecore.backend.TorchOps

* ``BackendOps`` is the abstract operation surface.
* ``NumpyOps`` is always available and uses NumPy plus SciPy sparse.
* ``JaxOps`` is optional and follows JAX tracing and dtype configuration.
* ``TorchOps`` is optional and follows PyTorch tensor, dtype, device, and autograd semantics.
* ``CuPyOps`` is optional and follows CuPy/CuPy sparse semantics.

Choosing a backend
------------------

Users normally choose a backend by constructing a ``Context``:

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

Backend portability means SpaceCore calls the same abstract operations and keeps
the same space/operator model. It does not hide backend-specific dtype defaults,
optional dependency availability, device placement, sparse support, tracing, or
autograd behavior.

Autodoc
-------

.. autoclass:: spacecore.backend.BackendOps
   :members:

.. autoclass:: spacecore.backend.NumpyOps
   :members:
   :inherited-members:

.. autoclass:: spacecore.backend.JaxOps
   :members:
   :inherited-members:

.. autoclass:: spacecore.backend.TorchOps
   :members:
   :inherited-members:


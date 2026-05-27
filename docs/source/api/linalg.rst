Linear algebra API
==================

Linear algebra routines solve systems, estimate eigenpairs, and apply matrix
functions through :class:`~spacecore.linop.LinOp` objects. They use
space-aware vector operations and avoid materializing dense operators unless a
method explicitly projects to a small Krylov subspace.

.. autosummary::
   :nosignatures:

   spacecore.linalg.cg
   spacecore.linalg.lsqr
   spacecore.linalg.lanczos_smallest
   spacecore.linalg.stochastic_lanczos
   spacecore.linalg.power_iteration
   spacecore.linalg.expm_multiply
   spacecore.linalg.CGResult
   spacecore.linalg.LSQRResult
   spacecore.linalg.LanczosResult
   spacecore.linalg.StochasticLanczosResult
   spacecore.linalg.PowerIterationResult
   spacecore.linalg.ExpmMultiplyResult

Solvers
-------

.. autofunction:: spacecore.linalg.cg

.. autoclass:: spacecore.linalg.CGResult
   :members:
   :undoc-members:

.. autofunction:: spacecore.linalg.lsqr

.. autoclass:: spacecore.linalg.LSQRResult
   :members:
   :undoc-members:

Eigenvalue algorithms
---------------------

.. autofunction:: spacecore.linalg.lanczos_smallest

.. autofunction:: spacecore.linalg.stochastic_lanczos

.. autoclass:: spacecore.linalg.LanczosResult
   :members:
   :undoc-members:

.. autoclass:: spacecore.linalg.StochasticLanczosResult
   :members:
   :undoc-members:

.. autofunction:: spacecore.linalg.power_iteration

.. autoclass:: spacecore.linalg.PowerIterationResult
   :members:
   :undoc-members:

Matrix functions
----------------

.. autofunction:: spacecore.linalg.expm_multiply

.. autoclass:: spacecore.linalg.ExpmMultiplyResult
   :members:
   :undoc-members:

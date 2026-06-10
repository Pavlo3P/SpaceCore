Linear Algebra API
==================

Linear algebra routines operate on SpaceCore spaces and operators. Preconditions
are mathematical: square, Hermitian, positive-definite, and residual statements
refer to the domain and codomain inner products, not only coordinate arrays.

Linear solves
-------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.cg
   spacecore.linalg.CGResult

* ``cg`` solves ``A x = b`` for Hermitian positive-definite ``A : X -> X`` with residuals measured in ``X.norm``.
* ``CGResult`` stores ``x``, convergence flag, iteration count, residual norm, and status details.

Least squares
-------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.lsqr
   spacecore.linalg.LSQRResult

* ``lsqr`` solves least-squares problems for ``A : X -> Y`` using ``X.inner`` and ``Y.inner``.
* ``LSQRResult`` stores the solution, convergence data, and residual diagnostics.

Eigenvalue and spectral methods
-------------------------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.lanczos_smallest
   spacecore.linalg.power_iteration
   spacecore.linalg.LanczosResult
   spacecore.linalg.PowerIterationResult

* ``lanczos_smallest`` approximates the smallest eigenpair of a Hermitian ``A : X -> X``.
* ``power_iteration`` estimates the dominant eigenpair of a self-adjoint action or quadratic-form Hessian.

Matrix functions
----------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.expm_multiply
   spacecore.linalg.ExpmMultiplyResult

* ``expm_multiply`` approximates ``exp(t A) v`` for square Hermitian ``A : X -> X`` using Krylov projection.

Autodoc
-------

.. autofunction:: spacecore.linalg.cg
.. autoclass:: spacecore.linalg.CGResult
   :members:

.. autofunction:: spacecore.linalg.lsqr
.. autoclass:: spacecore.linalg.LSQRResult
   :members:

.. autofunction:: spacecore.linalg.lanczos_smallest
.. autoclass:: spacecore.linalg.LanczosResult
   :members:

.. autofunction:: spacecore.linalg.power_iteration
.. autoclass:: spacecore.linalg.PowerIterationResult
   :members:

.. autofunction:: spacecore.linalg.expm_multiply
.. autoclass:: spacecore.linalg.ExpmMultiplyResult
   :members:

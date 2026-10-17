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
   spacecore.linalg.LinalgResult

* ``LinalgResult`` is returned by every solver below: ``x``, ``converged``, ``num_iters`` and
  ``residual_norm``, plus ``eigenvalue`` or ``normal_residual_norm`` where the solver produces them.

Least squares
-------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.lsqr

Eigenvalue and spectral methods
-------------------------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.lanczos_smallest
   spacecore.linalg.power_iteration

Matrix functions
----------------

.. autosummary::
   :nosignatures:

   spacecore.linalg.expm_multiply

Native elements and JAX
-----------------------

``lanczos_smallest`` and ``expm_multiply`` retain native space elements and use
``zeros``, ``add``, ``scale``, and ``inner`` together with the operator action.
They do not require a coordinate shape or ``flatten``/``unflatten`` methods.
Built-in dense spaces retain their vectorized basis operations. Custom elements
are held directly during eager execution; on JAX they must be registered pytrees
with fixed structure, leaf shapes, and dtypes throughout the recurrence, including
``space.zeros()``. Basis capacity is fixed by the static ``max_iter`` argument.
Mathematical linear combinations always use space algebra, even when the
representation has multiple array leaves.

Lanczos requires a nonzero starting element. A zero start raises ``ValueError``
eagerly. Under JAX tracing, it returns a NaN eigenvalue, a zero eigenvector,
``num_iters=0``, and ``converged=False``. This replaces the previous coordinate
vector fallback. ``expm_multiply`` accepts zero and returns zero with
``num_iters=0`` and a zero ``residual_norm``.

Autodoc
-------

.. autofunction:: spacecore.linalg.cg
.. autoclass:: spacecore.linalg.LinalgResult
   :members:

.. autofunction:: spacecore.linalg.lsqr

.. autofunction:: spacecore.linalg.lanczos_smallest

.. autofunction:: spacecore.linalg.power_iteration

.. autofunction:: spacecore.linalg.expm_multiply

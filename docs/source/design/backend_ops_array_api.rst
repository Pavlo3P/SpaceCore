BackendOps Array API substrate
==============================

Call ``BackendOps`` methods (``ops.matmul``, ``ops.reshape``, ``ops.eigh``);
``ops.xp``, the backend's Array API namespace, is how those methods are
implemented and is an escape hatch, not the portable API.

``ops.xp`` is ``array_api_compat.numpy`` / ``.torch`` / ``.cupy`` for NumPy,
Torch, CuPy and ``jax.numpy`` for JAX. Library modules are also exposed
directly: ``ops.np`` / ``ops.sp`` (NumPy and SciPy; on CuPy, ``cupy`` and
``cupyx.scipy``, plus ``ops.cp``), ``ops.jax`` / ``ops.jnp`` /
``ops.jsparse``, ``ops.torch``.

Where each method is implemented
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Implementation
     - Methods
   * - Once in ``BackendOps``, through ``xp``
     - ``reshape``, ``broadcast_to``, ``stack``, ``conj``, ``real``,
       ``imag``, ``abs``, ``sign``, ``sqrt``, ``exp``, ``log``, ``where``,
       ``isfinite``, ``isnan``, ``sum``, ``mean``, ``min``, ``max``,
       ``prod``, ``sort``, ``argsort``, ``argmin``, ``argmax``, ``diag``,
       ``diagonal``, ``tril``, ``triu``, ``take``
   * - Once in ``BackendOps``, through an Array API name or ``xp.linalg``
     - ``transpose`` (``permute_dims``), ``concatenate`` (``concat``),
       ``eigh``, ``eigvalsh``, ``solve``, ``svd``, ``cholesky``, ``norm``
   * - Composed from other ops
     - ``trace`` = ``sum(diagonal(x), axis=-1)``: trailing two axes, batched
   * - Per backend (abstract in ``BackendOps``)
     - ``dense_array``, ``sparse_array``, ``sanitize_dtype``, ``assparse``,
       ``sparse_matmul``, ``allclose_sparse``, ``logsumexp``, ``ix_``,
       ``fori_loop``, ``while_loop``, ``scan``, ``cond``
   * - Per backend where semantics differ
     - ``index_set`` / ``index_add`` (in-place default; JAX overrides),
       ``vmap`` (loop default; JAX and Torch native)

``TorchOps`` overrides several ``xp``-based methods for Torch semantics,
among them ``matmul``, ``where``, ``concatenate``, ``diagonal``, ``sum``,
``eigh``, ``solve``, ``svd``, ``cholesky``, ``norm``.

Backend tuning arguments
------------------------

Mathematical options are explicit arguments. Backend tuning goes in a
``backend_kwargs`` dict, accepted by ``matmul``, ``eigh``, ``eigvalsh``,
``solve``, ``svd``, ``cholesky`` and forwarded to the library call
(``asarray`` and ``astype`` forward extra keyword arguments the same way).
Anything else uses ``ops.xp`` or the library handles.

.. code-block:: python

   import numpy as np
   from spacecore import NumpyOps

   ops = NumpyOps()
   A = ops.asarray([[2.0, 1.0], [1.0, 3.0]])

   w, V = ops.eigh(A)                                    # portable
   L = ops.cholesky(A, backend_kwargs={"upper": False})  # forwarded to xp.linalg
   X = ops.xp.linalg.matrix_power(A, 3)                  # escape hatch, not portable
   assert np.allclose(L @ L.T, A) and np.allclose(X, A @ A @ A)

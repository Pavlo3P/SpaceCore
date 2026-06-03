BackendOps Array API substrate
==============================

``BackendOps`` remains SpaceCore's declared numerical API. Users and internals
should continue to call explicit methods such as ``ops.matmul``,
``ops.reshape``, and ``ops.eigh`` rather than reaching through to backend
libraries directly.

Concrete backends expose an Array API compatible ``xp`` namespace as the shared
implementation substrate for common dense-array operations. NumPy and PyTorch
use ``array-api-compat`` wrappers; JAX uses ``jax.numpy``, which already exposes
the Array API spellings SpaceCore relies on. ``xp`` is intentionally an escape
hatch for advanced backend-specific code; it is not the preferred portable
SpaceCore API.

Audit categories
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Category
     - Methods
     - Notes
   * - Direct ``xp`` delegation
     - ``reshape``, ``broadcast_to``, ``stack``, ``conj``, ``real``, ``imag``,
       ``abs``, ``sign``, ``sqrt``, ``exp``, ``log``, ``where``, ``isfinite``,
       ``isnan``
     - Implemented once in ``BackendOps``.
   * - Delegation with name or namespace adaptation
     - ``transpose``, ``concatenate``, ``trace``, ``take``, ``eigh``,
       ``eigvalsh``, ``solve``, ``svd``, ``cholesky``, ``norm``
     - SpaceCore keeps existing method names while adapting to names such as
       ``concat`` or backend ``linalg`` namespaces where needed.
   * - Common Array API delegation
     - ``sum``, ``mean``, ``min``, ``max``, ``prod``, ``sort``, ``argsort``,
       ``argmin``, ``argmax``, ``diag``, ``diagonal``, ``tril``, ``triu``
     - The base method uses Array API spellings such as ``axis``,
       ``keepdims``, ``concat``, and ``permute_dims``.
   * - Deliberately backend-specific
     - ``sanitize_dtype``, ``assparse``, ``sparse_matmul``, ``logsumexp``,
       ``index_set``, ``index_add``, ``ix_``, ``fori_loop``, ``while_loop``,
       ``scan``, ``cond``, ``allclose_sparse``, sparse/dense type predicates
     - These operations have sparse formats, mutation semantics, tracing rules,
       or dtype/device behavior that is not captured by the common dense-array
       namespace.

Backend tuning arguments
------------------------

Portable mathematical options stay as explicit method arguments. Backend
tuning should be passed only to selected heavier methods, currently through a
``backend_kwargs`` dictionary on operations such as ``matmul``, ``eigh``,
``solve``, ``eigvalsh``, ``svd``, and ``cholesky``. Exotic backend features
belong behind ``ops.xp`` or the backend-specific handles such as ``ops.np`` and
``ops.jax``.

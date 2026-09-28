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

Array types and editor hints
----------------------------

Concrete backend operations retain their native dense array type. The same is
true for an explicitly constructed context:

.. code-block:: python

   ops = sc.NumpyOps()
   x = ops.asarray([1.0, 2.0])       # numpy.typing.NDArray[Any]
   y = ops.sqrt(x)                  # numpy.typing.NDArray[Any]
   ctx = sc.Context(ops)
   z = ctx.asarray([3.0, 4.0])      # numpy.typing.NDArray[Any]
   z.astype(np.float32)             # native NumPy completions

``JaxOps`` yields ``jax.Array`` and ``TorchOps`` yields ``torch.Tensor``.
Install the corresponding backend in the editor's Python environment to get
its native completions. SpaceCore ships its typing metadata in the wheel.

For portable code, use ``BackendOps`` and ``DenseArray``. A heterogeneous
``list[BackendOps]`` exposes the common array protocol. A context resolved from
a string or ambient state also exposes this common protocol: a registry entry
can change at runtime, so its native type cannot be inferred safely from its
name. Space and operator APIs continue to expose their portable representations.
CuPy and sparse results retain their existing portable protocols.

These annotations track the array family, not exact shape, dtype, device, sparse
layout, or gradient state. Runtime space checks still validate those properties.
``spacecore.ArrayLike`` means an object with array metadata; unlike NumPy's
``ArrayLike``, it does not include Python lists. Constructors such as
``ops.asarray`` accept lists separately.

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


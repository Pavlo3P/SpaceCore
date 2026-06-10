Conversion Policy
=================

Context-bound objects expose ``convert(new_ctx)``. In ``0.3.x`` conversion is
explicit and target-context driven: the requested context controls backend,
dtype, and ``enable_checks``.

What moves
----------

Spaces are reconstructed in the target context. Shape, structure, tolerances,
and geometry definitions are preserved. Geometry data that stores arrays, such
as ``WeightedInnerProduct.weights``, is converted with ``target.asarray``.

Operators are reconstructed in the target context when they implement
``_convert``. Matrix-backed operators convert stored dense or sparse matrices.
Algebraic operators convert their operands. Matrix-free operators preserve their
Python callables and rebuild only their domain, codomain, and context; the
callables must already be valid for the target backend.

What may fail
-------------

Conversion may fail when the target backend lacks a required sparse format,
when dtype conversion is unsupported, when backend-specific callables cannot run
on the target arrays, or when structural validation rejects the converted data.
Optional backends are available only when their packages are installed.

Backend differences
-------------------

NumPy conversion creates NumPy arrays and SciPy sparse objects. JAX conversion
creates JAX arrays and follows JAX dtype configuration, including
``jax_enable_x64``. Torch conversion creates tensors and follows PyTorch dtype
and device semantics; SpaceCore does not silently move tensors across devices.
CuPy conversion creates CuPy arrays and CuPy sparse objects when CuPy is
installed.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   src = sc.Context(sc.NumpyOps(), dtype=np.float32)
   dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), src)
   Y = X.convert(dst)
   print(Y.dtype == np.dtype("float64"))

Expected output:

.. code-block:: text

   True

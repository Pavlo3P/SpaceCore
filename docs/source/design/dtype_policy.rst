Dtype Behavior
==============

Each ``Context`` stores a backend-normalized ``dtype``. Array construction
through ``ctx.asarray(...)`` and ``ctx.assparse(...)`` uses that dtype unless a
method explicitly receives another dtype.

Dtype Resolution
----------------

When a constructor infers several compatible contexts, SpaceCore joins their
dtypes using backend-compatible promotion and then sanitizes the result for the
selected backend.

.. code-block:: python

   import spacecore as sc

   X = sc.DenseCoordinateSpace((3,), ctx=sc.Context(sc.NumpyOps(), dtype="float32"))
   Y = sc.DenseCoordinateSpace((3,), ctx=sc.Context(sc.NumpyOps(), dtype="float64"))

   Z = sc.ProductSpace((X, Y))

The product space resolves to one backend family and a dtype capable of
representing both inputs.

Explicit Conversion
-------------------

During explicit conversion, the target context dtype wins:

.. code-block:: python

   target = sc.Context(sc.NumpyOps(), dtype="float64", enable_checks=True)
   converted = obj.convert(target)

The converted object uses ``target.dtype``. If exact dtype matters, make it
part of the ``Context`` you construct.

Backend Defaults
----------------

Every ``BackendOps`` implementation defines dtype normalization through
``sanitize_dtype(dtype)``. That method normalizes dtype objects into
backend-native dtype representations and determines the backend default when
``dtype=None``.

For ``NumpyOps``, ``sanitize_dtype(None)`` currently resolves to
``numpy.float64``. For ``JaxOps``, the default depends on JAX configuration: if
``jax_enable_x64=True`` the default is ``float64``; otherwise the default is
``float32``.

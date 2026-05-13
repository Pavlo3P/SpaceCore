Dtype policy
============

Each ``Context`` stores a backend-normalized ``dtype``. Array construction
through ``ctx.asarray(...)`` and ``ctx.assparse(...)`` uses that dtype.

Dtype resolution is a separate concern from context resolution. Context
resolution determines which context will be assigned to a new object. Once that
context is fixed, input objects carrying their own contexts may need to be
converted to it. During that conversion, the backend is determined by the
resolved context, but dtype handling is controlled by the dtype policy.

For example, if an input object has dtype ``float32`` and is converted to a
context on another backend, SpaceCore must decide whether to preserve the
original dtype or cast the object to the dtype supplied by the resolved context.
Conceptually, the dtype policy controls whether conversion prioritizes dtype
preservation or dtype unification.

Dtype resolution
----------------

When a constructor infers several compatible contexts, SpaceCore joins their
dtypes using NumPy result-type promotion and then sanitizes the result for the
selected backend.

.. code-block:: python

   import spacecore as sc

   X = sc.VectorSpace((3,), ctx=sc.Context(sc.NumpyOps(), dtype="float32"))
   Y = sc.VectorSpace((3,), ctx=sc.Context(sc.NumpyOps(), dtype="float64"))

   Z = sc.ProductSpace((X, Y))

The product space resolves to one backend family and a dtype capable of
representing both inputs.

Preserve or convert
-------------------

The ``dtype_resolution_policy`` regulates how dtype is chosen when a new context
is normalized relative to an existing one.

Use ``spacecore.set_dtype_resolution_policy(...)`` to set it and
``spacecore.get_dtype_resolution_policy()`` to inspect the active value.

.. dropdown:: ``dtype_resolution_policy`` types

   * ``convert``: when an object is converted to the resolved context, use the
     dtype that the resolved context provides. This prioritizes unifying values
     under the target context.
   * ``keep_native``: when an object is converted to another backend, preserve
     the object's dtype by converting it to the equivalent dtype in the target
     backend. This prioritizes preserving the source object's numerical dtype.

The default is ``keep_native``.

.. dropdown:: Dtype policy checkpoints

   * Context creation sanitizes dtype through the selected backend.
   * Array construction uses the context dtype.
   * Multi-input constructors join compatible inferred dtypes.
   * Conversion either preserves the native dtype or follows target context
     resolution, depending on policy.

Use an explicit ``Context`` when dtype is part of the numerical contract of an
algorithm.

Backend defaults
----------------

Every ``BackendOps`` implementation defines its own dtype normalization rule
through ``sanitize_dtype(dtype)``. That method both normalizes dtype objects
into backend-native dtype representations and determines the backend default
when ``dtype=None``.

For ``NumpyOps``, ``sanitize_dtype(None)`` currently resolves to
``numpy.float64``. For ``JaxOps``, the default depends on JAX configuration: if
``jax_enable_x64=True`` the default is ``float64``; otherwise the default is
``float32``.

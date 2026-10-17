Context, dtype and conversion
=============================

What a context owns
-------------------

A ``Context`` owns two pieces of execution policy:

* ``ops`` - a ``BackendOps`` instance such as ``NumpyOps``;
* ``dtype`` - the backend-normalized default representation dtype.

It does not own array memory, devices, gradient state, or sparse storage; those
belong to the backend arrays. It does not own the validation level either:
``check_level`` is a property of each space, operator and functional, set at
construction or taken from the ambient level (:doc:`checking_policy`).

Normalization
-------------

``spacecore.normalize_context(ctx)`` accepts a concrete ``Context``, backend
name, backend family, or ``None``. Concrete contexts are copied and their dtype
is sanitized through ``ops.sanitize_dtype``. Backend names such as ``"numpy"``
and ``"torch"`` create a new context with that backend. ``None`` returns the
global default context. Constructors that receive several context-bearing inputs
resolve one backend family and one dtype through backend promotion; an explicit
``ctx=...`` takes priority over inference.

Dtype and scalar field
----------------------

The context dtype is the default array representation, not the mathematical
scalar field. ``Context.asarray`` and ``Context.assparse`` use it unless given an
explicit dtype. Spaces use ``space.dtype`` for zeros, ones, unflattened arrays
and exact representation checks; operators use their context dtype for stored
matrices and scalar helper values.

Every public space exposes ``space.field`` as ``"real"`` or ``"complex"``,
derived from the dtype: ``float32``/``float64`` give spaces over
:math:`\mathbb{R}`, ``complex64``/``complex128`` spaces over :math:`\mathbb{C}`.
There is no constructor-level field override; the field changes only when the
representation dtype does.

Membership checks separate the two questions. ``FieldCheck`` checks real versus
complex compatibility. ``DTypeCheck`` requires exact dtype equality, including
precision, so real data passes the field check of a complex space but fails its
representation check until it is represented in the space's complex dtype.

Geometry can restrict the field further. ``WeightedInnerProduct`` requires real,
positive, finite weights even in a complex coordinate space.
``EuclideanElementwiseJordanSpace`` requires a real scalar field and Euclidean
geometry; complex elementwise Jordan spaces use ``ElementwiseJordanSpace``.

Dense and sparse conversion never narrow complex to real silently: a complex
representation is rejected for a non-complex target dtype even when its imaginary
entries are zero. Discarding the imaginary part is a named user action, such as
passing ``x.real``. NumPy, JAX, Torch and CuPy all run this guard before their
backend conversion call.

Solvers use the operator domain and codomain inner products: Hermitian and
positive-definite assumptions are with respect to those inner products, not the
coordinate matrix. Complex problems need complex-compatible dtypes.

Where conversion happens
------------------------

User arrays passed to operations such as ``space.add`` or ``A.apply`` are not
converted. The object's check level decides how much of backend, shape, dtype,
recursive membership and numerical validity is checked; under ``"none"`` values
are forwarded and backend errors may surface later.

SpaceCore converts only in constructors and on request: ``ctx.asarray(...)``
creates dense arrays, ``ctx.assparse(...)`` sparse arrays where supported, and
``obj.convert(new_ctx)`` rebuilds a context-bound object under a target context.

``convert(new_ctx)`` takes backend and dtype from the target context; the object
keeps its own ``check_level``.

* Spaces are reconstructed in the target context with their shape, structure,
  tolerances and geometry; geometry data that stores arrays, such as
  ``WeightedInnerProduct.weights``, converts with ``target.asarray``.
* Operators, functionals, spaces and sets are rebuilt from their declared
  constructor fields (:doc:`extending`): the context is replaced, bound objects
  convert recursively and dense arrays move with ``target.asarray``. Classes with
  other needs override ``_convert``; sparse matrices convert with ``assparse``.
* Algebraic operators convert their operands. Matrix-free operators keep their
  Python callables and rebuild only their domain, codomain and context, so the
  callables must already work on the target backend.

Conversion may fail when the target backend lacks a required sparse format, when
a dtype conversion is unsupported, when backend-specific callables cannot run on
the target arrays, or when structural validation rejects the converted data.
NumPy conversion creates NumPy arrays and SciPy sparse objects; JAX follows JAX
dtype configuration, including ``jax_enable_x64``; Torch follows PyTorch dtype and
device semantics, and SpaceCore never moves tensors across devices; CuPy creates
CuPy arrays and sparse objects. Optional backends are available only when their
packages are installed.

Guarantees
----------

A constructed space or operator stores one normalized ``Context`` and uses it for
its own helper methods. It does not guarantee that a future user-supplied array
has that context unless its check level validates it or the array was created
through the same context.

Errors
------

Invalid representation or field usage usually raises ``TypeError`` from
membership checks, dtype conversion or sparse conversion. Invalid Euclidean-Jordan
field usage raises ``ValueError``.

Examples
--------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   src = sc.Context(sc.NumpyOps(), dtype=np.float32)
   dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), src, check_level="cheap")
   Y = X.convert(dst)
   print(Y.dtype == np.dtype("float64"), Y.check_level)

   try:
       sc.EuclideanElementwiseJordanSpace((2,), sc.Context(sc.NumpyOps(), dtype=np.complex128))
   except ValueError as exc:
       print(exc)

Expected output:

.. code-block:: text

   True cheap
   EuclideanElementwiseJordanSpace requires a real scalar field.

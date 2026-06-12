Dtype and Scalar-Field Behavior
===============================

Each ``Context`` stores a backend-normalized ``dtype``. This is the default
array representation dtype, not the mathematical scalar field. ``Context.asarray``
and ``Context.assparse`` use that dtype unless a backend method receives an
explicit dtype argument.

Stored and inferred dtype
-------------------------

A concrete ``Context`` sanitizes its dtype through ``ops.sanitize_dtype``.
Constructors that receive several context-bearing inputs resolve one backend
family and one dtype through backend promotion. Explicit ``ctx=...`` takes
priority over inference.

Spaces use ``space.dtype`` for zeros, ones, unflattened arrays, and exact
representation checks. Operators use their context dtype for stored matrices
and scalar helper values.

Scalar field
------------

Every public space exposes ``space.field`` as either ``"real"`` or
``"complex"``. The field is derived from the normalized context dtype:

* ``float32`` and ``float64`` represent spaces over :math:`\mathbb{R}`;
* ``complex64`` and ``complex128`` represent spaces over :math:`\mathbb{C}`.

There is no constructor-level field override in 0.4.0. A space's field changes
only when its context representation dtype changes. This keeps the Stage 1
contract explicit while exact dtype membership remains strict.

Membership checks separate the two questions. ``FieldCheck`` checks real versus
complex mathematical compatibility. ``DTypeCheck`` still requires exact dtype
equality, including precision, under the 0.4.0 Stage 1 policy. Real-valued data
is mathematically compatible with a complex field, but it still fails the exact
representation check until represented with the space's complex dtype.

Real and complex spaces
-----------------------

Most coordinate spaces accept real or complex fields. Geometry can add extra
restrictions. ``WeightedInnerProduct`` requires real, positive, finite weights
even in a complex coordinate space. ``EuclideanElementwiseJordanSpace``
requires a real scalar field and Euclidean geometry; complex elementwise Jordan
spaces use ``ElementwiseJordanSpace`` instead.

No silent complex-to-real narrowing
-----------------------------------

Dense and sparse conversion rejects a complex representation when the target
dtype is non-complex, even if the current imaginary entries happen to be zero.
Discarding the imaginary part must be a named user action, such as passing
``x.real`` or the result of a backend real-part operation before conversion.
NumPy, JAX, Torch, and optional CuPy use the same SpaceCore guard before their
backend conversion call.

Linear algebra assumptions
--------------------------

Solvers use the operator domain and codomain inner products. Hermitian and
positive-definite assumptions are with respect to those inner products, not just
the coordinate matrix. Complex problems require complex-compatible dtypes when
inputs or scalars are complex.

Expected errors
---------------

Invalid representation or field usage usually appears as ``TypeError`` from
membership checks, backend dtype conversion, or sparse conversion. Invalid
Euclidean-Jordan field usage raises ``ValueError``.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
   try:
       sc.EuclideanElementwiseJordanSpace((2,), sc.Context(sc.NumpyOps(), dtype=np.complex128))
   except ValueError as exc:
       print(exc)

Expected output:

.. code-block:: text

   EuclideanElementwiseJordanSpace requires a real scalar field.

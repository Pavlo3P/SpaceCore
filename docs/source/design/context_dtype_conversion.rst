Context, dtype and conversion
=============================

A ``Context`` is a frozen pair ``(ops, dtype)``: the backend operations and the
dtype of the arrays it creates. Arrays, devices, autograd state and
``check_level`` are not part of it; the check level belongs to each object
(:doc:`checking_policy`).

Normalization
-------------

``sc.normalize_context(ctx, dtype=None)`` turns any context specification into
a ``Context``:

====================================  ======================================================
``ctx``                               result
====================================  ======================================================
a ``Context``                         a copy, dtype re-sanitized by ``ops.sanitize_dtype``
a backend name or ``BackendFamily``   a new context for that backend with ``dtype``
``None``                              the active default context
====================================  ======================================================

``dtype`` is ignored, with a warning, for a ``Context`` or ``None``.

A constructor with several context-bearing inputs uses an explicit ``ctx=`` when
given. Otherwise the inputs must share one backend family (else ``ValueError``)
and their dtypes are promoted by that backend's ``result_type``.

Dtype and scalar field
----------------------

The dtype is the representation; ``space.field`` (``"real"`` or ``"complex"``)
is derived from it and has no separate override.

* ``ctx.asarray`` and ``ctx.assparse`` always produce ``ctx.dtype``.
* ``float32``/``float64`` give a real space, ``complex64``/``complex128`` a
  complex one.
* ``FieldCheck`` tests real/complex compatibility; ``DTypeCheck`` requires the
  exact dtype, precision included. Real data passes the field check of a complex
  space and fails its dtype check.
* Conversion never narrows complex to real: converting a complex array to a real
  dtype raises ``TypeError`` even when every imaginary part is zero. Pass
  ``x.real`` to discard it.
* ``WeightedInnerProduct`` weights must be real, finite and positive, also in a
  complex space (``ValueError``, checked from ``standard``).
* ``EuclideanElementwiseJordanSpace`` requires a real field (``ValueError``); use
  ``ElementwiseJordanSpace`` for complex data.

Membership failures raise ``SpaceValidationError``, a subclass of both
``TypeError`` and ``ValueError``.

Where conversion happens
------------------------

SpaceCore converts arrays only in constructors and on request; operations such
as ``space.add`` or ``A.apply`` never convert their arguments. The object's check
level decides how much of an argument is validated; under ``"none"`` it is
forwarded unchanged and a backend error may surface later.

* ``ctx.asarray(x)``, ``ctx.assparse(x)``: a dense or sparse array in ``ctx``.
* ``obj.convert(new_ctx)``: the object rebuilt in ``new_ctx``. It returns ``obj``
  when the context is unchanged and keeps ``obj.check_level``.

How ``convert`` rebuilds an object:

* By default from its declared ``_fields`` (:doc:`extending`): ``ctx`` is
  replaced, fields with a ``convert`` method (spaces, operators, geometries such
  as ``WeightedInnerProduct``) convert recursively, dense arrays go through
  ``new_ctx.asarray``. A class with other needs overrides ``_convert``, as sparse
  operators do with ``assparse``.
* ``MatrixFreeLinOp`` keeps its callables and converts only its spaces and
  context, so the callables must already run on the target backend.

Conversion fails when the target backend lacks a needed sparse format or dtype,
when user callables cannot run on the target arrays, or when the converted data
fails validation. JAX follows ``jax_enable_x64``; Torch tensors are never moved
across devices.

Example
-------

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

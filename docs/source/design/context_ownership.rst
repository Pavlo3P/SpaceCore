Context Ownership and Normalization
===================================

A ``Context`` owns three pieces of execution policy:

* ``ops`` - a ``BackendOps`` instance such as ``NumpyOps``;
* ``dtype`` - the backend-normalized default dtype;
* ``enable_checks`` - whether spaces and operators run membership checks.

It does not own array memory, devices, gradient state, or sparse storage. Those
belong to the backend arrays themselves.

Normalization
-------------

``spacecore.normalize_context(ctx)`` accepts a concrete ``Context``, backend
name, backend family, or ``None``. Concrete contexts are copied and their dtype
is sanitized by their backend. Backend names such as ``"numpy"`` and
``"torch"`` create a new context with that backend. ``None`` returns the global
default context.

``Context.asarray(x)`` and ``Context.assparse(x)`` convert user input into the
context backend and dtype. Space constructors keep the normalized context and
use it for later array construction and validation.

Accepted as-is versus converted
-------------------------------

User arrays passed to operations such as ``space.add`` or ``A.apply`` are not
silently converted. When checks are enabled, they are validated against the
space's backend, shape, and dtype. When checks are disabled, they are forwarded
to backend operations and backend errors may surface later.

Constructors and explicit conversion are the places where SpaceCore converts:
``ctx.asarray(...)`` creates dense arrays, ``ctx.assparse(...)`` creates sparse
arrays where supported, and ``obj.convert(new_ctx)`` rebuilds context-bound
objects under a target context.

Guarantees
----------

A constructed space or operator guarantees that it stores one normalized
``Context`` and uses that context for its own helper methods. It does not
guarantee that every future user-supplied array has that context unless checks
are enabled or the user created the array through the same context.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
   X = sc.DenseCoordinateSpace((2,), ctx)
   x = ctx.asarray([1.0, 2.0])
   X.check_member(x)

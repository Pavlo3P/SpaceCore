Context guide
=============

This tutorial follows ``tutorials/2_Context.ipynb``. It explains what
``Context`` represents, how it is used by spaces and operators, what
``enable_checks`` controls, and how context is chosen when new objects are
created.

A useful mental model is:

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

``BackendOps`` describes how numerical operations are performed. ``Context``
describes which backend, dtype, and checking policy are active. ``Space`` and
``LinOp`` carry that context and use it to manipulate arrays.

What Context signifies
----------------------

``Context`` is the runtime numerical policy attached to library objects. It is
the triple:

.. math::

   \texttt{Context}
   =
   (\texttt{ops}, \texttt{dtype}, \texttt{enable\_checks}).

It answers three questions:

1. Which backend is active?
2. Which dtype should be used?
3. Should runtime validation checks be enforced?

``Context`` is not itself a space or an operator. It carries the numerical rules
under which those objects live.

A context stores:

.. dropdown:: Context fields

   * ``ops``: backend operation object, such as ``NumpyOps`` or ``JaxOps``
   * ``dtype``: backend-normalized dtype for new arrays
   * ``enable_checks``: validation switch for space and operator membership

Basic methods
-------------

Typical methods are:

* ``ctx.asarray(x)``
* ``ctx.assparse(x)``
* ``ctx.convert(x)``
* ``ctx.assert_dense(x)``
* ``ctx.assert_sparse(x)``

The first three perform conversion into the backend and dtype specified by the
context. The last two are explicit validation helpers.

.. code-block:: python

   import numpy as np
   from spacecore.backend import Context, NumpyOps

   ctx = Context(NumpyOps(), dtype=np.float64, enable_checks=True)

   x = ctx.asarray([1, 2, 3])
   print(x.dtype)
   ctx.assert_dense(x)

The enable_checks flag
----------------------

``enable_checks`` controls whether context-bound spaces and operators run their
membership checks automatically. At the space level, ``check_member(x)`` runs
the actual validation logic only when:

.. math::

   \texttt{space.ctx.enable\_checks} = \texttt{True}.

This affects shape, dtype, backend representation, and structure checks such as
Hermitian symmetry.

.. code-block:: python

   import numpy as np
   from spacecore.backend import Context, NumpyOps
   from spacecore.space import DenseCoordinateSpace

   X_checked = DenseCoordinateSpace((2, 2), Context(NumpyOps(), enable_checks=True))
   X_unchecked = DenseCoordinateSpace((2, 2), Context(NumpyOps(), enable_checks=False))

   bad = np.zeros((3,))

   try:
       X_checked.check_member(bad)
   except Exception as exc:
       print(type(exc).__name__, exc)

   X_unchecked.check_member(bad)  # validation is skipped

``enable_checks=False`` is the default because some checks can be expensive,
for example checking matrix symmetry. Enable checks while debugging or when
user-facing validation is more important than runtime overhead.

Global context
--------------

You can provide context explicitly through ``ctx=...`` or implicitly through the
global default context. The helper functions are ``set_context`` and
``get_context``.

.. code-block:: python

   from spacecore import set_context, get_context
   from spacecore.backend import Context, NumpyOps

   set_context(Context(NumpyOps(), dtype="float64", enable_checks=False))
   print(get_context())

Use explicit contexts in library code and tests. The global context is most
useful in notebooks and scripts.

Context resolution order
------------------------

When SpaceCore needs to choose a context, the intended priority is:

.. math::

   \text{explicit context}
   \succ
   \text{inferred context from other objects}
   \succ
   \text{global default context}.

More precisely:

1. If an explicit context is passed, it is used.
2. Otherwise, SpaceCore tries to infer a context from other given objects.
3. If inference fails, the global default context is used.

Context inference can come from objects that already carry ``.ctx`` or from
backend-native arrays recognized by registered backends. Multiple inferred
contexts must be backend-compatible. If dtypes differ, SpaceCore chooses the
most general dtype among the input contexts. The inferred ``enable_checks`` flag
is the logical conjunction of source flags:

.. math::

   \texttt{enable\_checks}
   =
   \bigwedge_i \texttt{ctx}_i.\texttt{enable\_checks}.

Resolving contexts directly
---------------------------

Most users get context resolution automatically through constructors such as
``DenseCoordinateSpace(...)``, ``ProductSpace(...)``, and ``DenseLinOp(...)``. When an
algorithm needs to resolve a context before constructing an object, use the
public helper ``resolve_context_priority``:

.. code-block:: python

   import spacecore as sc

   X = sc.DenseCoordinateSpace((3,), ctx="numpy")
   ctx = sc.resolve_context_priority(None, X)

   assert ctx == X.ctx

The first argument is the explicit priority context. If it is not ``None``, it
wins over inferred contexts:

.. code-block:: python

   explicit = sc.Context(sc.NumpyOps(), dtype="float64", enable_checks=False)
   ctx = sc.resolve_context_priority(explicit, X)

   assert ctx == explicit

This helper is the supported public entry point for context-priority resolution.
Do not access the internal context manager singleton from user code.

Practical rule
--------------

* Pass ``ctx=...`` when you want full control.
* Rely on inference when composing objects that already carry a context.
* Rely on the global default only for convenience.

Why Context is separate from BackendOps
---------------------------------------

``BackendOps`` describes the backend interface itself. ``Context`` packages that
backend together with dtype and checking policy. This is why spaces and
operators usually carry a ``Context``, not a bare ``BackendOps`` instance.

Typical workflow:

.. code-block:: python

   import numpy as np
   from spacecore.backend import Context, NumpyOps
   from spacecore.space import HermitianSpace

   ctx = Context(NumpyOps(), dtype=np.complex128, enable_checks=True)
   H = HermitianSpace(3, ctx=ctx)

Summary
-------

``Context`` determines the backend, dtype, and validation policy. You can
provide it explicitly, let SpaceCore infer it, or let constructors fall back to
the global default.

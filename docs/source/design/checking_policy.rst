Checking policy
===============

Each bound object (space, operator, functional) validates at its own
``check_level``, one of ``"none"``, ``"cheap"``, ``"standard"``, ``"strict"``
(the literal type ``spacecore.CheckLevel``). The :class:`~spacecore.Context`
carries only backend ops and dtype; passing ``check_level=`` to it raises
``TypeError``.

.. code-block:: python

   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype="float64")
   X = sc.DenseCoordinateSpace((3,), ctx=ctx, check_level="standard")
   X.check_member(X.ctx.asarray([1.0, 2.0, 3.0]))
   assert X.check_level == "standard"

Checks catch errors; they never coerce inputs. A failed check raises.

Levels
------

Each level runs everything the levels below it run.

============  ==================================================================
Level         Adds
============  ==================================================================
``none``      Nothing optional. Constructor invariants that keep internal
              storage coherent (operator layout, squareness) still run.
``cheap``     Backend representation, shape/rank, scalar field, exact dtype,
              tree structure and arity, per-leaf checks.
``standard``  Hermitian membership, stored-representer membership, scalar
              output shape, positive weights of ``WeightedInnerProduct``, the
              Hermitian requirement of ``cg``, ``lanczos_smallest`` and
              ``expm_multiply``. The default.
``strict``    Numerical probes: matrix-free adjoint dot-test and declared
              Hermitian test, ``cg`` positive-curvature probe, spectral probes
              of ``OperatorInnerProduct``.
============  ==================================================================

Use ``"strict"`` while debugging, ``"standard"`` in user-facing code, and
``"cheap"`` or ``"none"`` in trusted hot loops.

Setting the level
-----------------

* Per object: ``check_level=`` at construction; read it back from
  ``obj.check_level``.
* Ambient default for objects built without one:
  :func:`spacecore.set_check_level` (process-wide) or
  :func:`spacecore.use_check_level` (``with`` block, scoped to the current
  thread or async task); read it with :func:`spacecore.get_check_level`.
* Composites (sums, compositions, ``TreeSpace``) built without a level take the
  least strict level of their parts: ``"strict"`` with ``"cheap"`` gives
  ``"cheap"``.

.. code-block:: python

   with sc.use_check_level("strict"):
       Y = sc.DenseCoordinateSpace((3,), ctx=ctx)
   assert Y.check_level == "strict"

Under tracing
-------------

While a backend traces (``jax.jit``, ``jax.vmap``, ``torch.compile``), an object
checks at the lower of its own level and
:func:`spacecore.get_traced_check_level`, default ``"none"``. Raise the cap with
:func:`spacecore.set_traced_check_level`: ``"cheap"`` keeps shape and dtype
checks, which run once at trace time; ``"standard"`` and above compare values,
which a tracer cannot do.

Where checks run
----------------

Users implement ``_<op>_core`` methods; the public method (``apply``,
``value``, ``inner``, ...) is a checked wrapper that validates inputs and
outputs, then calls the core. Composites call the cores of their parts, so
validation happens once, at the outer boundary.

For implementers:

* Input/output membership: decorate with ``@checked_method``.
* A ``SpaceCheck`` subclass declares ``minimum_level``; a space runs the checks
  its level includes.
* Any other check (scalar output shape, a numerical probe): guard it with
  ``self._checks_at_least(level)``, which applies the tracing cap.

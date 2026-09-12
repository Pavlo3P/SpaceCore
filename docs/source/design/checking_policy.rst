Checking policy
===============

SpaceCore uses ``ContextBound.check_level`` as its public runtime-validation
policy. The public type is ``spacecore.CheckLevel``, a literal type with four
ordered values: ``"none"``, ``"cheap"``, ``"standard"``, and ``"strict"``. A
literal keeps construction simple and makes invalid spellings visible to static
type checkers without introducing a separate policy object.

The level belongs to the *bound object* — space, operator, functional — and not
to the :class:`~spacecore.Context`. A context fixes backend ops and dtype; two
objects sharing both may still need different strictness, so the policy is set
per object, with an ambient default for objects that do not name one.

.. code-block:: python

   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype="float64")
   X = sc.DenseCoordinateSpace((3,), ctx=ctx, check_level="standard")

   x = X.ctx.asarray([1.0, 2.0, 3.0])
   X.check_member(x)

Levels
------

``none``
   Skips optional runtime membership and output checks. Constructor invariants
   needed to keep internal storage coherent, such as an operator tensor's
   layout, still run. Backend errors may otherwise surface later.

``cheap``
   Adds deterministic local checks: backend representation, shape/rank, scalar
   field, exact dtype, tree structure and arity, tree leaf interface checks, and
   domain/codomain membership at that same level. These checks are suitable for
   performance-sensitive trusted code that still needs interface validation.

``standard``
   Adds linear or near-linear mathematical validation: stored representer
   membership, configured Hermitian membership, and scalar functional output
   shape. TreeSpace traversal also applies these standard checks to each
   leaf. This is the normal choice for user-facing libraries and is the
   compatibility target for the old enabled Boolean policy.

``strict``
   Includes all standard checks and enables expensive mathematical probes.
   Current runtime probes include matrix-free adjoint consistency and CG
   Hermitian/positive-curvature preconditions. Exhaustive basis, metric,
   spectral, batched/single, and cross-backend conformance belongs in dedicated
   development tests rather than every hot method call.

Choosing a level
----------------

* Development and debugging: use ``"strict"`` when numerical probes are useful,
  or ``"standard"`` for lower overhead.
* Performance-sensitive trusted code: use ``"cheap"`` or ``"none"``.
* User-facing libraries: usually use ``"standard"``.

An object constructed without an explicit ``check_level`` takes the ambient
default, read with :func:`spacecore.get_check_level`. Move that default
process-wide with :func:`spacecore.set_check_level`, or for a block with
:func:`spacecore.use_check_level`, which is scoped to the current thread or
async task.

Where checks run
----------------

Spaces dispatch their ``SpaceCheck`` objects by minimum level. The same
dispatch is used by ``check_member``, ``checked_method``, and batched trailing
shape validation, so spaces, LinOps, functionals, and solver inputs share one
policy. Tree leaf checks recurse under the converted leaf
contexts. Linear-operator square/layout invariants that prevent incoherent
internal state remain unconditional.

When a context is inferred from several source objects, SpaceCore selects the
least expensive source level. For example, combining ``"strict"`` and
``"cheap"`` contexts produces a ``"cheap"`` inferred policy.

Migration from context-carried levels
-------------------------------------

Before 0.4.3 the level rode on the ``Context``, and the long-deprecated
``enable_checks=`` Boolean was still accepted there. Both are gone: ``Context``
is now exactly ``(ops, dtype)``, and passing either keyword to it raises
``TypeError``.

.. code-block:: python

   # 0.4.3 and later: the level is set on the object
   ctx = sc.Context(sc.NumpyOps())
   X = sc.DenseCoordinateSpace((3,), ctx=ctx, check_level="standard")

   # ...or moved for everything constructed in a block
   with sc.use_check_level("strict"):
       Y = sc.DenseCoordinateSpace((3,), ctx=ctx)

The level is always one of the four literals; the Boolean spelling is not
accepted in its place. Read an object's level back from ``obj.check_level``.

Implementation convention
-------------------------

Plain input and output membership checks use ``@checked_method``. Individual
``SpaceCheck`` classes declare a minimum level. Inline validation uses
``self._checks_at_least(level)`` only for checks that are not ordinary space
membership, such as scalar output shape or a strict numerical probe.

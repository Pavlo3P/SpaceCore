Checking policy
===============

SpaceCore uses ``Context.check_level`` as its public runtime-validation policy.
The public type is ``spacecore.CheckLevel``, a literal type with four ordered
values: ``"none"``, ``"cheap"``, ``"standard"``, and ``"strict"``. A literal
keeps context construction simple and makes invalid spellings visible to static
type checkers without introducing a separate policy object.

.. code-block:: python

   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype="float64", check_level="standard")
   X = sc.DenseCoordinateSpace((3,), ctx=ctx)

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

The process-wide default context remains ``"none"`` for compatibility. A
direct ``Context(...)`` defaults to ``"standard"``, matching the previous
direct-constructor default.

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

Migration from ``enable_checks``
--------------------------------

``enable_checks`` remains as a deprecated compatibility keyword:

* ``enable_checks=True`` maps to ``check_level="standard"``;
* ``enable_checks=False`` maps to ``check_level="none"``;
* passing both keywords raises ``TypeError``.

.. code-block:: python

   # New spelling
   ctx = sc.Context(sc.NumpyOps(), check_level="standard")

   # Deprecated equivalent
   legacy_ctx = sc.Context(sc.NumpyOps(), enable_checks=True)

``ctx.enable_checks`` remains a deprecated Boolean view and is true for
``cheap``, ``standard``, and ``strict`` contexts.

Implementation convention
-------------------------

Plain input and output membership checks use ``@checked_method``. Individual
``SpaceCheck`` classes declare a minimum level. Inline validation uses
``self._checks_at_least(level)`` only for checks that are not ordinary space
membership, such as scalar output shape or a strict numerical probe.

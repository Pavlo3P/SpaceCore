Conversion policy
=================

SpaceCore is built around the chain

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

Spaces and operators therefore always live under a numerical policy: a backend,
a dtype, and a checking policy. As soon as objects may come from different
sources, the library must decide which backend to use, which dtype to use, and
whether backend changes should happen silently.

SpaceCore treats backend conversion as an explicit context operation. Objects
that inherit from ``ContextBound`` provide:

.. code-block:: python

   converted = obj.convert(target_context)

The target may be a ``Context``, backend family string, backend enum value, or
``None``. Passing ``None`` resolves to the default context, subject to dtype
policy.

Why conversion is visible
-------------------------

Backend changes can affect execution mode, sparse support, dtype semantics, and
compilation behavior. A JAX-backed operator and a NumPy-backed operator may
expose the same SpaceCore methods, but they are not interchangeable at the
array level.

Without an explicit conversion policy, common construction patterns become
ambiguous. For example, if a ``ProductSpace`` is created from several spaces and
one of the inputs uses a JAX context while the global context is NumPy, the
library must choose between inferring from the inputs or converting inputs to
the global default context. The policy layer makes that decision deterministic.

Context resolution
------------------

When a new context-bound object is created, SpaceCore resolves its context in a
fixed order:

1. Use an explicit context provided through ``ctx=...``. If the explicit
   context is given as a string, missing context parameters are filled from
   defaults.
2. If no explicit context is provided, infer a context from input objects that
   carry a ``ctx`` attribute.
3. A common context can be inferred only when all context-carrying inputs use
   the same backend family.
4. The inferred context uses the shared backend and the most general dtype among
   the inferred dtypes. The inferred ``enable_checks`` flag is enabled only if
   all inferred contexts have checks enabled.
5. If no context can be inferred, use the global default context set by
   ``spacecore.set_context()``.

Once the context is resolved, it is assigned to the new object. Inputs that
carry their own contexts are adapted to the backend of the resolved context.
Their dtype is handled separately by the dtype policy.

User code can apply this same priority rule through
``spacecore.resolve_context_priority(priority_ctx, *objects)``. The helper is
the public entry point for context-priority resolution; the internal context
manager singleton is not part of the user-facing API.

Policy modes
------------

The ``resolution_policy`` regulates what happens when an object with a native
context is converted to a target context with a different backend family. This
policy is used when conversion is enforced through the context manager.

Use ``spacecore.set_resolution_policy(...)`` to set it and
``spacecore.get_resolution_policy()`` to inspect the active value.

.. dropdown:: ``resolution_policy`` types

   * ``warning``: if the object has a native context and the target context has
     a different backend family, conversion is allowed but a warning is issued.
     This is the default behavior.
   * ``error``: if the object has a native context and the target context is
     backend-incompatible, conversion is rejected. This is useful when
     accidental backend migration must be forbidden.
   * ``silent``: if the object has a native context and the target context is
     backend-incompatible, conversion proceeds without warning. This is useful
     in controlled pipelines where automatic conversion is expected.

The default is ``warning``.

Summary
-------

Context resolution and backend conversion are related but separate decisions:

* Context resolution decides which context the new object receives.
* Conversion adapts context-carrying inputs to that resolved context.
* ``resolution_policy`` controls whether backend-incompatible conversion warns,
  errors, or proceeds silently.
* Dtype handling during conversion is governed separately by the dtype policy.

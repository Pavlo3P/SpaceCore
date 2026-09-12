Context API
===========

``Context`` packages backend operations and the default dtype. Spaces,
operators, and functionals store a normalized context and use it for array
construction and conversion.

Validation policy is *not* part of a context: ``check_level`` belongs to the
bound object, so two objects on the same backend and dtype may validate at
different strictness. Use ``check_level="none"``, ``"cheap"``, ``"standard"``,
or ``"strict"`` on the object, or move the ambient default with
``set_check_level`` / ``use_check_level``. The exported ``spacecore.CheckLevel``
literal is available for annotations. See :doc:`../design/checking_policy`.

Context
-------

.. autosummary::
   :nosignatures:

   spacecore.Context

.. autoclass:: spacecore.Context
   :members:

Context helpers
---------------

.. autosummary::
   :nosignatures:

   spacecore.get_context
   spacecore.set_context
   spacecore.use_context
   spacecore.get_check_level
   spacecore.set_check_level
   spacecore.use_check_level
   spacecore.normalize_context
   spacecore.normalize_ops
   spacecore.resolve_context_priority
   spacecore.register_ops

* ``get_context`` and ``set_context`` manage the global default context, and
  ``use_context`` overrides it for a block, scoped to the current thread or
  async task.
* ``get_check_level`` / ``set_check_level`` / ``use_check_level`` do the same for
  the ambient validation level applied to newly constructed bound objects.
* ``normalize_context`` turns backend names, families, concrete contexts, or ``None`` into a context.
* ``resolve_context_priority`` chooses a common context for constructors.
* ``register_ops`` adds a custom backend implementation.

.. autofunction:: spacecore.get_context
.. autofunction:: spacecore.set_context
.. autofunction:: spacecore.use_context
.. autofunction:: spacecore.get_check_level
.. autofunction:: spacecore.set_check_level
.. autofunction:: spacecore.use_check_level
.. autofunction:: spacecore.normalize_context
.. autofunction:: spacecore.normalize_ops
.. autofunction:: spacecore.resolve_context_priority
.. autofunction:: spacecore.register_ops

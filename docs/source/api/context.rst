Context API
===========

``Context`` packages backend operations, default dtype, and runtime validation
policy. Spaces and operators store a normalized context and use it for array
construction and checks.

Context
-------

.. autosummary::
   :nosignatures:

   spacecore.backend.Context

.. autoclass:: spacecore.backend.Context
   :members:

Context helpers
---------------

.. autosummary::
   :nosignatures:

   spacecore.get_context
   spacecore.set_context
   spacecore.normalize_context
   spacecore.normalize_ops
   spacecore.resolve_context_priority
   spacecore.register_ops

* ``get_context`` and ``set_context`` manage the global default context.
* ``normalize_context`` turns backend names, families, concrete contexts, or ``None`` into a context.
* ``resolve_context_priority`` chooses a common context for constructors.
* ``register_ops`` adds a custom backend implementation.

.. autofunction:: spacecore.get_context
.. autofunction:: spacecore.set_context
.. autofunction:: spacecore.normalize_context
.. autofunction:: spacecore.normalize_ops
.. autofunction:: spacecore.resolve_context_priority
.. autofunction:: spacecore.register_ops

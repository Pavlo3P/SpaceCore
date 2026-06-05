Checking policy
===============

SpaceCore spaces can validate membership before operations. Checks are attached
to spaces and run only when ``Context.enable_checks`` is true.

.. code-block:: python

   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype="float64", enable_checks=True)
   X = sc.DenseCoordinateSpace((3,), ctx=ctx)

   x = X.ctx.asarray([1.0, 2.0, 3.0])
   X.check_member(x)

Built-in checks
---------------

.. dropdown:: Validation categories

   * backend family and dense array representation
   * shape
   * dtype
   * square matrix structure
   * Hermitian matrix structure
   * product element structure and component validity; tuple is the default
     representation, and registered pytree/dataclass product elements are
     validated through the product structure adapter

Where checks run
----------------

Spaces call ``check_member`` inside operations such as ``add``, ``inner``,
``flatten``, and capability-specific methods such as ``spectral_apply``. Linear operators call domain and codomain checks
before their ``apply`` and ``rapply`` methods when checking is enabled.

For exploratory use, enabled checks produce clearer errors. For tight numerical
loops, disabled checks reduce validation overhead.

Implementation convention
-------------------------

Methods that perform simple membership validation should use
``@checked_method`` rather than inline ``if self._enable_checks`` branches. This
keeps validation policy visible at the method signature and avoids duplicating
the same guard throughout spaces, operators, and functionals.

Inline ``if self._enable_checks`` blocks are reserved for checks that are not
plain membership checks, such as dense-array assertions, custom output-shape
comparisons, or the implementation of ``_check_member`` itself.

Inferred checking policy
------------------------

When a context is inferred from several source objects, the inferred
``enable_checks`` flag is the conjunction of the source flags:

.. math::

   \texttt{enable\_checks}
   =
   \bigwedge_i \texttt{ctx}_i.\texttt{enable\_checks}.

In other words, inferred checks remain enabled only if all source contexts have
checks enabled. If any source context has checks disabled, the inferred context
also disables checks.

The default ``enable_checks`` value used by the context manager is ``False``.

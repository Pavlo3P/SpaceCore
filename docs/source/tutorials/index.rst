Tutorials
=========

This learning path takes a new user from contexts and spaces to a small
space-aware solve.

Recommended order
-----------------

1. :doc:`spaces` - define one vector space and check its geometry.
2. :doc:`linops` - build maps :math:`A : X \to Y`, use ``apply`` and metric adjoints.
3. :doc:`context` - make backend ownership explicit.
4. :doc:`backend_ops` - write backend-agnostic helper code.
5. :doc:`conversion_policy` - rebuild spaces and operators under another context.
6. :doc:`weighted_tikhonov` - worked inverse problem with non-Euclidean adjoints.
7. :doc:`regularized_ot` - longer worked example. This page is retained as an advanced example.

Tutorial status for the 0.3.1 docs baseline
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 28 20 20

   * - Page
     - Purpose
     - Status
     - Notes
   * - ``spaces.rst``
     - First space and weighted geometry
     - updated
     - NumPy-only, cumulative
   * - ``linops.rst``
     - Dense/diagonal maps, adjoints, batching
     - updated
     - NumPy-only, cumulative
   * - ``context.rst``
     - Context ownership and checks
     - updated
     - NumPy baseline, optional backend note
   * - ``backend_ops.rst``
     - Portable helper functions over ``BackendOps``
     - updated
     - NumPy doctestable path
   * - ``conversion_policy.rst``
     - Explicit target-context conversion
     - updated
     - NumPy-to-NumPy dtype conversion shown
   * - ``weighted_tikhonov.rst`` and ``tutorials/weighted_tikhonov.ipynb``
     - Weighted inverse problem
     - added
     - NumPy-only, script and notebook code cells covered by tests
   * - ``regularized_ot.rst``
     - Advanced optimal-transport example
     - retained
     - user-modified in current worktree; not overwritten in this pass

.. toctree::
   :maxdepth: 1

   spaces
   linops
   context
   backend_ops
   conversion_policy
   weighted_tikhonov
   regularized_ot

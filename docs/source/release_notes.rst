Release notes
=============

Version 0.1.4
-------------

Changes
~~~~~~~

Backend support
^^^^^^^^^^^^^^^

* Added optional PyTorch support through ``TorchOps`` and the ``torch`` backend
  family.
* Added PyTorch backend tests and smoke coverage.
* Expanded the portable ``BackendOps`` interface with NumPy-like helpers for
  metadata, array construction, broadcasting, reductions, indexing, safety
  checks, and linear algebra.
* Added backend-specific docstrings for backend operation interfaces.

Context and conversion
^^^^^^^^^^^^^^^^^^^^^^

* Added ``spacecore.resolve_context_priority(...)`` as the public wrapper around
  SpaceCore's context-priority resolution logic.
* Exported ``resolve_context_priority`` from the top-level ``spacecore`` package
  and the contextual helper package.
* Added explicit tests for the public context-priority wrapper.
* Strengthened dtype, context conversion, and ``enable_checks=True`` coverage.

Spaces and validation
^^^^^^^^^^^^^^^^^^^^^

* Refactored space membership validation into modular ``SpaceCheck`` classes.
* Added public validation check classes such as ``BackendCheck``,
  ``DTypeCheck``, ``ShapeCheck``, ``HermitianCheck``, ``SquareMatrixCheck``,
  ``ProductStructureCheck``, and ``ProductComponentCheck``.
* Added focused tests for space validation checks.
* Improved product-space context handling and conversion behavior.

Linear operators
^^^^^^^^^^^^^^^^

* Optimized linear-operator hot paths for dense, sparse, block-diagonal,
  stacked, sum-to-single, and product operators.
* Added JIT-focused linear-operator tests.
* Added sparse linear-operator tests and broadened dense/operator conversion
  coverage.

Documentation and examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Added a Sphinx documentation site with API reference, tutorials, design
  notes, custom styling, and GitHub Pages deployment workflow.
* Added documentation pages for backend ops, contexts, spaces, linear
  operators, conversion policy, dtype policy, and checking policy.
* Added a regularized optimal transport tutorial and generated tutorial images.
* Added the regularized optimal transport notebook example.
* Updated the README with newer usage, documentation, and backend information.
* Documented ``resolve_context_priority`` in the README, Context API reference,
  context tutorial, and conversion policy design note.
* Added release notes to the Sphinx documentation.

Testing and CI
^^^^^^^^^^^^^^

* Reorganized tests into backend, context, integration, linear-operator, and
  space-focused packages.
* Added NumPy, JAX, and PyTorch smoke tests.
* Added a documentation build workflow.
* Updated CI coverage for the expanded backend and validation behavior.

Packaging
^^^^^^^^^

* Bumped the package version to ``0.1.4``.
* Made Sphinx read the release version from ``pyproject.toml``.
* Added ``pytorch`` to package keywords.

Version 0.1.3
-------------

Previous experimental release.

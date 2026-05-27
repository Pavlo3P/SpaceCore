Release notes
=============

Version 0.2.0
-------------

SpaceCore 0.2.0 is a major API expansion after
``3a1e382f13ef0496f8f54dc55db81aea95444775``. It migrates backend operations
toward the Array API, adds lazy linear-operator algebra, introduces functionals
and iterative solvers, broadens batching support, and substantially improves
documentation and validation.

Changes
~~~~~~~

Backend support
^^^^^^^^^^^^^^^

* Refactored ``BackendOps`` around ``array-api-compat`` so backend-specific
  code shares a common Array API-oriented implementation.
* Added optional CuPy support through ``CuPyOps`` and the ``cupy`` backend
  family.
* Broadened backend operation coverage for array creation, dtype conversion,
  sparse conversion, indexing, reductions, linear algebra, loop primitives,
  tree helpers, and vectorized mapping.
* Added backend loop tests for ``fori_loop``, ``while_loop``, ``cond``, and
  tree/stack behavior used by JIT-compatible algorithms.
* Added complex-dtype helpers on backend ops, including centralized complex
  dtype detection and real-dtype extraction.
* Added JAX pytree registration support for new operator, space, and
  functional objects.

Context and checking
^^^^^^^^^^^^^^^^^^^^

* Fixed contextual import cycles by moving contextual implementation details
  behind private modules while preserving public exports.
* Added ``ContextBound`` support for context-aware conversion and object
  binding.
* Centralized conversion, context normalization, backend registration, and
  context-resolution helpers in the contextual manager.
* Extended ``checked_method`` to support validation against ``self`` and
  multiple input argument positions.
* Replaced repeated manual ``if self._enable_checks`` membership checks in
  spaces with ``checked_method`` where the decorator fits.
* Added and documented reusable space-validation checks, including backend,
  dtype, shape, Hermitian, square-matrix, product-structure, and
  product-component checks.
* Improved ``enable_checks`` behavior and test coverage across spaces,
  operators, context conversion, and functionals.

Spaces
^^^^^^

* Added ``BatchSpace`` for batched elements with explicit batch shape and batch
  axis metadata.
* Improved ``VectorSpace``, ``HermitianSpace``, and ``ProductSpace`` docstrings,
  conversion behavior, validation, and batching support.
* Made the default linalg initial vector space-aware by normalizing with
  ``A.domain.norm`` instead of assuming a Euclidean inner product.

Linear operators
^^^^^^^^^^^^^^^^

* Added lazy linear-operator algebra:

  * ``A @ B`` and ``make_composed`` for composition.
  * ``A + B`` and ``make_sum`` for sums.
  * scalar multiplication and ``make_scaled`` for scaled operators.
  * ``IdentityLinOp``, ``ZeroLinOp``, ``ScaledLinOp``, ``SumLinOp``, and
    ``ComposedLinOp``.

* Added dense materialization via ``to_dense`` for core linear operators and
  operator algebra.
* Added ``DiagonalLinOp`` and improved dense, sparse, and diagonal operator
  handling for complex adjoints.
* Added public ``LinOp.is_hermitian()`` structural checks where verification is
  cheap and reliable.
* Made ``DenseLinOp.is_hermitian()`` and ``SparseLinOp.is_hermitian()`` return
  ``None`` for custom space geometries instead of applying an incorrect
  Euclidean matrix-symmetry test.
* Added product-structured operators and batched lifting:

  * ``ProductLinOp``
  * ``BlockDiagonalLinOp``
  * ``StackedLinOp``
  * ``SumToSingleLinOp``
  * ``vapply`` and ``rvapply`` paths for batched operator application.

* Optimized dense, sparse, block-diagonal, stacked, sum-to-single, and
  product-operator batched paths.
* Improved linear-operator equality, representation, conversion, and JAX
  pytree behavior.

Functionals
^^^^^^^^^^^

* Added the ``Functional`` abstraction for scalar-valued maps on spaces.
* Added linear functional implementations:

  * ``LinearFunctional``
  * ``InnerProductFunctional``
  * ``MatrixFreeLinearFunctional``

* Added quadratic forms:

  * ``QuadraticForm``
  * ``LinOpQuadraticForm``

* Added ``Functional.compose`` and ``ComposedFunctional`` for pull-backs along
  linear operators.
* Added ``LinOpQuadraticForm`` Hermiticity validation through the public
  ``LinOp.is_hermitian()`` contract instead of reaching into private operator
  attributes.
* Added value, gradient, batched value, batched gradient, conversion, and
  pytree coverage for functionals.

Linear algebra
^^^^^^^^^^^^^^

* Added JIT-compatible iterative solvers:

  * ``cg`` for Hermitian positive-definite systems.
  * ``lsqr`` for rectangular least-squares problems.
  * ``power_iteration`` for dominant eigenpair estimates.
  * ``lanczos_smallest`` for smallest Ritz eigenpair estimates.

* Renamed ``stochastic_lanczos`` to ``lanczos_smallest`` and kept
  ``stochastic_lanczos`` as a deprecated alias.
* Added ``LanczosResult`` with residual estimate, Krylov dimension, and
  convergence flag.
* Added ``expm_multiply`` for Krylov matrix-exponential actions
  ``exp(t * A) @ v`` on Hermitian operators.
* Added ``ExpmMultiplyResult`` with result vector, Krylov dimension, projected
  residual estimate, and convergence flag.
* Factored shared Lanczos basis and tridiagonal construction for reuse by
  ``lanczos_smallest`` and ``expm_multiply``.
* Hoisted the Lanczos coefficient zero template out of the loop body.
* Vectorized Lanczos reorthogonalization for plain ``VectorSpace`` domains while
  preserving the slower geometry-aware path for custom spaces.
* Added a weighted-inner-product regression test to prevent applying the
  Euclidean Lanczos fast path to non-Euclidean spaces.
* Made ``lanczos_smallest`` reject operators that are structurally known to be
  non-Hermitian.
* Clarified solver contracts for ``domain == codomain`` square requirements,
  Hermiticity enforcement, tolerance semantics, JAX static arguments, complex
  scalar behavior, ill-conditioning caveats, and power-iteration convergence
  caveats.
* Renamed the linalg helper ``safe_inverse`` to ``safe_inverse_nonneg`` to make
  its nonnegative-domain semantics explicit.

Documentation
^^^^^^^^^^^^^

* Added API reference pages for backend ops, spaces, linear operators,
  functionals, and linear algebra.
* Added the linear algebra API page covering solvers, eigenvalue algorithms,
  matrix-function routines, and result types.
* Added a JAX integration design note documenting trace-time operator algebra
  and recommended JIT usage.
* Added and updated tutorials for backend operations, linear operators, and
  matrix-free linalg workflows.
* Added cacheability and JIT traceability audit documents.
* Added a committed JAXPR fixture for ``lanczos_smallest`` regression tracking.
* Added docstring migration tooling, a migration baseline, and broad NumPy-style
  docstring coverage for public APIs.
* Added ``numpydoc`` validation configuration and doctest integration.
* Reworked public docstrings for solvers, spaces, operators, functionals,
  backends, and contextual helpers.

Testing and CI
^^^^^^^^^^^^^^

* Added broad tests for backend ops delegation, backend loop primitives, CuPy
  ops, context resolution, ``checked_method``, functionals, linalg solvers,
  operator algebra, batched lifting, dense materialization, diagonal operators,
  and JAX pytree/JIT behavior.
* Added cross-backend linalg tests for NumPy, JAX, Torch, and optional CuPy.
* Added ``expm_multiply`` tests against dense SciPy ground truth, complex time,
  group behavior, linearity, residual estimates, and JAX JIT.
* Added CI execution of the JIT audit script in ``--check`` mode.
* Added nonblocking documentation lint/audit steps for the docstring migration.
* Added development dependencies for testing, coverage, linting, and docstring
  validation.

Packaging
^^^^^^^^^

* Bumped the package version to ``0.2.0``.
* Made the top-level ``__version__`` resolve from package metadata instead of a
  hand-maintained constant.
* Added optional dependency groups for JAX, Torch, CuPy, examples, docs, and
  development tooling.
* Updated top-level exports for new backends, operators, functionals, solvers,
  result types, validation checks, and contextual helpers.

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

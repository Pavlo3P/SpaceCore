Release notes
=============

Version 0.3.1
-------------

SpaceCore 0.3.1 is a stabilization release for the ``0.3.x`` API. It focuses
on release-candidate verification, documentation consistency, executable
tutorials, package artifacts, and public API audit cleanup. It does not add new
solver families and does not include SDPLab-specific downstream validation.

Fixes
~~~~~

* Updated active tutorial notebooks to use current public APIs:
  ``MatrixFreeLinOp`` for action-defined operators, ``lanczos_smallest`` for
  smallest-Ritz-eigenpair estimation, and ``DenseVectorSpace`` for dense vector
  construction.
* Removed invalid notebook output metadata that produced nbformat validation
  warnings during execution.
* Updated the removed-API audit to skip ``.venv*`` directories so installed
  third-party packages are not reported as SpaceCore migration findings.

Documentation
~~~~~~~~~~~~~

* Reworked API reference pages for backend, context, spaces, linear operators,
  functionals, and linalg.
* Added design notes for context ownership, batching, and capability dispatch.
* Clarified conversion and dtype policies around explicit target contexts.
* Clarified adjoint language so metric adjoints are not described as merely
  coordinate transposes outside Euclidean coordinate spaces.
* Added a public docstring audit record for the ``0.3.1`` release candidate.

Examples and tutorials
~~~~~~~~~~~~~~~~~~~~~~

* Added a SpaceCore-only weighted Tikhonov worked example that exercises
  weighted spaces, metric adjoints, lazy operator algebra, conjugate gradients,
  and an independent dense NumPy reference solve.
* Added tests for the weighted Tikhonov example.
* Re-executed the active tutorial notebooks and worked example as part of the
  release-candidate gate.

Testing and packaging
~~~~~~~~~~~~~~~~~~~~~

* Documentation CI builds Sphinx with warnings treated as errors.
* Release-candidate verification covers the full test suite, strict docs build,
  public API audit, source distribution and wheel build, ``twine check``, clean
  wheel installation, and installed-package smoke testing.

Known limitations
~~~~~~~~~~~~~~~~~

* Solver coverage remains intentionally narrow: CG, LSQR, power iteration,
  Lanczos smallest-eigenpair estimation, and matrix-exponential actions.
* Optional backend support depends on installed optional dependencies. CuPy is
  not required for the core release-candidate gate.
* The regularized optimal transport tutorial is illustrative and depends on
  optional JAX/Optax packages; it is not a production OT solver.
* SDPLab-specific downstream validation is intentionally out of scope for this
  release-candidate check.

Version 0.3.0
-------------

Released 2026-06-05: `GitHub release <https://github.com/Pavlo3P/SpaceCore/releases/tag/v0.3.0>`_.

SpaceCore 0.3.0 is a breaking release in the unstable ``0.x`` series. Space
capabilities are recomputed from actual structure, dtype, and inner product,
including after ``convert()``.

Prominent migration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   space.eigh(x)
   # -> space.spectral_decompose(x)  # eigenvalues and frame
   # -> space.spectrum(x)            # eigenvalues only

Migration table
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 0.2.x
     - 0.3.0
   * - ``sc.VectorSpace((n,))``
     - ``sc.DenseVectorSpace((n,))``
   * - ``sc.VectorSpace((d, d))``
     - ``sc.DenseCoordinateSpace((d, d))``
   * - ``space.eigh(x)``
     - ``space.spectral_decompose(x)`` or ``space.spectrum(x)``
   * - ``ProductInnerProductSpace(...)``
     - ``ProductSpace(...)``
   * - ``StackedInnerProductSpace(...)``
     - ``StackedSpace(...)``
   * - ``BatchSpace`` and ``space.batch(...)``
     - leading-axis batched arrays with ``vapply(...)`` / ``rvapply(...)``
   * - ``op.vapply(xs, batch_space=...)``
     - ``op.vapply(xs)``
   * - global context conversion policies
     - explicit ``Context`` construction and ``obj.convert(ctx)``
   * - global dtype preservation policies
     - target-context dtype during explicit conversion

Added
~~~~~

* ``spectrum``, ``spectral_decompose``, and ``from_spectrum`` as the Jordan
  spectral contract.
* ``ElementwiseJordanSpace`` and ``EuclideanElementwiseJordanSpace``.
* ``ProductStructure``, ``TupleStructure``, ``PytreeStructure``, and
  ``ProductSpace.from_template``.
* ``ProductSpectralDecomposition`` for product spectral data independent of
  product element structure.
* ``StackedSpace`` and vectorizable axis-aware checks.
* ``InnerProduct.validate_for(space)`` with strict ``WeightedInnerProduct``
  construction validation.
* ``scripts/api_audit.py`` for repository and downstream API migration audits.

Changed and removed
~~~~~~~~~~~~~~~~~~~

* ``VectorSpace`` is abstract.
* Previous concrete ``VectorSpace`` use cases moved to ``DenseVectorSpace`` and
  ``DenseCoordinateSpace``.
* ``DenseVectorSpace`` is now a plain one-dimensional dense vector space with
  star and no Jordan capability by default.
* Complex elementwise Jordan spaces no longer claim
  ``EuclideanJordanAlgebraSpace``.
* ``eigh`` was removed from spaces.
* Specialized public product and stacked constructors were replaced by
  auto-dispatching ``ProductSpace(...)`` and ``StackedSpace(...)`` factories.
* ``BatchSpace``, ``Space.batch``, and ``batch_space=`` arguments were removed
  from public batching APIs. Use leading-axis vectorization through ``vapply``
  and ``rvapply``.
* Global context-policy and dtype-policy APIs were removed. Conversion now
  follows the requested target ``Context`` directly.

Version 0.2.0
-------------

SpaceCore 0.2.0 is a major API expansion. The backend layer now sits on the
Array API standard. Operators gained a lazy algebra with adjoint views,
composition, sums, and scaling. A new :class:`Functional` hierarchy provides
scalar-valued maps with gradients and pull-backs. A new :mod:`spacecore.linalg`
module ships four JIT-compatible iterative solvers. Spaces, operators, and
functionals share a single validation pattern via ``checked_method``, and the
public API is documented to numpydoc standard with doctest coverage.

This release introduces breaking renames; see :ref:`migration-0-2`.

Highlights
~~~~~~~~~~

* Array API backend layer with optional CuPy support.
* Lazy operator algebra: ``A @ B``, ``A + B``, ``A.H``, plus
  :class:`IdentityLinOp`, :class:`ZeroLinOp`, :class:`MatrixFreeLinOp`, and
  ``make_*`` factories with algebraic simplification.
* :class:`Functional` hierarchy with linear and quadratic forms, plus
  ``Functional.compose`` for pull-back along linear operators.
* New :mod:`spacecore.linalg` module with iterative solvers: :func:`cg`,
  :func:`lsqr`, :func:`power_iteration`, :func:`lanczos_smallest`,
  :func:`expm_multiply`.
* Geometry-aware solvers honor the declared ``Space.inner`` instead of assuming
  Euclidean.
* Unified ``checked_method`` decorator across :class:`Space`, :class:`LinOp`,
  and :class:`Functional`.
* Comprehensive numpydoc-style docstrings, doctests, and a JAX integration
  design note.

Backend
~~~~~~~

* Migrated :class:`BackendOps` to the Array API standard via
  ``array-api-compat``.
* Added :class:`CuPyOps` and the ``cupy`` backend family as an optional install
  (``pip install 'spacecore[cupy]'``).
* Centralized complex-dtype handling on :class:`BackendOps`:

  * :meth:`BackendOps.is_complex_dtype` for backend-aware complex detection.
  * :meth:`BackendOps.real_dtype` for extracting the real dtype matching a
    complex one.

* Broadened backend coverage for array creation, dtype conversion, sparse
  conversion, indexing, reductions, linear algebra, loop primitives
  (``fori_loop``, ``while_loop``, ``cond``), tree helpers, and vectorized
  mapping.
* Registered JAX pytrees for operator, space, and functional types so they pass
  through ``jax.jit``, ``jax.vmap``, and ``jax.grad`` boundaries.

Context and checking
~~~~~~~~~~~~~~~~~~~~

* Restructured ``_contextual`` to hide implementation details while keeping the
  public free-function API (:func:`set_context`, :func:`get_context`,
  :func:`resolve_context_priority`, :func:`register_ops`, and the
  resolution-policy accessors).
* Extended :func:`~spacecore._checks.checked_method` to support validation
  against ``self`` and multiple input argument positions.
* Replaced manual ``if self._enable_checks`` guards with ``checked_method``
  across :class:`Space`, :class:`LinOp`, and :class:`Functional`. Inline guards
  are now reserved for non-membership checks such as dense-array assertions and
  custom output-shape checks.
* Added reusable space-validation checks documented at
  ``docs/source/design/checking_policy.rst``: backend, dtype, shape, Hermitian,
  square-matrix, product-structure, and product-component checks.

Spaces
~~~~~~

* Added :class:`BatchSpace` for batched elements with explicit batch shape and
  batch-axis metadata.
* Improved :class:`DenseCoordinateSpace`, :class:`HermitianSpace`, and
  :class:`ProductSpace` conversion behavior, validation, batching support, and
  docstrings.

Linear operators
~~~~~~~~~~~~~~~~

* **Lazy operator algebra.** Added composition, addition, scaling, and adjoint
  view with algebraic simplification:

  * ``A @ B`` composes operators.
  * ``A + B`` sums operators.
  * ``alpha * A`` scales an operator.
  * ``A.H`` returns a cached adjoint view satisfying ``A.H.H is A``.

  Simplification rules eliminate ``I``, ``Zero``, ``alpha = 0``, ``alpha = 1``,
  and flatten nested sums.

* Added :class:`IdentityLinOp`, :class:`ZeroLinOp`, :class:`MatrixFreeLinOp`,
  and :class:`DiagonalLinOp`.
* Added structural :meth:`LinOp.is_hermitian` reporting ``True``, ``False``,
  or ``None`` (unknown) without applying incorrect Euclidean assumptions for
  custom space geometries.
* Added :meth:`LinOp.to_dense` for materializing operators as backend arrays.
* Added product-structured operators and batched lifting:

  * :class:`ProductLinOp`
  * :class:`BlockDiagonalLinOp`
  * :class:`StackedLinOp`
  * :class:`SumToSingleLinOp`
  * ``vapply`` / ``rvapply`` paths for batched operator application.

* Improved linear-operator equality, representation, conversion, and JAX
  pytree behavior.

Functionals
~~~~~~~~~~~

* Added :class:`Functional` as an abstract base for scalar-valued maps on
  spaces, with :meth:`value`, :meth:`grad`, :meth:`hess_apply`, and batched
  counterparts.
* Added linear functional implementations:

  * :class:`LinearFunctional`
  * :class:`InnerProductFunctional`
  * :class:`MatrixFreeLinearFunctional`

* Added quadratic forms:

  * :class:`QuadraticForm`
  * :class:`LinOpQuadraticForm`

* Added :meth:`Functional.compose` and :class:`ComposedFunctional` for
  pull-backs along linear operators, with specializations that preserve the
  concrete functional type when possible.

Linear algebra
~~~~~~~~~~~~~~

The :mod:`spacecore.linalg` module is new in 0.2.0. It provides
JIT-compatible iterative solvers and structured result types.

* Added iterative solvers:

  * :func:`cg` for Hermitian positive-definite systems.
  * :func:`lsqr` for rectangular least-squares problems.
  * :func:`power_iteration` for dominant-eigenpair estimates of a
    :class:`LinOp` or :class:`QuadraticForm`.
  * :func:`lanczos_smallest` for smallest-Ritz-eigenpair estimates of
    Hermitian operators.
  * :func:`expm_multiply` for Krylov matrix-exponential actions
    ``exp(t A) v`` on Hermitian operators, with complex ``t`` supported for
    Schrodinger-type evolution.

* Added structured result types :class:`CGResult`, :class:`LSQRResult`,
  :class:`PowerIterationResult`, :class:`LanczosResult`, and
  :class:`ExpmMultiplyResult`, each carrying convergence diagnostics and a
  compact ``__repr__``.
* Solvers are geometry-aware: norms, inner products, and the default initial
  vector use ``Space.inner`` and ``Space.norm`` rather than assuming Euclidean
  geometry. This makes the solvers correct on custom inner products such as
  RKHS or weighted spaces.

Documentation
~~~~~~~~~~~~~

* Reworked public docstrings to numpydoc standard with runnable doctests for
  solvers, spaces, operators, functionals, backends, and contextual helpers.
* Clarified solver contracts: ``domain == codomain`` square requirements,
  Hermiticity enforcement, tolerance semantics, JAX static arguments, complex
  scalar behavior, ill-conditioning caveats, and convergence assumptions.
* Added API reference pages for backend ops, spaces, linear operators,
  functionals, and linear algebra.
* Added a JAX integration design note documenting trace-time operator algebra
  and recommended JIT usage at
  ``docs/source/design/jax_integration.rst``.
* Added tutorials for backend operations, linear operators, and matrix-free
  linalg workflows.

Testing and CI
~~~~~~~~~~~~~~

* Added cross-backend tests covering NumPy, JAX, Torch, and optional CuPy.
* Added tests for backend ops delegation, backend loop primitives, CuPy ops,
  context resolution, ``checked_method``, functionals, linalg solvers,
  operator algebra, batched lifting, dense materialization, diagonal
  operators, and JAX pytree/JIT behavior.
* Added CI execution of a JIT-traceability audit script in ``--check`` mode
  and a coverage floor of 70% via ``pytest-cov``.
* Added nonblocking documentation lint and audit steps for the docstring
  migration.

Packaging
~~~~~~~~~

* Bumped the package version to ``0.2.0``.
* ``spacecore.__version__`` now resolves from package metadata via
  ``importlib.metadata`` instead of a hand-maintained constant.
* Added optional dependency groups: ``[jax]``, ``[torch]``, ``[cupy]``,
  ``[examples]``, ``[docs]``, ``[dev]``.
* Added an explicit ``__all__`` at the top level covering new backends,
  operators, functionals, solvers, result types, validation checks, and
  contextual helpers.

.. _migration-0-2:

Migration from 0.1.x
~~~~~~~~~~~~~~~~~~~~

* ``BackendOps.eps`` is now a method ``eps(dtype)`` rather than a property.
  Callers must pass a dtype, typically ``ctx.dtype``.
* The implementation attribute ``DenseLinOp.A`` is now a
  :class:`functools.cached_property` backed by ``_A``. The public attribute
  access ``op.A`` is unchanged.
* :meth:`LinOp.__eq__` now returns ``NotImplemented`` instead of raising
  ``NotImplementedError`` on the base class, so ``op == None`` and
  ``op in some_list`` no longer raise.
* Several module-internal helpers in ``spacecore._contextual`` moved to
  private modules. Use the public functions re-exported from
  :mod:`spacecore._contextual` (``set_context``, ``get_context``,
  ``resolve_context_priority``, ``register_ops``, ``set_resolution_policy``,
  and the dtype-policy accessors) rather than importing from internal modules.

Known limitations
~~~~~~~~~~~~~~~~~

* :func:`cg`, :func:`lsqr`, and :func:`power_iteration` do not structurally
  validate operator properties (positive-definiteness, full Hermiticity) and
  may silently produce incorrect results on inputs that violate their
  preconditions. See each function's ``Notes`` section for details.
* Operator algebra runs Python-level simplification at construction time. For
  maximum JIT efficiency, assemble operator expressions outside the
  ``jax.jit`` boundary; see the JAX integration design note.
* :class:`MatrixFreeLinOp` stores its callables in pytree auxiliary data.
  Constructing one inside a JIT-traced function with a new lambda each call
  triggers retracing. Construct outside the traced region with a stable
  callable reference.
* The CuPy backend is provided as a preview. Coverage of non-standard
  operations and sparse handling may evolve in a subsequent release.

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

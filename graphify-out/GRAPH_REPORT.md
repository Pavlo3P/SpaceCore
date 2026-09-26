# Graph Report - spacecore  (2026-06-23)

## Corpus Check
- 82 files · ~51,809 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2219 nodes · 4893 edges · 85 communities (83 shown, 2 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 338 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Validation Checks|Validation Checks]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Tree Structures|Tree Structures]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Conj Reshape|Conj Reshape]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Validation Checks|Validation Checks]]
- [[_COMMUNITY_Using Array|Using Array]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Linear Solvers|Linear Solvers]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Copy Add|Copy Add]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Tree Structures|Tree Structures]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Over Given|Over Given]]
- [[_COMMUNITY_Linear Operators|Linear Operators]]
- [[_COMMUNITY_Reshape Ravel|Reshape Ravel]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Backend Operations|Backend Operations]]
- [[_COMMUNITY_Scan Run|Scan Run]]
- [[_COMMUNITY_Vector Spaces|Vector Spaces]]
- [[_COMMUNITY_Diagonal Trace|Diagonal Trace]]

## God Nodes (most connected - your core abstractions)
1. `BackendOps` - 136 edges
2. `LinOp` - 90 edges
3. `DenseArray` - 77 edges
4. `TreeSpace` - 75 edges
5. `DenseCoordinateSpace` - 48 edges
6. `Functional` - 46 edges
7. `DenseArray` - 40 edges
8. `TorchOps` - 39 edges
9. `Context` - 38 edges
10. `EuclideanElementwiseJordanSpace` - 37 edges

## Surprising Connections (you probably didn't know these)
- `BackendOps` --uses--> `BackendOps`  [INFERRED]
  spacecore/backend/_context.py → spacecore/backend/_ops.py
- `DType` --uses--> `BackendOps`  [INFERRED]
  spacecore/backend/_context.py → spacecore/backend/_ops.py
- `ArrayLike` --uses--> `BackendOps`  [INFERRED]
  spacecore/backend/_context.py → spacecore/backend/_ops.py
- `LinOp` --uses--> `SpaceCoreOps`  [INFERRED]
  spacecore/linalg/_cg.py → spacecore/linalg/_utils.py
- `Any` --uses--> `SpaceCoreOps`  [INFERRED]
  spacecore/linalg/_cg.py → spacecore/linalg/_utils.py

## Import Cycles
- 1-file cycle: `spacecore/kernels/core/__init__.py -> spacecore/kernels/core/__init__.py`
- 1-file cycle: `spacecore/kernels/specs/__init__.py -> spacecore/kernels/specs/__init__.py`

## Communities (85 total, 2 thin omitted)

### Community 0 - "Backend Operations"
Cohesion: 0.07
Nodes (38): BackendCheck, _dtype_of(), DTypeCheck, FieldCheck, HermitianCheck, Check that a value is a dense array for a space backend.      Parameters     ---, Raised when an object is not a member of a space., Check that a value has the canonical shape of a space.      Parameters     ----- (+30 more)

### Community 1 - "Backend Operations"
Cohesion: 0.05
Nodes (49): Space capability with a canonical involution/star operation.      Parameters, Return the canonical star/involution of ``x``., StarSpace, base(), ops(), Convert the base space and rebuild the stacked space., Inner-product operations for stacks whose base supports them., Return ``sum_i base.inner(x[i], y[i])`` as a scalar. (+41 more)

### Community 2 - "Backend Operations"
Cohesion: 0.05
Nodes (40): KernelSpec, Any, Any, Any, block_diagonal_dense_apply_applicable(), block_diagonal_dense_apply_generic(), block_diagonal_dense_apply_optimized(), Block-diagonal dense-leaf apply kernel.  The generic path for a ``BlockDiagonalL (+32 more)

### Community 3 - "Backend Operations"
Cohesion: 0.06
Nodes (29): LinOp, Apply the adjoint map to an element of ``self.codomain``., Apply without adding validation beyond the concrete implementation., Apply the adjoint without adding validation beyond the implementation., Apply to a batch without adding validation beyond the implementation., Apply the adjoint to a batch without adding validation beyond the implementation, Apply this linear operator to ``x``., Apply the adjoint of this linear operator to ``y``. (+21 more)

### Community 4 - "Backend Operations"
Cohesion: 0.09
Nodes (22): Any, ArrayLike, DenseArray, DType, Index, SparseArray, Sparse array type tuple using PyTorch.          Returns         -------, Check whether an object is a dense PyTorch tensor.          Input:             x (+14 more)

### Community 5 - "Vector Spaces"
Cohesion: 0.06
Nodes (22): ABC, Base space abstractions and inner-product geometry., EuclideanInnerProduct, InnerProduct, Validate diagonal weights against a coordinate space., Geometry of a space: inner product plus Riesz maps., Return the inner product of ``x`` and ``y``., Map a coordinate element to its dual. Euclidean default: identity. (+14 more)

### Community 6 - "Backend Operations"
Cohesion: 0.05
Nodes (31): JAX backend implementation and pytree registration helpers., JaxOps, BackendOps implementation for the JAX ecosystem.      This backend uses JAX for, Dense array type using JAX.          Returns         -------             Concret, Sparse array type tuple using JAX.          Returns         -------, Convert input to a sparse array using JAX.          Input:             x: Dense,, Multiply sparse and dense arrays using JAX.          Input:             a: Spars, Vectorize a function using ``jax.vmap``. (+23 more)

### Community 7 - "Vector Spaces"
Cohesion: 0.06
Nodes (27): Present the public ``StackedSpace`` label, not the private dispatch subclass., Return the stacked zero element., Return the stacked all-ones element., Return the stacked sum ``x + y``., Return the leading-axis batch sum of stacked elements., Return the stacked scalar product ``a * x``., Return the leading-axis batch scalar product of stacked elements., Flatten the whole stacked element to one coordinate vector. (+19 more)

### Community 8 - "Backend Operations"
Cohesion: 0.11
Nodes (27): Context, Select backend operations, representation dtype, and validation policy.      A c, _C, Return ``count×<base descriptor>`` (e.g. ``8×ℝ^3``)., core_kernels(), Class decorator that binds the named core-kernel set onto an operator.      Inst, Evaluate over a leading batch axis. Input must have shape ``(N,) + domain.shape`, Evaluate ``domain.inner(representer, xs[i])`` without a Python loop. (+19 more)

### Community 9 - "Validation Checks"
Cohesion: 0.05
Nodes (22): Compute a stable log-sum-exp reduction., Return the number of dimensions of x., Permute dimensions of x., Insert singleton dimensions into x., Move axes of x to new positions., Stack arrays along a new axis (delegates to xp.stack)., Stack arrays horizontally / column-wise (delegates to xp.hstack).          Conca, Stack arrays vertically / row-wise (delegates to xp.vstack).          Inputs are (+14 more)

### Community 10 - "Backend Operations"
Cohesion: 0.05
Nodes (21): BackendOps, Public numerical contract for SpaceCore backends.      Common dense-array operat, Backend family identifier., Broadcast x to shape (delegates to xp.broadcast_to)., Whether this backend supports sparse arrays., Vectorize ``fn`` over array axes using a Python-loop fallback., Whether ``vmap`` is implemented by the backend rather than a Python loop., Vectorize a scalar Python function over its array arguments.          Returns a (+13 more)

### Community 11 - "Linear Operators"
Cohesion: 0.07
Nodes (26): _DenseMode, Domain space of this linear operator., Codomain space of this linear operator., DenseLinOp, Select the dense computation mode once for this operator., Stored dense tensor representation of this operator.          The returned array, Apply the dense operator to ``x``., r"""Apply the adjoint dense operator to ``y``.          Euclidean spaces use the (+18 more)

### Community 12 - "Backend Operations"
Cohesion: 0.07
Nodes (21): Reject implicit loss of a complex representation during conversion., Machine epsilon for dtype., Return whether ``dtype`` is a complex floating type.          Parameters, Return the real floating dtype with the same precision as ``dtype``.          Pa, Convert input to a dense backend array (delegates to xp.asarray)., Cast x to dtype, returning x unchanged when dtype is None., Create an uninitialized array (delegates to xp.empty)., Create a zero-filled array (delegates to xp.zeros). (+13 more)

### Community 13 - "Vector Spaces"
Cohesion: 0.09
Nodes (21): _components(), _from_components(), Return a leafwise all-ones element when leaf spaces support it., Return the componentwise sum., Return the componentwise leading-axis batch sum., Return the componentwise scalar product., Return the componentwise batched scalar product., Concatenate leaf coordinate vectors into one dense vector. (+13 more)

### Community 14 - "Backend Operations"
Cohesion: 0.08
Nodes (23): DenseCoordinateSpace, Return the configured inner product without membership checks., Return ``X`` as a dense one-dimensional coordinate vector., Reshape a flat coordinate vector into this space's canonical shape., Flatten a leading-axis batch of dense elements to ``(N, size)``., Unflatten rows of shape ``(N, size)`` into dense space elements., Run member checks for a single element while allowing batched input.          Va, Convert this dense coordinate space to ``new_ctx`` without changing shape. (+15 more)

### Community 15 - "Backend Operations"
Cohesion: 0.06
Nodes (24): ContextBound, Return the execution context bound to this object., Return this object's runtime validation level., Deprecated internal Boolean view retained for compatibility., Return whether this object runs checks assigned to ``level``., Return ``x`` asserted as a dense array when cheap checks are enabled.          H, Return the terse backend half of a repr (``backend='numpy', dtype=float64``)., Return the class label shown in reprs.          Defaults to the runtime class na (+16 more)

### Community 16 - "Linear Operators"
Cohesion: 0.08
Nodes (24): DenseVectorSpace, ElementwiseJordanSpace, Elementwise Jordan algebra for real or complex dense coordinates.      Parameter, Return elementwise conjugation., Return the elementwise Jordan product., Return ``x`` as its Jordan spectrum under elementwise product., Return the trivial spectral decomposition ``(x, None)``., Reconstruct an elementwise Jordan element from its spectrum. (+16 more)

### Community 17 - "Vector Spaces"
Cohesion: 0.07
Nodes (21): Functional, Return the pull-back ``self o A``.          Parameters         ----------, Raise if ``x`` is not in the domain., Return structural equality when implemented by a subclass.          Mirrors :cla, Return the ``domain → scalar-field`` descriptor for this functional., Return a bounded ``ClassName(domain → field)`` form for nesting., Flatten this functional for pytree registration., Rebuild this functional from pytree data. (+13 more)

### Community 18 - "Vector Spaces"
Cohesion: 0.06
Nodes (23): _convert_space_element(), MatrixFreeLinearFunctional, Stored domain element ``c`` defining ``ell_c(x) = <c, x>``., Return ``domain.inner(representer, x)``., Return whether another inner-product functional has the same representer., Convert a value recursively into a possibly tree-valued space., Convert the domain and representer to ``new_ctx``., Linear functional defined by user-supplied evaluation callables.      ``MatrixFr (+15 more)

### Community 19 - "Backend Operations"
Cohesion: 0.06
Nodes (19): Build open-mesh index arrays., Raise ``TypeError`` unless both ``a`` and ``b`` are sparse for this backend., Return whether ``x`` carries a complex representation., Return x.dtype after verifying x is a backend array., Return x.shape as a tuple., Return the total number of elements in x., Remove singleton dimensions from x., Dense array type accepted by this backend. (+11 more)

### Community 20 - "Vector Spaces"
Cohesion: 0.08
Nodes (21): HermitianSpace, r"""Project ``x`` onto the Hermitian subspace as :math:`(X + X^*) / 2`., Return the canonical star operation for Hermitian elements: identity., Return the Hermitian Jordan product ``(xy + yx) / 2``., Return the Hermitian eigenvalue spectrum of ``x``., Return the Hermitian eigendecomposition ``(evals, evecs)``., Reconstruct a Hermitian element from eigenvalues and eigenvectors., Reshape dense coordinates and symmetrize the result. (+13 more)

### Community 21 - "Backend Operations"
Cohesion: 0.08
Nodes (19): Select the sparse computation mode once for this operator., Return the complex conjugate of a backend sparse array., Stored sparse matrix representation of this operator.          The returned spar, Forward action ``y = A @ x`` in Euclidean coordinates.          x must have shap, Metric-aware adjoint action.          y must have shape cod.shape (dense)., Return the stored sparse matrix representation without copying.          The ret, Materialize the stored sparse matrix as a dense 2D coordinate matrix.          U, Materialize the stored sparse matrix as a dense operator tensor.          The re (+11 more)

### Community 22 - "Vector Spaces"
Cohesion: 0.09
Nodes (20): Any, BlockDiagonalLinOp, BlockMatrixLinOp, Check that each block maps the corresponding pair of tree leaves., Apply each block to the matching direct-product component., Apply each block's metric adjoint to the matching component., Apply each block over a tree of leading-axis batches., Apply each metric adjoint over a tree of leading-axis batches. (+12 more)

### Community 23 - "Backend Operations"
Cohesion: 0.11
Nodes (16): BackendFamily, Backend contexts and operation implementations., CuPy backend implementation., NumPy backend implementation., NumpyOps, BackendOps implementation for the NumPy/SciPy ecosystem.      This backend uses, Protocol, Shape (+8 more)

### Community 24 - "Linear Operators"
Cohesion: 0.08
Nodes (20): _DiagonalMode, DiagonalLinOp, Dense tensor representation of this diagonal operator., Apply the diagonal operator to ``x``., Apply the adjoint diagonal operator to ``y``., Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``;, Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codo, Return the flattened dense diagonal matrix representation. (+12 more)

### Community 25 - "Linear Operators"
Cohesion: 0.10
Nodes (30): expm_multiply(), ExpmMultiplyResult, Store the result returned by :func:`expm_multiply`.      Parameters     --------, Return a compact summary without printing the full vector., r"""     Compute :math:`\exp(t A) v` by Krylov projection.      Require ``A`` to, _build_tridiagonal(), _check_lanczos_max_iter(), _lanczos_basis_and_tridiag() (+22 more)

### Community 26 - "Linear Operators"
Cohesion: 0.08
Nodes (17): _AdjointViewLinOp, Apply this operator to a batch of domain elements.          Parameters         -, Apply the adjoint operator to a batch of codomain elements.          Parameters, Hermitian-adjoint view of a linear operator.      ``A.H`` represents the adjoint, Return ``op.rapply(y)``., Return ``op.apply(x)``., Return ``op.rvapply(ys)`` over a batch., Return ``op.vapply(xs)`` over a batch. (+9 more)

### Community 27 - "Linear Operators"
Cohesion: 0.14
Nodes (22): Return whether contexts match for algebra, ignoring validation checks., _same_math_context(), _flatten_sum_terms(), make_composed(), make_sum(), Flatten nested lazy sums into a tuple of terms., Return a locally simplified lazy sum of linear operators.      This factory perf, Original operator viewed as the adjoint of this adjoint view. (+14 more)

### Community 28 - "Linear Operators"
Cohesion: 0.12
Nodes (30): adjoint_apply_core(), adjoint_rapply_core(), adjoint_vapply_core(), batched_zeros(), compose_chain(), composed_apply_core(), composed_rapply_core(), composed_vapply_core() (+22 more)

### Community 29 - "Vector Spaces"
Cohesion: 0.11
Nodes (29): EuclideanElementwiseJordanSpace, Real elementwise Euclidean Jordan algebra.      Parameters     ----------     sh, _LeafwiseHostMixin, _LeafwiseInnerProductMixin, _LeafwiseJordanMixin, _LeafwiseStarMixin, Type-only declarations of the TreeSpace host surface used by leaf mixins., Inner-product operations for trees whose leaves all support them. (+21 more)

### Community 30 - "Linear Operators"
Cohesion: 0.10
Nodes (20): Functional, ComposedFunctional, make_functional_composed(), Flatten this functional for pytree registration., Rebuild this functional from pytree data., Convert the composed functional and operator to ``new_ctx``., Raise unless ``F`` can be composed with ``A``., Return the pull-back ``F o A`` with local specializations.      Parameters     - (+12 more)

### Community 31 - "Linear Operators"
Cohesion: 0.16
Nodes (21): Any, Codomain, Context, Domain, LinOp, Self, Context, Domain (+13 more)

### Community 32 - "Linear Operators"
Cohesion: 0.10
Nodes (17): Any, LinOp, TreeSpace, Apply component adjoints and return a domain product element., Apply component adjoints without checks and rebuild domain representation., Apply this sum-to-single operator over a structured product batch., Apply over a product batch without membership checks., r"""     Represent a sum of leaf operators from a tree domain.      If ``dom = X (+9 more)

### Community 33 - "Backend Operations"
Cohesion: 0.09
Nodes (19): BackendOps, CuPyOps, Multiply a CuPy sparse matrix by a CuPy dense array., BackendOps implementation for CuPy GPU arrays.      This backend uses CuPy for d, Compute log-sum-exp using ``cupyx.scipy.special``., Return a CuPy copy of ``x`` (mutation primitive for index ops)., Accumulate ``values`` into ``y`` at ``index`` via ``cupy.add.at``., Build open-mesh indices using CuPy. (+11 more)

### Community 34 - "Vector Spaces"
Cohesion: 0.07
Nodes (15): r"""Represent a finite direct product as a Python tree.      ``TreeSpace`` repre, Return the immutable optree structure definition., Return paths in the same deterministic order as ``leaf_spaces``., Return the number of ordered leaves., Present the public ``TreeSpace`` label, not the private dispatch subclass., Return ``Tree(<leaf descriptors>)``, abbreviating wide trees., Rebuild the configured Python tree from ordered leaves., Return the additive identity. (+7 more)

### Community 35 - "Linear Operators"
Cohesion: 0.12
Nodes (19): LinearFunctional, r"""     Represent a linear scalar-valued map.      Parameters     ----------, LinOpQuadraticForm, Raise when ``Q`` is structurally known to be non-Hermitian., Return ``1/2 * <x, Qx> + linear(x) + a``., Return the gradient with respect to ``domain.inner``.          This is the Riesz, Return the Hessian action ``Q x`` under the Hermitian assumption., Evaluate the Riesz gradient over a leading batch axis. (+11 more)

### Community 36 - "Backend Operations"
Cohesion: 0.18
Nodes (24): ContextConflictError, ContextError, ContextInferenceError, UnknownBackendError, enforce_convert_policy(), get_context(), normalize_context(), normalize_ops() (+16 more)

### Community 37 - "Vector Spaces"
Cohesion: 0.11
Nodes (24): Deprecated Boolean view of :attr:`check_level`., Run all membership checks with a shared member/batched shape policy., _run_checks(), check_level_at_least(), enabled_to_level(), level_to_enabled(), minimum_check_level(), normalize_check_level() (+16 more)

### Community 38 - "Tree Structures"
Cohesion: 0.11
Nodes (17): asarray(), EagerControlFlowMixin, Run a while-loop eagerly, converting the predicate to ``bool`` each step., Run a scan primitive eagerly, stacking per-step outputs at the end.          ``u, Select a branch eagerly using Python truth-value conversion., Grab a representative leaf to infer leading length., Take per-step slice ``xs[i]`` along axis 0 for each leaf., Stack per-step outputs into a single pytree of arrays (axis 0). (+9 more)

### Community 39 - "Backend Operations"
Cohesion: 0.12
Nodes (22): _action_from_linop(), _action_from_quadratic_form(), power_iteration(), _power_iteration_core(), PowerIterationResult, r"""     Estimate the dominant eigenpair of a self-adjoint action.      Accept a, Store the result returned by :func:`power_iteration`.      Parameters     ------, Run the backend-loop implementation of power iteration. (+14 more)

### Community 40 - "Linear Operators"
Cohesion: 0.12
Nodes (14): Any, Apply this stacked operator over a batch and preserve codomain structure., Apply over a batch without checks and rebuild codomain representation., Apply the adjoint stacked operator over a structured product batch., r"""     Represent operators from one domain as a tree-valued map.      If ``dom, Apply the adjoint over a product batch without membership checks., Convert spaces and component operators to ``new_ctx``., Return dense adjoint matrices for the exact flat-vector fast path. (+6 more)

### Community 41 - "Conj Reshape"
Cohesion: 0.13
Nodes (4): Any, Self, DenseArray, Define the portable dense-array surface used by core abstractions.      The prot

### Community 42 - "Vector Spaces"
Cohesion: 0.11
Nodes (17): Return a shape-aware descriptor like ``ℝ^5`` or ``ℝ^(2, 3)``., Summarize leaves rather than dumping their full array contents., Summarize spectral arrays rather than dumping their full contents., field_symbol(), format_dtype(), _format_scalar(), Any, Shared helpers for compact, Jupyter-friendly ``__repr__`` output.  The goal is a (+9 more)

### Community 43 - "Backend Operations"
Cohesion: 0.12
Nodes (13): Return the subset of ``member_checks`` that apply at ``level``.          Cached, Raise if ``x`` is not a valid element of this space., General space capability: context ownership and membership checks.      Paramete, Return the mathematical scalar field derived from the context dtype.          ``, Algebraic-equality comparison, run only after the backend gate passes., Return the scalar-field glyph (``ℝ``/``ℂ``) for this space., Return a compact math descriptor used in reprs and operator arrows.          The, Return every ``SpaceCheck`` this instance must satisfy.          Walks the MRO a (+5 more)

### Community 44 - "Backend Operations"
Cohesion: 0.22
Nodes (9): _context_type(), Contextual, Infer context from `.ctx` first, then registered backend arrays., Resolve the target context for ``x``., Resolve explicit context first, then compatible inferred contexts., Resolve contexts and backend registrations., Any, BackendOps (+1 more)

### Community 45 - "Linear Operators"
Cohesion: 0.09
Nodes (12): r"""     Lazy zero map between two spaces.      ``ZeroLinOp(X, Y)`` represents t, Return the zero element of the codomain., Return the zero element of the domain., Return the batched zero element of the codomain., Return the batched zero element of the domain., Return the dense tensor representation of the zero map.          The returned ar, Return whether the zero map is Hermitian.          Returns         -------, Return whether another zero map has the same spaces. (+4 more)

### Community 46 - "Backend Operations"
Cohesion: 0.18
Nodes (15): _context_with_check_level(), leaf_spaces(), ops(), _prod_int(), Return ``ctx`` with an explicit validation policy when requested., Return the integer product of a shape tuple., Build a tuple-structured tree from ordered leaf spaces.          Parameters, Build a tree space from an example Python tree value. (+7 more)

### Community 47 - "Linear Operators"
Cohesion: 0.10
Nodes (11): IdentityLinOp, r"""     Lazy identity map on a space.      ``IdentityLinOp(X)`` represents the, Return ``x`` after domain validation., Return ``x`` after codomain validation., Return ``xs`` after batched domain validation., Return ``xs`` after batched codomain validation., Return the dense tensor representation of this identity map.          The return, Return whether another identity map has the same space. (+3 more)

### Community 48 - "Vector Spaces"
Cohesion: 0.16
Nodes (11): CoordinateSpace, Finite coordinate vector space capability.      Parameters     ----------     sh, Return a dense one-dimensional coordinate vector., Flatten a leading-axis batch of space elements to shape ``(N, size)``., Unflatten rows of shape ``(N, size)`` into a leading-axis batch., Return the leading-axis batch sum of ``x`` and ``y``., Return the leading-axis batch scalar product ``a * x``., Return ``count`` leading-axis copies of this leaf space as one space. (+3 more)

### Community 49 - "Vector Spaces"
Cohesion: 0.16
Nodes (10): InnerProductSpace, Vector space capability with an inner-product geometry.      Parameters     ----, JordanAlgebraSpace, Vector space capability with a Jordan product and spectral calculus.      Parame, Return the Jordan product of ``x`` and ``y``., Return Jordan-algebraic eigenvalues of ``x``., Return spectral data sufficient to reconstruct ``x``., Reconstruct an element from spectral data. (+2 more)

### Community 50 - "Linear Operators"
Cohesion: 0.10
Nodes (11): r"""     Lazy scalar multiple of a linear operator.      ``ScaledLinOp(alpha, A), Return ``scalar * op.apply(x)``., Return ``conj(scalar) * op.rapply(y)``., Return ``scalar * op.vapply(xs)``., Return whether this scaled operator is structurally Hermitian.          For a re, Flatten this operator for pytree registration., Rebuild this operator from pytree data., Convert the operand to ``new_ctx`` while preserving the scalar. (+3 more)

### Community 51 - "Linear Operators"
Cohesion: 0.12
Nodes (15): _is_one_scalar(), is_scalar_like(), _is_zero_scalar(), make_scaled(), Return whether ``value`` is scalar-like one., Return a locally simplified scalar multiple of a linear operator.      This fact, Return whether ``value`` can be used as a scalar multiplier for a ``LinOp``., Return whether two scalar-like values compare equal. (+7 more)

### Community 52 - "Linear Operators"
Cohesion: 0.11
Nodes (10): ComposedLinOp, r"""     Lazy composition of two linear operators.      ``ComposedLinOp(A, B)``, Return ``left.apply(right.apply(x))``., Return ``right.rapply(left.rapply(z))``., Return ``left.vapply(right.vapply(xs))``., Return whether this composition is structurally Hermitian.          A Gram produ, Return whether another composition has the same operands, in order., Flatten this operator for pytree registration. (+2 more)

### Community 53 - "Backend Operations"
Cohesion: 0.13
Nodes (10): Return a mutable copy of ``x``.          Mutable backends override this with the, Accumulate ``values`` into ``y`` at ``index`` in place.          Backend mutatio, Return ``x`` with ``x[index]`` set to ``values``.          With ``copy=True`` a, Return ``x`` with ``values`` accumulated into ``x[index]``.          With ``copy, Select between x and y by condition (delegates to xp.where)., Elementwise maximum (delegates to xp.maximum)., Elementwise minimum (delegates to xp.minimum)., Clip x into [a_min, a_max] (delegates to xp.clip). (+2 more)

### Community 54 - "Linear Operators"
Cohesion: 0.20
Nodes (17): _broadcast_space_element(), composed_functional_value_core(), inner_product_value_core(), inner_product_vvalue_core(), linear_grad_core(), linear_vgrad_core(), linop_quadratic_grad_core(), linop_quadratic_value_core() (+9 more)

### Community 55 - "Vector Spaces"
Cohesion: 0.20
Nodes (11): core_normalize(), normalize(), Return ``obj``'s bound check-free core when it is safe to use.      The core is, Resolved check-free linear-space ops for a solver's hot loop.      Each op is re, Return ``1 / value`` where ``value > 0`` and zero otherwise.      This helper is, Normalize a space member and return ``(unit, norm)``., Normalize ``x`` through resolved check-free space ops; return ``(unit, norm)``., resolve_core() (+3 more)

### Community 56 - "Linear Operators"
Cohesion: 0.14
Nodes (10): Context, LinOp, TreeSpace, Build a tuple-structured block-diagonal operator., Convert spaces and blocks while retaining explicit tree layouts., Validate a nonempty block collection and its shared execution policy., Flatten row-major blocks for JAX pytree registration., Build a one-row block matrix from a sequence of operators. (+2 more)

### Community 57 - "Backend Operations"
Cohesion: 0.16
Nodes (10): Return ``x`` after verifying that it is a dense array for this backend., Return ``x`` after verifying that it is a sparse array for this backend., Convert ``x`` to a dense backend array using this context's dtype.          Para, Convert ``x`` to a sparse backend object using this context's dtype.          Pa, Convert an existing backend array to this context.          Dense inputs are con, Return whether another object has the same execution context.          Parameter, Any, ArrayLike (+2 more)

### Community 58 - "Vector Spaces"
Cohesion: 0.23
Nodes (10): _euclidean_elementwise_jordan_incompatibility(), _is_euclidean_elementwise_jordan_compatible(), Resolve the public geometry aliases for elementwise coordinate spaces., Return the violated Euclidean-elementwise invariant, if any., Return whether elementwise coordinates have Euclidean-Jordan geometry., Raise if a Euclidean elementwise Jordan space would be untruthful., _resolve_elementwise_geometry(), _validate_euclidean_elementwise_jordan() (+2 more)

### Community 59 - "Vector Spaces"
Cohesion: 0.13
Nodes (7): r"""Bind ordered leaves to a :class:`TreeSpace`.      ``TreeElement`` is an expl, Reconstruct the Python tree value represented by this element., Expose element leaves as JAX pytree children., Rebuild an element from JAX pytree children., Flatten a matching Python tree value into ordered leaves., Bind a matching Python tree value to this space., TreeElement

### Community 60 - "Linear Operators"
Cohesion: 0.12
Nodes (9): r"""     Lazy finite sum of linear operators with common spaces.      ``SumLinOp, Return ``sum_i ops[i].apply(x)``., Return ``sum_i ops[i].rapply(y)``., Return ``sum_i ops[i].vapply(xs)``., Return whether another sum has the same operands, in order., Flatten this operator for pytree registration., Rebuild this operator from pytree data., Convert all operands to ``new_ctx``. (+1 more)

### Community 61 - "Linear Operators"
Cohesion: 0.24
Nodes (11): Core apply / evaluation kernels and their binding rules.  The check-free cores o, core_kernel_names(), CoreKernelSet, get_core_kernels(), Rules that bind *core apply kernels* to operator implementations.  A *core kerne, Record ``kernel_set`` so operators can bind it by name.      Re-registering the, Return the registered :class:`CoreKernelSet` named ``name``., Return every registered core-kernel-set name, in registration order. (+3 more)

### Community 62 - "Validation Checks"
Cohesion: 0.17
Nodes (13): lsqr(), LSQRResult, Store the result returned by :func:`lsqr`.      Parameters     ----------     x, Return a compact summary without printing the full solution array., r"""     Solve :math:`\min_x \|A x - b\|` by LSQR.      Allow ``A`` to map betwe, Return ``A``'s bound check-free adjoint core (or checked ``rapply``)., Return whether iteration ``k`` should refresh convergence diagnostics., Return the absolute-plus-relative convergence threshold. (+5 more)

### Community 63 - "Using Array"
Cohesion: 0.14
Nodes (9): Normalize a dtype specifier using NumPy.          Input:             dtype: Opti, Convert input to a sparse array using SciPy.          Input:             x: Dens, r"""         Build open mesh index arrays using NumPy.          Input:, Compare sparse arrays elementwise within tolerances using SciPy.          Input:, Dense array type using NumPy.          Returns         -------             Concr, Sparse array type tuple using SciPy.          Returns         -------, Any, DType (+1 more)

### Community 64 - "Linear Operators"
Cohesion: 0.24
Nodes (13): Concrete core kernels for :class:`spacecore.linop.SparseLinOp`.  A sparse coordi, sparse_apply_core(), sparse_euclidean_rapply_core(), sparse_euclidean_rvapply_core(), sparse_rapply_core(), sparse_rvapply_core(), sparse_vapply_core(), Exception (+5 more)

### Community 65 - "Vector Spaces"
Cohesion: 0.21
Nodes (13): check_maxiter(), default_initial_vector(), default_maxiter(), _defining_class(), Return the real part of ``space.inner(x, y)`` (checked; for one-shot use)., Return a deterministic unit vector in ``A.domain`` using its geometry., Run expensive development-time Hermitian and positive-curvature probes., Return the default Krylov iteration count for ``A``. (+5 more)

### Community 66 - "Linear Solvers"
Cohesion: 0.21
Nodes (10): cg(), CGResult, Store the result returned by :func:`cg`.      Parameters     ----------     x :, Return a compact summary without printing the full solution array., r"""     Solve :math:`A x = b` by conjugate gradients.      Require ``A`` to be, Iterative linear algebra solvers and Krylov algorithms., Return ``A``'s bound check-free forward core (or checked ``apply``)., resolve_apply() (+2 more)

### Community 67 - "Linear Operators"
Cohesion: 0.15
Nodes (6): MatrixFreeLinOp, Apply the forward callable.          Parameters         ----------         x:, Apply the adjoint callable.          Parameters         ----------         y:, Convert this matrix-free operator to ``new_ctx``.          Parameters         --, Linear operator defined by user-supplied forward and reverse callables.      ``M, Probe the declared adjoint identity on deterministic space elements.

### Community 68 - "Linear Operators"
Cohesion: 0.35
Nodes (10): diagonal_apply_core(), diagonal_euclidean_rapply_core(), diagonal_euclidean_rvapply_core(), diagonal_rapply_core(), diagonal_rvapply_core(), diagonal_vapply_core(), Concrete core kernels for :class:`spacecore.linop.DiagonalLinOp`.  A diagonal op, metric_rapply() (+2 more)

### Community 69 - "Copy Add"
Cohesion: 0.18
Nodes (7): Multiply sparse and dense arrays using SciPy.          Input:             a: Spa, Compute a stable log-sum-exp reduction using SciPy.          Input:, Return a NumPy copy of ``x`` (mutation primitive for index ops)., Accumulate ``values`` into ``y`` at ``index`` via ``numpy.add.at``., ArrayLike, DenseArray, Index

### Community 70 - "Vector Spaces"
Cohesion: 0.24
Nodes (7): _format_path(), Validate TreeElement ownership and leaf arity., Validate each TreeElement leaf against its corresponding space., Format an optree leaf path as a compact Python-style location., _TreeLeafCheck, _TreeStructureCheck, SpaceCheck

### Community 71 - "Tree Structures"
Cohesion: 0.20
Nodes (6): Store leafwise Jordan spectral data in deterministic leaf order.      Parameters, Flatten spectral data for JAX pytree registration., Rebuild spectral data from JAX pytree children., Return leafwise spectral data independent of tree structure., Reconstruct a tree element from leafwise spectral data., TreeSpectralDecomposition

### Community 72 - "Backend Operations"
Cohesion: 0.22
Nodes (4): Positive infinity as a cached backend scalar., NaN as a cached backend scalar., Pi as a cached backend scalar., Euler's number as a cached backend scalar.

### Community 73 - "Linear Operators"
Cohesion: 0.44
Nodes (8): dense_apply_core(), dense_euclidean_rapply_core(), dense_euclidean_rvapply_core(), dense_rapply_core(), dense_rvapply_core(), dense_vapply_core(), Concrete core kernels for :class:`spacecore.linop.DenseLinOp`.  A dense coordina, Any

### Community 74 - "Backend Operations"
Cohesion: 0.25
Nodes (5): Run a counted loop primitive., Run a while-loop primitive., Run backend-compatible conditional branch selection., R, T

### Community 75 - "Backend Operations"
Cohesion: 0.29
Nodes (4): Convert input to a backend sparse array., Multiply a sparse array by a dense array., Compare sparse arrays elementwise within tolerances., SparseArray

### Community 76 - "Over Given"
Cohesion: 0.29
Nodes (3): Mean over given axes (delegates to xp.mean)., Minimum over given axes (delegates to xp.min)., Maximum over given axes (delegates to xp.max).

### Community 77 - "Linear Operators"
Cohesion: 0.29
Nodes (7): _DenseMode, Private computation modes for dense coordinate operators., _DiagonalMode, Private computation modes for diagonal coordinate operators., Private computation modes for sparse coordinate operators., _SparseMode, Enum

### Community 78 - "Reshape Ravel"
Cohesion: 0.33
Nodes (3): Flatten x to one dimension., Reshape x (delegates to xp.reshape)., Return ``sum(conj(x) * y)`` over flattened inputs.          Matches NumPy, JAX,

### Community 79 - "Vector Spaces"
Cohesion: 0.47
Nodes (6): CapabilitySet, Return structural capabilities advertised by one leaf space., Return capabilities shared by every tree leaf., _space_capabilities(), _tree_capabilities(), Space

### Community 80 - "Vector Spaces"
Cohesion: 0.33
Nodes (4): QuadraticForm, Represent a scalar quadratic objective on a space.      Parameters     ---------, Apply the Hessian action at ``x`` when available., Return the gradient with respect to ``domain.inner`` when available.

### Community 81 - "Backend Operations"
Cohesion: 0.40
Nodes (4): Validate and normalize the context after dataclass initialization.          Rais, BackendOps, CheckLevel, DType

### Community 82 - "Scan Run"
Cohesion: 0.40
Nodes (4): Run a scan primitive., Carry, X, Y

## Knowledge Gaps
- **35 isolated node(s):** `BackendOps`, `DType`, `Self`, `BackendFamily`, `Carry` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BackendOps` connect `Backend Operations` to `Backend Operations`, `Backend Operations`, `Vector Spaces`, `Backend Operations`, `Validation Checks`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Vector Spaces`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Over Given`, `Reshape Ravel`, `Backend Operations`, `Scan Run`, `Diagonal Trace`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `LinOp` connect `Backend Operations` to `Backend Operations`, `Backend Operations`, `Linear Operators`, `Backend Operations`, `Linear Operators`, `Backend Operations`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Backend Operations`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Validation Checks`, `Vector Spaces`, `Linear Solvers`, `Linear Operators`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `ComposedLinOp` connect `Linear Operators` to `Backend Operations`, `Backend Operations`, `Backend Operations`, `Backend Operations`, `Linear Operators`, `Linear Operators`, `Linear Operators`, `Linear Operators`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `BackendOps` (e.g. with `Context` and `Any`) actually correct?**
  _`BackendOps` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `LinOp` (e.g. with `_DenseMode` and `_DiagonalMode`) actually correct?**
  _`LinOp` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `TreeSpace` (e.g. with `_StackedEuclideanJordanAlgebraSpace` and `_StackedEuclideanJordanStarSpace`) actually correct?**
  _`TreeSpace` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Any` (e.g. with `LinOp` and `Codomain`) actually correct?**
  _`Any` has 3 INFERRED edges - model-reasoned connections that need verification._
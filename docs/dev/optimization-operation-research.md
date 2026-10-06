# Operations for general optimization

Research and design synthesis — 26 September 2026.

## 1. Conclusion and scope

A framework can express a broad range of optimization algorithms independently of object storage, provided each algorithm declares the mathematical operations it needs. There is no useful, universally sufficient list of elementary operations that also guarantees efficient implementations and convergence for arbitrary representations.

The useful target is a **capability vocabulary**: a small algebraic foundation and optional interfaces for geometry, differentiation, constraints, subproblems, uncertainty, and search. An algorithm consumes a subset. A representation and problem jointly supply that subset.

This report derives that vocabulary from optimization families, rather than from SpaceCore's existing implementation. It covers smooth and nonsmooth continuous optimization, constrained and conic methods, stochastic optimization, manifolds, derivative-free methods, mixed integer/global search, and several structured extensions. “Most” means coverage of these major families; it is not a measured percentage of all published algorithms.

The operation names below are proposed semantic contracts, not a demand to implement one class per row. Source links identify supporting papers and solver designs. The grouping, interface recommendations, and implementation order are this report's synthesis.

### Three separate questions

1. **Expressibility:** Can the iteration be written using available operations?
2. **Faithfulness:** Do these operations implement the mathematics the iteration assumes?
3. **Practicality:** Can they do so within the time, memory, and accuracy budget?

Supplying an `add` method answers only part of the first question. Truncating its result can change the algorithm. Computing an exact answer by expanding an exponentially large tensor can make it unusable.

## 2. Mathematical objects to distinguish

| Object | Meaning | Why the distinction matters |
|---|---|---|
| Search point `x` | Candidate solution in a domain | A permutation, manifold point, or neural model need not form a vector space |
| Direction `v` | An admissible infinitesimal displacement | On a manifold it belongs to a tangent space at a particular point |
| Covector `a` | Linear functional on directions | Derivatives naturally act on directions; converting them to gradients requires geometry |
| Scalar | Objective, step, coefficient, or diagnostic | Optimization objectives need an ordering, usually real values |
| Linear operator `A` | Map between vector spaces | Application does not require a stored matrix |
| Nonlinear map `F` | Residual, constraint, model, or parametrization | Its derivative is a point-dependent linear map |
| Set `C` | Feasible candidates | Membership, projection, and linear minimization are different capabilities |
| Oracle result | Computed value with accuracy/status information | Exact, approximate, and stochastic answers have different meanings |
| Solver state | History, multipliers, estimates, active sets | It need not have the same representation as the solution |

For real-valued optimization over complex data, specify the underlying real differential. A safe convention is `Df(x)[v] = Re(inner(grad_f(x), v))`. A real-linear derivative is not necessarily complex-linear. This must be respected by automatic differentiation adapters and adjoints.

## 3. Operation catalogue

### A. Construction, algebra, and products

These are the common foundation for vector-space algorithms:

| Operation | Contract / role |
|---|---|
| `zero(space)` | Additive identity; construction without an existing dense array |
| `add(x, y)`, `scale(a, x)` | Addition and scalar action with declared closure |
| `lincomb(coefficients, elements)` | Linear combination; mathematically derived, potentially faster than repeated additions |
| `difference(x, y)` | Vector difference where it exists; not assumed on every domain |
| `update(x, step, direction)` | Affine update in a vector domain, or explicitly specified domain update |
| `product(parts)`, `component(x, i)`, `replace(x, i, value)` | Structured variables and block algorithms |
| `copy` / ownership policy | Protect stored iterates and histories from later mutation |
| `validate` / compatibility | Domain, scalar field, representation, and geometry compatibility |

Subtraction and `axpy(a,x,y)` are derived from addition/scaling. They may still deserve optimized implementations. Arbitrary multiplication of two elements is **not** a vector-space operation.

An initial iterate can be user-supplied. Random construction, a basis, dimension, indexing, and dense conversion should be optional. Some algorithms require them; many do not.

### B. Duality, norms, and geometry

| Operation | Contract / role |
|---|---|
| `pair(a, v)` | Evaluate a covector on a direction |
| `norm(v)`, `dual_norm(a)` | Size of primal directions and dual residuals |
| `inner(u, v)` | Hilbert geometry when available |
| `lower(x, v)` | Metric map from tangent vectors to covectors |
| `raise(x, a)` | Inverse metric action, when defined |
| `distance(x, y)` | Domain-specific distance; not always a norm of subtraction |

With a fixed positive definite metric matrix `M`, `lower(v)=Mv` and `raise(a)=M⁻¹a`. Gradient descent uses `raise(df(x))`. A normed space need not have an inner product or a linear inverse metric. Banach-space algorithms may instead use a duality map or a Bregman update.

Do not universally replace a norm by `sqrt(inner(v,v))`: a representation may have a more stable native norm. Different algorithm state components can use different spaces and norms.

### C. Evaluation and differentiation

| Operation | Required by / meaning |
|---|---|
| `value(x)` | Objective-based methods; allows extended-real values for domains |
| `value_and_derivative(x)` | Fused evaluation, preserving shared work |
| `differential(x)` | Covector `Df(x)` |
| `gradient(x, geometry)` | Geometry-dependent vector representing the differential |
| `directional_derivative(x, v)` | Line searches and directional models |
| `evaluate(F, x)` | Nonlinear residuals and constraints |
| `linearize(F, x)` | A reusable derivative operator at a fixed point |
| `jvp(x, v)` | `DF(x)v` without materializing a Jacobian |
| `vjp(x, a)` | Dual pullback `DF(x)*a` |
| `hessian_action(x, v)` | Second derivative action; declare whether result is dual or primal |
| `weighted_hessian_action(F, x, a, v)` | `D²(pair(a,F))(x)[v, ·]`; needed for constraint curvature |
| `subgradient(x)` | A valid subgradient under a stated nonsmooth convention |
| `generalized_jacobian_action(x, v)` | Semismooth Newton variants; an arbitrary subgradient is insufficient |

A VJP is a dual map. A metric adjoint additionally uses the domain and codomain metrics. For fixed real Hilbert geometries, `J_adjoint = raise_X ◦ J* ◦ lower_Y`.

Full matrices are optional materializations. Established interfaces already accept Hessian-vector products, and nonlinear least-squares interfaces accept Jacobians represented as linear operators. [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

Higher-order methods add multilinear derivative actions. They do not justify requiring an arbitrary derivative order from every function.

### D. Operators and numerical linear algebra

Useful operator operations are `apply`, dual pullback or metric adjoint, composition, sums, scaling, identities, and block assembly. Optional structural facts include symmetry, positive definiteness, sparsity, and a norm bound.

Numerical capabilities include:

- Linear solves, with the spaces and residual norm specified.
- Least-squares solves, including rank-deficient cases where supported.
- Preconditioner application; an approximate inverse is not an exact solve.
- Positive definite, indefinite, and nonsymmetric solve strategies.
- Reusable factorization or solver preparation, plus multiple right-hand sides.
- Orthogonalization, small Gram matrices, eigen/singular-value actions, and spectral transformations where a method requires them.
- Trust-region quadratic subproblem solves, including negative-curvature handling.

A KKT system can be indefinite even when the objective is convex. CG is therefore not a universal inner solver. Conversely, matrix-free Newton need not construct a Hessian matrix.

Preparation should be tied to the operator's parameters, geometry, and version. A changed ADMM penalty, changed Jacobian, or mutated matrix can invalidate a factorization.

PETSc/TAO provides a useful precedent for separating optimization methods from matrix implementations, derivative callbacks, and linear solver/preconditioner choices. [TAO manual](https://petsc.org/release/manual/tao/).

### E. Sets, proximal maps, and monotone operators

These capabilities belong to the function/set/operator together with its geometry:

| Operation | Mathematical meaning |
|---|---|
| `contains(C, x)` | Membership; allow unknown when exact certification is unavailable |
| `violation(C, x)` | Quantitative infeasibility with a defined scale |
| `project(C, x, metric)` | Closest feasible point in the specified metric |
| `prox(f, x, step, metric)` | Minimize `f(z) + ||z-x||²_metric/(2 step)` |
| `conjugate_value(f, a)` | Fenchel conjugate with the declared dual pairing |
| `prox_conjugate(f, a, step)` | Proximal operation in the appropriate dual geometry |
| `linear_minimize(C, a)` | Minimize `pair(a,z)` over `C` |
| `support(C, a)` | Supremum of `pair(a,z)` over `C`, possibly infinite |
| `separate(C, x)` | Valid separating hyperplane when applicable |
| `resolvent(T, x, step)` | `(I + step*T)⁻¹x` for a Hilbert-space operator |
| `operator_value(T, x)` | Forward evaluation for splitting or variational inequalities |

Projection is the proximal map of a set indicator. In a fixed Hilbert geometry, `prox_(step*f*)(y) = y - step*prox_(f/step)(y/step)`, identifying primal and dual by the Riesz map. General metric/dual implementations must account for that identification. [Parikh–Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).

A linear minimization oracle enables conditional-gradient methods and can be much cheaper than projection. Away-step variants additionally need an active decomposition and atom selection. [Jaggi, Revisiting Frank–Wolfe](https://proceedings.mlr.press/v28/jaggi13.html).

Do not infer that every differentiable function has an accessible prox, or that prox of a sum/composition is available from the component prox operations. Nonconvex proximal problems can be nonunique; selection and accuracy must be explicit.

### F. Bregman and information geometry

For a differentiable distance-generating function `h`, define

`D_h(z,x) = h(z) - h(x) - pair(dh(x), z-x)`.

The central operation is

`bregman_step(x,a,step) = argmin_{z in C} pair(a,z) + D_h(z,x)/step`.

Optional components are `h`, `dh`, inverse mirror map, and divergence evaluation. A specialized update can avoid computing them individually. Mirror-Prox adds repeated evaluation of a monotone operator and Bregman steps; its operator need not be an objective gradient. [He–Juditsky–Nemirovski, Composite Mirror-Prox](https://arxiv.org/abs/1311.1098).

Natural gradient uses a different contract: solve a point-dependent metric system. Degenerate Fisher/QGT metrics require a declared gauge, regularization, or pseudoinverse convention. NetKet's QGT implementations demonstrate matrix-free metric action and solves on parameter trees. [NetKet QGT/SR guide](https://netket.readthedocs.io/en/stable/user-guides/sr.html).

### G. Manifold geometry

Optional operations are tangent-space construction, tangent projection where an embedding exists, point-dependent inner products, retraction, vector transport, Riemannian gradient, and Riemannian Hessian action. Exponential/logarithmic maps are stronger optional capabilities; many methods only need a retraction.

Transport is needed when a method combines directions stored at different points. A Riemannian Hessian generally includes connection/curvature terms; projecting an ordinary Hessian-vector product is not a universal substitute. [Pymanopt manifold contracts](https://pymanopt.org/docs/stable/manifolds.html).

A generic `update` can dispatch to affine motion or retraction, but this alone does not turn a Euclidean optimizer into a correct Riemannian variant. Its history, derivative interpretation, and convergence conditions also need appropriate geometry.

### H. General constraints, cones, and complementarity

For `min f(x)` subject to `c(x)=0` and `g(x)≤0`, useful operations include constraint values, Jacobian actions/pullbacks, multiplier algebra, and weighted constraint Hessian actions.

SQP and interior-point methods use the Hessian of the **Lagrangian**, incorporating multiplier-weighted constraint curvature. Ipopt's interface explicitly requests this quantity and derivative sparsity information. [Ipopt interface](https://coin-or.github.io/Ipopt/INTERFACES.html).

Additional capabilities depend on the chosen formulation:

- Bound handling: feasible intervals, active coordinates, restriction/free-variable maps.
- Barriers: interior membership, barrier value/gradient/Hessian action, admissible step length.
- Cones: dual cone, complementarity pairing, cone projection or barrier/scaling operations.
- Symmetric cones: Jordan algebra operations and associated scaling when required.
- Nonsymmetric cones: suitable barriers and derivative operations; a symmetric-cone formula cannot be assumed.
- KKT solve or constrained quadratic subproblem solve, merit/filter evaluation, and restoration mechanisms.

Conic optimization includes nonnegative, second-order, semidefinite, exponential, and power cones. Their computational requirements differ. [Clarabel supported cones](https://clarabel.org/stable/api_cone_types/).

A projection method, a linear-oracle method, and a barrier method require different access to the same feasible set.

### I. Stochastic and coordinate-dependent operations

Stochastic methods need explicit randomness and sampling semantics:

- Sample an index, batch, scenario, or trajectory; advance reproducible RNG state.
- Evaluate sampled values/derivatives, and identify the distribution being sampled.
- Average or reduce estimates, with sample count and uncertainty information where available.
- Evaluate component functions for finite-sum methods; maintain reference/full gradients for variance reduction.
- Reuse a sample set when an inner iteration needs a fixed stochastic operator.

Independent sampling, correlated MCMC, biased approximation, and deterministic evaluation are different contracts. An estimated standard error is not a certified deterministic bound.

Adam/RMSProp/AdaGrad additionally need coordinatewise second-moment accumulation and diagonal scaling. These operations require a chosen coordinate structure; arbitrary vector addition and inner products do not supply them. Ordinary Adam changes under a general change of basis. [Kingma–Ba, Adam](https://arxiv.org/abs/1412.6980).

Expose coordinate algebra only where meaningful: componentwise products, absolute squares, square roots, division, clipping, and diagonal application. For complex parameters, explicitly define the real second-moment convention.

### J. Blocks, local models, and structured subproblems

Block-coordinate methods need more than a tree container:

- Select blocks and define restriction/injection operations.
- Construct the objective with the other blocks fixed.
- Evaluate partial derivatives or solve/update the selected block.
- Reconcile shared constraints and overlaps, if present.

Alternating least squares and tensor sweeps use local environments and local linear or eigenproblems. These are specialized ways to realize block operations. MPS gauge transformations preserve the represented tensor; truncation generally does not. [TensorNetwork MPS overview](https://tensornetwork.org/mps/).

Other structured extensions follow the same pattern:

- Optimal control: trajectory rollout, dynamics derivatives, adjoint sweeps, Riccati or structured KKT solves.
- Dynamic programming: states, admissible actions, transitions, stage costs, Bellman minimization, terminal values; discounting or finite-horizon assumptions.
- Bilevel optimization: inner solution plus sensitivity actions, often a transposed linear solve under differentiability and regularity assumptions.
- Majorization-minimization/proximal Newton: construct a model, solve it to a requested accuracy, evaluate acceptance conditions.

These extensions should expose meaningful local problems. A universal callback named `do_the_optimization` provides no reusable mathematical structure.

### K. Derivative-free and population methods

| Family | Additional operations |
|---|---|
| Nelder–Mead | Ordered function evaluations, an initial simplex, affine combinations |
| Pattern/direct search | Poll directions or moves, step scale, feasibility and acceptance |
| Model-based derivative-free trust region | Interpolation samples, local coordinates/model fitting, poisedness checks, model subproblem |
| CMA-ES | Gaussian sampling, weighted means, covariance updates, covariance factorization/actions |
| Differential evolution | Population selection, scaled differences, crossover structure, bounds handling |
| Simulated annealing | Proposal kernel, energy comparison, temperature schedule, acceptance randomness |
| Genetic search | Domain-specific mutation, crossover, selection, validity/repair |
| Bayesian optimization | Surrogate fit/update, posterior uncertainty, acquisition evaluation, acquisition optimization |

There is no universal “random point” distribution over an arbitrary domain. The distribution or proposal kernel is part of the algorithm's meaning.

CMA-ES adapts a search distribution; Bayesian optimization constructs and optimizes an acquisition function from a probabilistic model. These need state and operations beyond an objective evaluator. [Hansen, CMA-ES tutorial](https://arxiv.org/abs/1604.00772), [BoTorch overview](https://botorch.org/docs/overview).

### L. Discrete optimization and global certificates

Discrete/global methods need another set of capabilities:

- Enumerate or generate admissible moves, branches, or assignments.
- Test integrality and domain-specific feasibility.
- Propagate constraints and reduce variable domains.
- Solve relaxations and obtain valid lower bounds for minimization.
- Construct feasible incumbents and upper bounds.
- Generate valid cuts/separating inequalities and pricing columns where applicable.
- Partition search regions, manage search nodes, and prune with certified bounds.
- For rigorous continuous global methods, interval bounds or other valid enclosures of objective/constraints.

LP simplex also needs a basis representation, basis and transposed-basis solves, reduced costs, ratio tests, and pivots. These are not supplied by a generic gradient interface.

SCIP separates branching, separation, propagation, pricing, and constraint handlers into extensible components. [SCIP project](https://scipopt.org/).

Global search heuristics can run without lower-bound certificates. Claiming global optimality requires additional mathematical evidence. General nonconvex black-box optimization cannot acquire that evidence merely from a finite sequence of function evaluations without further assumptions.

Multiobjective methods add objective vectors, dominance/order rules, archives, or explicit scalarizations. These should not be silently coerced into a single scalar objective.

### M. Approximation, termination, and execution

Every expensive oracle should be able to report:

- Result and success/failure status.
- Accuracy requested and accuracy actually obtained.
- Residual or error measure and the norm in which it is measured.
- Whether the measure is a bound, estimate, or heuristic.
- Relevant work counters, sample counts, and reusable state.

Compression belongs here as an explicit operation: `compress(x, tolerance/budget)` with an error report when available. It must not secretly redefine exact addition. A fixed error tolerance need not preserve convergence; the admissible schedule depends on the outer method.

Stopping measures are algorithm-specific: gradient norm, projected-gradient/proximal residual, primal/dual feasibility, complementarity, a duality gap, a Frank–Wolfe gap, statistical uncertainty, or a global bound gap. A small change in the iterate alone is not a general optimality certificate.

Execution capabilities include optional batching, fused evaluation, compilation, asynchronous/distributed reductions, cancellation, and serialization. They should preserve the mathematical operation contract. Dense conversion, automatic differentiation, GPU support, and JIT are optional implementation capabilities.

## 4. Algorithm coverage matrix

Each row names a representative formulation. Variants can add or remove requirements. Algebra, appropriate scalar arithmetic, and state management are implicit where applicable.

| Algorithm family | Essential additional capabilities | Important boundary |
|---|---|---|
| Gradient descent / heavy ball / accelerated gradient | Gradient, update; value for backtracking | Acceleration assumptions and coefficients depend on problem class |
| Nonlinear CG | Gradient, inner products, line search | Descent safeguards and line-search conditions |
| L-BFGS | Gradient, differences, pairings, initial inverse-Hessian action | Curvature pairs and positive-curvature safeguards |
| Newton–Krylov | Gradient, Hessian action, inner solve, globalization | Hessian can be indefinite |
| Trust-region Newton | Value, gradient, Hessian action, norm, constrained quadratic solve | Negative curvature and actual/predicted reduction |
| Gauss–Newton / Levenberg–Marquardt | Residual, JVP/VJP, least-squares or damped normal solve | Gauss–Newton omits residual curvature |
| Subgradient descent | Subgradient, update, step schedule | Nonsmooth stationarity differs from smooth gradient tests |
| Bundle / cutting-plane | Value/subgradient or separation, stored affine models, master solve | Master optimization is a substantive dependency |
| Proximal point | Prox or resolvent | Access can be as difficult as another optimization problem |
| Proximal gradient / FISTA | Smooth gradient, prox, updates | Compatible geometry; standard FISTA needs convex assumptions |
| Projected gradient | Gradient, projection | Membership alone is insufficient |
| Douglas–Rachford / Peaceman–Rachford | Two resolvents/prox maps, affine combinations | Relaxation and convergence assumptions differ |
| PDHG / Chambolle–Pock | Forward/adjoint linear operator, primal and conjugate prox | Step control uses operator norm in the chosen geometries |
| ADMM | Coupled residual, multiplier update, two block minimization oracles | General block steps are not automatically simple prox calls |
| Forward–backward splitting | Forward operator, resolvent | Forward operator conditions matter |
| Extragradient / Mirror-Prox | Operator evaluations, projection or Bregman steps | Operator need not be a gradient |
| Mirror descent / dual averaging | Dual accumulation and mirror/Bregman update | Geometry and domain restrictions are explicit |
| Frank–Wolfe | Gradient, linear minimization, feasible convex combinations | Approximate oracle needs a stated quality measure |
| Natural gradient / SR | Differential, metric action/solve | Singular metrics and estimation require policy |
| SGD / stochastic momentum | Sampled gradient, RNG/batch contract | Noise assumptions determine guarantees |
| SVRG / SAGA | Indexed component gradients, reference gradient/history | Different state/memory requirements |
| Adam / RMSProp / AdaGrad | Sampled gradient, coordinate moments and scaling | Coordinate dependent |
| Block coordinate / ALS | Restrict/inject, partial model, local update/solve | Separability and coupling determine validity |
| Riemannian gradient | Tangent gradient, retraction, point-dependent metric | Point and tangent representations differ |
| Riemannian CG / L-BFGS | Above plus transport and compatible history updates | Euclidean history cannot be reused blindly |
| Riemannian trust region | Above plus Riemannian Hessian and tangent trust-region solve | Curvature-aware second derivative |
| Penalty / augmented Lagrangian | Constraint values/derivatives, inner unconstrained solve | Inner accuracy and penalty updates |
| SQP | Constraint linearization, Lagrangian curvature or approximation, QP solve | Active sets, restoration, globalization |
| Interior point / conic IPM | Barriers/scaling, feasible steps, KKT solve | Cone-dependent operations |
| Semismooth Newton | Residual, generalized derivative action, linear solve | Requires appropriate semismoothness |
| LP simplex / active-set QP | Coordinate/basis access, basis solves, pivots/working-set updates | Stronger structure than abstract vector algebra |
| Nelder–Mead / direct search | Values, simplex or poll generator, moves | Evaluation-only does not mean geometry-free |
| CMA-ES / population search | Sampling, ranking, population/distribution updates | Covariance and crossover structure vary by method |
| Bayesian optimization | Probabilistic model, acquisition, inner optimizer | Uncertainty model is part of the algorithm |
| Branch-and-bound/cut/price | Relaxation bounds, branches, feasibility; cuts/pricing as needed | Certified bounds for certified pruning |
| Dynamic programming | Transition and cost model, Bellman update | Requires exploitable sequential state structure |
| Bilevel / differentiable optimization | Inner solve, sensitivity or implicit derivative action | Solution map may be nonsmooth or nonunique |
| Multiobjective search | Objective vector, dominance/scalarization, archive | Pareto goals differ from one scalar minimizer |

PDHG's forward/adjoint and proximal structure is directly visible in the original method. ADMM's general formulation instead exposes block minimization under a linear coupling. [Chambolle–Pock](https://optimization-online.org/2010/06/2646/), [Boyd et al., ADMM](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html).

## 5. What can be derived, and what must be supplied?

Derive inexpensive algebraic conveniences, but do not confuse mathematical definability with a computational implementation.

| Given | Can derive | Qualification |
|---|---|---|
| Addition and scaling | Subtraction, affine combinations | Native fused implementation can be substantially better |
| Inner product | Norm | Native norm may be more stable |
| Differential and inverse metric | Gradient | Metric must be specified and invertible, or policy supplied |
| JVP/VJP and metrics | Metric adjoint / normal operator | Transpose and adjoint conventions must match |
| Derivatives of two composable maps | Chain-rule JVP/VJP | No materialized Jacobian needed |
| Prox of a convex function | Conjugate prox through Moreau | Correct dual geometry and scaling |
| Set projection | Indicator prox | In the same metric |
| Block-separable prox problems | Product prox | Coupled metrics can destroy separability |
| Function evaluator plus coordinate perturbations | Finite-difference estimates | Extra evaluations, smoothness, scale, noise; not exact derivatives |
| Operator application | Iterative linear solve | Solvability, conditioning, algorithm assumptions, and stopping remain |
| Generic optimizer | Approximate prox/subproblem | This is a nested solve, with cost and error, not a free identity |

In particular, there is no general cheap derivation of projection from membership, prox from a gradient, an inverse from an operator application, or a global certificate from local derivatives.

## 6. Representation independence: what is actually promised?

The defensible promise is:

> An algorithm can be reused unchanged across representations that implement its required operations with compatible mathematical semantics and declared accuracy.

It is not a promise of the same cost, numerical trajectory, or convergence guarantee for every object.

### MPS example

Unrestricted-rank MPS can represent sums, but ranks may grow. Keeping a fixed rank requires truncation or a manifold/variational formulation. These choices lead to different contracts:

1. Ambient vector algorithm with exact operations and growing representation cost.
2. Inexact ambient algorithm with controlled truncation errors.
3. Fixed-rank manifold algorithm with tangent geometry and retraction.
4. Alternating local optimization over tensor cores.

### Neural representation example

For `x = model(theta)`, parameter addition generally does not implement addition of the represented objects:

`model(theta1 + theta2) != model(theta1) + model(theta2)`.

Optimizing `f(model(theta))` therefore requires the parametrization and its derivative, or a native oracle that implements the requested operation in object space. A parameter gradient is not automatically the ambient object gradient. A prox in parameter space is not automatically the prox of the object-space objective.

This is the main limit to the idea that changing only storage always suffices. The framework should make the chosen formulation visible.

## 7. Recommended framework structure

Keep ownership aligned with mathematics:

| Owner | Responsibilities |
|---|---|
| Domain / geometry | Valid points, tangent/dual spaces, algebra where available, metrics, retractions |
| Function / map | Values and supported derivatives, native proximal or other problem-specific oracles |
| Set | Membership, violation, and whichever of projection/LMO/separation/barrier it supports |
| Operator | Application, pullback/adjoint, optional structure and specialized solves |
| Representation implementation | Storage, exact kernels, explicit approximation, conversion |
| Algorithm | Update equations, state, step control, stopping, accuracy requests |
| Execution implementation | Batching, compilation, device/distribution, workspace reuse |

An oracle is simply a callable providing a well-defined mathematical operation. Start with small protocols or callbacks and concrete implementations. Do not require one universal `Optimizable` object implementing every capability.

An illustrative requirement declaration could be:

```
L-BFGS: vector updates + differential/gradient + pairing + initial metric action
PDHG:   two Hilbert spaces + K/adjoint + prox_g + prox_f_conjugate
SQP:    objective derivatives + constraints/linearization + QP subproblem
CMA-ES: point coordinates + objective + distribution sampling/update
B&B:    branch generator + relaxation bounds + feasible incumbents
```

Check these requirements before iteration. Report the missing operation specifically. Do not quietly densify, approximate, change geometry, or swap the optimization problem to fill a missing capability.

Function composition should propagate only justified capabilities: values and smooth derivatives can compose by known rules; proximal accessibility and convexity require additional conditions.

## 8. Implementation priorities

The full catalogue is a roadmap, not one release's implementation list.

### Phase 1: broad continuous methods

Implement native algebra/products, norms/pairings, explicit geometry, objective evaluation, gradient/differential conventions, linear operators/adjoints, JVP/VJP, Hessian actions, and prepared linear solves. Add basic result/status and stopping contracts.

Validate with gradient descent, L-BFGS, Newton–Krylov, and Gauss–Newton. Use both ordinary arrays and a non-array representation.

### Phase 2: nonsmooth and constrained first-order methods

Add proximal maps, projection, conjugate handling, linear minimization, and block subproblem callbacks. Validate with proximal gradient, PDHG, ADMM, and Frank–Wolfe.

These methods deliberately exercise different constraint-access models.

### Phase 3: geometry and stochastic representations

Add Bregman steps, manifolds, explicit parameter maps, stochastic evaluation contracts, coordinate moments, and approximation budgets. Validate with mirror descent, a manifold optimizer, Adam on parameters, and natural gradient/SR with a metric solve.

### Phase 4: general constrained optimization

Add nonlinear constraints, weighted Hessian actions, KKT systems, barriers/cones, and structured quadratic subproblems. Validate one SQP or augmented-Lagrangian path and a conic/interior-point path.

### Phase 5: independent search extensions

Add derivative-free search, population/distribution interfaces, surrogate optimization, and branch/cut/price contracts when those families become actual implementation targets. Reuse objective and domain infrastructure without making continuous methods depend on search-tree machinery.

### SpaceCore implications, kept separate from the general result

SpaceCore already has native `VectorSpace` operations, inner-product geometry, operator application/adjoints, scalar functionals, and some quadratic Hessian actions. They form useful starting points.

Current source checks identify specific boundaries:

- `Functional` binds its domain type to `CoordinateSpace`; general evaluation should not require coordinates.
- `FunctionalScaledOperator.linearize_at` is a useful concrete derivative example, but not a general nonlinear-map derivative interface.
- Generic `InnerProduct.riesz`/`riesz_inverse` default to identity; non-Euclidean geometry should not silently inherit that meaning.
- The earlier representation research documents mandatory densification paths and coordinate assumptions in solver defaults.

Relevant source: [functional base](../../spacecore/functional/_base.py), [operator family](../../spacecore/opfamily.py), [inner-product geometry](../../spacecore/space/base/_inner_product.py).

Do not expand the backend array API until it contains every algorithm's mathematics. Many operations in this report belong to sets, maps, geometries, or specialized subproblem implementations.

## 9. Acceptance criteria

The architecture should pass mathematical and representation tests, not just method-presence tests:

1. Algebra laws and closure at the advertised accuracy.
2. Adjoint identity in the declared geometry.
3. Directional finite-difference checks for derivatives on suitable deterministic test problems.
4. Hessian symmetry where mathematically applicable.
5. Proximal optimality residuals on independently solvable examples.
6. Feasibility and oracle gaps for projections and linear minimization.
7. Manifold validity and tangent/retraction consistency.
8. Agreement across exact representations of the same mathematical problem.
9. Explicit reports when approximation or sampling limits achievable accuracy.
10. Representative algorithms execute without accessing storage internals or forcing dense conversion.
11. Instrumentation verifies expensive preparation is reused only while valid.
12. A local convergence status cannot be mistaken for a global optimality certificate.

The report adds a design document only. No solver implementations or numerical guarantees have been added by this research.

# JIT Traceability Audit

This audit was generated with `scripts/jit_audit.py`. The script:

- wraps each solver with `jax.jit`;
- calls each jitted wrapper twice with shape/dtype-stable values;
- calls it again with a changed static iteration argument;
- calls it again with a changed operator/domain shape;
- writes a `jax.make_jaxpr` fixture for `lanczos_smallest` to
  `tests/fixtures/jaxpr_lanczos_smallest.txt`.

The script also enables `jax_log_compiles`. In addition to JAX's compile logs,
it uses a trace-time counter, which increments only when JAX retraces the Python
wrapper.

## Summary

| Solver | Traces cleanly | Recompiles/retraces on same shape/dtype values | Retraces when expected | Evidence |
| --- | --- | --- | --- | --- |
| `cg` | Yes | No. Trace count stayed at 1 after two calls. | Yes. Static `maxiter` change raised trace count to 2; shape change raised it to 3. | `scripts/jit_audit.py` output: `{'solver': 'cg', 'traces_after_two_same_shape_calls': 1, 'traces_after_static_change': 2, 'traces_after_shape_change': 3, ...}` |
| `lsqr` | Yes | No. Trace count stayed at 1 after two calls. | Yes. Static `maxiter` change raised trace count to 2; shape change raised it to 3. | `scripts/jit_audit.py` output for `lsqr`. |
| `lanczos_smallest` | Yes | No. Trace count stayed at 1 after two calls. | Yes. Static `max_iter` change raised trace count to 2; shape change raised it to 3. | `scripts/jit_audit.py` output for `lanczos_smallest`. |
| `power_iteration` | Yes | No. Trace count stayed at 1 after two calls. | Yes. Static `maxiter` change raised trace count to 2; shape change raised it to 3. | `scripts/jit_audit.py` output for `power_iteration`. |
| `expm_multiply` | Yes | No. Trace count stayed at 1 after two calls. | Yes. Static `max_iter` change raised trace count to 2; shape change raised it to 3. | `scripts/jit_audit.py` output: `{'solver': 'expm_multiply', 'traces_after_two_same_shape_calls': 1, 'traces_after_static_change': 2, 'traces_after_shape_change': 3, ...}` |

## Findings

### 1. Lanczos full reorthogonalization now uses a vectorized exact-VectorSpace path

`tests/fixtures/jaxpr_lanczos_smallest.txt` captures the current JAXPR for
`lanczos_smallest(max_iter=3)`. Grep evidence:

```text
grep -n "scan\\|while\\|scatter\\|dot_general" tests/fixtures/jaxpr_lanczos_smallest.txt
```

The initial audit fixture showed:

- a top-level `while` at line 61 for the Krylov iteration;
- a nested `scan` at line 143 corresponding to
  `ops.fori_loop(0, max_iter + 1, fill_coeff, coeffs_full)`;
- repeated `scatter` operations inside that scan.

This is correct, but for the common exact `VectorSpace` case it is more IR than
needed. Mathematically, Euclidean reorthogonalization coefficients are
`conj(V) @ w`, which can be one `einsum`/matmul node.

Important constraint: this replacement is **not valid for arbitrary
`Space.inner`**. A weighted space, RKHS, or any custom geometry must keep using
`Space.inner(v_j, w)`. Therefore the optimization should be guarded to the exact
`VectorSpace` type only, not subclasses.

The implemented exact-`VectorSpace` path now lowers the coefficient computation
to a single `dot_general` at fixture line 143. Remaining `scan` operations in
the fixture are the fixed-size tridiagonal construction loops, not the
reorthogonalization coefficient loop.

### 2. `cg` and `lsqr` use `ops.cond` correctly for periodic diagnostics

Both solvers trace without errors and do not retrace on value-only changes. The
`ops.cond(..., lambda _: ..., lambda _: ..., operand)` pattern is valid under
the installed JAX version. Both branches return matching shapes and dtypes.

### 3. `power_iteration` dispatch is trace-time, not data-dependent

`power_iteration` branches on whether the first argument is a `LinOp` or
`QuadraticForm`. That branch is Python-level and happens at trace time. This is
acceptable because the object type is static in the pytree structure; changing
from a `LinOp` to a `QuadraticForm` should retrace.

### 4. Algebra construction inside JIT is possible but theoretical

The algebra factories use Python `isinstance` checks for symbolic simplification.
Those checks are not data-dependent on traced arrays. If users construct new
operator expressions inside a jitted function, that Python algebra executes at
trace time and contributes to trace cost. This is a usage-pattern concern, not a
solver bug: normal use passes already-built operators into jitted numerical
kernels.

### 5. Constants are mostly static by design

Iteration counts (`maxiter`, `max_iter`) are static in the audit wrappers.
Changing them retraces, which is expected because loop bounds and fixed-size
work arrays change. Scalars such as tolerances are currently Python arguments
converted through `ops.asarray`; changing them may retrace unless callers pass
them through a wrapper as array values. This is acceptable for the current API.

## Implemented Change

- Added a Euclidean fast path for Lanczos reorthogonalization when
  `type(A.domain) is VectorSpace`: compute all coefficients with
  `ops.einsum("jn,n->j", ops.conj(V_), w)`.
- Kept the existing `Space.inner` loop for all non-exact `VectorSpace` domains
  to preserve Space geometry.

This is the only validated change from this audit. The broader replacement
`V @ w` is rejected for non-Euclidean spaces because it would regress the
geometry-correct Lanczos recurrence.

## Follow-Up TODO

1. Consider a benchmark for exact `VectorSpace` Lanczos before/after the
   reorthogonalization fast path.
2. Consider vectorizing fixed-size tridiagonal construction if the remaining
   construction `scan` nodes show up in JAX profiling.
3. If users report trace-time issues from constructing algebra expressions
   inside `jax.jit`, document that operators should be built outside the jitted
   function.

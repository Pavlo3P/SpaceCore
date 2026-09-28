# Backend array typing investigation

## Result

Backend operations and explicitly constructed contexts now preserve native dense
array types: `NDArray[Any]` for NumPy, `jax.Array` for JAX, and `torch.Tensor` for
Torch. Portable `DenseArray` arguments accept native arrays in both Pyright and
mypy. Optional backend exports, elementwise operations, and checked methods have
visible signatures. The distribution includes `py.typed`.

## Reproduced causes

The initial consumer reproduction produced six Pyright errors. Independent
failures explained both missing hints and false warnings:

| Cause | Observed consequence | Correction |
| --- | --- | --- |
| Protocol operators allowed keyword parameters | NumPy, JAX, and Torch arrays were rejected as `DenseArray` | Make operator/index arguments positional-only |
| Eager mixin declared `asarray` and `stack` as `Any` | Those declarations shadowed the backend implementations in the static MRO | Remove the declarations; cast the host only inside the mixin helper |
| Array results always used the portable protocol | Native methods such as `astype`, `block_until_ready`, and `detach` disappeared | Specialize a covariant result type on each typed backend and carry it through `Context` |
| Elementwise methods were installed with `setattr` | `sqrt`, `conj`, and other operations appeared absent | Declare ordinary methods with explicit signatures |
| Optional classes were inserted into `globals()` | `sc.JaxOps` and `sc.TorchOps` appeared absent | Add explicit `TYPE_CHECKING` imports and exports while preserving runtime optional-import guards |
| `checked_method` returned `Callable[..., Any]` | Decorated methods lost parameters and return types | Preserve the original callable type with a bounded type variable |
| No `py.typed` marker | Installed-library annotations were not reliably discoverable | Package the standard marker |

Callable compatibility includes parameter kinds and names. A protocol promising
`x.__getitem__(key=...)` cannot be implemented by a positional-only native
method. The correction follows the [Python callable specification](https://typing.python.org/en/latest/spec/callables.html).

The marker follows the [typing distribution specification](https://typing.python.org/en/latest/spec/distributing.html).
The native aliases follow [NumPy's typing API](https://numpy.org/doc/stable/reference/typing.html)
and [JAX's public Array interface](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html).

## Why this design

`BackendOps[DenseT]` describes the dense result family. Its input contracts remain
portable protocols, preserving existing backend-independent algorithms and
custom backend compatibility. An unspecified type parameter defaults to
`DenseArray`, so a heterogeneous `list[BackendOps]` still has useful metadata
and operation hints. Covariance allows concrete backends to flow into this
portable API. This follows the [generic typing rules](https://typing.python.org/en/latest/spec/generics.html).

`typing-extensions>=4.12` supplies type-variable defaults on supported Python
3.11/3.12. It is a small runtime dependency; no checker plugin is needed.

Alternatives considered:

- Replacing array protocols with `Any` removes warnings but also removes error
  detection and completion. Negative regression cases guard against this.
- A union of every native backend array loses the relation between the selected
  backend and its output and couples portable typing to optional packages.
- A separate stub tree duplicates signatures. Inline generic annotations and
  ordinary forwarding methods keep the implementation and contract together.
- Shape/dtype generics would require an additional contract for promotion,
  reductions, sparse layouts, and ambient context changes. This fix promises
  the array family and keeps those existing runtime checks.
- Signature preservation with a callable type variable retains generic methods
  without rewriting the validation wrapper around a `ParamSpec`.

String and ambient contexts deliberately retain portable results. Backend names
are registered dynamically, so a literal name alone cannot safely prove its
native array class.

## Runtime consistency

Testing the new result annotations exposed a pre-existing contract violation:
NumPy elementwise operations on zero-dimensional arrays, vector `matmul`,
scalar `einsum`, and SciPy `logsumexp` could return NumPy scalars. Their outputs
now go through the existing `_as_array` helper, as reductions already did.
Array-valued results retain values and dtype and become zero-dimensional arrays
when necessary. `logsumexp(return_sign=True)` normalizes both tuple members.
CuPy uses the same helper; JAX and Torch native arrays pass through unchanged.

The forwarding methods retain keyword forwarding. Triangular operations pass
`k` by keyword because Torch's Array API adapter requires that spelling.

## Verification

The consumer fixtures in `tests/typing` contain exact `assert_type` assertions
and intentionally invalid calls. Pyright's unnecessary-ignore check and mypy's
unused-ignore check ensure the invalid cases still fail rather than turning
into `Any`. Fixtures cover protocol acceptance, constructors, stacking,
contractions, inherited methods, explicit contexts, backend lists, decorated
methods, and native-only array methods. They are excluded from pytest execution.

Run the commands in `CONTRIBUTING.md`. CI checks both full optional-backend and
NumPy-only installations. The full runtime suite, backend scalar-result tests,
JIT audit, docstring audit, and an isolated wheel consumer also validate the fix.

Local verification on 2026-09-26:

- Pyright 1.1.414 and mypy 2.3.1: all consumer assertions and negative cases pass.
- Pyright on the changed backend/type/decorator/context implementation: no errors
  or warnings.
- Runtime suite: 3,640 passed, 170 skipped, one expected failure; NumPy 2.4.2,
  JAX 0.10.2, and Torch 2.11.0 on Python 3.13.
- A clean Python 3.12 wheel installation with NumPy 2.5.3 and no optional
  backends: both consumer type checks, marker discovery, and import/result checks
  pass from outside the checkout.
- Ruff, docstring audit, JIT audit, and the CI Sphinx build with warnings treated
  as errors pass. Python 3.11 consumer syntax is also checked by the static tools;
  the CI matrix supplies actual Python 3.11/3.12 runtime coverage.

## Limits

- This is a backend typing repair, not a claim that every SpaceCore module passes
  strict checking. Consumer checks use mypy's `--follow-imports=silent` to focus
  diagnostics on the public contract.
- Space/operator APIs continue to describe portable representations; they do
  not infer a native array class from ambient state. Call `ctx.assert_dense`
  with an explicit typed context when a native backend boundary is needed.
- Sparse and CuPy results retain portable protocols. Native sparse format and
  CuPy typing need usable upstream metadata and separate coverage. CuPy runtime
  tests require a CUDA environment and were not run locally.
- Native dtype, shape, device, and gradient state are not statically tracked.
  NumPy results use `NDArray[Any]` because operations can change dtype.
- Backend namespace escape hatches (`xp`, `np`, `jnp`, etc.) remain dynamically
  typed. The supported typed route is the declared `BackendOps` surface.

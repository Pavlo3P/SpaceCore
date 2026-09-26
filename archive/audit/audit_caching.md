# Cacheability Audit

This audit covers the requested hot paths and separates construction-time state
from per-call work. Evidence comes from direct source inspection with line
numbers and grep checks such as:

```text
grep -R "_dom_is_flat\\|_cod_is_flat\\|flatten_batch\\|unflatten_batch\\|coeffs_full = ops.zeros\\|fori_loop(0, max_iter + 1" -n spacecore/linop spacecore/linalg
```

## Candidates

| Location | Per-call work | Evidence | Cost of caching | Benefit | Recommendation |
| --- | --- | --- | --- | --- | --- |
| `spacecore/linop/_dense.py:76-81` | Reshapes `x` when the domain is not flat, multiplies by cached `_A2`, and reshapes/unflattens output. | `_A2`, `_A2T`, `_A2H`, `_dom_is_flat`, `_cod_is_flat`, sizes are already constructed at lines 46-57. | Additional caching would duplicate shape tuples only; < 100 bytes. | Negligible. The expensive matrix multiply dominates. | Don't cache more. Existing construction-time cache is appropriate. |
| `spacecore/linop/_dense.py:92-97` | Reshapes `y`, multiplies by cached `_A2H`, reshapes/unflattens output. | `_A2H` is already cached at line 53; flat flags are cached at lines 54-55. | Additional shape tuple cache only; < 100 bytes. | Negligible relative to matvec. | Don't cache more. |
| `spacecore/linop/_dense.py:190-222` | Leading-axis batched reshape and batched dense matmul using `flatten_batch` / `unflatten_batch`. | `vapply` and `rvapply` call the domain/codomain batch flattening helpers directly. | A general cache keyed by input shape would be an unbounded dict and would not be a pytree leaf. | Low unless repeatedly applying custom non-vector spaces with the same leading batch shape. | Don't cache. Avoid unbounded mutable instance state for a narrow path. |
| `spacecore/linop/_sparse.py:74-96` | Reshapes vectors, sparse matvec with cached `_AH`, output reshape/unflatten. | `_AH` and flat flags are cached at lines 47-52. | Additional cache would be shape tuples only. | Negligible relative to sparse matvec. | Don't cache more. |
| `spacecore/linop/_sparse.py:210-242` | Leading-axis batched sparse matmul using `flatten_batch` / `unflatten_batch`. | Same pattern as dense batched application. | Unbounded dict if keyed by input shape; not JAX-friendly. | Low for the common vector-space fast path. | Don't cache. |
| `spacecore/linop/_diagonal.py:146-153` | Leading-axis vectorized diagonal multiply on every `vapply`/`rvapply`. | The methods multiply by broadcast-compatible diagonal arrays. | No reusable derived object beyond the diagonal itself. | Low: the elementwise multiply dominates for large arrays. | Don't cache. Keep stateless and JAX-safe. |
| `spacecore/linop/_algebra.py:283-297` | `SumLinOp.apply/rapply` loops over operands and accumulates results. | Lines 286-288 and 294-296. | No reusable derived object; caching partial sums would depend on input. | None. Work is mathematical operator application. | Don't cache. |
| `spacecore/linop/_algebra.py:397-405` | `ComposedLinOp.vapply` / `rvapply` delegates through the right and left operators. | Vectorized paths compose the two vectorized applications directly. | No reusable derived object; caching would depend on input shape and nested operator state. | Low and only for repeated vectorized composition with the same shapes. | Don't cache. |
| `spacecore/linop/_algebra.py:878-894` | `_AdjointViewLinOp` delegates to the wrapped operator. | Lines 881, 886, 890, 894. | Nothing useful to cache. | None. Delegation is already minimal. | Don't cache. |
| `spacecore/linalg/_lanczos.py:108-112` | Builds `e0` and normalized `e0_unit` once per `lanczos_smallest` call. | Lines 108-112. | Caching on `A.domain` would add mutable state to spaces and complicate JAX pytree semantics. Memory is `O(n)`. | Low: once per solver call, not per iteration. | Don't cache. |
| `spacecore/linalg/_lanczos.py:163-170` | Allocates `coeffs_full = zeros(max_iter + 1)` inside every Lanczos iteration, then fills it with a `fori_loop`. | Grep shows `coeffs_full = ops.zeros((max_iter + 1,), dtype=ctx.dtype)` at line 163 inside `body_fun`, followed by `ops.fori_loop(0, max_iter + 1, ...)` at line 170. | One extra closure-captured zero vector of length `max_iter + 1`; bytes are about `(max_iter + 1) * itemsize`, usually a few hundred bytes. | Moderate: avoids allocating the same small vector `m` times per Krylov call. This is also the path Task 3 will revisit for trace cleanliness. | Cache/hoist a zero template outside `body_fun` and reuse it as the initial coefficient vector. |

## Recommended

- Hoist Lanczos `coeffs_full` zero-vector allocation out of `body_fun`.
  This removes one small allocation per Krylov iteration with negligible memory
  cost and no API change.

Everything else is either already cached (`_A2`, `_A2H`, `_AH`, flat flags) or
would require mutable shape caches for small tuple/reflection work. Those are
not worth the added state, especially for JAX pytree compatibility.

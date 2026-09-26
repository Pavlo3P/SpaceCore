# Task: Generalize `eigh` to a Jordan-algebra `spectrum`

Replace the special-cased `eigh` (defined on `HermitianSpace`, raises on `VectorSpace`) with a uniform spectral contract grounded in Jordan algebras: a vector's spectrum is itself, a Hermitian matrix's spectrum is its eigenvalues, a product's spectrum is the concatenation of component spectra. Make every path batch-safe so spectra vectorize over leading axes.

Do phases in order. Run the full suite after each. Keep `eigh` as a deprecated alias for backward compatibility.

## Governing contract (applies to every phase)

**All spectral outputs have shape `(..., rank)`: eigenvalues live on the last axis, all leading axes are batch.** This single invariant drives the rest:

- `VectorSpace`: element `(..., n)` → spectrum `(..., n)` (its own coordinates; `rank = n`).
- `HermitianSpace`: element `(..., d, d)` → spectrum `(..., d)` (`rank = d`).
- `ProductSpace`: component spectra each `(..., rank_i)` → **concatenate along the last axis** (`axis=-1`) to `(..., Σ rank_i)`. Never default `concatenate` (axis=0) — that would merge the batch axis and silently corrupt batched spectra.
- All reconstruction (`from_spectrum`, `eig_to_dense`) operates on trailing axes and broadcasts over leading ones.

**`spectrum` vs `spectral_decompose` is a hard semantic boundary, not a convenience:**

- **`spectrum` (eigenvalues only) suffices for spectral *scalars*** — `logdet`, `trace_exp`, `trace_f` — because they are `reduce(f(λ))`.
- **`spectral_decompose` (eigenvalues + frame) is *required* for anything producing an *element*** — PSD projection, spectral clipping, `matrix_log`/`matrix_exp`, and the *gradients* of spectral functions — because they must reassemble via the frame.

So `spectral_decompose` is **load-bearing for SDP barrier gradients**, not optional polish. Implement it fully for vector and Hermitian.

## Phase 1 — Spectral contract on `Space`

**Do.** Add to the `Space` base:
```python
def spectrum(self, x):
    """Jordan-algebraic eigenvalues of x, shape (..., rank). Default: not defined."""
    raise NotImplementedError(f"{type(self).__name__} defines no spectrum.")

def spectral_decompose(self, x):
    """Return (eigvals, frame), eigvals shape (..., rank).

    `frame` is a *reconstruction* frame, not literally the Jordan idempotents:
    for HermitianSpace it is the eigenvector matrix Q (with Jordan idempotents
    c_i = q_i q_i^*); `from_spectrum` reconstructs the element from (eigvals, frame).
    """
    raise NotImplementedError

def from_spectrum(self, eigvals, frame):
    """Reconstruct an element from spectral data (inverse of spectral_decompose)."""
    raise NotImplementedError
```
Document the Jordan-algebra framing: spectrum of a direct sum is the union of spectra; the vector algebra (elementwise product) has the vector as its own spectrum; the symmetric/Hermitian-matrix algebra has eigenvalues. Document that `frame` is a reconstruction frame, and the relation to Jordan idempotents ($c_i = q_i q_i^*$), so no caller assumes `frame` is a set of orthogonal idempotents.

**Acceptance.** Base contract present; default raises; the `(..., rank)` shape contract and the `spectrum`/`spectral_decompose` boundary are documented.

## Phase 2 — Per-space implementations

**Do.**
1. **`VectorSpace`** (Jordan algebra of elementwise product):
   - `spectrum(x) = x` (returning the same object is fine as an optimization but is **not** part of the contract).
   - `spectral_decompose(x) = (x, None)` (frame trivial).
   - `from_spectrum(eigvals, frame) = eigvals`.
   - Remove the `eigh` raise; `eigh` becomes a deprecated alias that warns and calls `spectral_decompose`.
   - Shape: element `(..., n)` → spectrum `(..., n)`; already fits the contract, no reshape.
2. **`HermitianSpace`:**
   - `spectrum(x) = ops.eigh(x)[0]` → `(..., d)`.
   - `spectral_decompose(x) = ops.eigh(x)` → `(evals (..., d), evecs (..., d, d))`.
   - `from_spectrum(evals, evecs) = eig_to_dense(evals, evecs)`.
   - Keep `eigh` as an alias of `spectral_decompose` with a `DeprecationWarning`.
   - **Batch-safety fix (both required):**
     - `symmetrize`: `x.T.conj()` → `ops.conj(ops.swapaxes(x, -1, -2))`.
     - `eig_to_dense`: replace the 2D-only `(evecs * evals) @ evecs.T.conj()` (broken on two counts — `.T` and the `evecs * evals` broadcast) with an explicit batched reconstruction along the last two axes. Prefer an `einsum` that avoids materializing a diagonal:
       ```python
       ops.einsum("...ij,...j,...kj->...ik", evecs, evals, ops.conj(evecs))
       ```
       (morally `evecs @ diag(evals) @ evecs.conj().swapaxes(-1,-2)`).
   - `ops.eigh` is already batched in NumPy/JAX; verify `spectrum`/`spectral_decompose`/`from_spectrum` all work on `(N, d, d)` and general `(..., d, d)`.
3. **`ProductSpace`:**
   - `spectrum(x) = ops.concatenate([s.spectrum(p) for s, p in components], axis=-1)`. **`axis=-1` is mandatory** — default would merge the batch axis. Thread components through the structure adapter if Task A landed, else tuple unpacking.
   - `spectral_decompose` returns per-component decompositions in a structured form (list/tuple of `(evals_i, frame_i)` aligned to components); `from_spectrum` rebuilds componentwise via each component space's `from_spectrum`. This is component-delegating, not a single fused frame — document it as such.
4. **`StackedSpace`** (if present): `spectrum` lifts elementwise over the stack/leading axis (each copy's spectrum), output `(..., count, rank)` consistent with the trailing-`rank` contract; `spectral_decompose`/`from_spectrum` lift elementwise likewise.

**Acceptance.**
- `VectorSpace.spectrum(x)` is **allclose-equal** to `x` (value equality, not object identity); shape `(..., n)`.
- `HermitianSpace.spectrum` equals `ops.eigh(x)[0]` and works batched on `(N,d,d)` and `(...,d,d)`; `eig_to_dense`/`symmetrize` are batch-correct; `from_spectrum(spectral_decompose(x)) ≈ x` batched.
- `ProductSpace.spectrum` concatenates along the last axis and is shape-correct for batched component spectra.

## Phase 3 — Migrate internal callers + alias

**Do.**
1. Find internal `eigh` callers (`grep -rn "\.eigh(" spacecore/`) — `psd_proj`, any solver/functional. Route element-producing ones (`psd_proj`, projections, matrix functions) through `spectral_decompose`/`from_spectrum`; route scalar ones (logdet, trace functions) through `spectrum`. Keep behavior identical.
2. `eigh` stays as a thin alias emitting `DeprecationWarning` (warn-once per call site, mirror the existing pattern). Do not remove this release.
3. Confirm the SDP spectral functions split correctly across the boundary:
   - `logdet`, `trace_exp`, `trace_f` → `f(space.spectrum(x))` (scalars, `spectrum` suffices).
   - PSD projection, spectral clipping, `matrix_log`, `matrix_exp`, spectral-function gradients → `spectral_decompose` + `from_spectrum` (need the frame).
   Note any SDP operation that `spectrum` alone cannot express (expected: all the element-producing ones — confirm they use `spectral_decompose`).

**Acceptance.** Internal callers work via the new contract; scalar vs element operations use `spectrum` vs `spectral_decompose` respectively; `eigh` still works with a deprecation warning; no behavior change.

## Phase 4 — Tests

`tests/spaces/test_spectrum.py`:
- **`VectorSpace`:** `spectrum(x)` allclose-equals `x` (not `is`); shape `(..., n)`; `from_spectrum(spectral_decompose(x)) ≈ x`.
- **`HermitianSpace`:** `spectrum` equals `ops.eigh(x)[0]`; **batched `(N,d,d)` matches per-slice looping**; `from_spectrum(spectral_decompose(x)) ≈ x` batched; `eig_to_dense`/`symmetrize` correct on `(...,d,d)`; verify the `einsum` reconstruction equals the dense `Q diag(λ) Q^*` reference on a small case.
- **`ProductSpace`:** `spectrum` = last-axis concatenation of component spectra; **batched** product (each component carrying a leading batch axis) produces shape `(..., Σ rank_i)` — explicitly test that a batched product spectrum does not collapse the batch axis (regression for the `axis=-1` fix); mixed vector+Hermitian product.
- **Jordan-algebra law:** spectrum of a direct sum = union (concatenation) of component spectra.
- **Boundary:** a spectral scalar (e.g. `logdet`) computed via `spectrum`, and an element op (e.g. PSD projection) computed via `spectral_decompose`, both correct.
- **`eigh` alias** emits `DeprecationWarning` and returns the same as `spectral_decompose`.
- NumPy and JAX; batched cases under both.

**Acceptance.** All pass; deprecation warning fires; Hermitian spectrum batch-correct; batched product spectrum keeps its batch axis.

## Constraints
- **`(..., rank)` shape contract everywhere; product spectra concatenate on `axis=-1`.**
- Vector spectrum equality is **value** (allclose), never object identity.
- `spectral_decompose` is required (load-bearing for SDP gradients), not optional.
- `frame` is a reconstruction frame; document the Jordan-idempotent relation, don't claim `frame` *is* the idempotents.
- Hermitian reconstruction batch-safe via `einsum`/`swapaxes(-1,-2)`, never `.T` or a materialized `diag`.
- Backward compatible: `eigh` remains (deprecated), behavior preserved.
- Don't build a Jordan-algebra class hierarchy — implement the contract on existing spaces; the algebra framing is the documented law, not new types.
- JIT-safe and pytree-safe.
- Product spectrum threads the structure adapter if Task A is present; else tuple unpacking.

## Report
Per phase: tests passing/failing; internal `eigh` callers migrated and which side of the `spectrum`/`spectral_decompose` boundary each landed on; confirmation Hermitian spectrum and `eig_to_dense` are batch-correct (and the `einsum` matches the dense reference); confirmation batched product spectrum preserves the batch axis under `axis=-1`; that the `eigh` alias warns. Note that `ProductSpace.spectral_decompose`/`from_spectrum` are component-delegating, and any SDP spectral function `spectrum` alone can't express.
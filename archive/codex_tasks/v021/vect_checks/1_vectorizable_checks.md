# Task: Vectorizable, axis-aware space checks

Make every `SpaceCheck` operate on trailing ("core") axes so the same check validates a single element `(…shape)` and a leading-batched stack `(N, …shape)` or `(…, …shape)` identically. Unify `_check_member` and `_check_batched` into one path parameterized by whether leading dims are allowed.

Most checks already use trailing-axis ops (`HermitianCheck` uses `swapaxes(-1,-2)`); this task makes that a contract and fixes the ones that don't.

Do phases in order. Run the full suite after each.

## Phase 1 — Core-rank contract on `SpaceCheck`

**Do.** In `spacecore/space/_checks.py`, add to the `SpaceCheck` base:
```python
class SpaceCheck(ABC):
    core_rank: int = 0   # trailing axes that define one element; 0 = rank-agnostic
    def is_valid(self, space, x) -> bool: ...   # MUST use trailing axes only
```
- `is_valid` implementations must reference only trailing axes (`x.shape[-core_rank:]`, `swapaxes(-1,-2)`, reductions over the whole array) so leading dims are free.
- Add a helper on the base: `core_shape(space)` returns the trailing shape one element should have, and `leading_dims(x, space)` returns the batch dims.

**Acceptance.** Base contract compiles; existing checks still run.

## Phase 2 — Make each check trailing-axis correct

**Do.** Audit and fix each:
- `BackendCheck`, `DTypeCheck` → `core_rank = 0`, rank-agnostic (already batch-safe; no change beyond declaring rank).
- `ShapeCheck` → `core_rank = len(space.shape)`. Validate `x.shape[-core_rank:] == tuple(space.shape)` and leading dims are unconstrained (member mode requires exact, see Phase 3). Currently it likely checks full-shape equality — change to trailing-match.
- `SquareMatrixCheck` → `core_rank = 2`; check `x.shape[-1] == x.shape[-2]`. Likely already last-two-dims.
- `HermitianCheck` → `core_rank = 2`; already uses `swapaxes(-1,-2)` + `allclose`; confirm it validates `(d,d)`, `(N,d,d)`, `(…,d,d)` identically.
- `ProductStructureCheck`, `ProductComponentCheck` → recurse into components with the same member/batch mode (coordinate with Task A's structure adapter if landed; otherwise tuple).

**Acceptance.** Each check's `is_valid` works on a single element and a batched stack with no code change between the two calls. A Hermitian check passes `(4,4)` and `(8,4,4)`, fails a stack with one non-Hermitian slice.

## Phase 3 — Unify member and batched validation

**Do.**
1. Introduce one validation entry that both `_check_member` and `_check_batched` route through:
   ```python
   def _run_checks(space, x, *, allow_leading: bool):
       # member: allow_leading=False -> x.shape must equal core_shape exactly
       # batched: allow_leading=True -> x.shape[-core_rank:] must match; leading dims free
   ```
2. `_check_member(x)` = `_run_checks(x, allow_leading=False)`.
3. `_check_batched(space, xs)` (currently in `spacecore/_batching.py`) = `_run_checks(xs, allow_leading=True)`. Keep the existing product-tuple recursion.
4. Each check decides validity on trailing axes; the `allow_leading` flag only governs whether extra leading dims are permitted, not the per-axis logic.

**Acceptance.** `_check_member` rejects a batched stack (extra leading dim) as before; `_check_batched` accepts it; both reuse the same checks. Existing `enable_checks` tests pass.

## Phase 4 — Tests

`tests/spaces/test_vectorized_checks.py`:
- Each check validates single + batched identically (parametrize over shapes `(d,d)`, `(N,d,d)`, `(M,N,d,d)`).
- Hermitian: passes batched Hermitian, fails when one slice is non-Hermitian.
- Shape: accepts correct trailing dims with arbitrary leading; rejects wrong trailing dims; member mode rejects leading dims.
- DType/Backend: unaffected by batching.
- NumPy and JAX (JAX must work under `jit` where checks are disabled in trace, enabled eager).

**Acceptance.** All pass; existing check behavior for single elements unchanged.

## Constraints
- Trailing-axis only; never hard-code 2D where a batch could appear.
- `core_rank = 0` checks stay rank-agnostic.
- Member vs batched differ only by `allow_leading`; per-axis logic identical.
- No behavior change for single-element member checks (regression-safe).

## Report
Per phase: which checks needed changes vs already compliant; confirmation member/batched now share one path; the parametrized shapes each check was verified against.
# Task: Make `SpaceCheck.__call__` respect member-check semantics

## Problem

After vectorizable checks, each `SpaceCheck.is_valid(space, x)` is trailing-axis aware and may accept leading batch dimensions. This is correct for `_check_batched`, but it creates a semantic leak:

```python
ShapeCheck()(space, batched_x)
```

currently accepts `batched_x` because `__call__` delegates directly to `is_valid`.

Direct `SpaceCheck.__call__` should behave like a member check, not a batched check.

## Required change

In `spacecore/space/_checks.py`, change `SpaceCheck.__call__` so it delegates to the unified validation path with:

```python
allow_leading=False
```

not directly to `is_valid`.

The intended semantics are:

* `check(space, x)` means: validate one space element.
* `_check_member(space, x)` uses the same semantics.
* `_check_batched(space, xs)` is the only path that allows leading batch dimensions.
* `is_valid(space, x)` remains trailing-axis aware and batch-safe internally.

## Tests

Add regression tests:

1. `ShapeCheck()(space, x)` accepts `x.shape == space.shape`.
2. `ShapeCheck()(space, xs)` rejects `xs.shape == (N, *space.shape)`.
3. `_check_batched(space, xs)` still accepts `xs.shape == (N, *space.shape)`.
4. Hermitian and product checks preserve existing behavior.

## Acceptance

* Direct check calls no longer silently accept batched arrays.
* `_check_member`, `_check_batched`, and `SpaceCheck.__call__` have consistent semantics.
* Existing vectorized-check tests still pass.

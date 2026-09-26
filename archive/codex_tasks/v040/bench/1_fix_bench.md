Fix SpaceCore benchmark overhead attribution and reduce avoidable wrapper overhead.

Context:
`out.json` shows large SpaceCore overhead, especially for JAX. The likely issue is partly benchmark design: JAX/Torch SpaceCore calls are compared against NumPy “bare” baselines. Also, `check_level="cheap"` validation dominates small operations.

Tasks:

1. Fix benchmark baselines.

   * In `bench/_operations.py`, ensure each backend uses backend-native bare arrays and operations.
   * NumPy: `np.ndarray`, `x + y`, `a * x`, `A @ x`.
   * JAX: `jax.Array`, `x + y`, `a * x`, `A @ x`; use `.block_until_ready()` where needed.
   * Torch: `torch.Tensor`, `x + y`, `a * x`, `A @ x`.

2. Add benchmark matrix over check levels.

   * Run/report at least `check_level="none"` and `check_level="cheap"`.
   * Report overhead as:

     * backend bare time,
     * SpaceCore unchecked time,
     * SpaceCore checked time,
     * validation overhead = checked − unchecked.

3. Add hot-path unchecked kernels.

   * For space ops and dense linop ops, factor core computation into private methods, e.g. `_add_core`, `_scale_core`, `_inner_core`, `_apply_core`, `_rapply_core`, `_vapply_core`.
   * Public methods should preserve validation semantics.
   * Internal composed operators may call unchecked kernels after constructor-time compatibility checks.

4. JAX-specific benchmark fix.

   * Separate eager and `jax.jit` results.
   * Do not include compilation time in steady-state runtime.
   * Record compilation time separately.

5. Add regression tests.

   * Verify benchmark operation factories produce backend-native bare inputs.
   * Verify `check_level="none"` skips membership checks.
   * Verify public checked methods and unchecked cores return numerically identical results.

Acceptance:
The new benchmark report must clearly distinguish backend dispatch cost, SpaceCore abstraction cost, and validation cost.

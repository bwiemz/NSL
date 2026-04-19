# WRGA B.3.2 Option 3 -- source-AD Wengert handler for fused GatedLoRA forward

**Design date:** 2026-04-19
**Scope:** narrow wiring change; enables the B.3.2 trigger to be re-measured with the fused forward kernel actually firing under `--source-ad`.
**Not in scope:** any fused backward kernel work; any broader source-AD refactor; any optimization of source-AD's backward allocator.

## 1. Problem

When an NSL program uses `@adapter(type=gatedlora, ...)` and runs under `--source-ad --target cuda_sm80` with `NSL_WRGA_FUSED_CUDA=1`, the AST rewrite replaces `x @ self.w` with a single `Call` to `nsl_adapter_fused_gatedlora_matmul(x, W, A, B, scale, gate, kernel_handle)`. In inference mode (no `train` block) the fused kernel fires. In training mode (`train` block active, source-AD walks the primal), the fused kernel does NOT fire: `NSL_WRGA_GPU_LAUNCH_COUNTER=1 epochs=3` reports `[nsl-gpu-launch-count] 0`.

**Root cause:** `source_ad.rs::extract_expr` sees the unknown callee `nsl_adapter_fused_gatedlora_matmul`, silently returns `None`, and the train block's forward goes through an unfused fallback path. The user who opted into fusion via env var + target gets unfused execution and pays silent compile-time wiring debt.

**Why this blocks B.3.2:** the B.3.2 trigger condition (`backward > 2.5x fused_forward`) requires the fused forward kernel to actually be used in the training loop. The just-merged trigger measurement (2026-04-18) compared fused-inference-forward against all-unfused-training -- informative, but not the quantity the trigger was designed around. See `project_wrga_b32_measurement.md` addendum for the correction.

## 2. Approach

Add three coordinated pieces:

1. **New `PrimalOp` variant** `FusedGatedLoraMatmul { scale: f32 }` in `wengert.rs`. Inputs: `[x, W, A, B, gate]` (5 VarIds; `scale` is compile-time f32 from the adapter decorator; `kernel_handle` is codegen-internal and not visible to source-AD). Output: `[y]` (1 VarId, the fused matmul result).

2. **Wengert extractor handler** in `source_ad.rs::extract_expr` matching the callee name `nsl_adapter_fused_gatedlora_matmul`, extracting the 5 input vars (skipping the scale literal + kernel_handle literal), and emitting the `FusedGatedLoraMatmul` primal op.

3. **AD rule** in `apply_ad_rule` for `PrimalOp::FusedGatedLoraMatmul`. Emits the gradient set via primitive tensor ops:
   - `sig = Sigmoid(gate)`
   - `sig_prime = sig * (1 - sig)`  (computed via `SigmoidBackward` AdjointExpr pattern already supported)
   - `xa = Matmul(x, A)`
   - `xab = Matmul(xa, B)`
   - `dy_sig = dy * broadcast(sig)` (per-column multiplication)
   - `dx += Matmul(dy, W^T) + Matmul(Matmul(dy_sig, B^T), A^T) * scale`
   - `dW += Matmul(x^T, dy)`
   - `dA += Matmul(x^T, dy_sig) * Matmul(..., B^T)`  -- actually: `dA += Matmul(Matmul(x^T, dy_sig), B^T) * scale`
   - `dB += Matmul(xa^T, dy_sig) * scale`
   - `dgate += reduce_rows(dy * sig_prime * xab) * scale`

   Each of these is expressible as standard primitive ops (Matmul, Transpose, Mul, Sigmoid, ReduceSum) that already have AD support and GPU kernels. The adjoint emission follows the existing `accumulate_adjoint` pattern used for primitive ops.

**Forward behavior** is unchanged. The fused kernel fires because the codegen still emits the `nsl_adapter_fused_gatedlora_matmul` call; the Wengert list just now has a known primal op for it. Downstream NSL compilation (including PTX synthesis, prescan, kernel registry) is untouched.

**Backward behavior** is the unfused adapter-triple math, expressed as Wengert primitive ops. This is the SAME backward the AST rewrite currently produces when fusion is disabled (`--target cuda_sm70`). No new kernel; no semantic change. The only thing changing is that source-AD now keeps the forward fused instead of silently unfusing the whole train iteration.

## 3. Why the `CshaDispatchDecision::AlreadyEmitted` precedent doesn't apply verbatim

CSHA's fused backward KERNEL exists. Its dispatcher says "I've already emitted the fused backward for this chain; skip per-op AD on upstream chain members."

GatedLoRA's fused backward kernel does NOT exist in option 3. The option-3 handler says "I recognize this fused FORWARD; produce backward via primitive ops." The Wengert entry for `FusedGatedLoraMatmul` is the single node that owns both forward-primal recognition and backward-adjoint emission. There is no chain of upstream ops to suppress (the AST rewrite already collapsed the 7 underlying ops into one); there is no dedicated backward kernel to emit.

So option 3 adds ONE entry point, not the full dispatcher structure CSHA uses. It's the "recognize the fused call and know its gradients" case, which is strictly simpler.

## 4. Test discipline

Three new tests, all `#[cfg(feature = "cuda")]`:

1. **`gatedlora_fused_fires_in_train_block`** (launch-counter; `build_4_fused_cuda_actually_fires` pattern): NSL program with `@adapter(gatedlora)` + `train(epochs=3)` + `NSL_WRGA_GPU_LAUNCH_COUNTER=1`. Assert `[nsl-gpu-launch-count] >= 3` (one per epoch's forward). Catches the regression that was silent on 2026-04-18.

2. **`gatedlora_fused_backward_matches_unfused_reference`** (numerical fixture): same model at small shape (m=4, n=8, k=8, rank=2) run two ways -- once with `--target cuda_sm80` (fused forward, source-AD emits unfused backward via option 3's AD rule), once with `--target cuda_sm70` (fully unfused). After one training step with nonzero lr, compare `m.w`, `m.lora_A`, `m.lora_B`, `m.gate` tensors element-wise at 1e-4 tolerance. If the AD rule is wrong, weights diverge and this fires.

3. **`gatedlora_fused_ad_rule_has_all_five_input_grads`** (structural unit test on `source_ad.rs`): synthesize a Wengert list with a `FusedGatedLoraMatmul` op, run the AD rule, assert that adjoint expressions for all 5 inputs (x, W, A, B, gate) were emitted. Catches "AD rule forgets to accumulate gradient for input N" silently.

## 5. Silent-fallback diagnostic warning

As part of this spec (bundled for discoverability parallel to B.3.1's em-dash-caught-by-scale-sweep precedent): at the `source_ad.rs::extract_expr` "unsupported callee" fallback site, emit a one-line warning naming the unrecognized FFI. Prevents future silent regressions of the same class.

**Warning format:**

```
[source-ad] warning: unrecognized FFI callee '<name>' in train block; \
  falling back to unfused AST evaluation. If you expected a fused kernel, \
  check that source-AD has a handler for this FFI.
```

This item is small (one `eprintln!` change) and can ship with the option 3 commit. If the reviewer prefers it as a separate commit, it's atomic enough to split. Either way, it lands before the trigger re-measurement so any future silent fallback is immediately visible.

## 6. Post-land measurement + decision tree

Immediately after option 3 ships (and optionally after the silent-fallback warning is in), re-run `wrga_b32_fused_trigger_final` with the now-properly-wired fused forward. Record the new ratio and verdict in the measurement memory file. Decision tree (from the addendum; repeated here so the spec stands alone):

- **ratio > 2.5x:** proceed with B.3.2 kernel work (new milestone, fused backward kernel + integration fixtures; wiring is already done by this spec).
- **ratio in [1.5x, 2.5x]:** profile the backward; if matmul-bound proceed with kernel work; if allocator-bound file a separate allocator-optimization milestone.
- **ratio < 1.5x:** B.3.2 stays deferred; option 3 wiring remains in-tree as it enables fused forward in training (value independent of B.3.2).

The bench script prints the tree-branch mechanically -- no re-derivation at measurement time.

## 7. Explicit non-goals

Enumerated so no reviewer, subagent, or future-self expands scope:

- No fused backward kernel.
- No `cp.async` / multi-warp staging optimization of the fused forward (B.4 perf item).
- No new fusion decisions in `wrga_fusion.rs`.
- No change to the PTX synthesizer, the runtime FFI, or the kernel registry.
- No change to the AST rewrite pass in `wrga_adapter_rewrite.rs`.
- No fix for adapter-field CPU placement after `m.to(cuda)` (separate follow-up).
- No fix for the kernel profiler zero-duration bug (separate follow-up).
- No source-AD allocator optimization.

If any of these surface during implementation, they get filed as new work items and do not absorb into this spec.

## 8. Files touched

- `crates/nsl-codegen/src/wengert.rs`: add `PrimalOp::FusedGatedLoraMatmul { scale: f32 }` variant + `type_for_op` arm.
- `crates/nsl-codegen/src/source_ad.rs`:
  - Extractor: new match arm for callee `"nsl_adapter_fused_gatedlora_matmul"`.
  - AD rule: new arm in `apply_ad_rule` (or wherever per-op adjoints are produced) producing 5 input adjoints.
  - Warning line at unsupported-callee fallback.
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` (existing integration test file): add the 2 new CUDA-gated tests described in §4.
- `crates/nsl-codegen/tests/` (new file or existing): add structural unit test for the AD rule.

Estimated LOC: 200-350 across the above, dominated by the AD rule expansion.

## 9. Success criteria

- `gatedlora_fused_fires_in_train_block`: `[nsl-gpu-launch-count] >= epochs` under train block.
- `gatedlora_fused_backward_matches_unfused_reference`: weight-by-weight 1e-4 tolerance after one training step with nonzero lr.
- Existing 21 ptxas + 15 integration fixtures stay green.
- `wrga_b32_fused_trigger_final` completes and records a ratio in the memory file that we can then apply the decision tree to.

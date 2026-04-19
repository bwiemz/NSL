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

3. **AD rule** in `apply_ad_rule` for `PrimalOp::FusedGatedLoraMatmul`. Emits the gradient set via primitive tensor ops with explicit shape annotations:

   ```text
   # Forward:
   #   y[B, N] = x[B, K] @ W[K, N]
   #         + sigmoid(gate[N]) ⊙ (x @ A @ B)[B, N] * scale
   #
   # Given upstream adjoint dy[B, N]:

   sig[N]          = Sigmoid(gate)
   sig_prime[N]    = Mul(sig, Sub(Constant(1.0), sig))     # elementwise on gate dim
   xa[B, R]        = Matmul(x, A)                          # rank-R intermediate
   xab[B, N]       = Matmul(xa, B)                         # full-rank adapter output

   sig_bcast[B, N] = BroadcastTo(sig, target_shape=[B, N], broadcast_axes=[0])
   dy_sig[B, N]    = Mul(dy, sig_bcast)                    # elementwise per-column
   dy_sig_sc[B, N] = Mul(dy_sig, Constant(scale))          # scalar multiply, factored once

   # Input adjoints accumulated via `accumulate_adjoint`:
   dx[B, K]   += Matmul(dy, Transpose(W))
              +  Matmul(Matmul(dy_sig_sc, Transpose(B)), Transpose(A))
   dW[K, N]   += Matmul(Transpose(x), dy)
   dA[K, R]   += Matmul(Matmul(Transpose(x), dy_sig_sc), Transpose(B))
   dB[R, N]   += Matmul(Transpose(xa), dy_sig_sc)
   dgate[N]   += Mul(
                   ReduceSum(Mul(Mul(dy, xab), sig_prime), axis=0),   # [N]
                   Constant(scale)
                 )
   ```

   **Load-bearing details:**
   - `sig_bcast` is an elementwise broadcast with `broadcast_axes=[0]` (i.e., the batch row axis is introduced, the `[N]` column axis is preserved). Implementation must emit the correct broadcast axis; axis-swap bugs are caught by the unit test in §4 Test 4.
   - `dy_sig = Mul(dy, sig_bcast)` is elementwise-multiply, NOT a matmul. Any `Matmul` involving `sig` in the emitted Wengert graph is wrong.
   - The `scale` factor is applied exactly once (into `dy_sig_sc`) and then reused — not re-applied separately to `dA`, `dB`, and `dgate`. This avoids drift if `scale` is ever computed at runtime rather than a compile-time constant.
   - `dgate` reduces over `axis=0` (the batch/row dim `B`) to produce shape `[N]`. If the reduction is missing or wrong-axis, `dgate` ends up with shape `[B, N]`, which the Test 3 shape assertion catches.

   Each primitive op is already in `PrimalOp` and already has an AD-rule entry (verified: `Sigmoid`, `Matmul`, `Mul`, `Sub`, `Transpose`, `BroadcastTo`, `ReduceSum` all present). The adjoint emission follows the existing `accumulate_adjoint` pattern used for primitive ops — no new adjoint-expression kinds needed.

**Forward behavior** is unchanged. The fused kernel fires because the codegen still emits the `nsl_adapter_fused_gatedlora_matmul` call; the Wengert list just now has a known primal op for it. Downstream NSL compilation (including PTX synthesis, prescan, kernel registry) is untouched.

**Backward behavior** is the unfused adapter-triple math, expressed as Wengert primitive ops. This is the SAME backward the AST rewrite currently produces when fusion is disabled (`--target cuda_sm70`). No new kernel; no semantic change. The only thing changing is that source-AD now keeps the forward fused instead of silently unfusing the whole train iteration.

## 3. Why the `CshaDispatchDecision::AlreadyEmitted` precedent doesn't apply verbatim

CSHA's fused backward KERNEL exists. Its dispatcher says "I've already emitted the fused backward for this chain; skip per-op AD on upstream chain members."

GatedLoRA's fused backward kernel does NOT exist in option 3. The option-3 handler says "I recognize this fused FORWARD; produce backward via primitive ops." The Wengert entry for `FusedGatedLoraMatmul` is the single node that owns both forward-primal recognition and backward-adjoint emission. There is no chain of upstream ops to suppress (the AST rewrite already collapsed the 7 underlying ops into one); there is no dedicated backward kernel to emit.

So option 3 adds ONE entry point, not the full dispatcher structure CSHA uses. It's the "recognize the fused call and know its gradients" case, which is strictly simpler.

## 4. Test discipline

Three new tests, all `#[cfg(feature = "cuda")]`:

1. **`gatedlora_fused_fires_in_train_block`** (launch-counter; `build_4_fused_cuda_actually_fires` pattern): NSL program with `@adapter(gatedlora)` + `train(epochs=3)` + `NSL_WRGA_GPU_LAUNCH_COUNTER=1`. Assert `[nsl-gpu-launch-count] >= 3` (one per epoch's forward). Catches the regression that was silent on 2026-04-18.

2. **`gatedlora_fused_backward_matches_unfused_reference`** (numerical fixture): same model at small shape (m=4, n=8, k=8, rank=2) run two ways -- once with `--target cuda_sm80` (fused forward, source-AD emits unfused backward via option 3's AD rule), once with `--target cuda_sm70` (fully unfused). After **one training step with `lr = 1e-3`**, compare `m.w`, `m.lora_A`, `m.lora_B`, `m.gate` tensors element-wise at **1e-4 tolerance**. If the AD rule is wrong, weights diverge and this fires.

   **Tolerance rationale (pin inline at the fixture as a comment, following the B.3.1 Fixture D ULP precedent):**

   ```text
   // Tolerance 1e-4 on post-training-step weight comparison.
   //
   //   lr = 1e-3 (pinned; see below).
   //
   //   Fused-forward output differs from unfused-forward by ≤ 2e-4 per element
   //   (fused sigmoid ex2.approx + rcp.approx ULP ~9e-8, MMA f16-accumulator
   //   noise floor ~2e-4; see B.3.1 Fixture D derivation in
   //   wrga_adapter_runtime_equivalence.rs near GATEDLORA_FIXTURE_D_SRC).
   //
   //   That per-element output difference ε_y ~ 2e-4 propagates into dy with
   //   comparable magnitude, then into weight updates Δw = -lr * dy * (...),
   //   giving weight-level discrepancy ≈ lr * 2e-4 = 2e-7 per element at
   //   lr = 1e-3.
   //
   //   1e-4 tolerance therefore provides ~500× headroom at lr = 1e-3. If lr
   //   is raised in a future refactor (e.g., to exercise harder gradient
   //   signals), this tolerance needs recalibration: at lr = 1e-1 the
   //   expected weight drift is ~2e-5, still inside 1e-4 but only 5× headroom;
   //   at lr = 1.0 it is ~2e-4, right at the tolerance boundary.
   ```

   The load-bearing point: do not raise `lr` without also revisiting the tolerance. Both are pinned here.

3. **`gatedlora_fused_ad_rule_emits_correct_shapes`** (structural unit test on `source_ad.rs`): synthesize a Wengert list with a `FusedGatedLoraMatmul` op at concrete shapes (B=3, K=8, R=4, N=16, scale=1.5), run the AD rule, assert adjoints for all 5 inputs with their exact shapes:

   ```rust
   assert_eq!(wengert.adjoints.get(x_var).unwrap().shape(), &[3, 8]);
   assert_eq!(wengert.adjoints.get(w_var).unwrap().shape(), &[8, 16]);
   assert_eq!(wengert.adjoints.get(a_var).unwrap().shape(), &[8, 4]);
   assert_eq!(wengert.adjoints.get(b_var).unwrap().shape(), &[4, 16]);
   assert_eq!(wengert.adjoints.get(gate_var).unwrap().shape(), &[16]);
   ```

   Catches "AD rule forgets to accumulate gradient for input N" (presence bug) and "dgate has shape `[B, N]` because ReduceSum forgot the axis" (shape bug — the most plausible shape regression given how broadcast + reduction interact).

4. **`gatedlora_fused_backward_broadcasts_gate_on_correct_axis`** (structural unit test): feed `sig = [1.0, 2.0, 4.0, 8.0]` (shape `[4]`) and `dy = ones([3, 4])` (shape `[3, 4]`) through the Wengert graph the AD rule emits, and assert the resulting `dy_sig` tensor equals `[[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]]`. Catches the axis-swap failure mode where `sig` is accidentally broadcast along the wrong axis (producing e.g., `[[1, 1, 1, 1], [2, 2, 2, 2], ...]` which would still have shape `[3, 4]` but compute the wrong gradient). Explicit because shape-only checks pass both correct and swapped axis variants.

5. **`gatedlora_fused_unchanged_inference_behavior`** (regression): run the same `@adapter(gatedlora)` model in inference mode (no `train` block) before and after the option 3 change, assert bitwise-identical output tensors. Catches the class of bug where adding the Wengert extractor handler accidentally perturbs the inference codegen path (e.g., by activating source-AD hooks that weren't active before). Implementation note: compare against a recorded baseline from pre-option-3 main rather than running two binaries.

## 5. Silent-fallback diagnostic warning — SEPARATE COMMIT

At the `source_ad.rs::extract_expr` "unsupported callee" fallback site, emit a one-line warning naming the unrecognized FFI. Prevents future silent regressions of the same class.

**Warning format:**

```text
[source-ad] warning: unrecognized FFI callee '<name>' in train block; \
  falling back to unfused AST evaluation. If you expected a fused kernel, \
  check that source-AD has a handler for this FFI.
```

**Shipped as its own commit, not bundled with option 3**, for three reasons: (1) the warning is a general source-AD quality-of-life improvement applying to every unrecognized FFI, not just `nsl_adapter_fused_gatedlora_matmul`; bundling mis-scopes it. (2) If option 3 implementation hits an unexpected issue and gets rolled back, the diagnostic improvement — which is orthogonal — should not roll back with it. (3) The commit message for the diagnostic ("source-AD warns on unrecognized FFI callees to prevent silent fallback regressions") is a one-line institutional-memory contribution; bundling into option 3's commit message buries it.

**Ordering:** the diagnostic commit lands first (it helps future debugging even if option 3 isn't done yet), then option 3. Both ship in the same PR if the worktree supports multi-commit PRs; otherwise two PRs with the diagnostic merged first. The trigger re-measurement depends on option 3 landing, not on the diagnostic, but having the diagnostic in place protects the re-measurement itself against repeat-silent-fallback errors.

## 6. Post-land measurement + decision tree

Immediately after option 3 ships (and optionally after the silent-fallback warning is in), re-run `wrga_b32_fused_trigger_final` with the now-properly-wired fused forward. Record the new ratio and verdict in the measurement memory file. Decision tree (from the addendum; repeated here so the spec stands alone):

- **ratio > 2.5x:** proceed with B.3.2 kernel work (new milestone, fused backward kernel + integration fixtures; wiring is already done by this spec).
- **ratio in [1.5x, 2.5x]:** profile the backward; if matmul-bound proceed with kernel work; if allocator-bound file a separate allocator-optimization milestone.
- **ratio < 1.5x:** B.3.2 (fused backward kernel) stays deferred indefinitely. Option 3's source-AD wiring remains in-tree unconditionally — its value is **"fused forward actually fires in training loops, regardless of B.3.2's fate"**. Without option 3, the fused-forward PTX shipped by B.3.1 is inference-only and silently dormant under `--source-ad`; with option 3, it delivers whatever speedup it's worth inside real training. This value statement is explicit so future readers don't question "what was this wiring for if B.3.2 got deferred."

The bench script prints the tree-branch mechanically -- no re-derivation at measurement time.

## 7. Explicit non-goals

Enumerated so no reviewer, subagent, or future-self expands scope:

- No fused backward kernel.
- No `cp.async` / multi-warp staging optimization of the fused forward (B.4 perf item).
- No new fusion decisions in `wrga_fusion.rs`.
- No change to the PTX synthesizer, the runtime FFI, or the kernel registry.
- No change to the AST rewrite pass in `wrga_adapter_rewrite.rs`.
- No fix for adapter-field CPU placement after `m.to(cuda)` (separate follow-up).

**Verification gate before Test 1 dispatch:** before writing `gatedlora_fused_fires_in_train_block`, empirically verify that `m.gate_<site>__gatedlora` is on-device after the standard `m.to(cuda)` + init-train-block pattern used by B.3.1's fixtures. If `gate` is still on CPU, the fused FFI's device check fails and Test 1 fails for reasons unrelated to option 3 — the resolution is to lift the placement fix into option 3 (scope expansion) or write the test fixtures to explicitly reassign `gate` with `.to(cuda)` (fragile workaround). B.3.1's inference fixtures demonstrably fire fused kernels, so placement works for the inference path; the verification is that it works for the train-block path too. A two-line repro at the top of implementation confirms this before scope commits either way.
- No fix for the kernel profiler zero-duration bug (separate follow-up).
- No source-AD allocator optimization.

If any of these surface during implementation, they get filed as new work items and do not absorb into this spec.

## 8. Files touched

- `crates/nsl-codegen/src/wengert.rs`:
  - Add `PrimalOp::FusedGatedLoraMatmul { scale: f32 }` variant.
  - Add `type_for_op` arm returning `WengertType::Tensor` (same as other tensor-producing ops; shape-inference is handled elsewhere by the tensor-type system reading input shapes). Output shape is `[B, N]` where `B = input_shapes[0][0]` (from `x`) and `N = input_shapes[1][1]` (from `W`) — mirroring the standard `Matmul` output-shape rule. If the existing `type_for_op` only returns a `WengertType` discriminant (not a concrete shape), this is a 1-line addition matching the `Matmul` pattern; if `type_for_op` also computes output shape, ~5 lines deriving `[B, N]` from inputs `[x, W, A, B, gate]` positions.
- `crates/nsl-codegen/src/source_ad.rs`:
  - Extractor: new match arm for callee `"nsl_adapter_fused_gatedlora_matmul"`. Extracts the 5 tensor inputs; pulls `scale: f32` from the 5th argument (a float literal); ignores the 7th argument `kernel_handle` (codegen-internal, not AD-relevant).
  - AD rule: new arm in `apply_ad_rule` (or wherever per-op adjoints are produced) emitting the 5 input adjoints per §2's shape-annotated recipe.
  - Warning line at unsupported-callee fallback (separate commit; §5).
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` (existing integration test file): add Test 1 (`gatedlora_fused_fires_in_train_block`), Test 2 (`gatedlora_fused_backward_matches_unfused_reference`), and Test 5 (`gatedlora_fused_unchanged_inference_behavior`) from §4.
- `crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs` (new file): add Test 3 (shape assertions) and Test 4 (broadcast-axis unit test) from §4.

Estimated LOC: 200-350 across the above, dominated by the AD rule expansion in `source_ad.rs`.

## 9. Success criteria

- `gatedlora_fused_fires_in_train_block`: `[nsl-gpu-launch-count] >= epochs` under train block.
- `gatedlora_fused_backward_matches_unfused_reference`: weight-by-weight 1e-4 tolerance after one training step with nonzero lr.
- Existing 21 ptxas + 15 integration fixtures stay green.
- `wrga_b32_fused_trigger_final` completes and records a ratio in the memory file that we can then apply the decision tree to.

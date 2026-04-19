# WRGA B.3.2 Option 3 (revised) -- fused forward in training via method-body rewrite + source-AD handler

**Supersedes:** [`2026-04-19-wrga-b32-option3-source-ad-wiring-design.md`](2026-04-19-wrga-b32-option3-source-ad-wiring-design.md). That spec's §1 root-cause analysis was wrong; the intervention it proposed (source-AD Wengert handler for `nsl_adapter_fused_gatedlora_matmul`) is still correct but incomplete. The gap it missed is that `model_method_bodies` is populated from the *unrewritten* AST, so the fused FFI callee it planned to handle never appears in the code path source-AD walks. This revision closes that gap.

**Verification discipline:** follows WRGA paper Appendix B.5 ("verify the specific code path, not a proxy"). See `docs/research/NSL-WRGA-Research-Appendix-B2-addendum.md` §B.5.

**Institutional note for the sequence retrospective:** the original spec's *intervention* (Wengert handler + AD rule) was correct; only the *root cause* was mis-identified. When pre-dispatch verification invalidates a spec's root-cause analysis, examine the intervention independently before abandoning it — sometimes the correct intervention addresses a different (or additional) layer than the spec claimed. This finding is worth recording in close-out but does not warrant a new Appendix B entry on its own.

---

## 1. Problem (corrected)

**What the previous spec claimed:** source-AD sees unknown callee `nsl_adapter_fused_gatedlora_matmul` and falls back silently; adding a Wengert handler fixes it.

**What actually happens (verified by direct probe, 2026-04-19):**

1. `prescan_adapter_sites_from_decorators` populates `compiler.adapter_sites` before `compile_user_functions` runs (B.2.1 Task 5.5, `wrga_prescan.rs:47`, called at 4 entry points). The Cranelift-compiled method body DOES get rewritten with the fused FFI call.
2. Separately, `declaration.rs:490` populates `compiler.models.model_method_bodies` with `fn_def.clone()` — the **unrewritten** AST. This happens inside `declare_user_functions` at `entry_points.rs:260`, which runs *before* prescan.
3. Source-AD's inline expansion consumes `model_method_bodies` (not the Cranelift-compiled version) because it needs AST to walk. So in a `train(step): m.forward(x)` block, source-AD sees the original `x @ self.w` and reports `1/1 trainable tensor params connected` — not the 5/5 we'd expect if the fused FFI had been seen and its 5 inputs extracted.

**Confirmed empirically:** subagent ran `@adapter(gatedlora) + train(epochs=1)` under `NSL_WRGA_GPU_LAUNCH_COUNTER=1`, got `[nsl-gpu-launch-count] 0` + `source AD gradient summary: 1/1 trainable tensor params connected`. The fused kernel never fires in the training path.

**The gap, precisely:** `model_method_bodies` is a parallel copy of the AST used only by source-AD. It is populated *before* prescan and *never re-rewritten*. That parallel copy is the specific data structure whose staleness causes option 3's handler to be inert.

## 2. Approach (two phases, same PR)

### Phase 3e -- method-body rewrite pass

Apply the existing adapter rewrite pass to `compiler.models.model_method_bodies` immediately after `prescan_adapter_sites_from_decorators` runs. The rewrite replaces `x @ self.w` with the fused FFI call (or unfused triple, per the existing dispatch rules in `wrga_adapter_rewrite.rs`) for every method body whose model has entries in `adapter_sites`.

### Phase Option 3 -- source-AD handler (unchanged intervention from superseded spec)

Add `PrimalOp::FusedGatedLoraMatmul { scale: f32 }`, extractor match arm for callee `"nsl_adapter_fused_gatedlora_matmul"`, and AD rule emitting 5 input adjoints per the shape-annotated recipe from the superseded spec's §2. The shape recipe, tolerance rationale, and test discipline from that spec carry over verbatim; the only change is that the callee now actually appears in the AST source-AD walks, because phase 3e put it there.

## 3. Insertion point for 3e (pinned)

Consumer enumeration for `model_method_bodies` (conducted 2026-04-19, grep across full workspace):

| Location | Role | Relative to prescan |
|---|---|---|
| `declaration.rs:494` | Populator (write, local bodies) | BEFORE prescan |
| `entry_points.rs:927-933` | Populator (write, imported bodies via `compile_program_with_imports`) | BEFORE prescan |
| `stmt.rs:3761` | Read -> `extractor.set_model_method_bodies(...)` | Inside `compile_main` (train block) -- AFTER prescan |
| `stmt.rs:7051` | Read -> `extractor.set_model_method_bodies(...)` | Inside `compile_main` -- AFTER prescan |
| `stmt.rs:7614, 7643` | Read for forward pipe-chain analysis | Inside `compile_main` -- AFTER prescan |
| `source_ad.rs:1600-1679` | Read during inline expansion | During extractor walk -- AFTER prescan |

**No intermediate consumers exist** between `prescan_adapter_sites_from_decorators` (populates `adapter_sites`) and the first READ of `model_method_bodies` (which happens inside `compile_main`). The window is clean.

**Pinned insertion point:** a new function `rewrite_model_method_bodies_with_adapter_sites(&mut compiler)` is called at each of the 4 entry-point sites immediately after `prescan_adapter_sites_from_decorators`. Before `compile_user_functions` is acceptable; before `compile_main` is required; between them is fine. Pragmatic choice: immediately after prescan, as a one-line follow-up, so the two-phase "populate sites + rewrite bodies" is atomic in each entry point.

```rust
crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);  // NEW
compiler.compile_user_functions(&ast.stmts)?;
```

No data-structure sharing with intermediate consumers, no ordering surprises with `apply_vmap_transforms` / `register_batched_functions` / `compile_kernels` (those don't read `model_method_bodies`). Verified by the enumeration above.

## 4. Shared rewrite helper

`compile_user_functions` at `compiler/functions.rs:540-625` contains the rewrite setup for the Cranelift path: builds a `RewriteContext` with `target_sm`, `fused_kernel_order`, `fused_gatedlora_kernel_order`, `self_sym`, `field_symbols` (from `model_field_types` + `model_tensor_field_shapes`), plus the load-bearing `sigmoid` symbol insertion from B.3.1 Task 5.1. That block is ~85 LOC and MUST be shared with 3e, not duplicated — any drift between the two call sites would produce different ASTs in the Cranelift-compiled vs source-AD-walked bodies, which is exactly the bug class this spec exists to prevent.

**Extraction:** move the `RewriteContext` construction + per-statement rewrite loop into a public helper in `wrga_adapter_rewrite.rs` — e.g.,

```rust
pub fn rewrite_stmts_for_model<'a>(
    compiler: &'a mut crate::compiler::Compiler<'_>,
    model_name: &str,
    stmts: &[Stmt],
) -> Vec<Stmt>;
```

Then both `compile_user_functions` (functions.rs:540-625) and 3e's new pass call it. The helper's contract: given a model name and statements, return the rewritten statements using the compiler's current `adapter_sites` + kernel orders + field symbols. Changes to rewrite rules land in one place.

The extraction is a prerequisite to 3e, not optional. It ships as 3e's first internal step (commit 3e).

## 5. Test discipline (B.5-compliant)

Three tests, all `#[cfg(feature = "cuda")]`, ship in the 3e commit (the path is proven reachable by 3e alone; option 3's handler on top makes the gradients correct):

1. **`gatedlora_fused_fires_in_train_block`** (proxy-level, necessary but not sufficient per B.5):

   NSL program with `@adapter(gatedlora) + train(epochs=3) + NSL_WRGA_GPU_LAUNCH_COUNTER=1`. Assert `[nsl-gpu-launch-count] >= 3` (one per epoch).

   Catches the regression where 3e's rewrite pass doesn't fire (empty `adapter_sites`, wrong insertion point, or the shared helper's extraction introduced a bug).

2. **`source_ad_sees_fused_ffi_in_train_block`** (B.5-compliant direct probe):

   Same NSL program. Parse the source-AD stderr line. Assert it reports **`5/5 trainable tensor params connected`** (x, W, A, B, gate) -- not 1/5. The `5/5` signal can only be satisfied if source-AD's Wengert walk has seen the fused FFI call and extracted its 5 inputs -- exactly the code path 3e closes.

   Per B.5: this is the direct-probe assertion that only the claimed path can produce. Launch-counter alone could be satisfied by some other fusion-adjacent code path firing; the 5/5 count can only be satisfied by the specific `FusedGatedLoraMatmul` PrimalOp appearing in source-AD's extracted Wengert list.

3. **`cranelift_and_source_ad_see_same_rewritten_ast`** (anti-drift check):

   Structural test asserting that `compile_user_functions` and 3e's pass produce the same `Vec<Stmt>` when given the same input method body and the same `adapter_sites`. Guards against the failure mode this spec exists to prevent: two call sites of the rewrite pass drifting apart.

The five tests from the superseded spec §4 (launch-counter in train block, numerical-equivalence weights, structural shape assertions, broadcast-axis unit, inference-unchanged regression) carry over to the original option 3 commit on top of 3e. Those tests gate the AD rule's correctness; tests 1-3 above gate 3e's reachability.

## 6. Commit structure (single PR, two commits)

Both commits land in one PR (branch `feat/wrga-b32-fused-backward`). Intermediate state between commits 1 and 2 is transient — acceptable for PR-level review, not worth handling gracefully.

**Commit A (3e):**

- Extract shared `wrga_adapter_rewrite::rewrite_stmts_for_model` helper; refactor `compile_user_functions` to call it.
- Add `wrga_prescan::rewrite_model_method_bodies_with_adapter_sites` pass.
- Wire pass into the 4 entry-point sites.
- Ship tests 1 (launch-counter), 2 (5/5 params -- **this will FAIL at this commit** because source-AD still has no handler for the fused FFI it now sees; partially-connected warning/fallback fires), 3 (anti-drift).

   Test 2 failing at commit A is expected and acceptable. The test is written to document the post-PR expectation; commit B makes it pass. Add a `#[ignore]` attribute on test 2 in commit A with a comment explaining this, removed in commit B.

**Commit B (original option 3):**

- Add `PrimalOp::FusedGatedLoraMatmul`; `type_for_op` arm; extractor match arm; AD rule per the superseded spec's §2 shape-annotated recipe (which remains correct).
- Remove the `#[ignore]` from test 2; it now passes.
- Ship tests 4-8 from the superseded spec §4 (numerical equivalence, shape assertions, broadcast axis, inference-unchanged, silent-fallback warning structural).

PR acceptance: all 8 tests green, plus the existing 21 ptxas + 15 integration fixtures unchanged.

## 7. Post-PR: trigger re-measurement + decision tree

Unchanged from superseded spec §6. Re-run `wrga_b32_fused_trigger_final` with 3e + option 3 active. Expected: fused forward actually fires in the train block, ratio reflects (fused-forward-in-training vs unfused-backward-in-training) which is the trigger's literal claim.

Decision-tree branches (mechanical, per superseded spec §6 and memory addendum):

- ratio > 2.5x -> schedule B.3.2 kernel work.
- ratio in [1.5x, 2.5x] -> profile matmul-bound vs allocator-bound; decide accordingly.
- ratio < 1.5x -> B.3.2 fused backward kernel stays deferred; option 3 wiring remains in-tree because fused forward in training is valuable independent of B.3.2.

## 8. Files touched (revised)

Commit A (3e):

- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`: add `rewrite_stmts_for_model` public helper extracting the existing logic from functions.rs:540-625.
- `crates/nsl-codegen/src/compiler/functions.rs`: refactor lines 540-625 to call the new helper.
- `crates/nsl-codegen/src/wrga_prescan.rs`: add `rewrite_model_method_bodies_with_adapter_sites` function.
- `crates/nsl-codegen/src/compiler/entry_points.rs`: call new function at 4 sites (lines 272, 463, 626, 821 neighbourhoods).
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`: add tests 1, 2 (with `#[ignore]` initially), 3.

Estimated LOC for commit A: ~150-200.

Commit B (option 3):

- `crates/nsl-codegen/src/wengert.rs`: add `PrimalOp::FusedGatedLoraMatmul { scale: f32 }` variant + `type_for_op` arm.
- `crates/nsl-codegen/src/source_ad.rs`: extractor match arm + AD rule.
- `crates/nsl-codegen/src/wengert_lower.rs` (or wherever `apply_ad_rule` lives): AD rule's adjoint expression emission per superseded spec §2 recipe.
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`: remove `#[ignore]` from test 2; add tests 4-8 per superseded spec §4.
- `crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs` (new file): structural shape + broadcast-axis unit tests.

Estimated LOC for commit B: ~250-300 (dominated by AD rule expansion).

## 9. Non-goals (reiterated from superseded spec §7, unchanged)

- No fused backward kernel. Backward stays unfused adapter-triple.
- No `cp.async` / multi-warp optimization of fused forward (B.4).
- No new fusion decisions in `wrga_fusion.rs`.
- No change to PTX synthesizer, runtime FFI, or kernel registry.
- No fix for adapter-field CPU placement (verified unnecessary by Task 0).
- No fix for kernel profiler zero durations.
- No source-AD allocator optimization.

## 10. Success criteria

Commit A (3e): tests 1 and 3 green; test 2 ignored with comment.
Commit B (option 3): test 2 green (5/5 trainable params); tests 4-8 green; all pre-existing 21 ptxas + 15 integration fixtures unchanged.
Post-PR: `wrga_b32_fused_trigger_final` reports a ratio with `[nsl-gpu-launch-count] >= 13` (for N=13 train iterations).

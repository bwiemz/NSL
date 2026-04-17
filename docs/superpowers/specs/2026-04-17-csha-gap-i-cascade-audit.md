# CSHA Gap I — Cascade Audit

Pre-implementation audit of hidden cascades between today's state (Gaps A-H merged at `408319f`) and a working hd=32 toy pretrain. Pure code read — no fixes applied.

Worktree: `.worktrees/csha-gap-i-design`, branch `docs/csha-gap-i-design`.

Scope: find every blocker that will ALSO break once the stated I.1/I.2/I.3 are fixed. Previous gaps (E→F→G→H) each surfaced one more layer of bug; this audit tries to flush the rest.

## Stated blockers (recap)

| ID  | Blocker                                                              | Verdict     |
| --- | -------------------------------------------------------------------- | ----------- |
| I.1 | `FusionMark.config` inflates SMEM by carrying pipeline-level flags   | Will break  |
| I.2 | Dead-grad elim prunes `FusedCshaBackward` (result unreferenced)      | Will break  |
| I.3 | Forward save-key vs. backward mark-key can diverge                   | Might break |

## Suspect list — verdicts

### A. Output tensor allocation shapes + dtypes — **Will break**

`wengert_lower.rs:1287-1322` allocates the 7 gradient tensors via `nsl_tensor_zeros`, which per `nsl-runtime/src/tensor/creation.rs:10` produces **f32 (dtype=1)**. The Tier C backward kernel writes dq/dk/dv/dwq/dwk/dwv as **f16** (half, 2 bytes/elem). `dx` is correctly f32.

When the kernel does `st.global.u16` into an f32-sized allocation, only half the bytes get populated; subsequent f32 reads on the CPU side interpret raw f16 bits as f32, yielding garbage. Weight updates will corrupt all trainable weights within one step.

- Evidence: `tensor_from_shape_list` explicit "(f32, dtype=1)" comment at `creation.rs:10`.
- Scope: **Medium** — requires either an `nsl_tensor_zeros_f16_on(shape, device)` FFI OR a `dtype` parameter for existing zeros/zeros_on.
- Trigger: every fused-backward launch.

### B. Cache keying (cross-function stale entries) — **Might break**

`Compiler.csha_fused_bwd_cache` and `Compiler.csha_forward_saves` are **never** `.clear()`-ed (grep for `.clear()` returns zero). Cranelift `Value` IDs restart at 0 per function (`FunctionBuilderContext::new()` per function in `func.rs:34`). If one compile emits a `FusedCshaBackward` for layer A at `Value(42)` and a downstream function's forward also produces `Value(42)`, the second function's backward will hit stale entries.

Today the train-block forward+backward both run inside a single Cranelift function (`compile_train_block` → `compile_source_ad_grad_block`), and the lowerer `.remove()`s entries on consumption. So the common path is safe. But any second `grad` block in the module, or any future split of forward/backward into separate functions, would hit stale data.

- Evidence: `compiler/mod.rs:676` and `:698` (init only); no clear-on-function-boundary.
- Scope: **Tiny** — add `self.csha_fused_bwd_cache.clear(); self.csha_forward_saves.clear();` at the end of each function compile.

### C. Save-buffer lifetime — **Fine** (with a caveat)

Saves are stack-allocated Cranelift Values pulled from a 48-byte stack slot (`expr/advanced.rs:1579-1601`), stored in `Compiler.csha_forward_saves`, and consumed + freed in `wengert_lower.rs:1374-1383`. Both forward and backward run in the same Cranelift function → SSA Values stay valid. The free happens after the launch in the same function. No UAF.

**Caveat**: multi-step training loops reuse the same stack slot pointer every epoch iteration (the `nsl_csha_alloc_backward_activations_into` call re-runs each iteration and overwrites the slot). That's correct — the free precedes the next alloc. Safe.

### D. Adjoint accumulation for shared inputs — **Might break**

EmitFused accumulates into `v.q_out_var`, `v.k_out_var`, etc. (source_ad.rs:307-313) via `accumulate_adjoint`, which correctly SUMS (source_ad.rs:880-896). If a q-matmul output also feeds a skip connection or another non-attention branch, the fused gradient merges with the skip's per-op backward — that's the desired behaviour.

HOWEVER: `AlreadyEmitted` (source_ad.rs:351-354) does `continue` BEFORE `apply_ad_rule`. So matmul/RoPE/RMSNorm ops in the claim set are fully suppressed. Good. But any op in the chain whose output is ALSO used outside the chain (e.g. the RMSNorm output feeding BOTH attention and a branch in a sidecar head) will have its non-attention users left without a gradient — because we suppressed per-op AD for RMSNorm entirely.

Typical transformer topology doesn't do this, but toy pretrain experiments that e.g. probe intermediate activations will miss gradients.

- Scope: **Small** but hard to detect. Flag as a future audit item; for toy pretrain it's fine.

### E. `AlreadyEmitted` semantics — **Fine**

source_ad.rs:351-354 explicitly `continue`s the loop without calling `apply_ad_rule`. The claimed op's per-op AD never runs. Exactly what Gap D.1 intends. Verified.

### F. GPU residency of 7 outputs — **Will break** (partial)

`wengert_lower.rs:1316-1322` does `nsl_tensor_zeros` (CPU) → `nsl_tensor_to_device(t, 1)`. `nsl_tensor_to_device` allocates new device memory and copies. BUT the newly-allocated tensor is f32 by default (see finding A). Combined with A, this gives: f32-sized device buffer, f16 kernel writes → byte count mismatch AND wrong dtype.

Input tensors (q_ptr/k_ptr/v_ptr from `launch_inputs[2..=4]`) — these are Cranelift Values representing the SDPA op's inputs (q_rope, k_rope, v_proj). Their device residency depends on forward-side placement, which today goes through `nsl_flash_attention_csha_with_saves` and therefore must already be on device at the FA call site. Reasonable.

### G. Scale-bits consistency — **Fine**

Gap G's fix at `expr/advanced.rs` is the forward-side scale. Backward's scale at `wengert_lower.rs:1250-1261` uses the SAME pattern (fcvt_from_sint → sqrt → fdiv → fdemote F32 → bitcast I32 → sextend I64). Matches Gap G convention.

### H. Multi-layer models — **Might break**

Forward side keys `csha_forward_saves.insert(layer_key, ...)` using `extras_for_current_function` with fallback to `layer_at_index(ordinal)` (sorted BTreeMap order). Backward side uses `mark.layer` directly = the plan's boundary-discovered layer (also inserted into sorted `by_layer: BTreeMap` at `csha_apply.rs:476`). Both converge on sorted-key order — multi-layer should agree.

Risk: if a model's function name contains a substring that coincidentally matches one plan key but not the corresponding layer's call site, `extras_for_current_function` hits false-positive match and binds forward's save to the WRONG layer. Backward then reads an empty save record.

- Scope: **Small** — either tighten the substring match (require boundary markers like `__blocks_N_`) or plumb a proper layer-id through the call site.

### I. Wengert ordering — **Fine**

EmitFused appends `FusedCshaBackward` then 7 extract ops. Adjoint list is lowered top-to-bottom (wengert_lower.rs walk order), so launch lowers (populates cache) before extracts read. Correct.

**Subtle**: after `eliminate_dead_gradients` (I.2) drops `FusedCshaBackward`, the extracts come FIRST in the filtered list and the cache read fails with "no cache entry". This is the downstream symptom of I.2.

### J. Input list threading (weight pointers) — **Will break**

`source_ad.rs:272-273` builds `launch_inputs = vec![chain_key, do_var]; launch_inputs.extend(op.inputs)`. `op` is the SDPA op whose inputs are `[q_rope, k_rope, v_proj]`. So the total is 5 entries.

`wengert_lower.rs:1271-1279` indexes:
- `q_ptr = inputs[2]` ✓ (q_rope)
- `k_ptr = inputs[3]` ✓ (k_rope)
- `v_ptr = inputs[4]` ✓ (v_proj)
- `x_ptr = null` (inputs[5] unavailable)
- `wq_ptr = null`, `wk_ptr = null`, `wv_ptr = null`
- `norm_weight_ptr = null`

The backward PTX has null-guards on wq/wk/wv (`csha_hooks_backward.rs:813`) so it skips dWq/dWk/dWv accumulation → **weight gradients come back as ZEROS**. A toy pretrain's AdamW will see zero grads on attention weights → no learning.

- Evidence: `csha_hooks_backward.rs:348-350` + `:813`.
- Scope: **Medium** — need to thread wq/wk/wv (and x, norm_weight) into `launch_inputs` from the primal's Param/Matmul VarIds via the chain's `chain_varids`. Already-resolved in csha_apply.rs `CshaChainVarIds`.

### K. RMSNorm gamma gradient — **Will break** (if gamma is trainable)

The fused kernel's `emit_drmsnorm` (flash_attention_v2/phases/backward) computes dx but not dgamma. Per-op AD for RMSNorm would emit `NormGammaBackward` (ad_rules.rs:387-389) but that path is suppressed by `AlreadyEmitted`. grep for `dgamma|d_norm_weight|gamma_grad` in `flash_attention_v2/phases/backward` → no matches.

If the toy model's RMSNorm uses a trainable gamma (default in `nsl.nn.RMSNorm`), its weight gradient stays zero. Depending on the optimizer's sensitivity to a fully-zero gradient, this may or may not blow up numerically, but it's definitely wrong.

- Scope: **Medium** — either teach the fused backward kernel to emit dgamma (large PTX change) OR emit a standalone `NormGammaBackward` op in EmitFused AFTER the extracts.

### L. Cross-loss dependencies — **Fine**

`sum(out)` vs `CrossEntropy(logits, y)` both produce adjoints on the SDPA output VarId via normal per-op backward. The EmitFused arm consumes `y_bar(sdpa_out_var)` regardless of how that adjoint was seeded. No cascade.

## Additional finds (beyond A-L)

### M. Extracts lack data-dependency on the launch op — **Will break** (root cause of I.2 specifically)

`source_ad.rs:285-288`:
```rust
let r = self.emit_op(
    PrimalOp::CshaFusedBackwardExtract { component },
    vec![chain_key],
);
```

Extracts list only `chain_key` (a primal VarId) as input. `FusedCshaBackward`'s result VarId is NOT in the extracts' inputs. So the dead-grad worklist walk from `needed_vars` back through `op.inputs` NEVER visits the launch op. The launch is dropped; cache is never populated; all 7 extracts fail with "no cache entry".

This is the mechanical root cause of I.2. Fix: add the launch op's result VarId to each extract's inputs, OR mark `FusedCshaBackward` as "has side effects / never eliminate" in a new live-set.

- Scope: **Tiny** — one-line fix in source_ad.rs, OR add a `has_side_effects()` method to PrimalOp and consult it in eliminate_dead_gradients.

### N. Train block compiles forward + backward into ONE Cranelift function — **Fine**

Verified: `compile_train_block` runs primal lowering then adjoint lowering in sequence, both using the same `builder`. Forward saves as Cranelift Values stay valid across the whole function. No cross-function leak (today).

## Could not fully audit

1. **NslTensor-vs-raw-device-pointer marshaling** — forward FA FFIs (`nsl_flash_attention_csha`, line 347) pass `q_ptr as u64` directly as a kernel argument, but tests (`csha_cuda_launch_fused.rs:486`) pass raw device pointers via `nsl_test_cuda_alloc`. The compiled path passes NslTensor* pointers from `q_val` unchanged. Either (a) the PTX derefs an offset=0 field of the NslTensor struct on the device (only safe if the struct is in managed memory) or (b) there's an implicit conversion I missed. The backward FFI mirrors this pattern exactly, so IF forward works, backward should too. Needs a GPU smoke to verify.

2. **`extras_for_current_function` false-positive match** — substring match against the fn name may bind the wrong layer. Would need a concrete multi-layer toy to trigger.

3. **`active_heads = 0` override at backward launch** (`wengert_lower.rs:1328`) — hardcoded to 0, meaning the kernel will iterate all heads. If the forward training config sets active_heads > 0, backward and forward disagree on effective head count. Skipped deeper look because the CSHA toy pretrain is almost certainly using all heads.

## Ranked execution order

Fixes should land in this order to minimise rework:

1. **I.1** (clamp FusionMark.config to training_config) — unblocks validator, no downstream effects.
2. **M + I.2** (fix dead-grad elim + extract dependencies) — one combined change; without this the launch op never fires.
3. **A + F** (f16 allocation + GPU placement combined) — one change to `alloc_shape` helper in wengert_lower.rs; depends on a new `nsl_tensor_zeros_on_dtype` FFI or a dtype arg.
4. **J** (weight pointer threading) — once launch runs and outputs are dtype-correct, thread wq/wk/wv via chain_varids.
5. **K** (RMSNorm gamma grad) — post-launch correctness; emit standalone NormGammaBackward after extracts.
6. **I.3 + H** (layer key consistency) — only surfaces in multi-layer; fix after single-layer works.
7. **B** (cache clear on function boundary) — hygiene; defer until we have a second `grad` block in a module.

Landing 1-4 is the minimum for a hd=32 toy pretrain to learn anything. 5 is required for it to match a reference implementation.

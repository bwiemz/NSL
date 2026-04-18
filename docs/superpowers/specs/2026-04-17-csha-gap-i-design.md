# CSHA Gap I — Fused Backward Cascade Fixes (Design)

Implementation plan covering I.1 + I.2 + I.3 (stated blockers) plus the high-confidence "Will break" items surfaced by the cascade audit (A, J, K, M). Lands in the order recommended by the audit.

Audit companion: `2026-04-17-csha-gap-i-cascade-audit.md`.

## I.1 — Clamp SMEM validator config to training_config

**Problem**: `FusionMark.config` carries pipeline-level fusion flags (`fused_projections=true`, `fused_output_proj=true`), which inflate the backward SMEM layout past the 99 KB cap at hd=32. Forward compile fine; backward dispatch rejected.

**Root cause**: `csha_apply::collect_chain_dispatch_map_with_wengert` (csha_apply.rs:594-605) stores the full plan config on the mark. `csha_dispatch_for_op` (ad_rules.rs:67) validates THIS config. But the fused-backward kernel is actually built from the clamped `csha_training_config` (compiler/kernel.rs:543-552, no fusion flags, block_kv=32).

**Fix strategy**: in `collect_chain_dispatch_map_with_wengert`, attach the **training** config (not the plan config) when available. Read from `compiler.kernels.flash_attention_context.csha_training_config` via a parameter.

**Files**:
- `crates/nsl-codegen/src/csha_apply.rs` — extend signature to accept `Option<&FlashAttentionConfig>` training config.
- `crates/nsl-codegen/src/stmt.rs:4047-4053` — pass `self.kernels.flash_attention_context.as_ref().and_then(|c| c.csha_training_config.as_ref())`.

**Test**: existing `ad_csha_reverse_walk_wiring.rs` already constructs marks with arbitrary configs; add a case where plan config has `fused_projections=true` + training config clamps them off and assert `csha_dispatch_for_op` returns `EmitFused` (not `Fallback`).

**Dependencies**: none — lands first.

---

## I.2 + M — Keep FusedCshaBackward alive in dead-grad elimination

**Problem**: `eliminate_dead_gradients` (source_ad.rs:909-931) walks the worklist from `needed_vars` (param adjoint VarIds) back through `op.inputs`. `FusedCshaBackward`'s result VarId is a placeholder never referenced; its extracts' inputs are `[chain_key]`, a primal VarId that doesn't reach the launch op. Launch op gets pruned; extracts fail with "no cache entry".

**Root cause**: extract ops don't declare data-dependency on the launch op, AND `FusedCshaBackward` is treated as a pure (prunable) op despite having a side-effect (populating the cache).

**Fix strategy (combined)**:
1. Change extract ops to include the launch op's result as their first input:
   ```rust
   let launch_result = self.emit_op(PrimalOp::FusedCshaBackward{..}, launch_inputs);
   for component in 0u8..=6u8 {
       self.emit_op(
           PrimalOp::CshaFusedBackwardExtract { component },
           vec![launch_result, chain_key],  // was vec![chain_key]
       );
   }
   ```
2. Update extract lowerer to read `key_val = inputs[1]` (shift index by 1).
3. Verify the worklist walk: `needed` → param adjoint → accumulate chain → extract[0] → inputs[0]=launch_result → launch → inputs[chain_key, do_var, q, k, v]. All nodes kept alive.

**Files**:
- `crates/nsl-codegen/src/source_ad.rs:276-290` — thread launch VarId into extracts.
- `crates/nsl-codegen/src/wengert_lower.rs:1058-1114` — extract arm reads `inputs[1]` as key.
- Tests in `csha_gap_d_emit_fused.rs` + `ad_csha_reverse_walk_wiring.rs` — update fixtures.

**Test**: add a dead-grad elim test in `source_ad.rs` that feeds a Wengert list with a FusedCshaBackward + 7 extracts, runs `eliminate_dead_gradients` with the extract-derived adjoints as `needed`, and asserts the launch op survives.

**Dependencies**: none; can land in parallel with I.1.

---

## A + F — f16 allocation for 6 of the 7 gradient outputs

**Problem**: `nsl_tensor_zeros` creates f32 tensors. The Tier C backward kernel writes dq/dk/dv/dwq/dwk/dwv as **f16**. dx stays f32. Byte count + dtype mismatch; weight updates see garbage.

**Root cause**: no dtype-parameterised zeros FFI exists (grep for `nsl_tensor.*f16` in runtime → none). Existing `tensor_from_shape_list` hardcodes f32.

**Fix strategy**: add `nsl_tensor_zeros_f16_on(shape_list, device) -> i64` FFI. Implemented analogously to `nsl_tensor_zeros_on` (tensor/mod.rs:1318-1359) with `element_size=2` and `dtype=2` (define if not present).

**Files**:
- `crates/nsl-runtime/src/tensor/creation.rs` — add `tensor_from_shape_list_f16` helper.
- `crates/nsl-runtime/src/tensor/mod.rs` — add `nsl_tensor_zeros_f16_on` FFI.
- `crates/nsl-codegen/src/wengert_lower.rs:1287-1322` — use f16 variant for dq/dk/dv/dwq/dwk/dwv; keep f32 for dx.
- `crates/nsl-codegen/src/compiler/runtime_symbols.rs` (or wherever symbol tables live) — declare the new FFI so Cranelift can call it.

**Test**: allocate via the new FFI, verify `element_size() == 2` and `dtype == 2` (or whatever code we pick for f16). Launch the backward kernel with these outputs and verify written values are f16-decoded correctly.

**Dependencies**: must land before the first real numerical test. Order: after I.1+I.2.

---

## J — Thread weight pointers through EmitFused

**Problem**: `launch_inputs` only carries `[chain_key, do_var, q_rope, k_rope, v_proj]`. wq/wk/wv/x/norm_weight come through as NULL. Backward PTX null-guards them → zero weight gradients → no learning.

**Root cause**: source_ad.rs:273 does `launch_inputs.extend(op.inputs.iter().copied())` where `op` is SDPA (3 inputs only). Weight/x VarIds are in `chain_varids` but not threaded.

**Fix strategy**: populate `launch_inputs[5..10]` from `chain_varids`:
```rust
let mut launch_inputs = vec![chain_key, do_var];
launch_inputs.extend(op.inputs.iter().copied());  // q, k, v
if let Some(v) = &chain_varids {
    launch_inputs.push(v.x_norm_var);   // inputs[5] = x
    launch_inputs.push(v.wq_var);       // inputs[6] = wq
    launch_inputs.push(v.wk_var);       // inputs[7] = wk
    launch_inputs.push(v.wv_var);       // inputs[8] = wv
    // norm_weight (gamma) VarId — add CshaChainVarIds.norm_weight_var
    launch_inputs.push(v.norm_weight_var.unwrap_or(0));  // inputs[9]
}
```

**Files**:
- `crates/nsl-codegen/src/csha_apply.rs:284` — add `norm_weight_var: Option<VarId>` to `CshaChainVarIds`. Resolve from the RMSNorm op's inputs[1] if the plan detected a gamma param.
- `crates/nsl-codegen/src/source_ad.rs:272-280` — thread the 5 extra VarIds.

**Test**: a GPU smoke test that runs one fused-backward step, checks `dwq.sum().abs() > 0` (weights received non-zero gradients).

**Dependencies**: J needs A+F landed (else the zero-dtype buffer swallows the gradient even if the kernel writes it).

---

## I.3 — Layer-key consistency between forward saves and backward marks

**Problem**: Forward uses `extras_for_current_function(fn_name)` substring match with fallback to `extras_at_index(ordinal)`. Backward uses `mark.layer` directly. For single-layer toy models without a `blocks.N` prefix, the substring match can false-positive or miss, desyncing the save-key.

**Root cause**: two different lookup mechanisms for the same layer identity.

**Fix strategy**: make forward use the **same** key the plan used.
- Thread `last_csha_bridge.layer_at_index(ordinal)` as the authoritative key when substring match fails (already done in expr/advanced.rs:1480-1484).
- Add a debug assert on the forward side that the chosen layer_key is present in `last_csha_bridge.marks.iter().map(|m| m.layer)`, flagging any divergence at compile time.
- OR: tighten `extras_for_current_function` to require a boundary marker (`__` prefix or `.` separator) rather than plain substring.

**Files**:
- `crates/nsl-codegen/src/csha_apply.rs:174-186` — tighten substring check.
- `crates/nsl-codegen/src/expr/advanced.rs:1470-1488` — add debug_assert.

**Test**: `ad_csha_reverse_walk_wiring.rs` — add a case with a single-layer model whose function name contains no `blocks_N` substring, verify forward saves and backward mark agree on key.

**Dependencies**: can land anytime; low priority for single-layer.

---

## K — RMSNorm gamma gradient (post-MVP, but required for training correctness)

**Problem**: RMSNorm gamma is trainable; fused backward emits dx but not dgamma. Per-op AD's gamma path is suppressed by `AlreadyEmitted`. dgamma stays at initial zero.

**Root cause**: Tier C kernel specification didn't include dgamma. Backward phase files (`flash_attention_v2/phases/backward/`) have no `dgamma|d_norm_weight` symbol.

**Fix strategy (two options)**:
- **Option A (kernel-side)**: extend `emit_drmsnorm` to accumulate `d_norm_weight[i] += do_rms[row, i] * x_hat[row, i]` across rows. Adds one more output buffer (dwgamma: [head_dim]). Larger PTX change.
- **Option B (codegen-side)**: after emitting the 7 extract ops in EmitFused, emit a standalone `NormGammaBackward` op that reuses the saved x_raw + do_input. This keeps the fused kernel simple and leans on the existing per-op path.

**Recommendation**: Option B for Gap I; Option A as a later optimization.

**Files** (Option B):
- `crates/nsl-codegen/src/source_ad.rs:340` — after extracts, if `chain_varids.norm_weight_var.is_some()`, emit `NormGammaBackward` with inputs = [do_adjoint, x_raw_VarId].
- `crates/nsl-codegen/src/ad_rules.rs:388` — `NormGammaBackward` already has an expression lowering; verify it works without the full per-op RMSNorm seed.

**Test**: verify `dgamma.sum() > 0` after one fused backward launch, and that it matches (within 1e-3) a reference computed via the per-op path with CSHA disabled.

**Dependencies**: needs J landed (saved x_raw is already threaded via `saves.x_raw`). Can land independently of A+F.

---

## Landing order (minimum viable hd=32 toy pretrain)

1. I.1 — SMEM validator unblocks dispatch.
2. I.2 + M — launch op survives dead-grad elimination.
3. A + F — gradient outputs are f16 on GPU.
4. J — weight pointers threaded.
5. K — RMSNorm gamma grad (required for numerical correctness).
6. I.3 — single-layer key stability (may already work; gated by test).
7. B — cache clear on function boundary (hygiene; nice-to-have).

After steps 1-4, a toy pretrain will RUN without segfaulting but loss will drift because K is unaddressed. After step 5 it should match a reference tape-AD trajectory.

## Test rollup

- Unit tests per fix (above).
- Integration smoke: extend `csha_cuda_launch_fused.rs` with a 4-layer 2-step forward+backward+step cycle; assert loss decreases.
- Numerical gate: re-enable `NUMERICAL_GATE_ENABLED` in `csha_cuda_backward.rs` once the placeholder HBM-load replacement lands (cross-referenced from Tier C close-out).

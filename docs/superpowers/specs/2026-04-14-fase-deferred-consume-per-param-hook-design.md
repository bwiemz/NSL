# FASE Deferred Consume-Per-Parameter Hook — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Depends on:** Item #1 (Deferred codegen), Item #2 (numerical validation), Item #3 (two-phase clip)
**Scoped as:** Item #4 of the FASE roadmap (targeted peak-memory win; does NOT complete M36 proper)
**Follow-up to:** [2026-04-14-fase-deferred-codegen-integration-design.md](2026-04-14-fase-deferred-codegen-integration-design.md)

## Context

CFTP §2.4 claims FASE Deferred's "real win" is peak memory: only one parameter's gradient is live at a time during the final fused step.  Items #1–#3 implement FASE correctness but do not deliver this claim — the backward pass still materializes every parameter's gradient tensor before any consumption runs.

The full M36 memory planner (general compile-time slab assignment for all tensors) is on the broader roadmap as its own milestone and is multi-week work.  For this pass we scope to a targeted FASE-specific fix: a callback hook in the source-AD Wengert lowering fires immediately after each parameter gradient is produced, letting FASE consume-and-free the gradient before the next one is computed.  The existing caching allocator implicitly reuses the freed slab for the next same-sized gradient, giving scratch-buffer-equivalent behavior without explicit buffer management.

Tape-AD (`nsl_tape_backward`) is a single monolithic FFI that materializes all gradients at once; peak-memory relief is not available on the tape-AD path.  Users seeking the demo must compile with `--source-ad`.

## Goals

1. Add a `FnMut(VarId, Value, &mut FunctionBuilder)` callback parameter to `wengert_lower::compile_wengert_ops`.
2. Fire the callback exactly once per trainable parameter, at the Wengert step that first produces the parameter's adjoint tensor.
3. Wire FASE Deferred (both non-clipped and two-phase-clipped variants) to provide a callback that does `fase_emit_accumulate` + `nsl_tensor_free`.
4. Skip the now-redundant post-backward per-parameter accumulation loop and grads_list population when the hook is active.
5. Preserve byte-identical behavior for every non-hook path: FullBuffer, Passthrough, tape AD, and non-FASE code.

## Non-Goals

- **Tape-AD peak-memory relief.** Unsupported; documented as a known limitation.
- **Explicit scratch-buffer management.** Relies on the caching allocator's reuse of freed slabs.  When M36 proper lands, the planner can replace implicit reuse with explicit slot assignment without throwing away any code from this pass.
- **Peak-memory regression test.** Item #5 of the FASE roadmap.
- **`@fase_peak_memory` annotations** or new user-facing syntax.  The hook fires automatically when conditions are met.

## Design Decisions

### D1. Callback signature

```rust
pub fn compile_wengert_ops(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    wengert: &WengertList,
    primal_vars: &HashMap<VarId, Value>,
    on_param_grad: Option<&mut dyn FnMut(
        VarId,
        Value,
        &mut FunctionBuilder,
    ) -> Result<(), CodegenError>>,
) -> Result<...>
```

The `VarId` identifies the parameter whose adjoint was just produced (same `VarId` that `extractor.named_param_var_ids()` yields).  The `Value` is a Cranelift Value holding the runtime pointer to the newly-allocated gradient tensor.  The `&mut FunctionBuilder` lets the callback emit further IR into the same function — without it, the callback cannot call `compile_call_by_name` or any other emission helper.

Return type `Result<(), CodegenError>` lets the callback surface errors up through Wengert lowering without panicking.

### D2. When the hook fires

Inside `wengert_lower.rs`, at every op-emission site whose output becomes a parameter's adjoint tensor, check `if on_param_grad.is_some() && is_trainable_param(var_id)`, then invoke the callback with the adjoint's `VarId` and its `Value`.

"Parameter adjoint" identification uses the same mechanism the existing post-Wengert patch loop uses ([stmt.rs:4004](crates/nsl-codegen/src/stmt.rs#L4004)): the set of `VarId`s corresponding to trainable parameter leaves, resolved from `extractor.named_param_var_ids()` and passed into lowering via the compiler state or a similar channel.

After the callback returns, `compile_wengert_ops` considers the gradient tensor "consumed" — no entry is added to any internal `grad_vars` map that a later step would try to use again.  The callback is responsible for freeing the tensor.

### D3. FASE callback construction in `stmt.rs`

When both `fase_deferred == true` and `self.features.source_ad_enabled == true`, build a local closure that captures the runtime `accum_list` handle and the `VarId → accum_list index` mapping:

```rust
// Built once before compile_wengert_ops.
let var_id_to_accum_idx: HashMap<VarId, i64> = /* from named_param_var_ids */;

let accum = accum_list.expect("Deferred always allocates accum_list");
let accum_scale = fase_plan.recipe.accum_scale;

let mut cb = |var_id: VarId, grad_ptr: Value, b: &mut FunctionBuilder| -> Result<(), CodegenError> {
    let Some(&idx) = var_id_to_accum_idx.get(&var_id) else {
        // Not a trainable param this FASE path tracks; skip.
        return Ok(());
    };
    let idx_val = b.ins().iconst(cl_types::I64, idx);
    let m_partial = self.compile_call_by_name(b, "nsl_list_get", &[accum, idx_val])?;
    self.fase_emit_accumulate(b, m_partial, grad_ptr, accum_scale)?;
    self.compile_call_by_name(b, "nsl_tensor_free", &[grad_ptr])?;
    Ok(())
};
```

Pass `Some(&mut cb)` into `compile_wengert_ops`.

### D4. Paths that use the hook

| Path | Hook |
|---|---|
| Source AD + FASE Deferred (no grad_clip) | **On** |
| Source AD + FASE Deferred + grad_clip | **On** (accumulate step moves into the hook; sum_sq post-backward and Phase B unchanged) |
| Source AD + FASE FullBuffer | Off (uses raw `accum += g`, not pre-scaled) |
| Source AD + FASE Passthrough (N=1) | Off (no accumulation buffer) |
| Tape AD, any FASE mode | Off (monolithic backward; callback can't fire mid-backward) |

For all Off rows, `on_param_grad = None` and `compile_wengert_ops` is byte-identical to today.

### D5. Two-phase clip interaction

Item #3's Phase A currently fuses `m_partial += g/N` with `sum_sq(m_partial[k])`.  With the hook, the accumulation half moves into the callback (runs during backward).  The `sum_sq` half stays in the post-backward Phase A loop — it only needs `m_partial`, which remains live after the callback.

Net Phase A cost after this change: one `sum_sq` call per parameter (was two ops: accumulate + sum_sq).  Slightly faster.  The Phase B scale-then-step loop is unchanged.

### D6. Redundant-work elimination

When the hook is active:
1. The post-Wengert "seed grads_list with zeros then patch with actual gradients" loop ([stmt.rs:3944-4040](crates/nsl-codegen/src/stmt.rs#L3944-L4040)) must be SKIPPED.  Grads are already consumed; keeping this loop would either (a) be dead code emitting noise allocations, or (b) double-free if it tried to push already-freed tensors.
2. The per-micro-batch accumulation loop ([stmt.rs:4183](crates/nsl-codegen/src/stmt.rs#L4183)) must be SKIPPED for the Deferred branch when the hook is active.  The hook already did the accumulation.

Both skips are compile-time `if self.hook_active_for_this_path { ... }` checks — zero runtime cost.

### D7. Correctness invariants

1. **Each trainable parameter's gradient is produced and consumed exactly once.**  Verified by a unit test with a mock callback that counts invocations per `VarId`.
2. **Final parameter values match the non-hook path.**  Items #2's tests (`sgd_exact_equivalence`, `adamw_fase_deferred_pipeline_equivalence`, `adamw_deferred_with_grad_clip`) must pass at their existing tolerances with source AD active.  If those tests currently run on tape AD, a parallel source-AD variant is added.
3. **Non-hook paths are byte-identical in emitted IR.**  Existing snapshot tests must not change.

### D8. Components touched

| File | Change |
|---|---|
| `crates/nsl-codegen/src/wengert_lower.rs` | Add `on_param_grad` parameter to `compile_wengert_ops`; fire it at adjoint-emission sites for trainable-param `VarId`s; propagate through internal helpers.  ~80 LOC. |
| `crates/nsl-codegen/src/stmt.rs` | Build `VarId → accum_idx` map before lowering; construct callback closure when Deferred+source-AD; pass it through; gate the now-redundant post-Wengert patch loop and per-micro-batch accumulation loop on hook-active.  ~150 LOC. |
| `crates/nsl-codegen/src/wengert_lower.rs` (`#[cfg(test)] mod tests`) | Mock-callback test: tiny Wengert list with two trainable params, record `(VarId, call_order)`, assert one callback per param.  ~60 LOC. |
| `crates/nsl-codegen/tests/fase_numerical_validation.rs` | If item #2's tests don't already run on source AD, add a `#[test]` variant forcing `--source-ad` for the AdamW pipeline test — so this pass's change is covered end-to-end.  ~40 LOC. |
| memory note | Update FASE project memory. |

Total: ~330 LOC + memory update.

## Architecture

### Data flow (source AD + FASE Deferred)

```
train-block emitter (stmt.rs):
  │
  ├─ allocate param_list, accum_list (one slot per trainable param)
  ├─ allocate m-state-list, v-state-list (optimizer state)
  ├─ build var_id_to_accum_idx: HashMap<VarId, i64>
  ├─ construct callback closure capturing { accum, accum_scale, map, self }
  │
  └─ for each micro-batch:
        compile_wengert_ops(..., on_param_grad = Some(&mut cb))
           │
           └─ for each Wengert op in topological order:
                 emit primal op (forward)
                 ...
                 for each Wengert op in REVERSE topological order:
                    emit adjoint op (backward) → produces grad tensor
                    if is_param(var_id) && on_param_grad.is_some():
                        cb(var_id, grad_ptr, builder):
                            m_partial = nsl_list_get(accum, map[var_id])
                            fase_emit_accumulate(m_partial, grad_ptr, accum_scale)
                            nsl_tensor_free(grad_ptr)
                            // Allocator slab now free for the next grad.

        // NO post-Wengert grads_list patch loop (skipped).
        // NO per-micro-batch accumulation loop (skipped).

        if final_micro_batch:
            if two_phase_clip:
                # Phase A: fold per-param sum_sq — accumulate already done.
                total_sq = 0
                for each param k:
                    total_sq += nsl_tensor_sum_sq(accum_list[k])
                clip_factor = min(1, τ / (sqrt(total_sq) + 1e-6))
                # Phase B: scale + fused step.
                for each param k:
                    nsl_tensor_mul_scalar_inplace(accum_list[k], clip_factor)
                    fase_emit_final_step(..., bc_params)
            else:
                # Non-clip: per-param fused step.
                for each param k:
                    fase_emit_final_step(..., bc_params)
```

### Peak-memory math (NSLCoder-50M, 48.8M params, f32, N=4)

| State | Before #4 | After #4 |
|---|---|---|
| Parameters | 195 MB | 195 MB |
| Optimizer state (m, v, m_partial) | 585 MB | 585 MB |
| Activations (saved for backward) | ~200 MB | ~200 MB |
| Gradients in flight | **195 MB (all simultaneously)** | **~4 MB (largest single gradient)** |
| **Total peak** | **~1175 MB** | **~984 MB** |

~16% reduction on this model.  Larger relative wins on models where a single parameter dominates (LM heads for large vocabularies).

## Risks

1. **Finding the "parameter adjoint produced" site in wengert_lower.**  Large function with multiple code paths.  Implementation plan's Task 1 does the reconnaissance; if the site isn't unique and clean, surface it before writing the hook — may need to refactor first.
2. **VarId mapping completeness.**  `named_param_var_ids()` must match exactly what `extractor`'s post-Wengert loop uses today; any drift leaves orphan accum_list entries (stuck at zero) or spurious callback invocations.
3. **Callback borrow-checker friction.**  The closure captures `self` mutably (for `compile_call_by_name`).  Passing `&mut dyn FnMut` through `compile_wengert_ops` avoids `'static` bounds, but the inner helpers may need threading.  If the trait-object approach fights the borrow checker, fall back to a dedicated `ParamGradHandler` struct with an explicit method, passed by `&mut`.
4. **Tests currently using tape AD.**  If item #2's validation tests run on tape AD by default, this pass's hook won't fire — silent no-op.  Add a source-AD variant of the AdamW pipeline test (D8 row 4) so the hook is actually exercised end-to-end.
5. **FullBuffer accumulation on source AD.**  FullBuffer mode on source AD currently uses the same post-Wengert patch loop.  We must ensure the hook is strictly gated on Deferred mode — accidentally engaging it for FullBuffer would break that path's `accum += g` semantics.

## Success Criteria

- `cargo test -p nsl-codegen` all green, including the new mock-callback unit test and the source-AD variant of the AdamW pipeline test.
- No snapshot regressions for non-hook paths.
- `compile_wengert_ops` with `on_param_grad = None` is byte-identical in emitted IR to the pre-#4 version (sanity-checked by any non-hook test that previously compared IR).
- Final parameter trajectories for source-AD + FASE Deferred match the reference (from item #2) within existing tolerances.

## Follow-Ups

This spec closes item #4.  Remaining FASE items:

- **Item #5:** peak-memory regression test — measures actual allocator high-water-mark, compares to a budget, asserts the reduction this pass claims.  Depends on an allocator-introspection hook (exists or near-existing in the caching allocator).
- **Item #6:** `nsl check --training-report` CLI.  Independent.
- **Future M36 milestone:** when general-purpose planned allocation lands, the hook in this spec becomes a natural special case — the callback gets its gradient slot from M36's planner instead of a fresh heap alloc.  No rework.

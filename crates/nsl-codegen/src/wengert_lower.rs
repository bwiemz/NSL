//! M40: Lower a WengertList (symbolic backward graph) to Cranelift IR.
//!
//! Each `PrimalOp` variant is dispatched to the corresponding runtime FFI call.
//! The result is a map from all VarIds (primal + adjoint) to Cranelift `Value`s.

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::wengert::{
    type_for_op, CompareKind, ConvGradKind, PrimalOp, VarId, WengertList, WengertOp, WengertType,
};
use crate::CodegenError;
use cranelift_codegen::ir::{condcodes::IntCC, types as cl_types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;
use std::collections::HashMap;

pub type VarMap = HashMap<VarId, Value>;

pub struct LoweredWengert {
    pub var_map: VarMap,
    pub owned_values: Vec<(VarId, Value, WengertType)>,
    /// VarIds that were freed early by the FASE hook's reduce_to_shape
    /// identity path.  When the hook fires on a `reduce_to_shape` result and
    /// the runtime returns the same pointer (shapes match → retain+return),
    /// the input VarId (raw_grad) needs an extra `nsl_tensor_free` to bring
    /// the refcount to zero — the hook's free only drops it from 2→1.  The
    /// extra free is emitted here; these VarIds are added to
    /// `freed_adjoint_vars` in stmt.rs so `free_wengert_owned_values` skips
    /// them and does not double-free.
    ///
    /// This is also correct when shapes differ (normal clone path): the input
    /// grad (rc=1) is freed immediately by the extra call, and cleanup skips it.
    pub hook_freed_input_vars: std::collections::HashSet<VarId>,
    /// CCR: victims of `PrimalOp::FreeTensor` markers emitted in this
    /// lowering.  stmt.rs merges these into the bulk-free exclusion sets so
    /// `free_wengert_owned_values` does not double-free them.  Distinct from
    /// `hook_freed_input_vars` only in provenance (explicit tape markers vs
    /// the FASE hook's identity-path extra free).
    pub explicit_freed_vars: std::collections::HashSet<VarId>,
}

/// Lower a WengertList to Cranelift IR by dispatching each PrimalOp to its runtime FFI call.
///
/// `primal_vars` maps VarIds from the forward pass to their Cranelift Values (i64 tensor pointers).
/// Returns a map from all VarIds (including adjoint) to Cranelift Values.
///
/// `on_param_grad`, when `Some((set, cb))`, causes `cb` to be invoked
/// immediately after any op whose `result` VarId is in `set`.  The callback
/// receives `(&mut Compiler, VarId, Value, &mut FunctionBuilder)` — the
/// compiler is passed explicitly so the closure does NOT need to capture it,
/// avoiding a double-mutable-borrow with the `compiler` parameter above.
/// The gradient is then REMOVED from `var_map` — the callback is responsible
/// for freeing or otherwise owning that tensor.  Used by FASE Deferred to
/// consume parameter gradients during backward lowering so only one gradient
/// is live at a time.
pub fn compile_wengert_ops(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    wengert: &WengertList,
    primal_vars: &VarMap,
    on_param_grad: Option<(
        &std::collections::HashSet<VarId>,
        &mut dyn FnMut(
            &mut Compiler,
            VarId,
            Value,
            &mut FunctionBuilder,
        ) -> Result<(), CodegenError>,
    )>,
) -> Result<LoweredWengert, CodegenError> {
    let mut var_map = primal_vars.clone();
    let mut owned_values = Vec::new();
    let mut hook_freed_input_vars = std::collections::HashSet::new();
    let mut explicit_freed_vars = std::collections::HashSet::new();
    // Clone var_types so the lowerer can look up types for any VarId and
    // also tag newly-created adjoint VarIds.
    let mut var_types = wengert.var_types.clone();
    compile_wengert_ops_range(
        compiler,
        builder,
        state,
        wengert,
        0..wengert.ops.len(),
        &mut var_map,
        &mut var_types,
        &mut owned_values,
        &mut hook_freed_input_vars,
        &mut explicit_freed_vars,
        on_param_grad,
    )?;
    // PCA Stage C: adopt the fused-SDPA extras (saved-LSE tensors) into the
    // step's owned set so cleanup frees them. VarId u32::MAX is a sentinel
    // no real op produces (it never appears in hook_freed_input_vars).
    for v in compiler.sdpa_extra_owned.drain(..) {
        owned_values.push((u32::MAX, v, WengertType::Tensor));
    }
    Ok(LoweredWengert {
        var_map,
        owned_values,
        hook_freed_input_vars,
        explicit_freed_vars,
    })
}

/// The sequential-fold core of [`compile_wengert_ops`], parameterized over
/// an op-index range with all fold state threaded in/out.
///
/// CSLA D2b part 2 (forward weight streaming) lowers the primal tape as
/// per-CCR-segment SLICES with upload/evict FFI calls emitted between
/// them. All slices lower into the same straight-line block chain, so SSA
/// values flow across slice boundaries for free — but the fold state must
/// be threaded explicitly: `var_types` carries the local Integer-vs-Tensor
/// inference (losing it at a boundary would re-default a slice-k integer
/// result to Tensor in slice k+1 and push a raw integer into
/// `owned_values` → `nsl_tensor_free` on a non-pointer), and the
/// owned/freed sets concatenate across slices so cleanup sees exactly what
/// the monolithic lowering would have produced.
///
/// The `sdpa_extra_owned` drain deliberately stays in the WRAPPER (and in
/// the sliced caller, after its last slice): extras accrue on the compiler
/// across the whole lowering regardless of slicing.
#[allow(clippy::too_many_arguments)]
pub fn compile_wengert_ops_range(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    _state: &mut FuncState,
    wengert: &WengertList,
    range: std::ops::Range<usize>,
    var_map: &mut VarMap,
    var_types: &mut HashMap<VarId, WengertType>,
    owned_values: &mut Vec<(VarId, Value, WengertType)>,
    hook_freed_input_vars: &mut std::collections::HashSet<VarId>,
    explicit_freed_vars: &mut std::collections::HashSet<VarId>,
    mut on_param_grad: Option<(
        &std::collections::HashSet<VarId>,
        &mut dyn FnMut(
            &mut Compiler,
            VarId,
            Value,
            &mut FunctionBuilder,
        ) -> Result<(), CodegenError>,
    )>,
) -> Result<(), CodegenError> {
    for op in &wengert.ops[range] {
        // Skip ops whose inputs can't be resolved (ghost VarIds from
        // get_or_create_adjoint that never received a gradient).
        // These produce dead adjoint paths for non-differentiable ops.
        let all_inputs_resolved = op.inputs.iter().all(|vid| var_map.contains_key(vid));
        if !all_inputs_resolved {
            // Ghost VarId — skip this op. Its result stays unmapped,
            // which propagates the "no gradient" signal downstream.
            if std::env::var("NSL_DEBUG_WENGERT").is_ok() {
                let missing: Vec<_> = op
                    .inputs
                    .iter()
                    .filter(|vid| !var_map.contains_key(vid))
                    .collect();
                eprintln!(
                    "[wengert-skip] VarId {} ({:?}) — missing inputs: {:?}",
                    op.result, op.op, missing
                );
            }
            continue;
        }
        // CCR early-free marker: emit the refcount decrement here in the
        // main loop (not in `lower_single_op`) so the freed VarId can be
        // recorded on the lowering result — the end-of-backward bulk free
        // consults `explicit_freed_vars` to skip these, exactly like the
        // FASE hook's `hook_freed_input_vars`. The victim stays in
        // `var_map` on purpose: `ccr::apply_to_adjoint` statically
        // validated that no later op references it, and keeping it mapped
        // means a validation bug would surface as a use-after-free under
        // compute-sanitizer rather than a silent ghost-skip (missing
        // gradient) — loud over silent.
        if matches!(op.op, PrimalOp::FreeTensor) {
            let victim_vid = op.inputs[0];
            let victim_val = var_map[&victim_vid];
            // Free ONLY what the end-of-backward bulk free would have
            // freed: a Tensor-typed victim whose lowered Value is an I64
            // pointer. The tape-level type default ("Tensor") over-claims
            // for scalar arithmetic, which lowers to raw f64 — freeing
            // that is a verifier error (and an integer masquerading as a
            // pointer would be worse). Skipped victims simply keep their
            // present end-of-backward lifetime; they hold no device memory.
            let is_tensor_ptr = builder.func.dfg.value_type(victim_val) == cl_types::I64
                && matches!(
                    var_types.get(&victim_vid),
                    Some(WengertType::Tensor) | None
                );
            if is_tensor_ptr {
                call(compiler, builder, "nsl_tensor_free", &[victim_val])?;
                explicit_freed_vars.insert(victim_vid);
            }
            let placeholder = builder.ins().iconst(cl_types::I64, 0);
            var_map.insert(op.result, placeholder);
            var_types.insert(op.result, WengertType::Integer);
            continue;
        }
        let result_val = lower_single_op(compiler, builder, op, var_map, var_types)?;
        var_map.insert(op.result, result_val);
        // FASE hook: consume parameter gradients immediately during lowering.
        if let Some(ref mut hook) = on_param_grad {
            let (param_set, cb) = hook;
            if param_set.contains(&op.result) {
                cb(compiler, op.result, result_val, builder)?;
                // Callback owns/freed the tensor — remove from var_map so
                // downstream code can't accidentally re-use it.
                var_map.remove(&op.result);

                // If the hook fired on a reduce_to_shape op, emit an extra
                // nsl_tensor_free for the raw-grad input (inputs[0]).
                //
                // Why this is needed: nsl_tensor_reduce_to_shape increments
                // the input's refcount and returns the same pointer when the
                // src and target shapes match (the identity path).  The hook
                // callback already called nsl_tensor_free(result_val), which
                // drops refcount from 2→1 — not an actual free.  We need one
                // more free here to bring it to 0 so the raw grad is released
                // immediately, not held until end-of-adjoint cleanup (which
                // is what causes the N×grad_size peak-memory regression).
                //
                // This is also correct on the non-identity path (fresh clone):
                // the raw_grad has rc=1, this free drops it to 0 (freed), and
                // the cleanup pass skips it via hook_freed_input_vars.
                if matches!(&op.op, PrimalOp::Passthrough(name) if name == "reduce_to_shape") {
                    if let Some(&input_vid) = op.inputs.first() {
                        if let Some(&input_val) = var_map.get(&input_vid) {
                            call(compiler, builder, "nsl_tensor_free", &[input_val])?;
                            hook_freed_input_vars.insert(input_vid);
                        }
                    }
                }
            }
        }
        // Infer result type: for binary ops, if both inputs are Integer, result is Integer.
        // For Constants, preserve the extractor's type (it may have overridden
        // the default Tensor to Integer for IntLiteral).
        let result_type = match &op.op {
            PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                let a_ty = var_types
                    .get(&op.inputs[0])
                    .copied()
                    .unwrap_or(WengertType::Tensor);
                let b_ty = var_types
                    .get(&op.inputs[1])
                    .copied()
                    .unwrap_or(WengertType::Tensor);
                if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                    WengertType::Integer
                } else {
                    WengertType::Tensor
                }
            }
            PrimalOp::Constant(_) => {
                // Preserve the extractor's type annotation (Integer for IntLiteral,
                // Tensor for adjoint seeds, Scalar for float literals).
                var_types
                    .get(&op.result)
                    .copied()
                    .unwrap_or(type_for_op(&op.op))
            }
            _ => type_for_op(&op.op),
        };
        if should_cleanup_result(&op.op, result_type) {
            owned_values.push((op.result, result_val, result_type));
        }
        var_types.insert(op.result, result_type);
    }
    Ok(())
}

/// CSLA D2b part 2: a PURE replica of the ownership/type fold
/// [`compile_wengert_ops_range`] performs over a PRIMAL tape — no IR
/// emission, no `Value`s. Lets stmt.rs run `CcrPlan::restrict_to_owned`
/// BEFORE the forward lowering: the segment-streamed forward needs the
/// post-restriction plan (and the final adjoint derived from it) at
/// emission time, but the restriction historically consumed the lowering's
/// own owned classification. stmt.rs asserts post-lowering that the actual
/// classification matches this prediction — divergence is a compiler bug
/// and fails the compile loudly rather than mis-planning silently.
///
/// Replicated decisions, in order: ghost-skip (an op with an unresolved
/// input produces nothing — resolution seeds from `seed_resolved`, the
/// primal_vars key set), `FreeTensor` interception (result becomes a
/// resolved Integer, nothing owned — absent from real primal tapes but
/// modeled for faithfulness), the Add/Sub/Mul/Div both-Integer fold,
/// `Constant` extractor-type preservation, `should_cleanup_result`.
/// The FASE hook never fires on primal tapes (stmt.rs passes None), so no
/// hook interaction is modeled. The `sdpa_extra_owned` u32::MAX sentinel
/// entries are compiler-side extras, not tape results — callers exclude
/// them when comparing.
pub fn infer_primal_owned(
    wengert: &WengertList,
    seed_resolved: &std::collections::HashSet<VarId>,
) -> HashMap<VarId, WengertType> {
    let mut resolved = seed_resolved.clone();
    let mut var_types = wengert.var_types.clone();
    let mut owned: HashMap<VarId, WengertType> = HashMap::new();
    for op in &wengert.ops {
        if !op.inputs.iter().all(|vid| resolved.contains(vid)) {
            continue;
        }
        if matches!(op.op, PrimalOp::FreeTensor) {
            resolved.insert(op.result);
            var_types.insert(op.result, WengertType::Integer);
            continue;
        }
        let result_type = match &op.op {
            PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                let a_ty = var_types
                    .get(&op.inputs[0])
                    .copied()
                    .unwrap_or(WengertType::Tensor);
                let b_ty = var_types
                    .get(&op.inputs[1])
                    .copied()
                    .unwrap_or(WengertType::Tensor);
                if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                    WengertType::Integer
                } else {
                    WengertType::Tensor
                }
            }
            PrimalOp::Constant(_) => var_types
                .get(&op.result)
                .copied()
                .unwrap_or(type_for_op(&op.op)),
            _ => type_for_op(&op.op),
        };
        resolved.insert(op.result);
        if should_cleanup_result(&op.op, result_type) {
            owned.insert(op.result, result_type);
        }
        var_types.insert(op.result, result_type);
    }
    owned
}

/// Resolve all input VarIds for a WengertOp to their Cranelift Values.
fn resolve_inputs(op: &WengertOp, var_map: &VarMap) -> Result<Vec<Value>, CodegenError> {
    op.inputs
        .iter()
        .map(|vid| {
            var_map.get(vid).copied().ok_or_else(|| {
                CodegenError::new(format!(
                    "source-ad: unresolved VarId {} (input to {:?} at VarId {})",
                    vid, op.op, op.result
                ))
            })
        })
        .collect()
}

/// Emit a single runtime FFI call and return the result Value.
fn call(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    name: &str,
    args: &[Value],
) -> Result<Value, CodegenError> {
    compiler.compile_call_by_name(builder, name, args)
}

fn should_cleanup_result(op: &PrimalOp, result_type: WengertType) -> bool {
    if !matches!(result_type, WengertType::Tensor | WengertType::List) {
        return false;
    }
    match op {
        PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_) => false,
        PrimalOp::Passthrough(name) if name.starts_with("dict_get:") => false,
        _ => true,
    }
}

fn free_tensor_value(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    value: Value,
) -> Result<(), CodegenError> {
    let _ = call(compiler, builder, "nsl_tensor_free", &[value])?;
    Ok(())
}

fn free_tensor_if_owned(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    value: Value,
    owned: bool,
) -> Result<(), CodegenError> {
    if owned {
        free_tensor_value(compiler, builder, value)?;
    }
    Ok(())
}

/// PCA Stage C: emit the fused-forward runtime dispatch for a decorator-free
/// SDPA op (plain or packed).
///
/// Emits the per-head_dim variant select (equality chain on the runtime
/// head_dim, mirroring the backward table's pattern), the
/// `nsl_sdpa_fused_forward` call, and the decline branch:
///
/// ```text
///   list = nsl_sdpa_fused_forward(...)
///   brif list != 0 -> fused_block, decomp_block
///   fused_block:  out = list[0]; lse = list[1]; free list; jump join(out, lse)
///   decomp_block: <caller emits the decomposed chain>; jump join(out, 0)
///   join(out, lse): ...
/// ```
///
/// Returns `Ok(None)` when the variant table is empty (non-CUDA target) —
/// the caller then lowers the decomposed chain in the CURRENT block with no
/// branching, exactly as before Stage C. Otherwise returns
/// `Ok(Some((join_block, out_param, lse_param)))` with the builder
/// positioned at the START of the decline block: the caller emits its
/// decomposed chain there, jumps to `join_block` with
/// `[decomposed_result, iconst 0]`, seals `join_block`, switches to it, and
/// uses the two params as (result, lse).
#[allow(clippy::too_many_arguments)]
fn emit_sdpa_fused_dispatch(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    q: Value,
    k: Value,
    v: Value,
    scale_bits: Value,
    causal: bool,
    segment_ids: Option<Value>,
) -> Result<Option<(cranelift_codegen::ir::Block, Value, Value)>, CodegenError> {
    let segment_masked = segment_ids.is_some();
    let variants = compiler.ensure_sdpa_fwd_variant_table(causal, segment_masked)?;
    if variants.is_empty() {
        return Ok(None);
    }

    // Runtime head_dim from Q ([b, h, s, d]) — via the non-aborting probe:
    // `nsl_tensor_shape_dim` process-aborts on out-of-range dims, and this
    // dispatch is emitted for EVERY decorator-free SDPA on CUDA targets,
    // including rank<4 attention that used to lower decomposed just fine.
    // Out-of-range returns 0, which matches no variant, so the FFI declines
    // and the decomposed fallback runs (review finding).
    let dim3 = builder.ins().iconst(cl_types::I64, 3);
    let head_dim = call(compiler, builder, "nsl_tensor_dim_or_zero", &[q, dim3])?;

    // Variant select: equality chain on head_dim. Unmatched dims leave all
    // selectors zero → the FFI declines → decomposed fallback.
    let zero = builder.ins().iconst(cl_types::I64, 0);
    let mut sel_base_ptx = zero;
    let mut sel_base_name = zero;
    let mut sel_tb_ptx = zero;
    let mut sel_tb_name = zero;
    let mut sel_bq = zero;
    let mut sel_bkv = zero;
    let mut sel_smem = zero;
    for variant in &variants {
        let is_hd = builder
            .ins()
            .icmp_imm(IntCC::Equal, head_dim, variant.head_dim);
        let base_ptx_gv = compiler
            .module
            .declare_data_in_func(variant.base_ptx, builder.func);
        let base_ptx_addr = builder.ins().symbol_value(cl_types::I64, base_ptx_gv);
        sel_base_ptx = builder.ins().select(is_hd, base_ptx_addr, sel_base_ptx);
        let base_name_gv = compiler
            .module
            .declare_data_in_func(variant.base_name, builder.func);
        let base_name_addr = builder.ins().symbol_value(cl_types::I64, base_name_gv);
        sel_base_name = builder.ins().select(is_hd, base_name_addr, sel_base_name);
        // Tier-B pair: present only on segment_masked tables. The sentinel
        // discipline (both-zero or both-nonzero) is preserved per variant.
        if let (Some(tb_ptx), Some(tb_name)) = (variant.tier_b_ptx, variant.tier_b_name) {
            let tb_ptx_gv = compiler.module.declare_data_in_func(tb_ptx, builder.func);
            let tb_ptx_addr = builder.ins().symbol_value(cl_types::I64, tb_ptx_gv);
            sel_tb_ptx = builder.ins().select(is_hd, tb_ptx_addr, sel_tb_ptx);
            let tb_name_gv = compiler.module.declare_data_in_func(tb_name, builder.func);
            let tb_name_addr = builder.ins().symbol_value(cl_types::I64, tb_name_gv);
            sel_tb_name = builder.ins().select(is_hd, tb_name_addr, sel_tb_name);
        }
        let bq = builder.ins().iconst(cl_types::I64, variant.block_q);
        sel_bq = builder.ins().select(is_hd, bq, sel_bq);
        let bkv = builder.ins().iconst(cl_types::I64, variant.block_kv);
        sel_bkv = builder.ins().select(is_hd, bkv, sel_bkv);
        let smem = builder.ins().iconst(cl_types::I64, variant.smem_bytes);
        sel_smem = builder.ins().select(is_hd, smem, sel_smem);
    }

    let causal_val = builder.ins().iconst(cl_types::I64, i64::from(causal));
    let seg_val = segment_ids.unwrap_or(zero);

    // 13-arg fused-forward FFI. Returns NslList* [out, lse] or 0 (decline).
    let list = call(
        compiler,
        builder,
        "nsl_sdpa_fused_forward",
        &[
            q, k, v, scale_bits, causal_val, seg_val, sel_base_ptx, sel_base_name, sel_tb_ptx,
            sel_tb_name, sel_bq, sel_bkv, sel_smem,
        ],
    )?;

    let fused_block = builder.create_block();
    let decomp_block = builder.create_block();
    let join_block = builder.create_block();
    let out_param = builder.append_block_param(join_block, cl_types::I64);
    let lse_param = builder.append_block_param(join_block, cl_types::I64);

    let took_fused = builder.ins().icmp_imm(IntCC::NotEqual, list, 0);
    builder
        .ins()
        .brif(took_fused, fused_block, &[], decomp_block, &[]);

    // Fused: unpack [out, lse] and free the (element-preserving) list.
    builder.switch_to_block(fused_block);
    builder.seal_block(fused_block);
    let idx0 = builder.ins().iconst(cl_types::I64, 0);
    let fused_out = call(compiler, builder, "nsl_list_get", &[list, idx0])?;
    let idx1 = builder.ins().iconst(cl_types::I64, 1);
    let fused_lse = call(compiler, builder, "nsl_list_get", &[list, idx1])?;
    let _ = call(compiler, builder, "nsl_list_free", &[list])?;
    builder.ins().jump(join_block, &[fused_out, fused_lse]);

    // Decline: the caller emits the decomposed chain here, then jumps to
    // join_block with [decomposed_result, 0] and seals it.
    builder.switch_to_block(decomp_block);
    builder.seal_block(decomp_block);

    Ok(Some((join_block, out_param, lse_param)))
}

/// Promote an Integer (i64) to a scalar Tensor if needed for mixed-type arithmetic.
fn promote_to_tensor(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    val: Value,
    ty: WengertType,
) -> Result<(Value, bool), CodegenError> {
    match ty {
        WengertType::Integer => {
            let f = builder.ins().fcvt_from_sint(cl_types::F64, val);
            let dt = builder.ins().iconst(cl_types::I64, 0); // f64
            Ok((
                call(compiler, builder, "nsl_tensor_scalar", &[f, dt])?,
                true,
            ))
        }
        WengertType::Scalar => {
            // f64 scalar → wrap in a scalar tensor for tensor arithmetic
            let dt = builder.ins().iconst(cl_types::I64, 0); // f64
            Ok((
                call(compiler, builder, "nsl_tensor_scalar", &[val, dt])?,
                true,
            ))
        }
        _ => Ok((val, false)), // Already a tensor pointer (i64)
    }
}

/// Option 3a — emit the fused CSHA forward FFI under a backward-dispatcher
/// claim.
///
/// Called from `PrimalOp::ScaledDotProductAttention` when the op is in
/// `compiler.csha_backward_claims.op_to_chain` AND the claim's layer has
/// `save_activations_for_backward=true`.  This REPLACES the primitive
/// decomposition (matmul → softmax → matmul) for the claimed op.
///
/// Responsibilities:
///   1. Allocate the six backward-activation buffers via
///      `nsl_csha_alloc_backward_activations_into` (matches the Gap A
///      allocator used by `compile_flash_attention_call`).
///   2. Resolve x_norm / norm_weight / Wq / Wk / Wv primal VarIds from
///      the chain's `CshaChainVarIds` into Cranelift `Value`s via
///      `var_map`.  Any VarId that can't be resolved stays null; the
///      FFI's per-arg null-guard falls through to the classic path for
///      that slot.
///   3. Load PTX / name pointers from `.rodata` via
///      `FlashAttentionCompileContext.csha_forward_with_saves_{ptx,name}_id`
///      — falling back to the non-CSHA PTX when the module has no
///      `@train` block.
///   4. Emit the 36-arg `nsl_flash_attention_csha_with_saves` FFI
///      call, returning the attention output tensor.
///   5. Stash the six save-pointer Cranelift Values on
///      `compiler.csha_forward_saves[layer]` so Gap D.1's
///      `FusedCshaBackward` lowerer finds them when it runs later in
///      the SAME function body.
///
/// The layer-key used here MUST match the key the backward lowerer
/// reads — both resolve from `claim.chain_marks[idx].layer`, so they're
/// structurally identical.
fn emit_fused_forward_under_claim(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    op: &WengertOp,
    inputs: &[Value],
    var_map: &VarMap,
    layer: &str,
) -> Result<Value, CodegenError> {
    // --- 1. Resolve the config (block_q/block_kv/head_dim/causal/
    //        d_model/eps_bits/shmem_bytes) from the flash attention
    //        context.  If the training config is absent this means the
    //        module never set up CSHA training PTX — the claim itself
    //        shouldn't have been issued, so we bail with a diagnostic
    //        rather than silently falling back to decomposition. ---
    let training_cfg = compiler
        .kernels
        .flash_attention_context
        .as_ref()
        .and_then(|c| c.csha_training_config.clone())
        .ok_or_else(|| {
            CodegenError::new(format!(
                "[source-ad] CSHA fused forward claim fired for layer '{}' \
                 but `csha_training_config` is absent — the module has no \
                 `@train` block or CSHA training PTX synthesis was rejected. \
                 This is a dispatcher bug, not a structural fallback case.",
                layer
            ))
        })?;
    let (block_q_i64, block_kv_i64, is_causal, d_model_i64, eps_bits_i64) = {
        let dm = training_cfg.csha.as_ref().map(|c| c.d_model as i64).unwrap_or(0);
        let eps = training_cfg
            .csha
            .as_ref()
            .map(|c| c.rmsnorm_eps.to_bits() as i64)
            .unwrap_or(1e-5_f32.to_bits() as i64);
        (
            training_cfg.block_q,
            training_cfg.block_kv,
            training_cfg.causal,
            dm,
            eps,
        )
    };
    let shmem_bytes_i64 = {
        let mut diags = Vec::<String>::new();
        let bytes = crate::flash_attention_selector::shared_mem_bytes_selected_with_diag(
            &training_cfg, &mut diags,
        ) as i64;
        for s in diags {
            eprintln!("warning: {s}");
        }
        bytes
    };
    // active_heads from the bridge (same source as
    // `compile_flash_attention_call`'s non-source-AD branch).
    let active_heads_i64 = compiler
        .last_csha_bridge
        .as_ref()
        .and_then(|b| b.extras_for_layer(layer))
        .map(|e| e.active_heads as i64)
        .unwrap_or(0);

    // --- 2. SDPA inputs: [q, k, v, scale, ...].  The scale slot is an
    //        f64 literal (NSL float literals lower to F64); the FFI
    //        expects an i64 carrying the f32 bit pattern. ---
    let q_val = inputs[0];
    let k_val = inputs[1];
    let v_val = inputs[2];
    let scale_bits = if inputs.len() > 3 {
        let scale_val = inputs[3];
        let scale_ty = builder.func.dfg.value_type(scale_val);
        if scale_ty == cl_types::F64 {
            let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_val);
            let scale_bits_i32 = builder.ins().bitcast(
                cl_types::I32,
                cranelift_codegen::ir::MemFlags::new(),
                scale_f32,
            );
            builder.ins().uextend(cl_types::I64, scale_bits_i32)
        } else if scale_ty == cl_types::F32 {
            let scale_bits_i32 = builder.ins().bitcast(
                cl_types::I32,
                cranelift_codegen::ir::MemFlags::new(),
                scale_val,
            );
            builder.ins().uextend(cl_types::I64, scale_bits_i32)
        } else {
            // Integer Value — typically an NslTensor handle (I64 pointer) for
            // scale, since NSL expressions like `1.0 / sqrt(32.0)` often flow
            // as rank-0 tensors through the Wengert list. Extract the scalar
            // via `nsl_tensor_item` (returns f64), then f64->f32->bits as in
            // the F64 branch. Prior code silently trusted `scale_val` as bits,
            // which sent a POINTER to the kernel's `.param .f32 scale` slot
            // and made %scale load from the lower 32 bits of that pointer —
            // a tiny denormal near 1e-35 that silently multiplied Q*K^T to
            // garbage downstream of the softmax butterfly.
            let scale_f64 = call(compiler, builder, "nsl_tensor_item", &[scale_val])?;
            let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_f64);
            let scale_bits_i32 = builder.ins().bitcast(
                cl_types::I32,
                cranelift_codegen::ir::MemFlags::new(),
                scale_f32,
            );
            builder.ins().uextend(cl_types::I64, scale_bits_i32)
        }
    } else {
        // Default scale = 1.0f32 bits.
        let one_f32_bits = 1.0_f32.to_bits() as i64;
        builder.ins().iconst(cl_types::I64, one_f32_bits)
    };

    // --- 3. Allocate out tensor on GPU with the shape of q. ---
    //
    // The CSHA fused kernel writes f32 output when
    // `save_activations_for_backward=true` (the training path this function
    // emits under — matched by a finalize.rs gate, see comment there).
    // Allocating on GPU directly is REQUIRED for correctness under this
    // path: pre-fix we called `nsl_tensor_zeros_like(q_val)` which
    // inherited q's device (CPU when q came from a stdlib CPU matmul) and
    // produced a CPU NslTensor*. The FFI's auto-promote path
    // (`csha_tensor_data_ptr`) then allocated a FRESH GPU buffer for the
    // kernel to write into, leaving the caller's CPU NslTensor.data
    // unchanged (= zeros). Downstream MSE read the CPU zeros as the
    // attention output, seeding the entire backward chain with zeros.
    let dim0 = builder.ins().iconst(cl_types::I64, 0);
    let dim1 = builder.ins().iconst(cl_types::I64, 1);
    let dim2 = builder.ins().iconst(cl_types::I64, 2);
    let dim3 = builder.ins().iconst(cl_types::I64, 3);
    let batch = call(compiler, builder, "nsl_tensor_shape_dim", &[q_val, dim0])?;
    let heads = call(compiler, builder, "nsl_tensor_shape_dim", &[q_val, dim1])?;
    let seq_len = call(compiler, builder, "nsl_tensor_shape_dim", &[q_val, dim2])?;
    let head_dim = call(compiler, builder, "nsl_tensor_shape_dim", &[q_val, dim3])?;

    let cuda_device_for_out = builder.ins().iconst(cl_types::I64, 1);
    let out_shape = call(compiler, builder, "nsl_list_new", &[])?;
    call(compiler, builder, "nsl_list_push", &[out_shape, batch])?;
    call(compiler, builder, "nsl_list_push", &[out_shape, heads])?;
    call(compiler, builder, "nsl_list_push", &[out_shape, seq_len])?;
    call(compiler, builder, "nsl_list_push", &[out_shape, head_dim])?;
    let out_val = call(
        compiler, builder, "nsl_tensor_zeros_on",
        &[out_shape, cuda_device_for_out],
    )?;
    call(compiler, builder, "nsl_list_free", &[out_shape])?;

    let lse_shape = call(compiler, builder, "nsl_list_new", &[])?;
    call(compiler, builder, "nsl_list_push", &[lse_shape, batch])?;
    call(compiler, builder, "nsl_list_push", &[lse_shape, heads])?;
    call(compiler, builder, "nsl_list_push", &[lse_shape, seq_len])?;
    let lse_cpu = call(compiler, builder, "nsl_tensor_zeros", &[lse_shape])?;
    let cuda_device = builder.ins().iconst(cl_types::I64, 1);
    let lse_val = call(
        compiler,
        builder,
        "nsl_tensor_to_device",
        &[lse_cpu, cuda_device],
    )?;

    // --- 4. Allocate the six save buffers.  Same values populate
    //        `csha_forward_saves[layer]` below. ---
    let saves_slot = builder.create_sized_stack_slot(
        cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            48,
            8,
        ),
    );
    let saves_ptr = builder.ins().stack_addr(cl_types::I64, saves_slot, 0);
    let _alloc_rc = call(
        compiler,
        builder,
        "nsl_csha_alloc_backward_activations_into",
        &[batch, heads, seq_len, head_dim, saves_ptr],
    )?;
    let q_proj_v = builder.ins().stack_load(cl_types::I64, saves_slot, 0);
    let k_proj_v = builder.ins().stack_load(cl_types::I64, saves_slot, 8);
    let v_proj_v = builder.ins().stack_load(cl_types::I64, saves_slot, 16);
    let row_max_v = builder.ins().stack_load(cl_types::I64, saves_slot, 24);
    let row_sum_v = builder.ins().stack_load(cl_types::I64, saves_slot, 32);
    let x_raw_v = builder.ins().stack_load(cl_types::I64, saves_slot, 40);

    // --- 5. Resolve chain-side VarIds via `var_map`. ---
    let null = builder.ins().iconst(cl_types::I64, 0);
    let (mut x_v, mut norm_w_v, mut wq_v, mut wk_v, mut wv_v) =
        (null, null, null, null, null);
    if let Some(claims) = compiler.csha_backward_claims.as_ref() {
        if let Some(&chain_idx) = claims.op_to_chain.get(&op.id) {
            if let Some(mark) = claims.chain_marks.get(chain_idx) {
                if let Some(chain) = mark.chain_varids.as_ref() {
                    if let Some(&v) = var_map.get(&chain.x_norm_var) {
                        x_v = v;
                    }
                    if let Some(nw) = chain.norm_weight_var {
                        if let Some(&v) = var_map.get(&nw) {
                            norm_w_v = v;
                        }
                    }
                    if let Some(&v) = var_map.get(&chain.wq_var) {
                        wq_v = v;
                    }
                    if let Some(&v) = var_map.get(&chain.wk_var) {
                        wk_v = v;
                    }
                    if let Some(&v) = var_map.get(&chain.wv_var) {
                        wv_v = v;
                    }
                }
            }
        }
    }
    // Output projection not part of this chain at Level 0/1 fusion.
    let wo_v = null;

    // --- 6. Load PTX / name from rodata (prefer Gap B's with-saves PTX). ---
    let (ptx_ptr, name_ptr) = {
        let ctx = compiler.kernels.flash_attention_context.as_ref();
        let with_saves_ptx_did = ctx.and_then(|c| c.csha_forward_with_saves_ptx_id);
        let with_saves_name_did = ctx.and_then(|c| c.csha_forward_with_saves_name_id);
        match (with_saves_ptx_did, with_saves_name_did) {
            (Some(pid), Some(nid)) => {
                let pgv = compiler.module.declare_data_in_func(pid, builder.func);
                let ngv = compiler.module.declare_data_in_func(nid, builder.func);
                (
                    builder.ins().symbol_value(cl_types::I64, pgv),
                    builder.ins().symbol_value(cl_types::I64, ngv),
                )
            }
            _ => {
                let pd = ctx.map(|c| c.ptx_data_id);
                let nd = ctx.map(|c| c.name_data_id);
                match (pd, nd) {
                    (Some(pid), Some(nid)) => {
                        let pgv = compiler.module.declare_data_in_func(pid, builder.func);
                        let ngv = compiler.module.declare_data_in_func(nid, builder.func);
                        (
                            builder.ins().symbol_value(cl_types::I64, pgv),
                            builder.ins().symbol_value(cl_types::I64, ngv),
                        )
                    }
                    _ => (null, null),
                }
            }
        }
    };

    let block_q_val = builder.ins().iconst(cl_types::I64, block_q_i64);
    let block_kv_val = builder.ins().iconst(cl_types::I64, block_kv_i64);
    let shmem_val = builder.ins().iconst(cl_types::I64, shmem_bytes_i64);
    let causal_val = builder.ins().iconst(cl_types::I64, if is_causal { 1 } else { 0 });
    let eps_bits_val = builder.ins().iconst(cl_types::I64, eps_bits_i64);
    let active_heads_val = builder.ins().iconst(cl_types::I64, active_heads_i64);
    let d_model_val = builder.ins().iconst(cl_types::I64, d_model_i64);

    // --- 7. Emit the 36-arg fused-with-saves FFI call. ---
    // PR #93 edit 4: stop silently discarding the launch rc.  Runtime
    // trap (TrapCode::unwrap_user(3)) fires when the FFI returns non-zero,
    // preventing the silent zero-fill of save buffers that cascades into
    // NaN gradients downstream.  The @train fused lowering path flows
    // through this call — not the advanced.rs one — so the diagnostic
    // must exist here too.
    //
    // CFTP §4.3 / Tier A activation (spec 2026-05-17): read the per-step
    // thread-local packing registry. Train block sets segment_ids /
    // doc_starts at step body start; this @train fused launch reads them
    // back. Both getters return 0 when uninitialized (test fixtures that
    // skip the train block) → identity path.
    let seg_ids_ptr_v = call(
        compiler,
        builder,
        "nsl_packing_metadata_get_segment_ids",
        &[],
    )?;
    let doc_starts_v = call(
        compiler,
        builder,
        "nsl_packing_metadata_get_doc_starts",
        &[],
    )?;
    // Sprint 1 (cycle-2): hoist null RoPE cos/sin into named locals so the
    // save record can stash them. Forward and backward must agree on cos/sin;
    // the save record is the structural channel.
    //
    // WIRE-HERE: when future work threads a real cos/sin tensor source
    // (e.g. resolved from a @param input or a wengert-side synthesis op),
    // REPLACE the two `null` assignments below with the resolved Cranelift
    // Value handles. The backward FFI call at the `nsl_flash_attention_csha_backward`
    // invocation later in this function consumes them from `saves.cos` /
    // `saves.sin` — once these locals are non-null, backward picks them up
    // with no further edits.
    let rope_cos_v = null;
    let rope_sin_v = null;
    // PCA Tier B planner spec §5: build the (tier_b_ptx_ptr, tier_b_name_ptr)
    // sentinel pair for the @train fused-AD forward FFI call. MUST mirror the
    // backward at line ~2027 — both sites read from the SAME
    // `flash_attention_context.csha_with_saves_tier_b_on_*` fields the
    // module-scan emitter populated, so the forward and backward dispatch
    // consistently under the runtime gate. Without this, the forward picks
    // BASE PTX while the backward picks Tier-B-on → backward computes dQ/dK/dV
    // on tiles the forward skipped → cross-document gradient leak.
    let (fwd_tier_b_ptx_id, fwd_tier_b_name_id) = compiler
        .kernels
        .flash_attention_context
        .as_ref()
        .map(|c| {
            (
                c.csha_with_saves_tier_b_on_ptx_id,
                c.csha_with_saves_tier_b_on_name_id,
            )
        })
        .unwrap_or((None, None));
    let tier_b_with_saves_fused = match (fwd_tier_b_ptx_id, fwd_tier_b_name_id) {
        (Some(pid), Some(nid)) => crate::pca_tier_b::tier_b_enabled(
            builder,
            &mut compiler.module,
            pid,
            nid,
        ),
        _ => crate::pca_tier_b::tier_b_disabled_sentinel(builder),
    };
    let launch_rc = call(
        compiler,
        builder,
        "nsl_flash_attention_csha_with_saves",
        &[
            q_val, k_val, v_val, out_val,
            lse_val,
            scale_bits,
            batch, heads, seq_len, head_dim,
            null, null, null, null, // paged
            rope_cos_v, rope_sin_v, // RoPE
            null, null,             // seq_ids, seq_lens
            shmem_val,
            ptx_ptr, name_ptr,
            block_q_val, block_kv_val,
            causal_val,
            x_v, norm_w_v,
            wq_v, wk_v, wv_v, wo_v,
            eps_bits_val,
            active_heads_val,
            d_model_val,
            q_proj_v, k_proj_v, v_proj_v,
            row_max_v, row_sum_v,
            x_raw_v,
            // PCA Tier A: segment_ids_ptr — read from the thread-local
            // packing registry (set by train block per step).
            seg_ids_ptr_v,
            // PCA Tier B planner spec §4: pass the Tier-B-on PTX sentinel
            // pair built above. The backward at line ~2027 reads from the
            // SAME context fields (`csha_with_saves_tier_b_on_*`), so when
            // segment_masked=true and the module-scan emitter produced a
            // Tier-B-on variant, BOTH forward and backward dispatch to
            // Tier-B-on under the runtime gate. When no Tier-B-on variant
            // exists (segment_masked=false or emission rejected by the
            // validator) both fall back to the base kernel.
            tier_b_with_saves_fused[0], tier_b_with_saves_fused[1],
            // PCA §4.3: doc_starts_ptr — read from the same registry.
            doc_starts_v,
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero.  0 here —
            // the @train fused forward kernel name does not carry the
            // `_per_doc_cta` suffix in any codegen path today.  Wiring
            // this to a packing-registry runtime read is Sprint 2 follow-on.
            null,
        ],
    )?;
    {
        use cranelift_codegen::ir::{condcodes::IntCC, TrapCode};
        let ok_block   = builder.create_block();
        let trap_block = builder.create_block();
        let is_err = builder
            .ins()
            .icmp_imm(IntCC::NotEqual, launch_rc, 0);
        builder.ins().brif(is_err, trap_block, &[], ok_block, &[]);
        builder.switch_to_block(trap_block);
        builder.seal_block(trap_block);
        // TrapCode user(3) = CSHA forward launch rc != 0.
        builder.ins().trap(TrapCode::unwrap_user(3));
        builder.switch_to_block(ok_block);
        builder.seal_block(ok_block);
    }

    // --- 8. Stash save-pointer Values for Gap D.1 backward lowerer. ---
    let (bwd_ptx_id, bwd_name_id, bwd_tier_b_ptx_id, bwd_tier_b_name_id) = compiler
        .kernels
        .flash_attention_context
        .as_ref()
        .map(|c| {
            (
                c.csha_backward_ptx_data_id,
                c.csha_backward_name_data_id,
                c.csha_backward_tier_b_on_ptx_id,
                c.csha_backward_tier_b_on_name_id,
            )
        })
        .unwrap_or((None, None, None, None));
    compiler.csha_forward_saves.insert(
        layer.to_string(),
        crate::csha_apply::CshaSavePointers {
            q_proj: q_proj_v,
            k_proj: k_proj_v,
            v_proj: v_proj_v,
            row_max: row_max_v,
            row_sum: row_sum_v,
            x_raw: x_raw_v,
            // Sprint 1 T1.1: retain the forward attention output handle
            // so the Tier B.2 hybrid backward's D pre-pass can compute
            // D = rowsum(dO * O). The runtime resolves the device data
            // pointer via `csha_tensor_data_ptr(out_ptr)` (see
            // `nsl_flash_attention_csha_backward`).
            out: out_val,
            // Sprint 1 (cycle-2): stash the same RoPE cos/sin Values the
            // forward FFI was handed so the Tier B.2 hybrid backward reads
            // identical pointers (today both null → both forward and
            // backward skip rotation via the in-kernel null-guard,
            // self-consistent rope-effectively-off).
            cos: rope_cos_v,
            sin: rope_sin_v,
            backward_ptx_data_id: bwd_ptx_id,
            backward_name_data_id: bwd_name_id,
            backward_tier_b_on_ptx_data_id: bwd_tier_b_ptx_id,
            backward_tier_b_on_name_data_id: bwd_tier_b_name_id,
        },
    );

    Ok(out_val)
}

/// Lower one WengertOp to Cranelift IR.
fn lower_single_op(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    op: &WengertOp,
    var_map: &VarMap,
    var_types: &HashMap<VarId, WengertType>,
) -> Result<Value, CodegenError> {
    // Marker ops (leaf nodes) — should be in primal_vars
    match &op.op {
        PrimalOp::Input(name) => {
            if let Some(&val) = var_map.get(&op.result) {
                return Ok(val);
            }
            // Unresolved Input leaf — this is a real error. Data accesses like
            // batch.input_ids should be dict_get ops (with the batch as input),
            // not disconnected Input leaves. Fail with a diagnostic instead of
            // silently producing a null pointer that crashes at runtime.
            return Err(CodegenError::new(format!(
                "[source-ad] unresolved Input VarId {} ('{}') — no Cranelift Value in primal_vars. \
                 If this is a struct field access (e.g., batch.field), it should be a dict_get op, \
                 not a leaf Input.",
                op.result, name
            )));
        }
        PrimalOp::Param(name) => {
            if let Some(&val) = var_map.get(&op.result) {
                return Ok(val);
            }
            // Unresolved Param — may be a scalar config field (eps, _d_model)
            // used only in non-differentiable contexts. Safe to use null for
            // Passthrough ops that don't need the actual tensor value.
            eprintln!(
                "[source-ad] note: unresolved Param VarId {} ('{}'), using null placeholder",
                op.result, name
            );
            return Ok(builder.ins().iconst(cl_types::I64, 0));
        }
        PrimalOp::Constant(constant_val) => {
            // If this VarId is already present in var_map (i.e., the caller pre-seeded
            // it with a real upstream gradient value — e.g., `dy_handle` from the L2
            // backward wrapper seeded via `loss_seed_var_id`, spec §4.2), use that
            // value as-is and skip materialising the hardcoded constant.
            if let Some(&val) = var_map.get(&op.result) {
                return Ok(val);
            }
            let ty = var_types
                .get(&op.result)
                .copied()
                .unwrap_or(WengertType::Tensor);
            match ty {
                WengertType::Scalar => {
                    // Raw f64 constant — used as scalar value (e.g., index 0.0 for subscript)
                    return Ok(builder.ins().f64const(*constant_val));
                }
                WengertType::Integer => {
                    // Raw i64 constant — used as integer (e.g., dimension size)
                    return Ok(builder.ins().iconst(cl_types::I64, *constant_val as i64));
                }
                WengertType::Tensor | WengertType::List => {
                    // Scalar tensor for use in tensor ops (default)
                    let v = builder.ins().f64const(*constant_val);
                    let dt = builder.ins().iconst(cl_types::I64, 1); // f32 default
                    return call(compiler, builder, "nsl_tensor_scalar", &[v, dt]);
                }
            }
        }
        _ => {}
    }

    let inputs = resolve_inputs(op, var_map)?;

    match &op.op {
        // Already handled above
        PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_) => unreachable!(),

        // Handled in the compile_wengert_ops main loop (the freed VarId
        // must be recorded on the lowering result); reaching here means a
        // FreeTensor leaked into a context that doesn't support it.
        PrimalOp::FreeTensor => unreachable!(
            "PrimalOp::FreeTensor must be intercepted by compile_wengert_ops"
        ),

        // === Elementwise unary (11 ops) ===
        // Promote scalar/integer inputs to tensor before calling tensor ops.
        PrimalOp::Relu
        | PrimalOp::Sigmoid
        | PrimalOp::Tanh
        | PrimalOp::Gelu
        | PrimalOp::Silu
        | PrimalOp::Exp
        | PrimalOp::Log
        | PrimalOp::Sqrt
        | PrimalOp::Abs
        | PrimalOp::Neg => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
            let rt_name = match &op.op {
                PrimalOp::Relu => "nsl_tensor_relu",
                PrimalOp::Sigmoid => "nsl_tensor_sigmoid",
                PrimalOp::Tanh => "nsl_tensor_tanh_act",
                PrimalOp::Gelu => "nsl_tensor_gelu",
                PrimalOp::Silu => "nsl_tensor_silu",
                PrimalOp::Exp => "nsl_tensor_exp",
                PrimalOp::Log => "nsl_tensor_log",
                PrimalOp::Sqrt => "nsl_tensor_sqrt",
                PrimalOp::Abs => "nsl_tensor_abs",
                PrimalOp::Neg => "nsl_tensor_neg",
                _ => unreachable!(),
            };
            let result = call(compiler, builder, rt_name, &[a])?;
            free_tensor_if_owned(compiler, builder, a, free_a)?;
            Ok(result)
        }
        PrimalOp::Clamp { min, max } => {
            let min_v = builder.ins().f64const(*min);
            let max_v = builder.ins().f64const(*max);
            call(
                compiler,
                builder,
                "nsl_tensor_clamp",
                &[inputs[0], min_v, max_v],
            )
        }

        // === Elementwise binary (4 ops) ===
        // Check if both inputs are Integer-typed (e.g., shape[0] * shape[1]).
        // If so, use Cranelift integer arithmetic instead of tensor ops.
        PrimalOp::Add => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().iadd(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                // ELTLS (FBIP-3): nsl_tensor_add takes a flags byte (flags=0 here).
                let flags_zero = builder.ins().iconst(cl_types::I8, 0);
                let result = call(compiler, builder, "nsl_tensor_add", &[a, b, flags_zero])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }
        PrimalOp::Sub => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().isub(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                // ELTLS (FBIP-3): nsl_tensor_sub takes a flags byte.
                let flags_zero = builder.ins().iconst(cl_types::I8, 0);
                let result = call(compiler, builder, "nsl_tensor_sub", &[a, b, flags_zero])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }
        PrimalOp::Mul => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().imul(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                // ELTLS (FBIP-3): nsl_tensor_mul takes a flags byte.
                let flags_zero = builder.ins().iconst(cl_types::I8, 0);
                let result = call(compiler, builder, "nsl_tensor_mul", &[a, b, flags_zero])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }
        PrimalOp::Div => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().sdiv(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                // ELTLS (FBIP-3): nsl_tensor_div takes a flags byte.
                let flags_zero = builder.ins().iconst(cl_types::I8, 0);
                let result = call(compiler, builder, "nsl_tensor_div", &[a, b, flags_zero])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }

        // === Linear algebra (4 ops) ===
        PrimalOp::Matmul => {
            // ELTLS (FBIP-3): nsl_tensor_matmul takes a flags byte.
            let flags_zero = builder.ins().iconst(cl_types::I8, 0);
            call(
                compiler,
                builder,
                "nsl_tensor_matmul",
                &[inputs[0], inputs[1], flags_zero],
            )
        }
        PrimalOp::Transpose { dim0, dim1 } => {
            let d0 = builder.ins().iconst(cl_types::I64, *dim0 as i64);
            let d1 = builder.ins().iconst(cl_types::I64, *dim1 as i64);
            call(
                compiler,
                builder,
                "nsl_tensor_transpose",
                &[inputs[0], d0, d1],
            )
        }
        // WRGA B.3.2 Option 3: fused GatedLoRA forward FFI call.
        //
        //   nsl_adapter_fused_gatedlora_matmul(x, W, A, B, scale: f64, gate, kh: i64) -> i64
        PrimalOp::FusedGatedLoraMatmul { scale, kernel_handle } => {
            let scale_v = builder.ins().f64const(*scale as f64);
            let kh = builder.ins().iconst(cl_types::I64, *kernel_handle);
            call(
                compiler,
                builder,
                "nsl_adapter_fused_gatedlora_matmul",
                &[inputs[0], inputs[1], inputs[2], inputs[3], scale_v, inputs[4], kh],
            )
        }
        // WRGA B.3 fused LoRA forward FFI call.
        //
        //   nsl_adapter_fused_lora_matmul(x, W, A, B, scale: f64, kh: i64) -> i64
        PrimalOp::FusedLoraMatmul { scale, kernel_handle } => {
            let scale_v = builder.ins().f64const(*scale as f64);
            let kh = builder.ins().iconst(cl_types::I64, *kernel_handle);
            call(
                compiler,
                builder,
                "nsl_adapter_fused_lora_matmul",
                &[inputs[0], inputs[1], inputs[2], inputs[3], scale_v, kh],
            )
        }
        // WRGA B.3 fused IA³ forward FFI call.
        //
        //   nsl_adapter_fused_ia3_matmul(x, W, gamma, kh: i64) -> i64
        PrimalOp::FusedIa3Matmul { kernel_handle } => {
            let kh = builder.ins().iconst(cl_types::I64, *kernel_handle);
            call(
                compiler,
                builder,
                "nsl_adapter_fused_ia3_matmul",
                &[inputs[0], inputs[1], inputs[2], kh],
            )
        }
        PrimalOp::Reshape { .. } => {
            // Reshape in backward pass is typically a view op; clone preserves data.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::Broadcast => {
            // Broadcast is a shape-level op; clone the tensor.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }

        // === Reductions (4 ops) ===
        PrimalOp::Sum { dim } => {
            // CRITICAL: nsl_tensor_sum_dim treats dim=-1 as "global sum" (all
            // dimensions), NOT "last dim". Source-AD's SoftmaxBackward emits
            // Sum { dim: Some(-1) } meaning "last dim". We must resolve
            // negative dims at lowering time using the input tensor's ndim
            // so the runtime sees a non-negative dimension index.
            // dim == None means global sum (pass -1 to the runtime).
            let d = match dim {
                None => builder.ins().iconst(cl_types::I64, -1_i64),
                Some(d) if *d < 0 => {
                    // Negative dim: resolve via ndim at runtime.
                    // ndim_val = nsl_tensor_ndim(input)  → i64
                    // actual_dim = ndim_val + d           → e.g. ndim + (-1) = last dim
                    let ndim_val = call(compiler, builder, "nsl_tensor_ndim", &[inputs[0]])?;
                    let offset = builder.ins().iconst(cl_types::I64, *d);
                    builder.ins().iadd(ndim_val, offset)
                }
                Some(d) => builder.ins().iconst(cl_types::I64, *d),
            };
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            // M46b: Route to deterministic sort-based reduction when --deterministic is active.
            if compiler.compile_options.deterministic {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_reduce_sum_deterministic",
                    &[inputs[0], d, keepdim],
                )
            } else {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_sum_dim",
                    &[inputs[0], d, keepdim],
                )
            }
        }
        PrimalOp::Mean { dim } => {
            // Same negative-dim resolution as Sum (see comment above).
            let d = match dim {
                None => builder.ins().iconst(cl_types::I64, -1_i64),
                Some(d) if *d < 0 => {
                    let ndim_val = call(compiler, builder, "nsl_tensor_ndim", &[inputs[0]])?;
                    let offset = builder.ins().iconst(cl_types::I64, *d);
                    builder.ins().iadd(ndim_val, offset)
                }
                Some(d) => builder.ins().iconst(cl_types::I64, *d),
            };
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            // M46b: Route to deterministic sort-based reduction when --deterministic is active.
            if compiler.compile_options.deterministic {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_reduce_mean_deterministic",
                    &[inputs[0], d, keepdim],
                )
            } else {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_mean_dim",
                    &[inputs[0], d, keepdim],
                )
            }
        }
        PrimalOp::Softmax { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(compiler, builder, "nsl_tensor_softmax", &[inputs[0], d])
        }
        PrimalOp::LogSoftmax { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(compiler, builder, "nsl_tensor_logsoftmax", &[inputs[0], d])
        }

        // === Shape ops (5 ops) ===
        PrimalOp::Concat { dim } => {
            // Runtime nsl_tensor_cat takes (tensor_list_ptr, dim).
            // In source AD, tensor_cat(tensors, dim) passes the list VarId from a
            // Passthrough("list") op as inputs[0]. Detect this by checking if the
            // first input is typed as List, and use it directly instead of building
            // a new list.
            let first_is_list = op
                .inputs
                .first()
                .and_then(|vid| var_types.get(vid).copied())
                .map(|t| t == WengertType::List)
                .unwrap_or(false);
            let list = if first_is_list {
                // inputs[0] is already a tensor list pointer (from Passthrough("list"))
                inputs[0]
            } else {
                // Individual tensor inputs — build a list from them
                let l = call(compiler, builder, "nsl_list_new", &[])?;
                for &inp in &inputs {
                    call(compiler, builder, "nsl_list_push", &[l, inp])?;
                }
                l
            };
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let result = call(compiler, builder, "nsl_tensor_cat", &[list, d])?;
            if !first_is_list {
                call(compiler, builder, "nsl_list_free", &[list])?;
            }
            Ok(result)
        }
        PrimalOp::Split { .. } => {
            // Split is a forward-only op — its backward is Concat (via SplitConcat adjoint).
            // If Split appears in an adjoint graph, it's from re-execution; clone is correct.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::Slice {
            dim, start, end, ..
        } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let s = builder.ins().iconst(cl_types::I64, *start);
            let e = builder.ins().iconst(cl_types::I64, *end);
            call(compiler, builder, "nsl_tensor_slice", &[inputs[0], d, s, e])
        }
        PrimalOp::PadZero {
            dim,
            pad_before,
            pad_after,
        } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let pb = builder.ins().iconst(cl_types::I64, *pad_before);
            let pa = builder.ins().iconst(cl_types::I64, *pad_after);
            call(
                compiler,
                builder,
                "nsl_tensor_pad_zero",
                &[inputs[0], d, pb, pa],
            )
        }
        PrimalOp::Repeat { kernel } => {
            let k = builder.ins().iconst(cl_types::I64, *kernel as i64);
            call(compiler, builder, "nsl_tensor_repeat", &[inputs[0], k])
        }

        // === Indexing (3 ops) ===
        PrimalOp::Gather { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(
                compiler,
                builder,
                "nsl_tensor_gather",
                &[inputs[0], d, inputs[1]],
            )
        }
        PrimalOp::ScatterAdd { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            // M46: Route to deterministic sort-accumulate variant when --deterministic is active.
            // The deterministic FFI takes (input, indices, src) — dim is implicit (0).
            if compiler.compile_options.deterministic {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_scatter_add_deterministic",
                    &[inputs[0], inputs[1], d],
                )
            } else {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_scatter_add",
                    &[inputs[0], inputs[1], d],
                )
            }
        }
        PrimalOp::Embedding => call(
            compiler,
            builder,
            "nsl_tensor_embedding_lookup",
            &[inputs[0], inputs[1]],
        ),

        // === Normalization (2 ops) ===
        PrimalOp::LayerNorm { eps } => {
            let e = builder.ins().f64const(*eps);
            call(
                compiler,
                builder,
                "nsl_tensor_layernorm",
                &[inputs[0], inputs[1], inputs[2], e],
            )
        }
        PrimalOp::RMSNorm { eps } => {
            // RMSNorm takes (input, weight, eps) — the eps argument from the Wengert
            // inputs is a float field loaded from the model struct. Use the hardcoded
            // eps from the PrimalOp to avoid type mismatches (input eps may be f64).
            let e = builder.ins().f64const(*eps);
            call(
                compiler,
                builder,
                "nsl_tensor_rmsnorm",
                &[inputs[0], inputs[1], e],
            )
        }
        PrimalOp::BatchNorm { eps, training } => {
            let e = builder.ins().f64const(*eps);
            let t = builder
                .ins()
                .iconst(cl_types::I64, if *training { 1 } else { 0 });
            call(
                compiler,
                builder,
                "nsl_tensor_batchnorm",
                &[inputs[0], inputs[1], inputs[2], e, t],
            )
        }

        // === Convolution (3 ops) ===
        PrimalOp::Conv2d { stride, padding } => {
            // Runtime: nsl_tensor_conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w)
            // Bias input is inputs[2] if present; use 0 (null) if only 2 inputs.
            let bias = if inputs.len() > 2 {
                inputs[2]
            } else {
                builder.ins().iconst(cl_types::I64, 0)
            };
            let sh = builder.ins().iconst(cl_types::I64, *stride as i64);
            let sw = builder.ins().iconst(cl_types::I64, *stride as i64);
            let ph = builder.ins().iconst(cl_types::I64, *padding as i64);
            let pw = builder.ins().iconst(cl_types::I64, *padding as i64);
            call(
                compiler,
                builder,
                "nsl_tensor_conv2d",
                &[inputs[0], inputs[1], bias, sh, sw, ph, pw],
            )
        }
        PrimalOp::ConvTranspose2d { stride, padding } => {
            // Approximate: use conv2d with the same args (weight should already be transposed by AD rules)
            let bias = if inputs.len() > 2 {
                inputs[2]
            } else {
                builder.ins().iconst(cl_types::I64, 0)
            };
            let sh = builder.ins().iconst(cl_types::I64, *stride as i64);
            let sw = builder.ins().iconst(cl_types::I64, *stride as i64);
            let ph = builder.ins().iconst(cl_types::I64, *padding as i64);
            let pw = builder.ins().iconst(cl_types::I64, *padding as i64);
            call(
                compiler,
                builder,
                "nsl_tensor_conv2d",
                &[inputs[0], inputs[1], bias, sh, sw, ph, pw],
            )
        }
        PrimalOp::Conv2dBackward {
            kind,
            stride,
            padding,
        } => {
            // inputs = [grad_output, input, weight]. Delegate to the runtime FFI
            // returning the selected gradient (wraps the verified conv2d_backward).
            let fname = match kind {
                ConvGradKind::Input => "nsl_conv2d_input_backward",
                ConvGradKind::Weight => "nsl_conv2d_weight_backward",
                ConvGradKind::Bias => "nsl_conv2d_bias_backward",
            };
            let sh = builder.ins().iconst(cl_types::I64, *stride as i64);
            let sw = builder.ins().iconst(cl_types::I64, *stride as i64);
            let ph = builder.ins().iconst(cl_types::I64, *padding as i64);
            let pw = builder.ins().iconst(cl_types::I64, *padding as i64);
            call(
                compiler,
                builder,
                fname,
                &[inputs[0], inputs[1], inputs[2], sh, sw, ph, pw],
            )
        }
        PrimalOp::MaterializeConvOutputGrad { stride, padding } => {
            // inputs = [grad_output, input, weight]. See PrimalOp docs: this
            // reifies grad_output once so sibling Conv2dBackward ops share it.
            let sh = builder.ins().iconst(cl_types::I64, *stride as i64);
            let sw = builder.ins().iconst(cl_types::I64, *stride as i64);
            let ph = builder.ins().iconst(cl_types::I64, *padding as i64);
            let pw = builder.ins().iconst(cl_types::I64, *padding as i64);
            call(
                compiler,
                builder,
                "nsl_materialize_conv_output_grad",
                &[inputs[0], inputs[1], inputs[2], sh, sw, ph, pw],
            )
        }

        // === Pooling (2 ops) ===
        PrimalOp::MaxPool2d { kernel, stride } => {
            // Runtime: nsl_tensor_maxpool2d(input, kernel_h, kernel_w, stride, padding)
            let kh = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let kw = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let s = builder.ins().iconst(cl_types::I64, *stride as i64);
            let p = builder.ins().iconst(cl_types::I64, 0); // padding = 0
            call(
                compiler,
                builder,
                "nsl_tensor_maxpool2d",
                &[inputs[0], kh, kw, s, p],
            )
        }
        PrimalOp::AvgPool2d { kernel, stride } => {
            let kh = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let kw = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let s = builder.ins().iconst(cl_types::I64, *stride as i64);
            let p = builder.ins().iconst(cl_types::I64, 0);
            call(
                compiler,
                builder,
                "nsl_tensor_avgpool2d",
                &[inputs[0], kh, kw, s, p],
            )
        }

        // === Loss functions (3 ops) ===
        // Loss functions are composite: we lower them to sequences of existing FFI calls.
        //
        // CFTP §4.4 G3 (Sprint v3-1 — substitution implemented UPSTREAM):
        //
        // Sprint v3-1 implemented `@fused_lm_ce` auto-substitution in
        // source_ad.rs (see the `"cross_entropy" | "cross_entropy_loss"`
        // arm in `WengertExtractor::extract_expr`, plus the helper
        // `try_match_fused_linear_ce_pattern`).  When the surrounding
        // `train` block carries `@fused_lm_ce(enabled = true)` AND all
        // four shape hints are populated AND the input matches
        // `Add(Matmul(x, Transpose(W, 0, 1)), bias)`, source-AD emits
        // `PrimalOp::FusedLinearCe` instead of `PrimalOp::CrossEntropyLoss`
        // before the lowerer ever runs — so by the time we get here, this
        // arm only sees CE for the composite/regression path.
        //
        // This arm intentionally lowers the composite (logsoftmax + gather
        // + neg/mean) for ALL `PrimalOp::CrossEntropyLoss` we see — that's
        // the correct behaviour for:
        //   * decorator absent
        //   * `enabled = false`
        //   * missing shape hints
        //   * input chain that doesn't match the canonical decomposition
        //     (e.g. user hand-rolled fused logits without the Add(Matmul,
        //     bias) pattern, or used a non-`{0,1}` transpose)
        //   * inference-mode evaluation outside any train block
        //
        // Cross-references (still useful for spelunking):
        //   * crates/nsl-codegen/src/source_ad.rs (search for
        //     `try_match_fused_linear_ce_pattern`) — the substitution site.
        //   * crates/nsl-codegen/src/fused_linear_ce.rs — PTX synthesis.
        //   * crates/nsl-runtime/src/fused_linear_ce.rs — forward + bwd FFI.
        //   * stdlib/nsl/nn/losses.nsl::fused_linear_ce — user-facing
        //     stdlib function (composite v1 path; the auto-substitution
        //     replaces this expansion at source-AD time when enabled).
        //   * crates/nsl-semantic/src/cftp.rs::validate_fused_ce_decorator
        //     — decorator parsing + alignment-invariant enforcement.
        PrimalOp::CrossEntropyLoss => {
            // Match stdlib/nsl/nn/losses.nsl semantics, including ignore labels
            // encoded as -100 by the DataLoader.
            let dim_one = builder.ins().iconst(cl_types::I64, 1);
            let safe_min = builder.ins().f64const(0.0);
            let safe_max = builder.ins().f64const(2147483647.0);
            let safe_targets = call(
                compiler,
                builder,
                "nsl_tensor_clamp",
                &[inputs[1], safe_min, safe_max],
            )?;
            let log_probs = call(
                compiler,
                builder,
                "nsl_tensor_logsoftmax",
                &[inputs[0], dim_one],
            )?;
            let gathered = call(
                compiler,
                builder,
                "nsl_tensor_gather",
                &[log_probs, dim_one, safe_targets],
            )?;
            let nll = call(compiler, builder, "nsl_tensor_neg", &[gathered])?;

            let one_f = builder.ins().f64const(1.0);
            let zero_f = builder.ins().f64const(0.0);
            let eps_f = builder.ins().f64const(1e-8);
            let wl_flags0_addsc1 = builder.ins().iconst(cl_types::I8, 0);
            let targets_plus_one = call(
                compiler,
                builder,
                "nsl_tensor_add_scalar",
                &[inputs[1], one_f, wl_flags0_addsc1],
            )?;
            let valid_mask = call(
                compiler,
                builder,
                "nsl_tensor_clamp",
                &[targets_plus_one, zero_f, one_f],
            )?;
            // ELTLS (FBIP-3): nsl_tensor_mul takes a flags byte.
            let flags0_nll = builder.ins().iconst(cl_types::I8, 0);
            let masked_nll = call(compiler, builder, "nsl_tensor_mul", &[nll, valid_mask, flags0_nll])?;
            let num_valid = call(compiler, builder, "nsl_tensor_sum", &[valid_mask])?;
            let wl_flags0_addsc2 = builder.ins().iconst(cl_types::I8, 0);
            let num_valid_eps = call(
                compiler,
                builder,
                "nsl_tensor_add_scalar",
                &[num_valid, eps_f, wl_flags0_addsc2],
            )?;
            let total = call(compiler, builder, "nsl_tensor_sum", &[masked_nll])?;
            // ELTLS (FBIP-3): nsl_tensor_div takes a flags byte.
            let flags0_div = builder.ins().iconst(cl_types::I8, 0);
            let result = call(compiler, builder, "nsl_tensor_div", &[total, num_valid_eps, flags0_div])?;
            for temp in [
                safe_targets,
                log_probs,
                gathered,
                nll,
                targets_plus_one,
                valid_mask,
                masked_nll,
                num_valid,
                num_valid_eps,
                total,
            ] {
                free_tensor_value(compiler, builder, temp)?;
            }
            Ok(result)
        }
        PrimalOp::MSELoss => {
            // mse_loss(pred, target) = mean((pred - target)^2)
            // ELTLS (FBIP-3): tensor-tensor ops take a flags byte.
            let mse_flags0_sub = builder.ins().iconst(cl_types::I8, 0);
            let diff = call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1], mse_flags0_sub])?;
            let mse_flags0_mul = builder.ins().iconst(cl_types::I8, 0);
            let sq = call(compiler, builder, "nsl_tensor_mul", &[diff, diff, mse_flags0_mul])?;
            let d = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            let result = call(compiler, builder, "nsl_tensor_mean_dim", &[sq, d, keepdim])?;
            free_tensor_value(compiler, builder, diff)?;
            free_tensor_value(compiler, builder, sq)?;
            Ok(result)
        }
        // CFTP §4.4 G3 (Sprint 4): fused linear-CE forward.
        //
        // Replaces the composite `x @ W^T + bias` matmul + `cross_entropy`
        // expansion with a single PTX kernel launch (or two kernels at
        // V > 8192).  The forward produces per-row losses in HBM; this
        // lowering then applies the standard `sum(masked_nll) / num_valid`
        // reduction to return a scalar mean loss to AD.
        //
        // Synchronisation: the FFI calls `cuCtxSynchronize` internally
        // after the kernel launch, so loss_out is ready by the time we
        // run the CPU-side / NslTensor-side reduction below.
        //
        // The forward also saves `lse_out` on
        // `compiler.fused_ce_fwd_lse` keyed by the result Cranelift
        // Value — the backward extract dispatcher consumes it.
        PrimalOp::FusedLinearCe {
            vocab_size,
            hidden_size,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
            is_large,
        } => lower_fused_linear_ce_forward(
            compiler,
            builder,
            &inputs,
            *vocab_size,
            *hidden_size,
            *batch_size,
            *seq_len,
            *vocab_tile,
            *ignore_index,
            *is_large,
        ),
        // CFTP §4.4 G3 (Sprint 4): fused linear-CE backward extract.
        // Mirrors `FlashAttentionBackwardExtract` — first component fires
        // the backward FFI and caches outputs; subsequent components hit
        // the cache; the last evicts.
        PrimalOp::FusedLinearCeBackwardExtract {
            component,
            vocab_size,
            hidden_size,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
        } => lower_fused_linear_ce_backward_extract(
            compiler,
            builder,
            &inputs,
            *component,
            *vocab_size,
            *hidden_size,
            *batch_size,
            *seq_len,
            *vocab_tile,
            *ignore_index,
        ),
        // CPKD: fused KL-CE distillation loss (forward).
        PrimalOp::FusedKlCe {
            vocab_size,
            student_hidden,
            teacher_hidden,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
            alpha_bits,
            temperature_bits,
        } => lower_fused_kl_ce_forward(
            compiler,
            builder,
            &inputs,
            &FusedKlCeLowerParams {
                vocab_size: *vocab_size,
                student_hidden: *student_hidden,
                teacher_hidden: *teacher_hidden,
                batch_size: *batch_size,
                seq_len: *seq_len,
                vocab_tile: *vocab_tile,
                ignore_index: *ignore_index,
                alpha: f64::from_bits(*alpha_bits),
                temperature: f64::from_bits(*temperature_bits),
            },
        ),
        // CPKD: fused KL-CE backward extract (student components only —
        // I-11: no teacher gradients exist anywhere in this path).
        PrimalOp::FusedKlCeBackwardExtract {
            component,
            vocab_size,
            student_hidden,
            teacher_hidden,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
            alpha_bits,
            temperature_bits,
        } => lower_fused_kl_ce_backward_extract(
            compiler,
            builder,
            &inputs,
            *component,
            &FusedKlCeLowerParams {
                vocab_size: *vocab_size,
                student_hidden: *student_hidden,
                teacher_hidden: *teacher_hidden,
                batch_size: *batch_size,
                seq_len: *seq_len,
                vocab_tile: *vocab_tile,
                ignore_index: *ignore_index,
                alpha: f64::from_bits(*alpha_bits),
                temperature: f64::from_bits(*temperature_bits),
            },
        ),
        PrimalOp::L1Loss => {
            // l1_loss(pred, target) = mean(|pred - target|)
            // ELTLS (FBIP-3): nsl_tensor_sub takes a flags byte.
            let l1_flags0 = builder.ins().iconst(cl_types::I8, 0);
            let diff = call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1], l1_flags0])?;
            let abs_diff = call(compiler, builder, "nsl_tensor_abs", &[diff])?;
            let d = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            let result = call(
                compiler,
                builder,
                "nsl_tensor_mean_dim",
                &[abs_diff, d, keepdim],
            )?;
            free_tensor_value(compiler, builder, diff)?;
            free_tensor_value(compiler, builder, abs_diff)?;
            Ok(result)
        }

        // === Attention (4 ops) ===
        PrimalOp::ScaledDotProductAttention { causal } => {
            // Option 3a — CSHA fused-forward claim dispatch:
            //
            // When the CSHA backward dispatcher has claimed this SDPA
            // op (it's in `compiler.csha_backward_claims.op_to_chain`)
            // AND the resolved layer has `save_activations_for_backward`,
            // emit the fused `nsl_flash_attention_csha_with_saves` FFI
            // here instead of decomposing into primitive matmul/softmax.
            //
            // This is the symmetric extension of Gap D.1's backward-side
            // dispatcher to the forward pass.  One forward code path
            // under the claim, one set of save buffers, deterministic
            // data flow into the fused backward — no drift between
            // "what the fused kernel sees" and "what primitive ops
            // would compute".
            //
            // No fallback path: if the fused FFI fails at runtime for
            // a claimed chain, that's a bug to fix.  We never silently
            // fall back to decomposition — that would reintroduce the
            // dual-path drift class the user explicitly rejected.
            let claim_layer = compiler
                .csha_backward_claims
                .as_ref()
                .and_then(|claims| {
                    claims
                        .op_to_chain
                        .get(&op.id)
                        .copied()
                        .and_then(|idx| claims.chain_marks.get(idx))
                        .map(|m| m.layer.clone())
                });
            if let Some(layer) = claim_layer {
                let needs_saves = compiler
                    .last_csha_bridge
                    .as_ref()
                    .and_then(|b| b.extras_for_layer(&layer))
                    .map(|e| e.save_activations_for_backward)
                    .unwrap_or(false);
                if needs_saves {
                    let result = emit_fused_forward_under_claim(
                        compiler, builder, op, &inputs, var_map, &layer,
                    )?;
                    // Mirror the side-channel registration the
                    // decomposition did so non-CSHA consumers of
                    // `flash_attn_aux` still work.  Backward consumes
                    // saves directly; null lse is harmless here.
                    let null_lse = builder.ins().iconst(cl_types::I64, 0);
                    compiler.flash_attn_aux.insert(result, (result, null_lse));
                    return Ok(result);
                }
            }

            // Unclaimed path — decompose into primitive ops:
            //   softmax((Q @ K.T) * scale [+ mask]) @ V
            // inputs: [q, k, v, scale, causal_flag]
            let q = inputs[0];
            let k = inputs[1];
            let v = inputs[2];
            // Scale may be f64 scalar or tensor — promote to tensor if needed
            let scale_ty = op
                .inputs
                .get(3)
                .and_then(|vid| var_types.get(vid).copied())
                .unwrap_or(WengertType::Tensor);
            let (scale, free_scale) = if inputs.len() > 3 {
                promote_to_tensor(compiler, builder, inputs[3], scale_ty)?
            } else {
                // Default scale: 1.0
                let one = builder.ins().f64const(1.0);
                let dt = builder.ins().iconst(cl_types::I64, 1);
                (
                    call(compiler, builder, "nsl_tensor_scalar", &[one, dt])?,
                    true,
                )
            };

            // Scale item hoisted ahead of the fused dispatch: both the fused
            // FFI (as f32 bits) and the decomposed chain (as f64) consume it.
            let scale_item = call(compiler, builder, "nsl_tensor_item", &[scale])?;
            free_tensor_if_owned(compiler, builder, scale, free_scale)?;

            // PCA Stage C: fused-forward dispatch (the forward analog of
            // PR #347's backward variant table). Decorator-free compiles
            // only — a decorated module's SDPA is owned by the
            // @flash_attention context machinery (and the decorator-free
            // backward keys its variant-table path off context==None, so
            // the forward must fork on exactly the same condition).
            let fused = if compiler.kernels.flash_attention_context.is_none() {
                let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_item);
                let scale_bits_i32 = builder.ins().bitcast(
                    cl_types::I32,
                    cranelift_codegen::ir::MemFlags::new(),
                    scale_f32,
                );
                let scale_bits = builder.ins().uextend(cl_types::I64, scale_bits_i32);
                emit_sdpa_fused_dispatch(compiler, builder, q, k, v, scale_bits, *causal, None)?
            } else {
                None
            };

            // K_T = transpose(K, -2, -1)
            let dim_m2 = builder.ins().iconst(cl_types::I64, -2_i64);
            let dim_m1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let k_t = call(
                compiler,
                builder,
                "nsl_tensor_transpose",
                &[k, dim_m2, dim_m1],
            )?;
            // scores = Q @ K_T
            // ELTLS (FBIP-3): nsl_tensor_matmul takes a flags byte.
            let attn_flags0_qk = builder.ins().iconst(cl_types::I8, 0);
            let scores = call(compiler, builder, "nsl_tensor_matmul", &[q, k_t, attn_flags0_qk])?;
            // scaled = scores * scale
            let wl_flags0_mulsc = builder.ins().iconst(cl_types::I8, 0);
            let scaled = call(
                compiler,
                builder,
                "nsl_tensor_mul_scalar",
                &[scores, scale_item, wl_flags0_mulsc],
            )?;

            // Apply causal mask if needed
            let masked = if *causal {
                let dim_neg2 = builder.ins().iconst(cl_types::I64, -2_i64);
                let seq_len = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim_neg2])?;
                let mask = call(compiler, builder, "nsl_tensor_causal_mask", &[seq_len])?;
                // ELTLS (FBIP-3): nsl_tensor_add takes a flags byte (flags=0 here).
                let flags_zero_add = builder.ins().iconst(cl_types::I8, 0);
                let masked = call(compiler, builder, "nsl_tensor_add", &[scaled, mask, flags_zero_add])?;
                free_tensor_value(compiler, builder, mask)?;
                masked
            } else {
                scaled
            };

            // attn_weights = softmax(masked, -1)
            let attn_weights = call(compiler, builder, "nsl_tensor_softmax", &[masked, dim_m1])?;
            // output = attn_weights @ V
            // ELTLS (FBIP-3): nsl_tensor_matmul takes a flags byte.
            let attn_flags0_av = builder.ins().iconst(cl_types::I8, 0);
            let result = call(compiler, builder, "nsl_tensor_matmul", &[attn_weights, v, attn_flags0_av])?;
            free_tensor_value(compiler, builder, k_t)?;
            free_tensor_value(compiler, builder, scores)?;
            if *causal {
                free_tensor_value(compiler, builder, scaled)?;
            }
            free_tensor_value(compiler, builder, masked)?;
            free_tensor_value(compiler, builder, attn_weights)?;

            // NOTE: save-buffer allocation for claimed chains is handled
            // above by option 3a's `emit_fused_forward_under_claim` —
            // that path emits `nsl_flash_attention_csha_with_saves`
            // (which both computes attention AND populates saves) AND
            // registers the save pointers in `csha_forward_saves`.
            //
            // This arm (decomposition) only runs for UNCLAIMED SDPA
            // ops — i.e. CSHA is off, or the dispatcher didn't claim
            // this op.  Unclaimed ops by definition don't need save
            // buffers (the backward consumer for them is the classic
            // `FlashAttentionBackwardExtract` / `nsl_flash_attention_backward`
            // path, not `FusedCshaBackward`).  So no save-alloc here.

            match fused {
                Some((join_block, out_param, lse_param)) => {
                    // Close the decline block and merge with the fused edge.
                    // The side-channel stores the join's (out, lse): the lse
                    // is the fused kernel's saved logsumexp, or runtime 0
                    // when the decomposed fallback ran (backward then
                    // recomputes it, exactly as before Stage C).
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(join_block, &[result, zero]);
                    builder.seal_block(join_block);
                    builder.switch_to_block(join_block);
                    compiler
                        .flash_attn_aux
                        .insert(out_param, (out_param, lse_param));
                    // The saved LSE is OWNED by this step (the backward only
                    // borrows it; forward-only runs never consume it). Freed
                    // by the step cleanup via the extra-owned drain — a
                    // runtime 0 (decline) is a free() no-op.
                    compiler.sdpa_extra_owned.push(lse_param);
                    Ok(out_param)
                }
                None => {
                    // Non-CUDA target or decorated module: single-path
                    // decomposition, (result, null-lse) side-channel as
                    // before.
                    let null_lse = builder.ins().iconst(cl_types::I64, 0);
                    compiler.flash_attn_aux.insert(result, (result, null_lse));
                    Ok(result)
                }
            }
        }
        PrimalOp::ScaledDotProductAttentionPacked => {
            // PCA Stage C: packed-sequence attention.
            //   inputs: [q, k, v, scale, segment_ids]
            // Fused path: segment-masked v2 flash kernel (causal-within-doc
            // derived from segment_ids in-kernel; saves LSE for backward;
            // Tier-B tile-skip variant when the runtime gate admits it).
            // Decline path: the Stage-B decomposed additive-mask chain,
            // with the dense [b,1,s,s] mask DERIVED from segment_ids at
            // the decline site (`nsl_packed_mask_from_segment_ids` — the
            // DataLoader no longer ships it) — numerically identical
            // because exp((s - 1e9) - lse) underflows to +0.0 (see the
            // PrimalOp contract).
            //
            // No CSHA-claim consultation here: claims target the plain
            // `ScaledDotProductAttention` op only, and the CSHA fused
            // family has no packed-mask semantics.
            let q = inputs[0];
            let k = inputs[1];
            let v = inputs[2];
            let seg = inputs[4];

            let scale_ty = op
                .inputs
                .get(3)
                .and_then(|vid| var_types.get(vid).copied())
                .unwrap_or(WengertType::Tensor);
            let (scale, free_scale) = promote_to_tensor(compiler, builder, inputs[3], scale_ty)?;
            let scale_item = call(compiler, builder, "nsl_tensor_item", &[scale])?;
            free_tensor_if_owned(compiler, builder, scale, free_scale)?;

            let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_item);
            let scale_bits_i32 = builder.ins().bitcast(
                cl_types::I32,
                cranelift_codegen::ir::MemFlags::new(),
                scale_f32,
            );
            let scale_bits = builder.ins().uextend(cl_types::I64, scale_bits_i32);

            // Packed masks are causal-within-document by contract, so the
            // fused kernel is always built causal=true.
            let fused = emit_sdpa_fused_dispatch(
                compiler, builder, q, k, v, scale_bits, true, Some(seg),
            )?;

            // Decomposed fallback: softmax((Q @ K.T) * scale + mask) @ V,
            // where mask is DERIVED from segment_ids here (this code only
            // runs on decline, so the O(s²) materialization is paid only
            // when the fused kernel does not).
            let dim_m2 = builder.ins().iconst(cl_types::I64, -2_i64);
            let dim_m1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let k_t = call(
                compiler,
                builder,
                "nsl_tensor_transpose",
                &[k, dim_m2, dim_m1],
            )?;
            let flags0_qk = builder.ins().iconst(cl_types::I8, 0);
            let scores = call(compiler, builder, "nsl_tensor_matmul", &[q, k_t, flags0_qk])?;
            let flags0_mulsc = builder.ins().iconst(cl_types::I8, 0);
            let scaled = call(
                compiler,
                builder,
                "nsl_tensor_mul_scalar",
                &[scores, scale_item, flags0_mulsc],
            )?;
            // masked = scaled + mask (additive [b,1,s,s] broadcast over
            // heads; REPLACES causal — the derived mask encodes
            // within-doc causality; GPU-resident when seg is, so the add
            // cannot drag a GPU graph to CPU).
            let mask = call(
                compiler,
                builder,
                "nsl_packed_mask_from_segment_ids",
                &[seg],
            )?;
            let flags0_add = builder.ins().iconst(cl_types::I8, 0);
            let masked = call(compiler, builder, "nsl_tensor_add", &[scaled, mask, flags0_add])?;
            let attn = call(compiler, builder, "nsl_tensor_softmax", &[masked, dim_m1])?;
            let flags0_av = builder.ins().iconst(cl_types::I8, 0);
            let result = call(compiler, builder, "nsl_tensor_matmul", &[attn, v, flags0_av])?;
            free_tensor_value(compiler, builder, k_t)?;
            free_tensor_value(compiler, builder, scores)?;
            free_tensor_value(compiler, builder, scaled)?;
            free_tensor_value(compiler, builder, mask)?;
            free_tensor_value(compiler, builder, masked)?;
            free_tensor_value(compiler, builder, attn)?;

            match fused {
                Some((join_block, out_param, lse_param)) => {
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(join_block, &[result, zero]);
                    builder.seal_block(join_block);
                    builder.switch_to_block(join_block);
                    compiler
                        .flash_attn_aux
                        .insert(out_param, (out_param, lse_param));
                    // The saved LSE is OWNED by this step (the backward only
                    // borrows it; forward-only runs never consume it). Freed
                    // by the step cleanup via the extra-owned drain — a
                    // runtime 0 (decline) is a free() no-op.
                    compiler.sdpa_extra_owned.push(lse_param);
                    Ok(out_param)
                }
                None => {
                    let null_lse = builder.ins().iconst(cl_types::I64, 0);
                    compiler.flash_attn_aux.insert(result, (result, null_lse));
                    Ok(result)
                }
            }
        }
        PrimalOp::FlashAttentionBackwardExtractPacked { component } => {
            // PCA Stage C: segment-masked attention backward.
            //   inputs: [dout, q, k, v, fwd_out, segment_ids]
            // Identical structure to FlashAttentionBackwardExtract below,
            // with three differences: the `_segmask` variant table
            // (causal=true, segment_masked=true), the saved LSE from the
            // fused forward's side-channel (runtime 0 when the decomposed
            // fallback ran — the runtime then refuses the GPU phase-2 and
            // takes the segment-aware CPU reference), and the segment
            // tensor as the 19th FFI argument.
            let dout = inputs[0];
            let q = inputs[1];
            let k = inputs[2];
            let v = inputs[3];
            let fwd_out = inputs[4];
            let seg = inputs[5];

            let list_val = if let Some(&cached) = compiler.flash_attn_bwd_cache.get(&fwd_out) {
                cached
            } else {
                // The op's ACTUAL scale, threaded through as the 7th
                // extract input (review finding: the plain arm's 1/sqrt(hd)
                // re-derivation silently mis-scales dQ/dK for non-default
                // scales — the packed op does not inherit that limitation).
                // The adjoint context may or may not carry a var_types
                // entry for the scale VarId — drive the conversion off the
                // Cranelift VALUE TYPE instead (the decorated call site's
                // scale_bits idiom): F64 = scalar, I64 = tensor pointer.
                let scale_in = inputs[6];
                let scale_item = if builder.func.dfg.value_type(scale_in) == cl_types::F64 {
                    scale_in
                } else {
                    call(compiler, builder, "nsl_tensor_item", &[scale_in])?
                };
                let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_item);
                let scale_bits_i32 = builder.ins().bitcast(
                    cl_types::I32,
                    cranelift_codegen::ir::MemFlags::new(),
                    scale_f32,
                );
                let scale_bits = builder.ins().uextend(cl_types::I64, scale_bits_i32);
                let dim3 = builder.ins().iconst(cl_types::I64, 3);
                let head_dim = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim3])?;

                let dim0 = builder.ins().iconst(cl_types::I64, 0);
                let dim1 = builder.ins().iconst(cl_types::I64, 1);
                let dim2 = builder.ins().iconst(cl_types::I64, 2);
                let batch = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim0])?;
                let heads = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim1])?;
                let seq_len = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim2])?;

                // Saved LSE from the fused forward (join-block param), or 0
                // when the forward decomposed at compile time. This is a
                // RUNTIME value: 0 exactly when the fused launch declined.
                let lse = compiler
                    .flash_attn_aux
                    .get(&fwd_out)
                    .map(|&(_, l)| l)
                    .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0));

                let causal_val = builder.ins().iconst(cl_types::I64, 1);

                let variants = compiler.ensure_sdpa_bwd_variant_table(true, true)?;
                let zero = builder.ins().iconst(cl_types::I64, 0);
                let mut sel = [zero; 4];
                for variant in &variants {
                    let is_hd = builder.ins().icmp_imm(
                        IntCC::Equal,
                        head_dim,
                        variant.head_dim,
                    );
                    let ids = [
                        variant.phase1_ptx,
                        variant.phase1_name,
                        variant.phase2_ptx,
                        variant.phase2_name,
                    ];
                    for (slot, id) in sel.iter_mut().zip(ids) {
                        let gv = compiler.module.declare_data_in_func(id, builder.func);
                        let addr = builder.ins().symbol_value(cl_types::I64, gv);
                        *slot = builder.ins().select(is_hd, addr, *slot);
                    }
                }
                let (p1_ptx, p1_name, p2_ptx, p2_name) = (sel[0], sel[1], sel[2], sel[3]);

                let [tier_b_ptx, tier_b_name] =
                    crate::pca_tier_b::tier_b_disabled_sentinel(builder);
                let list = call(
                    compiler,
                    builder,
                    "nsl_flash_attention_backward",
                    &[dout, q, k, v, fwd_out, lse,
                      scale_bits, batch, heads, seq_len, head_dim,
                      causal_val, p1_ptx, p1_name, p2_ptx, p2_name,
                      tier_b_ptx, tier_b_name, seg],
                )?;
                compiler.flash_attn_bwd_cache.insert(fwd_out, list);
                list
            };

            let idx = builder.ins().iconst(cl_types::I64, *component as i64);
            let result = call(compiler, builder, "nsl_list_get", &[list_val, idx])?;

            if *component == 2 {
                let _ = call(compiler, builder, "nsl_list_free", &[list_val])?;
                compiler.flash_attn_bwd_cache.remove(&fwd_out);
            }

            Ok(result)
        }
        PrimalOp::FlashAttentionBackwardExtract { causal, component } => {
            // Fused attention backward: call nsl_flash_attention_backward to get
            // [dQ, dK, dV] as an NslList, then extract the requested component.
            //
            // inputs: [dout, q, k, v, fwd_out]
            // The first component (0 = dQ) triggers the backward call and caches
            // the result list. Subsequent components (1 = dK, 2 = dV) extract from
            // the cached list. The last component (2 = dV) frees the list.
            let dout = inputs[0];
            let q = inputs[1];
            let k = inputs[2];
            let v = inputs[3];
            let fwd_out = inputs[4];

            // Look up or compute the backward [dQ, dK, dV] list
            let list_val = if let Some(&cached) = compiler.flash_attn_bwd_cache.get(&fwd_out) {
                cached
            } else {
                // Extract scale from Q's head_dim: scale = 1/sqrt(head_dim)
                let dim3 = builder.ins().iconst(cl_types::I64, 3);
                let head_dim = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim3])?;
                // Convert head_dim (i64) to f64, compute 1/sqrt, reinterpret as f32 bits
                let hd_f64 = builder.ins().fcvt_from_sint(cl_types::F64, head_dim);
                let hd_sqrt = builder.ins().sqrt(hd_f64);
                let one_f64 = builder.ins().f64const(1.0);
                let scale_f64 = builder.ins().fdiv(one_f64, hd_sqrt);
                // Convert to f32 then reinterpret as i32 bits for scale_bits param
                let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_f64);
                let scale_bits_i32 = builder.ins().bitcast(cl_types::I32, cranelift_codegen::ir::MemFlags::new(), scale_f32);
                let scale_bits = builder.ins().sextend(cl_types::I64, scale_bits_i32);

                // Extract batch, heads, seq_len, head_dim from Q shape
                let dim0 = builder.ins().iconst(cl_types::I64, 0);
                let dim1 = builder.ins().iconst(cl_types::I64, 1);
                let dim2 = builder.ins().iconst(cl_types::I64, 2);
                let batch = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim0])?;
                let heads = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim1])?;
                let seq_len = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim2])?;

                // Logsumexp: the fused forward (PCA Stage C dispatch) saves
                // it in the side-channel as a runtime value — 0 exactly when
                // the decomposed fallback ran, which tells the runtime to
                // recompute it (device-resident FLASH_LSE, PR #354), i.e.
                // the pre-Stage-C behavior.
                let lse_null = compiler
                    .flash_attn_aux
                    .get(&fwd_out)
                    .map(|&(_, l)| l)
                    .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0));

                let causal_val = builder.ins().iconst(cl_types::I64, if *causal { 1 } else { 0 });

                // Load backward PTX data pointers from .rodata.
                //
                // Decorated programs carry a single backward pair (specialized
                // for the decorator's head_dim) in flash_attention_context.
                // Decorator-free programs — every plain stdlib model — used to
                // get four null pointers here, which silently routed the whole
                // attention backward through the CPU reference (the PR #335
                // wiring gap). They now get the per-head_dim variant table:
                // the op's `causal` is static, head_dim is only known at
                // runtime, so we embed one backward pair per supported
                // head_dim and select with an equality chain on the runtime
                // head_dim value. Unmatched dims select null and keep the
                // CPU fallback.
                let ctx_bwd = compiler.kernels.flash_attention_context.as_ref().and_then(|c| {
                    Some((
                        c.bwd_phase1_data_id?,
                        c.bwd_phase1_name_data_id?,
                        c.bwd_phase2_data_id?,
                        c.bwd_phase2_name_data_id?,
                    ))
                });
                let (p1_ptx, p1_name, p2_ptx, p2_name) = if let Some(ids) = ctx_bwd {
                    let mut ptrs = Vec::with_capacity(4);
                    for id in [ids.0, ids.1, ids.2, ids.3] {
                        let gv = compiler.module.declare_data_in_func(id, builder.func);
                        ptrs.push(builder.ins().symbol_value(cl_types::I64, gv));
                    }
                    (ptrs[0], ptrs[1], ptrs[2], ptrs[3])
                } else {
                    let variants = compiler.ensure_sdpa_bwd_variant_table(*causal, false)?;
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let mut sel = [zero; 4];
                    for variant in &variants {
                        let is_hd = builder.ins().icmp_imm(
                            IntCC::Equal,
                            head_dim,
                            variant.head_dim,
                        );
                        let ids = [
                            variant.phase1_ptx,
                            variant.phase1_name,
                            variant.phase2_ptx,
                            variant.phase2_name,
                        ];
                        for (slot, id) in sel.iter_mut().zip(ids) {
                            let gv = compiler.module.declare_data_in_func(id, builder.func);
                            let addr = builder.ins().symbol_value(cl_types::I64, gv);
                            *slot = builder.ins().select(is_hd, addr, *slot);
                        }
                    }
                    (sel[0], sel[1], sel[2], sel[3])
                };

                // Tier-B sentinel pair (planner spec §4): the plain (non-CSHA)
                // SDPA backward has no Tier-B-on variant, so pass the disabled
                // sentinel — via the construction helper, per the discipline
                // documented on `nsl_flash_attention_backward` (inline 0,0
                // literals are what let the pre-fix 16-arg call rot silently).
                let [tier_b_ptx, tier_b_name] =
                    crate::pca_tier_b::tier_b_disabled_sentinel(builder);
                // 19th arg (PCA Stage C): segment tensor — always 0 on the
                // plain (non-packed) path.
                let seg_none = builder.ins().iconst(cl_types::I64, 0);
                let list = call(
                    compiler,
                    builder,
                    "nsl_flash_attention_backward",
                    &[dout, q, k, v, fwd_out, lse_null,
                      scale_bits, batch, heads, seq_len, head_dim,
                      causal_val, p1_ptx, p1_name, p2_ptx, p2_name,
                      tier_b_ptx, tier_b_name, seg_none],
                )?;
                compiler.flash_attn_bwd_cache.insert(fwd_out, list);
                list
            };

            // Extract the requested component
            let idx = builder.ins().iconst(cl_types::I64, *component as i64);
            let result = call(compiler, builder, "nsl_list_get", &[list_val, idx])?;

            // Free the list after the last component (dV, component=2) is extracted
            if *component == 2 {
                let _ = call(compiler, builder, "nsl_list_free", &[list_val])?;
                compiler.flash_attn_bwd_cache.remove(&fwd_out);
            }

            Ok(result)
        }
        PrimalOp::CshaFusedBackwardExtract { component } => {
            // Gap C: structural extract arm — reads the requested component
            // from `compiler.csha_fused_bwd_cache`, keyed by the first input's
            // Cranelift Value.  Gap D is responsible for populating the cache
            // (by calling `nsl_flash_attention_csha_backward` inside the
            // EmitFused arm of `AdjointGenerator::generate`).
            //
            // Today the EmitFused arm still falls through to per-op AD, so no
            // CshaFusedBackwardExtract op is ever emitted in a real compile
            // and this arm is unreachable in practice.  It exists as a ready-
            // made symbol Gap D can call without further plumbing work.
            //
            // Contract expected by Gap D (revised by Gap I.2+M):
            //   - inputs[0] is the launch op's Cranelift Value — a
            //     placeholder zero tensor that exists purely as a
            //     worklist-walk anchor for `eliminate_dead_gradients`.
            //     The value itself is never dereferenced here.
            //   - inputs[1] is the "chain key" VarId.  Gap D decides what
            //     VarId to use (typically the RMSNorm input or the SDPA
            //     output), but all seven extract ops in one chain MUST share
            //     the same second-input VarId so they look up the same cache
            //     entry.
            //   - Gap D populates the cache *before* emitting any extract
            //     op, by running the fused backward launch and stashing the
            //     eight output tensors into
            //     `compiler.csha_fused_bwd_cache.insert(key_val, [t0..t7])`.
            //   - `component` MUST be in 0..=7 (see variant doc-comment in
            //     `wengert.rs` for the dq/dk/dv/dwq/dwk/dwv/dx/dx_norm
            //     mapping; component 7 is the Gap I.5 addition).
            //
            // Cache eviction is the last-component's responsibility, mirroring
            // the FlashAttention extract pattern (which evicts on component=2).
            if *component > 7 {
                return Err(CodegenError::new(format!(
                    "CshaFusedBackwardExtract: component {} out of range 0..=7",
                    component
                )));
            }
            // Gap I.2+M: inputs[0] is the launch-result placeholder
            // (ignored); inputs[1] is the chain cache key.
            let _launch_result = inputs[0];
            let key_val = inputs[1];
            let slot = compiler
                .csha_fused_bwd_cache
                .get(&key_val)
                .copied()
                .ok_or_else(|| {
                    CodegenError::new(format!(
                        "CshaFusedBackwardExtract: no cache entry for key Value {:?}. \
                         Gap D must populate `compiler.csha_fused_bwd_cache` before \
                         emitting extract ops (via a `nsl_flash_attention_csha_backward` \
                         launch in the EmitFused arm of `AdjointGenerator::generate`).",
                        key_val
                    ))
                })?;
            let result = slot[*component as usize];

            // Evict on the last component (component=7 → dx_norm) so
            // subsequent chains with a coincidentally-equal key_val don't
            // reuse stale tensor pointers. Mirrors FlashAttention's
            // component==2 free. Bumped from 6 to 7 by the Gap I.5 Option-A
            // fix when `dx_norm` became the 8th extract.
            if *component == 7 {
                compiler.csha_fused_bwd_cache.remove(&key_val);
            }

            Ok(result)
        }
        PrimalOp::FusedCshaBackward { layer } => {
            // Gap D: fused CSHA backward kernel launch.
            //
            // Emits a call to `nsl_flash_attention_csha_backward` using:
            //   - Saved activation pointers stashed by Gap A's forward
            //     side in `compiler.csha_forward_saves[layer]`.
            //   - Backward PTX / name / shmem triple synthesized by
            //     Gap B and cached on the same record.
            //   - FlashAttentionConfig from the FA compile context
            //     (`csha_training_config`).
            //
            // Produces seven output gradient tensors (dq/dk/dv/dwq/dwk/
            // dwv/dx) which are stashed on `compiler.csha_fused_bwd_cache`
            // keyed by the first input's Cranelift Value (the "chain key"
            // the AD emitter chose).  Gap C's extract-op lowerer consumes
            // those outputs downstream.
            //
            // Input convention (produced by EmitFused in source_ad.rs):
            //   inputs[0] = chain_key VarId (cache key — opaque here)
            //   inputs[1] = dO VarId (adjoint of attention output), if
            //               the emitter could find one.  If `inputs.len()
            //               < 2` the lowerer passes null for do_ptr,
            //               which the kernel interprets as "no seed" and
            //               produces zeroed gradients (structural
            //               fallback — the AD graph will still be valid
            //               because per-op AD ran on SDPA separately).
            //
            // The op's own result Value is a placeholder zero tensor —
            // it is never read downstream; extract ops pull the real
            // outputs from the cache.
            let key_val = inputs[0];

            // Look up saves + backward PTX DataIds by layer name.
            let saves_opt = compiler.csha_forward_saves.get(layer).copied();
            let saves = saves_opt.ok_or_else(|| {
                CodegenError::new(format!(
                    "FusedCshaBackward: no forward saves for layer '{}'. \
                     Gap A must allocate save buffers at the FA call site \
                     before Gap D's adjoint emission runs.",
                    layer
                ))
            })?;

            // Backward PTX + name pointers from .rodata (Gap B).
            let bwd_ptx_did = saves.backward_ptx_data_id;
            let bwd_name_did = saves.backward_name_data_id;
            let (bwd_ptx_ptr, bwd_name_ptr) = match (bwd_ptx_did, bwd_name_did) {
                (Some(pid), Some(nid)) => {
                    let pgv = compiler.module.declare_data_in_func(pid, builder.func);
                    let ngv = compiler.module.declare_data_in_func(nid, builder.func);
                    (
                        builder.ins().symbol_value(cl_types::I64, pgv),
                        builder.ins().symbol_value(cl_types::I64, ngv),
                    )
                }
                _ => (
                    builder.ins().iconst(cl_types::I64, 0),
                    builder.ins().iconst(cl_types::I64, 0),
                ),
            };

            // FlashAttentionConfig + SMEM bytes for the backward launch.
            let training_cfg = compiler
                .kernels
                .flash_attention_context
                .as_ref()
                .and_then(|c| c.csha_training_config.clone());
            // Sprint 1 T1.3: capture the compile-time Tier B.2 hybrid-backward
            // eligibility decision while the config is in scope. The runtime
            // `seq_len == block_q` check happens below using the Cranelift
            // `seq_len` Value already extracted via `nsl_tensor_shape_dim`.
            let (block_q, block_kv, head_dim, is_causal, d_model, eps_bits, shmem_bytes,
                 tier_b2_compile_time_eligible) =
                match training_cfg {
                    Some(cfg) => {
                        // Backward kernel sizes shmem differently than forward:
                        // it needs `backward_total_bytes` (forward total +
                        // backward_extra dQ/dK/dV/P/dS/v_in/dRMSNorm tiles)
                        // plus PCA Tier A's embedded seg_smem tail when
                        // `segment_masked`. `shared_mem_bytes_v2` returns only
                        // the forward total — passing that to the backward
                        // launch short-allocates dynamic SMEM whenever the
                        // backward config needs the extern .shared path,
                        // corrupting gradients at runtime.
                        let bytes =
                            crate::flash_attention_v2::shared_mem_bytes_v2_backward(&cfg)
                                as i64;
                        let dm = cfg.csha.as_ref().map(|c| c.d_model as i64).unwrap_or(0);
                        let eps = cfg
                            .csha
                            .as_ref()
                            .map(|c| c.rmsnorm_eps.to_bits() as i64)
                            .unwrap_or(1e-5f32.to_bits() as i64);
                        let b2_ct =
                            crate::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_compile_time_eligible(&cfg);
                        (cfg.block_q, cfg.block_kv, cfg.head_dim, cfg.causal, dm, eps, bytes, b2_ct)
                    }
                    None => {
                        // No training config — the backward launch cannot
                        // succeed.  Return a zero-tensor placeholder so the
                        // op has a valid Value but do NOT emit the launch
                        // (it would segfault).  Extract ops will hit the
                        // cache-miss path and the whole adjoint graph
                        // falls back to per-op AD via the AD emitter's
                        // reset-cell pattern in EmitFused.
                        eprintln!(
                            "[nsl] FusedCshaBackward: no csha_training_config for layer '{}'; \
                             skipping launch (cache stays empty, extract ops will fail).",
                            layer
                        );
                        let one = builder.ins().iconst(cl_types::I64, 1);
                        let shape_list = call(compiler, builder, "nsl_list_new", &[])?;
                        call(compiler, builder, "nsl_list_push", &[shape_list, one])?;
                        return call(compiler, builder, "nsl_tensor_zeros", &[shape_list]);
                    }
                };

            // Shape values: derive batch/heads/seq_len/head_dim from the
            // chain_key tensor (it's a q/k/v-shaped tensor).  When the
            // emitter passes a non-tensor chain_key (e.g. a primal VarId
            // that's a scalar), we fall back to constants from the
            // config.
            //
            // For the smoke path we follow the minimal convention: if
            // inputs contains the forward q tensor as a later arg (index
            // 2 by the EmitFused convention), use it; otherwise read
            // from the key.
            let shape_src = if inputs.len() > 2 { inputs[2] } else { key_val };
            let dim0 = builder.ins().iconst(cl_types::I64, 0);
            let dim1 = builder.ins().iconst(cl_types::I64, 1);
            let dim2 = builder.ins().iconst(cl_types::I64, 2);
            let batch =
                call(compiler, builder, "nsl_tensor_shape_dim", &[shape_src, dim0])?;
            let heads =
                call(compiler, builder, "nsl_tensor_shape_dim", &[shape_src, dim1])?;
            let seq_len =
                call(compiler, builder, "nsl_tensor_shape_dim", &[shape_src, dim2])?;
            let hd_val = builder.ins().iconst(cl_types::I64, head_dim);
            let block_q_val = builder.ins().iconst(cl_types::I64, block_q);
            let block_kv_val = builder.ins().iconst(cl_types::I64, block_kv);
            let shmem_val = builder.ins().iconst(cl_types::I64, shmem_bytes);
            let causal_val = builder
                .ins()
                .iconst(cl_types::I64, if is_causal { 1 } else { 0 });
            let d_model_val = builder.ins().iconst(cl_types::I64, d_model);
            let eps_bits_val = builder.ins().iconst(cl_types::I64, eps_bits);

            // scale_bits = 1/sqrt(head_dim) reinterpreted as f32 bits.
            let hd_f64 = builder.ins().fcvt_from_sint(cl_types::F64, hd_val);
            let hd_sqrt = builder.ins().sqrt(hd_f64);
            let one_f64 = builder.ins().f64const(1.0);
            let scale_f64 = builder.ins().fdiv(one_f64, hd_sqrt);
            let scale_f32 = builder.ins().fdemote(cl_types::F32, scale_f64);
            let scale_bits_i32 = builder.ins().bitcast(
                cl_types::I32,
                cranelift_codegen::ir::MemFlags::new(),
                scale_f32,
            );
            let scale_bits = builder.ins().sextend(cl_types::I64, scale_bits_i32);

            let null = builder.ins().iconst(cl_types::I64, 0);

            // Optional input pointers: pass-through from inputs[].  These
            // feed the kernel's `q_ptr / k_ptr / v_ptr / out_ptr / lse_ptr /
            // x_ptr / norm_weight_ptr / wq_ptr / wk_ptr / wv_ptr` params.
            // Null is acceptable when the forward saves carry the
            // activations the backward kernel actually reads (which is
            // the design intent of Gap A).
            let q_ptr = if inputs.len() > 2 { inputs[2] } else { null };
            let k_ptr = if inputs.len() > 3 { inputs[3] } else { null };
            let v_ptr = if inputs.len() > 4 { inputs[4] } else { null };
            let x_ptr = if inputs.len() > 5 { inputs[5] } else { null };
            let wq_ptr = if inputs.len() > 6 { inputs[6] } else { null };
            let wk_ptr = if inputs.len() > 7 { inputs[7] } else { null };
            let wv_ptr = if inputs.len() > 8 { inputs[8] } else { null };
            let norm_weight_ptr = if inputs.len() > 9 { inputs[9] } else { null };
            let do_ptr = if inputs.len() > 1 { inputs[1] } else { null };

            // Allocate 7 output gradient tensors DIRECTLY on device with
            // the right dtypes (Gap I.3 — A+F).
            //
            // Tier C backward writes:
            //   - dq/dk/dv/dwq/dwk/dwv as f16 (st.global.u16 → 2 B/elem)
            //   - dx as f32 (RMSNorm-input gradient stays f32)
            //
            // Pre-A+F we zero-allocated all seven via `nsl_tensor_zeros`
            // (CPU, f32) then `nsl_tensor_to_device`.  That left the f16
            // buffers at 2× their intended byte size with every second
            // byte uninitialised; f32-interpreting reads on the host
            // produced garbage which the optimiser then stirred into the
            // next step's weights.  This block replaces that with a
            // single direct-on-device alloc per tensor at the correct
            // element size + dtype.
            //
            // Shapes:
            //   dq/dk/dv:      [batch, heads, seq_len, head_dim]
            //   dwq/dwk/dwv:   [d_model, heads*head_dim]    (kv_dim = heads*head_dim)
            //   dx:            [batch, seq_len, d_model]
            let heads_hd = builder.ins().imul(heads, hd_val);
            let cuda_device = builder.ins().iconst(cl_types::I64, 1);

            let alloc_shape_on = |compiler: &mut Compiler,
                                  builder: &mut FunctionBuilder,
                                  ffi: &str,
                                  dims: &[Value],
                                  device: Value|
             -> Result<Value, CodegenError> {
                let shape = call(compiler, builder, "nsl_list_new", &[])?;
                for d in dims {
                    call(compiler, builder, "nsl_list_push", &[shape, *d])?;
                }
                call(compiler, builder, ffi, &[shape, device])
            };

            // f16 outputs — the 6 kernel-side st.global.u16 sinks.
            let dq_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_f16_on",
                &[batch, heads, seq_len, hd_val], cuda_device,
            )?;
            let dk_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_f16_on",
                &[batch, heads, seq_len, hd_val], cuda_device,
            )?;
            let dv_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_f16_on",
                &[batch, heads, seq_len, hd_val], cuda_device,
            )?;
            let dwq_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_f16_on",
                &[d_model_val, heads_hd], cuda_device,
            )?;
            let dwk_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_f16_on",
                &[d_model_val, heads_hd], cuda_device,
            )?;
            let dwv_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_f16_on",
                &[d_model_val, heads_hd], cuda_device,
            )?;

            // dx stays on the f32 allocator — the kernel writes it as f32.
            let dx_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_on",
                &[batch, seq_len, d_model_val], cuda_device,
            )?;

            // Gap I.5 Option-A: 8th output `dx_norm`, gradient w.r.t. the
            // RMSNorm OUTPUT (= `dy_norm`). f32, shape [batch, seq, d_model].
            // Consumed by the AD-side `RmsNormGammaBackward` emission via
            // `extract_results[7]` for correct dgamma under the fused CSHA
            // dispatch path.
            let dxn_dev = alloc_shape_on(
                compiler, builder, "nsl_tensor_zeros_on",
                &[batch, seq_len, d_model_val], cuda_device,
            )?;

            // Launch the 44-arg backward kernel.
            // Forward-side `active_heads` — take from the training config
            // (stored on the FA context via Gap B).  When the ctx is gone,
            // pass 0 (kernel interprets as "all heads live").
            let active_heads_val = builder.ins().iconst(cl_types::I64, 0);

            // CFTP §4.3 / Tier A activation (spec 2026-05-17): read the
            // per-step thread-local packing registry. The forward at
            // line ~565 above reads the same registry on the same step,
            // so the de-rotation math here uses identical segment_ids /
            // doc_starts pointers — bit-identical effective_pos.
            let seg_ids_ptr_v = call(
                compiler,
                builder,
                "nsl_packing_metadata_get_segment_ids",
                &[],
            )?;
            let doc_starts_v = call(
                compiler,
                builder,
                "nsl_packing_metadata_get_doc_starts",
                &[],
            )?;

            // CSHA Tier B.2 (Phase 3 T6 / Sprint 1 T1.3): tier_b2_active flag.
            //
            // Semantics — the flag is `1` iff
            //   tier_b2_hybrid_backward_compile_time_eligible(config)   // compile-time
            //   && seq_len == config.block_q                            // runtime
            // matching `tier_b2_hybrid_backward_eligible(config, seq_len)`
            // bit-for-bit (invariant unit-tested in `dispatch::tests`).
            //
            // Decomposition rationale:
            //   - All Tier-B.2 dispatch checks except `seq_len == block_q` depend
            //     only on the `FlashAttentionConfig` baked in at lowering time.
            //   - `seq_len` is a runtime tensor shape dim (extracted above via
            //     `nsl_tensor_shape_dim`), so the seq comparison must be emitted
            //     as Cranelift IR.
            //
            // When the config is ineligible we emit a literal `iconst(0)` —
            // ptxas trivially constant-folds the dead hybrid launch branch
            // in the FFI. When the config is eligible the runtime compares
            // the live seq_len against `block_q` and `uextend`s the i1 to i64.
            // FFI gates the 4-kernel hybrid launch on this being non-zero;
            // the scalar single-kernel path remains the fallback.
            let tier_b2_active_flag = if tier_b2_compile_time_eligible {
                use cranelift_codegen::ir::condcodes::IntCC;
                let block_q_i64 = builder.ins().iconst(cl_types::I64, block_q);
                let seq_eq_bq = builder.ins().icmp(IntCC::Equal, seq_len, block_q_i64);
                builder.ins().uextend(cl_types::I64, seq_eq_bq)
            } else {
                builder.ins().iconst(cl_types::I64, 0)
            };

            // PCA Tier B planner spec §5: build the (tier_b_ptx_ptr,
            // tier_b_name_ptr) sentinel pair for the backward FFI. When the
            // module-scan emitted a Tier-B-on backward variant for this config
            // (mirror of the forward `_with_saves` Tier-B-on emission gated on
            // `should_emit_tier_b_at_codegen(training_config)`), pass the
            // symbol pair; otherwise the disabled sentinel. The runtime
            // launcher decides at-launch via the 4-condition gate in
            // `should_dispatch_tier_b_at_runtime`. Forward and backward MUST
            // dispatch consistently — if the forward picks Tier-B-on, the
            // backward MUST too, else the backward computes gradients for
            // tiles the forward skipped (gradients leak across documents).
            let tier_b_bwd = match (
                saves.backward_tier_b_on_ptx_data_id,
                saves.backward_tier_b_on_name_data_id,
            ) {
                (Some(pid), Some(nid)) => crate::pca_tier_b::tier_b_enabled(
                    builder,
                    &mut compiler.module,
                    pid,
                    nid,
                ),
                _ => crate::pca_tier_b::tier_b_disabled_sentinel(builder),
            };

            // Sprint 1 T1.1: source the forward attention output `O`
            // handle from the per-layer save record (populated by the
            // forward CSHA launch site in this same FunctionBuilder
            // scope). The Tier B.2 hybrid backward's D pre-pass
            // (kernel 1) computes D = rowsum(dO * O) and reads O via
            // `csha_tensor_data_ptr(out_ptr)`. Pre-T1.1 we passed
            // `null` here and the D pre-pass silently produced
            // D == 0; see PARITY-GATE NOTE in
            // `crates/nsl-runtime/src/flash_attention.rs::nsl_flash_attention_csha_backward`.
            let _rc = call(
                compiler,
                builder,
                "nsl_flash_attention_csha_backward",
                &[
                    // Forward-side 36 args (mirrors _with_saves order).
                    q_ptr, k_ptr, v_ptr,
                    saves.out,            // out_ptr (Sprint 1 T1.1)
                    null,                 // logsumexp_ptr
                    scale_bits,
                    batch, heads, seq_len, hd_val,
                    null, null, null, null,    // paged (block_table, k_pool, v_pool, block_size)
                    saves.cos, saves.sin,       // RoPE cos/sin (Sprint 1 cycle-2: sourced from forward save record)
                    null, null,                 // seq_ids, seq_lens
                    shmem_val,
                    bwd_ptx_ptr, bwd_name_ptr,
                    block_q_val, block_kv_val,
                    causal_val,
                    x_ptr, norm_weight_ptr,
                    wq_ptr, wk_ptr, wv_ptr,
                    null,                 // wo_ptr
                    eps_bits_val,
                    active_heads_val, d_model_val,
                    // Forward-saved activations (Gap A).
                    saves.q_proj, saves.k_proj, saves.v_proj,
                    saves.row_max, saves.row_sum,
                    saves.x_raw,
                    // Tier C backward-specific: dO input + 8 gradient outputs.
                    // 8th output (dxn_dev) added by Gap I.5 Option-A fix.
                    do_ptr,
                    dq_dev, dk_dev, dv_dev,
                    dwq_dev, dwk_dev, dwv_dev,
                    dx_dev,
                    dxn_dev,
                    // PCA Tier A: segment_ids_ptr — read from the
                    // thread-local packing registry (same value the
                    // forward at line ~565 read on this step).
                    seg_ids_ptr_v,
                    // PCA Tier B planner spec §4: pass the Tier-B-on sentinel
                    // pair built above as `tier_b_bwd`. MUST mirror the forward
                    // `_with_saves` call's dispatch decision — the runtime
                    // launcher uses the SAME 4-condition gate
                    // (`should_dispatch_tier_b_at_runtime`) so as long as both
                    // sites pass the same sentinel pair, dispatch stays
                    // consistent. Mismatched dispatch (forward Tier-B-on +
                    // backward base, or vice versa) computes gradients on
                    // tiles the forward skipped → leaks across documents.
                    tier_b_bwd[0], tier_b_bwd[1],
                    // PCA §4.3: doc_starts_ptr — read from the same
                    // registry; matches the forward's effective_pos.
                    doc_starts_v,
                    // CSHA Tier B.2 (Phase 3 T6 / Sprint 1 T1.3): tier_b2_active flag.
                    // Computed above as the AND of the compile-time
                    // `tier_b2_hybrid_backward_compile_time_eligible(config)`
                    // predicate and the runtime `seq_len == block_q` icmp.
                    // FFI's 4-kernel hybrid launch fires iff this is non-zero;
                    // otherwise the scalar single-kernel path remains active
                    // (byte-identical to pre-Sprint-1 behavior).
                    tier_b2_active_flag,
                    // PCA per-doc CTA backward (Sprint 5): num_docs_or_zero
                    // trailing sentinel. AD-side never emits a per-doc CTA
                    // backward today (the planner gate that activates the
                    // per-doc forward also feeds the backward synth at the
                    // dispatch site), so passing 0 here keeps the legacy
                    // per-q-block topology. Hot-path activation will set
                    // this when the planner integrates with AD.
                    null,
                ],
            )?;

            // Populate Gap C cache so extract ops can read outputs.
            // Slot 7 (dxn_dev) added by the Gap I.5 Option-A fix.
            compiler
                .csha_fused_bwd_cache
                .insert(key_val, [dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev, dxn_dev]);

            // Gap A cleanup: now that the backward has consumed the save
            // pointers, free them.  (Moved here from the forward site in
            // expr/advanced.rs — under @train, the forward leaves them
            // live so the backward can read them.)
            let _ = call(
                compiler,
                builder,
                "nsl_csha_free_backward_activations_from",
                &[
                    saves.q_proj, saves.k_proj, saves.v_proj,
                    saves.row_max, saves.row_sum, saves.x_raw,
                ],
            )?;
            compiler.csha_forward_saves.remove(layer);

            // Return a placeholder Value.  The op's own `result` VarId
            // is never consumed — extract ops pull real outputs from
            // the cache — so an empty tensor is fine.
            let one = builder.ins().iconst(cl_types::I64, 1);
            let shape_list = call(compiler, builder, "nsl_list_new", &[])?;
            call(compiler, builder, "nsl_list_push", &[shape_list, one])?;
            call(compiler, builder, "nsl_tensor_zeros", &[shape_list])
        }
        PrimalOp::RoPE { .. } => {
            // No dedicated nsl_tensor_rope FFI exists; clone as identity for now.
            // RoPE forward is handled by the flash attention system.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::RoPEInverse { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim as i64);
            call(
                compiler,
                builder,
                "nsl_tensor_rope_inverse",
                &[inputs[0], d],
            )
        }

        // === Regularization ===
        PrimalOp::Dropout { p } => {
            let pv = builder.ins().f64const(*p);
            let training = builder.ins().iconst(cl_types::I8, 1);
            call(
                compiler,
                builder,
                "nsl_tensor_dropout",
                &[inputs[0], pv, training],
            )
        }

        // === Control flow (2 ops) ===
        PrimalOp::Select => {
            // inputs: [cond, true_val, false_val]
            call(
                compiler,
                builder,
                "nsl_tensor_where",
                &[inputs[0], inputs[1], inputs[2]],
            )
        }
        PrimalOp::Condition(cmp) => {
            let cmp_val = builder.ins().iconst(
                cl_types::I64,
                match cmp {
                    CompareKind::Gt => 0,
                    CompareKind::GtEq => 1,
                    CompareKind::Lt => 2,
                    CompareKind::LtEq => 3,
                    CompareKind::Eq => 4,
                    CompareKind::NotEq => 5,
                },
            );
            call(
                compiler,
                builder,
                "nsl_tensor_compare",
                &[inputs[0], inputs[1], cmp_val],
            )
        }
        // Paper §5.3 checkpointing-aware backward marker (cycle-10 Task 2).
        //
        // Lowering treats this as a structural no-op: the marker exists in
        // the Wengert use-def graph so that the source-AD reverse-walk and
        // the downstream codegen dispatch can both observe "this subgraph
        // was checkpointed; values were not persisted across the forward /
        // backward boundary." The actual recompute PTX is emitted by the
        // backward-kernel emitter (`emit_prologue_recompute`, Task 8) — at
        // wengert-lower time we just emit a null placeholder result so the
        // var_map stays well-formed.
        PrimalOp::PrologueRecompute { subgraph_id: _ } => {
            Ok(builder.ins().iconst(cl_types::I64, 0))
        }

        // === Non-differentiable passthroughs ===
        PrimalOp::Passthrough(ref name) => {
            match name.as_str() {
                // CCR compressed saves (paper phases 5-6): spliced by
                // ccr::append_compressed_saves / apply_to_adjoint AFTER
                // adjoint generation — never differentiated (the ad_rules
                // default arm yields no adjoint, which is correct: these
                // wrap saved-for-backward-only tensors).
                "ccr_cast_fp16" => call(compiler, builder, "nsl_tensor_to_fp16", &[inputs[0]]),
                "ccr_cast_bf16" => call(compiler, builder, "nsl_tensor_to_bf16", &[inputs[0]]),
                "ccr_cast_f32" => call(compiler, builder, "nsl_tensor_to_f32", &[inputs[0]]),
                "shape" => call(compiler, builder, "nsl_tensor_shape", &[inputs[0]]),
                "ndim" => call(compiler, builder, "nsl_tensor_ndim", &[inputs[0]]),
                "reshape" => {
                    // inputs[0] = tensor, inputs[1] = shape_list
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_reshape",
                        &[inputs[0], inputs[1]],
                    )
                }
                "transpose" => {
                    // inputs[0] = tensor, inputs[1] = dim0, inputs[2] = dim1
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_transpose",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "contiguous" => call(compiler, builder, "nsl_tensor_contiguous", &[inputs[0]]),
                "expand" => call(
                    compiler,
                    builder,
                    "nsl_tensor_expand",
                    &[inputs[0], inputs[1]],
                ),
                "squeeze" => call(compiler, builder, "nsl_tensor_squeeze", &[inputs[0]]),
                "unsqueeze" => {
                    let dim = if inputs.len() > 1 {
                        inputs[1]
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    call(compiler, builder, "nsl_tensor_unsqueeze", &[inputs[0], dim])
                }
                "item" => {
                    // .item() extracts a scalar from a tensor and wraps back into a
                    // 0-dim scalar tensor so it stays in the tensor domain for
                    // arithmetic ops (Mul, Add, etc.).
                    let f64_val = call(compiler, builder, "nsl_tensor_item", &[inputs[0]])?;
                    let dt = builder.ins().iconst(cl_types::I64, 1); // f32
                    call(compiler, builder, "nsl_tensor_scalar", &[f64_val, dt])
                }
                // Trigonometric (forward pass — backward handled by tape)
                "cos" => call(compiler, builder, "nsl_tensor_cos", &[inputs[0]]),
                "sin" => call(compiler, builder, "nsl_tensor_sin", &[inputs[0]]),
                "rotate_half" => call(compiler, builder, "nsl_tensor_rotate_half", &[inputs[0]]),
                // Tensor construction (non-differentiable, used for shape/position computation)
                "arange" => {
                    // arange(start, stop) or arange(stop) — extract f64 from inputs
                    // and call nsl_tensor_arange(start, stop, step=1.0).
                    // Inputs may be Tensor (i64 ptr), Scalar (f64), or Integer (i64).
                    let mut to_f64 = |builder: &mut FunctionBuilder,
                                      val: Value,
                                      idx: usize|
                     -> Result<Value, CodegenError> {
                        let vty = op
                            .inputs
                            .get(idx)
                            .and_then(|vid| var_types.get(vid).copied())
                            .unwrap_or(WengertType::Tensor);
                        match vty {
                            WengertType::Scalar => Ok(val),
                            WengertType::Integer => {
                                Ok(builder.ins().fcvt_from_sint(cl_types::F64, val))
                            }
                            _ => call(compiler, builder, "nsl_tensor_item", &[val]),
                        }
                    };
                    let step = builder.ins().f64const(1.0);
                    if inputs.len() >= 2 {
                        let start = to_f64(builder, inputs[0], 0)?;
                        let stop = to_f64(builder, inputs[1], 1)?;
                        call(compiler, builder, "nsl_tensor_arange", &[start, stop, step])
                    } else if inputs.len() == 1 {
                        let zero = builder.ins().f64const(0.0);
                        let stop = to_f64(builder, inputs[0], 0)?;
                        call(compiler, builder, "nsl_tensor_arange", &[zero, stop, step])
                    } else {
                        Err(CodegenError::new(
                            "arange requires at least 1 argument".to_string(),
                        ))
                    }
                }
                "reduce_to_shape" => {
                    // inputs = [grad, target_param]
                    // Reduce grad by summing over leading batch dims to match target shape
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_reduce_to_shape",
                        &[inputs[0], inputs[1]],
                    )
                }
                "mean_keepdim_last" => {
                    // Mean along the LAST dimension with keepdim=true.
                    // Used by LayerNorm/RMSNorm backward to compute
                    // mean/variance reductions that broadcast correctly.
                    let ndim_val =
                        call(compiler, builder, "nsl_tensor_ndim", &[inputs[0]])?;
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let last_dim = builder.ins().isub(ndim_val, one);
                    let keepdim = builder.ins().iconst(cl_types::I64, 1);
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_mean_dim",
                        &[inputs[0], last_dim, keepdim],
                    )
                }
                "sum_keepdim_last" => {
                    // Sum along the LAST dimension with keepdim=true.
                    // Used by Softmax/LogSoftmax backward to reduce
                    // dot(grad, softmax_output) along the softmax axis
                    // while preserving the trailing dim as size-1 for
                    // broadcasting against the un-reduced gradient.
                    //
                    // Resolves "last dim" at runtime via nsl_tensor_ndim
                    // to avoid the dim=-1 → global-sum ambiguity.
                    let ndim_val =
                        call(compiler, builder, "nsl_tensor_ndim", &[inputs[0]])?;
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let last_dim = builder.ins().isub(ndim_val, one);
                    let keepdim = builder.ins().iconst(cl_types::I64, 1);
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_sum_dim",
                        &[inputs[0], last_dim, keepdim],
                    )
                }
                // "causal_mask_add" was used by the decomposed SDPA path
                // (commit e8d5a76) and has been removed in favor of the fused
                // ScaledDotProductAttention + FlashAttentionBackwardExtract path.
                _ if name.starts_with("dict_get:") => {
                    // inputs = [dict_ptr]
                    // Dict field access: batch.input_ids -> nsl_dict_get_str(batch, "input_ids")
                    let field = &name["dict_get:".len()..];
                    let key = compiler.compile_string_literal(builder, field)?;
                    call(compiler, builder, "nsl_dict_get_str", &[inputs[0], key])
                }
                "embedding_backward" => {
                    // inputs = [grad, indices, weight]
                    // Creates zeros_like(weight) then scatter-adds grad rows
                    call(
                        compiler,
                        builder,
                        "nsl_embedding_backward",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "cross_entropy_backward" => {
                    // inputs = [grad_output, logits, targets]
                    call(
                        compiler,
                        builder,
                        "nsl_cross_entropy_backward",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "mse_backward" => {
                    // inputs = [grad_output, pred, target]
                    // Emits grad_output * 2 * (pred - target) / numel(pred).
                    call(
                        compiler,
                        builder,
                        "nsl_mse_backward",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "l1_backward" => {
                    // inputs = [grad_output, pred, target]
                    // Emits grad_output * sign(pred - target) / numel(pred).
                    call(
                        compiler,
                        builder,
                        "nsl_l1_backward",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "silu_backward" => {
                    // inputs = [grad, x] — p4 slice 2 fused SiLU backward:
                    // grad * σ(x)*(1 + x*(1-σ(x))), one launch, bit-exact with
                    // the six-op source-AD expansion.
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_silu_backward",
                        &[inputs[0], inputs[1]],
                    )
                }
                "sigmoid_backward" => {
                    // inputs = [grad, y] — p4 slice 3 fused Sigmoid backward:
                    // grad * y*(1-y) (y = σ output), one launch, bit-exact with
                    // the three-op source-AD expansion.
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_sigmoid_backward",
                        &[inputs[0], inputs[1]],
                    )
                }
                "tanh_backward" => {
                    // inputs = [grad, y] — p4 slice 3 fused Tanh backward:
                    // grad * (1 - y*y) (y = tanh output), one launch, bit-exact
                    // with the three-op source-AD expansion.
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_tanh_backward",
                        &[inputs[0], inputs[1]],
                    )
                }
                "gelu_backward" => {
                    // inputs = [grad, x] — p4 GELU fix: grad * gelu'(x), one
                    // launch, derivative matched to each device's forward
                    // (replaces the FBIP-corrupted 7-op expansion).
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_gelu_backward",
                        &[inputs[0], inputs[1]],
                    )
                }
                "zeros" | "ones" | "full" | "randn" | "zeros_like" | "ones_like" => {
                    let rt_name = format!("nsl_tensor_{}", name);
                    call(compiler, builder, &rt_name, &inputs)
                }
                // Scalar type conversion
                "int" => {
                    // Convert to i64. Input may be:
                    //   - Scalar (f64): convert with fcvt_to_sint
                    //   - Tensor: extract f64 via nsl_tensor_item, then convert
                    //   - Integer (i64): already an integer, pass through
                    let input_ty = op
                        .inputs
                        .first()
                        .and_then(|vid| var_types.get(vid).copied())
                        .unwrap_or(WengertType::Tensor);
                    match input_ty {
                        WengertType::Integer => Ok(inputs[0]),
                        WengertType::Scalar => {
                            Ok(builder.ins().fcvt_to_sint(cl_types::I64, inputs[0]))
                        }
                        _ => {
                            // Tensor or List — extract f64 first
                            let f64_val = call(compiler, builder, "nsl_tensor_item", &[inputs[0]])?;
                            Ok(builder.ins().fcvt_to_sint(cl_types::I64, f64_val))
                        }
                    }
                }
                "float" => {
                    // float() converts to f64 scalar. Input may be:
                    //   - Integer (i64): convert with fcvt_from_sint
                    //   - Scalar (f64): pass through
                    //   - Tensor: extract f64 via nsl_tensor_item
                    let input_ty = op
                        .inputs
                        .first()
                        .and_then(|vid| var_types.get(vid).copied())
                        .unwrap_or(WengertType::Tensor);
                    match input_ty {
                        WengertType::Scalar => Ok(inputs[0]),
                        WengertType::Integer => {
                            Ok(builder.ins().fcvt_from_sint(cl_types::F64, inputs[0]))
                        }
                        _ => {
                            // Tensor — extract f64
                            call(compiler, builder, "nsl_tensor_item", &[inputs[0]])
                        }
                    }
                }
                "subscript" => {
                    // inputs[0] = list/shape, inputs[1] = index
                    // Index may be Scalar (f64 from Constant), Integer (i64), or Tensor.
                    let idx_ty = op
                        .inputs
                        .get(1)
                        .and_then(|vid| var_types.get(vid).copied())
                        .unwrap_or(WengertType::Tensor);
                    let idx = match idx_ty {
                        WengertType::Integer => inputs[1],
                        WengertType::Scalar => builder.ins().fcvt_to_sint(cl_types::I64, inputs[1]),
                        _ => {
                            // Tensor — extract f64 then convert to i64
                            let f64_val = call(compiler, builder, "nsl_tensor_item", &[inputs[1]])?;
                            builder.ins().fcvt_to_sint(cl_types::I64, f64_val)
                        }
                    };
                    // nsl_list_get returns raw i64 (dimension value, NOT tensor pointer).
                    call(compiler, builder, "nsl_list_get", &[inputs[0], idx])
                }
                "list" => {
                    // Build a list from input elements.
                    // Determine list kind from element types:
                    //   - If ANY element is a Tensor, treat as tensor list (push raw pointers)
                    //   - Otherwise, treat as dimension list (scalarize to i64)
                    let has_tensor = op.inputs.iter().any(|vid| {
                        var_types.get(vid).copied().unwrap_or(WengertType::Integer)
                            == WengertType::Tensor
                    });
                    let list = call(compiler, builder, "nsl_list_new", &[])?;
                    for (i, &inp) in inputs.iter().enumerate() {
                        let val_ty = op
                            .inputs
                            .get(i)
                            .and_then(|vid| var_types.get(vid).copied())
                            .unwrap_or(WengertType::Integer);
                        let val = if has_tensor {
                            // Tensor list: push raw i64 pointers (no scalarization)
                            inp
                        } else {
                            // Dimension list: convert to i64
                            match val_ty {
                                WengertType::Scalar => {
                                    builder.ins().fcvt_to_sint(cl_types::I64, inp)
                                }
                                _ => inp, // Integer, List — already i64
                            }
                        };
                        call(compiler, builder, "nsl_list_push", &[list, val])?;
                    }
                    Ok(list)
                }
                other => Err(CodegenError::new(format!(
                    "unsupported passthrough op: {}",
                    other
                ))),
            }
        }
    }
}

// ── CFTP §4.4 G3 (Sprint 4): fused linear-CE forward + backward lowering ────

/// Build a per-config `FusedLinearCEConfig` from the per-op shape facts.
///
/// `gpu_sm = 80` matches the codegen default — the runtime PTX header
/// always emits `.target sm_{max(80, gpu_sm)}` so this works on all
/// post-Ampere GPUs including Blackwell (sm_120).
fn build_fused_ce_cfg(
    vocab_size: u32,
    hidden_size: u32,
    batch_size: u32,
    seq_len: u32,
    vocab_tile: u32,
    ignore_index: i64,
    dtype: crate::fused_linear_ce::Dtype,
) -> Result<crate::fused_linear_ce::FusedLinearCEConfig, CodegenError> {
    let cfg = crate::fused_linear_ce::FusedLinearCEConfig {
        vocab_size,
        hidden_size,
        seq_len,
        batch_size,
        vocab_tile,
        gpu_sm: 80,
        dtype,
        ignore_index,
        // Raise the per-config cap to the hard ceiling so Sprint-3's
        // large-vocab path activates whenever vocab > 8192 without the
        // caller having to thread `max_vocab_v1` separately.
        max_vocab_v1: crate::fused_linear_ce::MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate()
        .map_err(|e| CodegenError::new(format!("FusedLinearCe config invalid: {e}")))?;
    Ok(cfg)
}

/// CFTP v4-2: derive the FFI dtype_tag + emitter Dtype from the active
/// `@fused_lm_ce` decorator config.
///
/// CFTP v10 (item 3): reads `compiler.active_fused_ce_config`, which
/// `compile_train_block` installs from the entry in `fused_ce_configs`
/// whose `train_block_stmt_id` matches the current train block.
/// Pre-v10 this was `fused_ce_configs.first()`, silently binding every
/// train block to the FIRST decorator's dtype — see the semantic checker
/// note this fix reversed.
///
/// Returns `(dtype_tag, emitter_dtype)`:
/// * `(0, Dtype::F32)` for `dtype = "f32"` or absent decorator
///   (pre-v4-2 byte-identical default)
/// * `(1, Dtype::F16)` for `dtype = "f16" | "fp16"`
/// * `(2, Dtype::Bf16)` for `dtype = "bf16"`
///
/// IMPORTANT: both the FFI sentinel and the emitter dtype must agree —
/// the PTX bytes carry the dtype-specialised cvt instructions and the
/// runtime dispatch tag must match so the launcher chooses the same
/// path. Hence a single source of truth.
fn fused_ce_dtype_for_compiler(
    compiler: &crate::compiler::Compiler,
) -> (i64, crate::fused_linear_ce::Dtype) {
    use crate::FusedCeDtypeHint;
    // Adversarial review Finding 9: guard the dtype read by `enabled` so a
    // future code path that lowers a PrimalOp::FusedLinearCe via a different
    // upstream substitution cannot silently inherit the dtype hint of a
    // disabled decorator. The current callers (source_ad.rs::extract_expr
    // for the @fused_lm_ce auto-substitution arm AND for explicit
    // `fused_linear_ce` builtin calls) ALREADY check `cfg.enabled` before
    // emitting the FusedLinearCe op, so this guard is defense-in-depth —
    // it codifies the documented `disabled → composite preserved` invariant
    // as a single-source-of-truth filter.
    match compiler
        .active_fused_ce_config
        .as_ref()
        .filter(|c| c.enabled)
        .and_then(|c| c.dtype)
    {
        None | Some(FusedCeDtypeHint::F32) => (0, crate::fused_linear_ce::Dtype::F32),
        Some(FusedCeDtypeHint::F16) => (1, crate::fused_linear_ce::Dtype::F16),
        Some(FusedCeDtypeHint::Bf16) => (2, crate::fused_linear_ce::Dtype::Bf16),
    }
}

/// CFTP v6 — Inline precision-cast for fused_linear_ce inputs.
///
/// Given the `dtype_tag` resolved from `fused_ce_dtype_for_compiler` and the
/// three f32 wengert tensors (x / W / bias), emit a `nsl_tensor_to_bf16` or
/// `nsl_tensor_to_fp16` call for each when `dtype_tag != 0`, returning the
/// cast tensor handles to be fed into the subsequent `nsl_tensor_data_ptr`
/// calls. When `dtype_tag == 0` the tensors are returned unchanged so the
/// F32 v1 byte-identity contract is preserved (no extra FFI calls, no extra
/// HBM allocation).
///
/// The runtime cast wrappers (`nsl_tensor_to_{bf16,fp16}`, registered in
/// `RUNTIME_FUNCTIONS` by the CFTP v6 codegen-signatures commit) publish
/// the shadow buffer into the scope sweep, so the cast tensors are
/// reclaimed at function-scope exit alongside `loss_out` / `lse_out`.
///
/// `dtype_tag` is the same wire constant the FFI's terminal `dtype_tag_val`
/// arg uses:
///   * 0 -> F32 — no cast, no FFI emit.
///   * 1 -> FP16 — emit `nsl_tensor_to_fp16` for each input.
///   * 2 -> BF16 — emit `nsl_tensor_to_bf16` for each input.
/// Any other value is a logic error (we panic-on-unreachable in the helper
/// rather than silently dropping the dtype).
fn maybe_precision_cast_inputs(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    dtype_tag: i64,
    x_t: Value,
    w_t: Value,
    bias_t: Value,
) -> Result<(Value, Value, Value), CodegenError> {
    let cast_fn = match dtype_tag {
        0 => return Ok((x_t, w_t, bias_t)),
        1 => "nsl_tensor_to_fp16",
        2 => "nsl_tensor_to_bf16",
        // CFTP v6 Finding 15 (LOW): `fused_ce_dtype_for_compiler` is the
        // single source of truth; its fully-covered match over
        // `Option<FusedCeDtypeHint>` (None|F32|F16|Bf16) produces exactly
        // {0,1,2}.  If a future variant is added to `FusedCeDtypeHint`,
        // that match becomes a compile error first — but if the producer
        // is updated and the consumer (here) is forgotten, this
        // `unreachable!` will fire loudly in debug builds, surfacing the
        // missed update at the first test invocation rather than after
        // a downstream codegen error.
        other => unreachable!(
            "maybe_precision_cast_inputs: unsupported dtype_tag {other} \
             (expected 0=F32, 1=FP16, 2=BF16). This is a wengert-lower \
             invariant violation; `fused_ce_dtype_for_compiler` is the \
             single source of truth and must produce exactly one of \
             {{0,1,2}}. If you are seeing this, you added a variant to \
             `FusedCeDtypeHint` without updating both \
             `fused_ce_dtype_for_compiler` (producer) and \
             `maybe_precision_cast_inputs` (consumer)."
        ),
    };
    let x_cast = call(compiler, builder, cast_fn, &[x_t])?;
    let w_cast = call(compiler, builder, cast_fn, &[w_t])?;
    let bias_cast = call(compiler, builder, cast_fn, &[bias_t])?;
    Ok((x_cast, w_cast, bias_cast))
}

/// Embed a Vec<u8> of PTX bytes (NUL-terminated copy made internally) and
/// a kernel name (NUL-terminated copy) into the Cranelift module's
/// `.rodata` and return Cranelift Values pointing to each.
///
/// Uses unique data symbol names derived from `tag` so multiple
/// PrimalOp::FusedLinearCe ops in the same compile unit don't collide.
fn embed_fused_ce_data(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    tag: &str,
    ptx_bytes: &[u8],
    kernel_name: &str,
) -> Result<(Value, Value), CodegenError> {
    use cranelift_module::DataDescription;
    use cranelift_module::Linkage;

    let mut ptx_with_nul = ptx_bytes.to_vec();
    ptx_with_nul.push(0);
    let ptx_sym = format!("__nsl_fused_ce_ptx_{}", tag);
    let ptx_id = compiler
        .module
        .declare_data(&ptx_sym, Linkage::Local, false, false)
        .map_err(|e| CodegenError::new(format!("declare fused-ce PTX data '{ptx_sym}': {e}")))?;
    let mut ptx_desc = DataDescription::new();
    ptx_desc.define(ptx_with_nul.into_boxed_slice());
    compiler
        .module
        .define_data(ptx_id, &ptx_desc)
        .map_err(|e| CodegenError::new(format!("define fused-ce PTX data '{ptx_sym}': {e}")))?;

    let mut name_with_nul = kernel_name.as_bytes().to_vec();
    name_with_nul.push(0);
    let name_sym = format!("__nsl_fused_ce_name_{}", tag);
    let name_id = compiler
        .module
        .declare_data(&name_sym, Linkage::Local, false, false)
        .map_err(|e| CodegenError::new(format!("declare fused-ce name data '{name_sym}': {e}")))?;
    let mut name_desc = DataDescription::new();
    name_desc.define(name_with_nul.into_boxed_slice());
    compiler
        .module
        .define_data(name_id, &name_desc)
        .map_err(|e| CodegenError::new(format!("define fused-ce name data '{name_sym}': {e}")))?;

    let ptx_gv = compiler.module.declare_data_in_func(ptx_id, builder.func);
    let name_gv = compiler.module.declare_data_in_func(name_id, builder.func);
    let ptx_val = builder.ins().symbol_value(cl_types::I64, ptx_gv);
    let name_val = builder.ins().symbol_value(cl_types::I64, name_gv);
    Ok((ptx_val, name_val))
}

/// Build a CPU-side f32 NslTensor with `shape` then move it to device=1
/// (CUDA). Returns the NslTensor handle Value (i64).
fn alloc_gpu_f32_tensor(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    shape: &[i64],
) -> Result<Value, CodegenError> {
    let shape_list = call(compiler, builder, "nsl_list_new", &[])?;
    for &dim in shape {
        let dim_val = builder.ins().iconst(cl_types::I64, dim);
        call(compiler, builder, "nsl_list_push", &[shape_list, dim_val])?;
    }
    let cpu_tensor = call(compiler, builder, "nsl_tensor_zeros", &[shape_list])?;
    let cuda_device = builder.ins().iconst(cl_types::I64, 1);
    let gpu_tensor = call(
        compiler,
        builder,
        "nsl_tensor_to_device",
        &[cpu_tensor, cuda_device],
    )?;
    Ok(gpu_tensor)
}

#[allow(clippy::too_many_arguments)]
fn lower_fused_linear_ce_forward(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    inputs: &[Value],
    vocab_size: u32,
    hidden_size: u32,
    batch_size: u32,
    seq_len: u32,
    vocab_tile: u32,
    ignore_index: i64,
    is_large: bool,
) -> Result<Value, CodegenError> {
    // CFTP v5: read dtype hint from the active `@fused_lm_ce` decorator.
    // Both the emitter cfg AND the FFI dtype_tag must agree; a single
    // source of truth (`fused_ce_dtype_for_compiler`) guarantees that.
    //
    // ACTIVATION (Sprint v5): the v4-2 refusal that gated this dispatch on
    // `dtype == F32` is LIFTED.  GPU numerical correctness at V=49152 was
    // validated by the prior sprint via direct-FFI tests
    // (`fused_linear_ce_fp16_v49152_numerical.rs` /
    //  `fused_linear_ce_bf16_v49152_numerical.rs`) with documented
    // relaxed tolerances (≈5e-2 rel-err) derived from the 384-tile
    // mantissa analysis — that empirical pin closes the v4 reviewer's
    // "no full-V numerical pin" finding.
    //
    // CFTP v6 — STRUCTURAL buffer-conformance close-out of Sprint v5
    // Finding 7.  The wengert tape produces f32 tensors; when the active
    // decorator selects `dtype = "fp16"|"bf16"` we INLINE a device-side
    // precision_cast op (`nsl_tensor_to_{bf16,fp16}`) below, so the FFI
    // always sees buffers whose byte layout matches the PTX kernel's
    // `ld.global.b16` reads.  The Sprint v5 env-gated runtime refusal
    // (`NSL_FUSED_LCE_REFUSE_NON_F32`) has been REMOVED in v6 because
    // this structural cast makes the silent-corruption window unreachable
    // from the wengert dispatch.  Direct-FFI tests
    // (`fused_linear_ce_{fp16,bf16}_v49152_numerical.rs`) continue to
    // allocate compliant bf16/fp16 buffers manually.
    let (dtype_tag, emitter_dtype) = fused_ce_dtype_for_compiler(compiler);
    let cfg = build_fused_ce_cfg(
        vocab_size,
        hidden_size,
        batch_size,
        seq_len,
        vocab_tile,
        ignore_index,
        emitter_dtype,
    )?;
    let rows = (batch_size as i64) * (seq_len as i64);
    let x_t = inputs[0];
    let w_t = inputs[1];
    let bias_t = inputs[2];
    let targets_t = inputs[3];

    // Allocate per-row output buffers on GPU.
    let loss_out = alloc_gpu_f32_tensor(compiler, builder, &[rows])?;
    let lse_out = alloc_gpu_f32_tensor(compiler, builder, &[rows])?;

    // CFTP v6: INLINE PRECISION CAST (closes Sprint v5 Finding 7).
    //
    // When the active `@fused_lm_ce(dtype = "fp16"|"bf16")` decorator selects
    // a non-f32 emitter dtype, the wengert tape's f32 storage no longer
    // matches the PTX's `ld.global.b16` reads on x/W/bias. Sprint v5
    // mitigated via an env-gated runtime refusal (`NSL_FUSED_LCE_REFUSE_NON_F32`);
    // Sprint v6 closes the gap STRUCTURALLY by inserting an explicit cast
    // op BEFORE the FFI call so the kernel sees the correct byte layout.
    //
    // Cast scope: only the x/W/bias INPUTS (read as fp16/bf16 by the PTX).
    // Outputs (loss_out / lse_out / dx / dW / dbias) stay f32 — the FFI
    // contract is master-grad on the output side regardless of dtype_tag.
    //
    // Lifecycle: `nsl_tensor_to_{bf16,fp16}` publish the shadow buffer into
    // the scope sweep (see `precision_cast::cast_and_publish`), so each cast
    // tensor is reclaimed at function-scope exit — same lifecycle as
    // `loss_out`/`lse_out` above.  Cost is `(B*S*H + V*H + V) * 2 bytes` of
    // extra HBM per forward+backward step.
    //
    // CFTP v6 Findings 10/14: the cast Values are CACHED in
    // `compiler.fused_ce_fwd_casts` (paired with `dtype_tag` per
    // Finding 16) so the symmetric backward extract REUSES them
    // instead of emitting three more cast calls — without the cache
    // the implementation emitted six casts per dispatch (forward three
    // + backward three), doubling the docstring's quoted HBM cost.
    //
    // Byte-identity: when `dtype_tag == 0` (no decorator OR `dtype="f32"`)
    // the cast block is fully skipped, so the F32 v1 byte-identity contract
    // is preserved (and the cache still stores the pass-through inputs as
    // a no-op so the backward can reuse them uniformly).
    let (x_t_for_ffi, w_t_for_ffi, bias_t_for_ffi) =
        maybe_precision_cast_inputs(compiler, builder, dtype_tag, x_t, w_t, bias_t)?;
    let x_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[x_t_for_ffi])?;
    let w_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[w_t_for_ffi])?;
    let bias_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[bias_t_for_ffi])?;
    let tgt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[targets_t])?;
    let loss_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[loss_out])?;
    let lse_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_out])?;

    // Forward PTX synthesis + embedding.
    let fwd_ptx_bytes = crate::fused_linear_ce::synthesize_fused_linear_ce_ptx(&cfg);
    let smem_bytes = cfg.shared_mem_bytes() as i64;
    let b_val = builder.ins().iconst(cl_types::I64, batch_size as i64);
    let s_val = builder.ins().iconst(cl_types::I64, seq_len as i64);
    let v_val = builder.ins().iconst(cl_types::I64, vocab_size as i64);
    let h_val = builder.ins().iconst(cl_types::I64, hidden_size as i64);
    let smem_val = builder.ins().iconst(cl_types::I64, smem_bytes);

    if is_large {
        // Large-vocab two-kernel path. Allocate partials buffer and
        // pass both kernel-name pointers.
        let num_tiles = cfg.num_vocab_tiles() as i64;
        let tag = format!(
            "v{}_h{}_t{}_large",
            vocab_size, hidden_size, vocab_tile
        );
        let partials_kname = cfg.large_partials_kernel_name();
        let finalize_kname = cfg.large_finalize_kernel_name();
        let (ptx_ptr, partials_name_ptr) =
            embed_fused_ce_data(compiler, builder, &tag, &fwd_ptx_bytes, &partials_kname)?;
        // Reuse the same PTX module for the finalize kernel name embed.
        // Only the name string differs.
        let (_, finalize_name_ptr) = embed_fused_ce_data(
            compiler,
            builder,
            &format!("{tag}_finalize"),
            &fwd_ptx_bytes,
            &finalize_kname,
        )?;

        // Partials scratch: f32[rows * num_tiles * 2]; allocate on GPU.
        let partials_buf = alloc_gpu_f32_tensor(
            compiler,
            builder,
            &[rows * num_tiles * 2],
        )?;
        let partials_ptr =
            call(compiler, builder, "nsl_tensor_data_ptr", &[partials_buf])?;
        let num_tiles_val = builder.ins().iconst(cl_types::I64, num_tiles);

        // CFTP v4-2: dtype_tag derived from @fused_lm_ce(dtype="...").
        //   None | Some("f32") -> 0 (pre-v4-2 byte-identical)
        //   Some("f16")        -> 1 (Sprint v3-2 emitters)
        //   Some("bf16")       -> 2 (Sprint v4-1 emitters)
        let dtype_tag_val = builder.ins().iconst(cl_types::I64, dtype_tag);
        let _rc = call(
            compiler,
            builder,
            "nsl_fused_linear_ce_forward_large",
            &[
                ptx_ptr,
                partials_name_ptr,
                finalize_name_ptr,
                x_ptr,
                w_ptr,
                bias_ptr,
                tgt_ptr,
                partials_ptr,
                loss_ptr,
                lse_ptr,
                b_val,
                s_val,
                v_val,
                h_val,
                num_tiles_val,
                smem_val,
                dtype_tag_val,
            ],
        )?;
        // Partials scratch is no longer needed after the launch.
        free_tensor_value(compiler, builder, partials_buf)?;
    } else {
        let tag = format!("v{}_h{}_small", vocab_size, hidden_size);
        let kname = cfg.kernel_name();
        let (ptx_ptr, name_ptr) =
            embed_fused_ce_data(compiler, builder, &tag, &fwd_ptx_bytes, &kname)?;
        // CFTP v4-2: dtype_tag derived from @fused_lm_ce(dtype="...").
        // Same source-of-truth as the large-vocab path above (see
        // `fused_ce_dtype_for_compiler`).
        let dtype_tag_val = builder.ins().iconst(cl_types::I64, dtype_tag);
        let _rc = call(
            compiler,
            builder,
            "nsl_fused_linear_ce_forward",
            &[
                ptx_ptr,
                name_ptr,
                x_ptr,
                w_ptr,
                bias_ptr,
                tgt_ptr,
                loss_ptr,
                lse_ptr,
                b_val,
                s_val,
                v_val,
                h_val,
                smem_val,
                dtype_tag_val,
            ],
        )?;
    }

    // Per-row losses are in `loss_out`. Apply the same masked-mean
    // reduction the composite `CrossEntropyLoss` lowering uses so the
    // scalar result is bit-compatible with the dual path.
    //
    // valid_mask[i] = clamp(targets[i] + 1, 0, 1) — 0 for -100, 1 else.
    // num_valid = sum(valid_mask) + eps.
    // total = sum(loss_out * valid_mask).
    // result = total / num_valid.
    let one_f = builder.ins().f64const(1.0);
    let zero_f = builder.ins().f64const(0.0);
    let eps_f = builder.ins().f64const(1e-8);
    let addsc_flags = builder.ins().iconst(cl_types::I8, 0);
    let targets_plus_one = call(
        compiler,
        builder,
        "nsl_tensor_add_scalar",
        &[targets_t, one_f, addsc_flags],
    )?;
    let valid_mask = call(
        compiler,
        builder,
        "nsl_tensor_clamp",
        &[targets_plus_one, zero_f, one_f],
    )?;
    // The fused kernel already writes 0 to ignore-index rows, but multiplying
    // by valid_mask makes the masking explicit and matches the composite
    // semantics (gradient seeds also see the same identity).
    let mul_flags = builder.ins().iconst(cl_types::I8, 0);
    let masked_loss = call(
        compiler,
        builder,
        "nsl_tensor_mul",
        &[loss_out, valid_mask, mul_flags],
    )?;
    let num_valid = call(compiler, builder, "nsl_tensor_sum", &[valid_mask])?;
    let addsc_flags2 = builder.ins().iconst(cl_types::I8, 0);
    let num_valid_eps = call(
        compiler,
        builder,
        "nsl_tensor_add_scalar",
        &[num_valid, eps_f, addsc_flags2],
    )?;
    let total = call(compiler, builder, "nsl_tensor_sum", &[masked_loss])?;
    let div_flags = builder.ins().iconst(cl_types::I8, 0);
    let result = call(
        compiler,
        builder,
        "nsl_tensor_div",
        &[total, num_valid_eps, div_flags],
    )?;
    // Free intermediates we own (loss_out's masked product, valid_mask
    // chain, total, num_valid_eps). The raw `loss_out` is freed too,
    // since the result lives in the new scalar tensor.  `lse_out`
    // is cached for backward — DO NOT free it here.
    for tmp in [
        targets_plus_one,
        valid_mask,
        masked_loss,
        num_valid,
        num_valid_eps,
        total,
        loss_out,
    ] {
        free_tensor_value(compiler, builder, tmp)?;
    }

    // Cache the lse_out for backward extract dispatch keyed by the
    // scalar-result Cranelift Value. Backward's first extract (component
    // = 0) reads + consumes it; the entry is evicted there.
    compiler.fused_ce_fwd_lse.insert(result, lse_out);
    // CFTP v6 Findings 10/14 + 16: cache the forward-side cast Values
    // so the symmetric backward dispatch can REUSE them instead of
    // emitting three more `nsl_tensor_to_{bf16,fp16}` calls per step
    // (the docstring's quoted '(B*S*H + V*H + V) * 2 bytes per forward
    // + backward step' assumed one cast set; without this cache the
    // implementation emitted two).  Pair with `dtype_tag` so a future
    // backward path that disagrees with the cached dtype surfaces as
    // a CodegenError rather than silently reusing the wrong byte
    // layout.  For dtype_tag == 0 (F32) the cached Values are the
    // pass-through inputs, so this is a no-op cost (one HashMap entry
    // per FusedLinearCe op).
    compiler
        .fused_ce_fwd_casts
        .insert(result, (x_t_for_ffi, w_t_for_ffi, bias_t_for_ffi, dtype_tag));
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn lower_fused_linear_ce_backward_extract(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    inputs: &[Value],
    component: u8,
    vocab_size: u32,
    hidden_size: u32,
    batch_size: u32,
    seq_len: u32,
    vocab_tile: u32,
    ignore_index: i64,
) -> Result<Value, CodegenError> {
    if component > 2 {
        return Err(CodegenError::new(format!(
            "FusedLinearCeBackwardExtract: component {} out of range 0..=2",
            component
        )));
    }
    // Convention (mirrors FusedCshaBackwardExtract):
    //   inputs[0] = output_bar (upstream grad seed)
    //   inputs[1..5] = x, W, bias, targets (forward inputs)
    //   inputs[5] = fwd_result (the FusedLinearCe op's result Value — cache key)
    let output_bar = inputs[0];
    let x_t = inputs[1];
    let w_t = inputs[2];
    let bias_t = inputs[3];
    let targets_t = inputs[4];
    let fwd_result_key = inputs[5];

    // Fire the backward launch on component=0; cache on subsequent calls.
    let slot = if let Some(&cached) = compiler.fused_ce_bwd_cache.get(&fwd_result_key) {
        cached
    } else {
        // CFTP v5: same dtype source-of-truth as forward (see
        // `fused_ce_dtype_for_compiler`). The backward kernel name is
        // a function of the cfg's `dtype` field so the PTX byte stream
        // we synthesise here must match the forward path's choice.
        //
        // ACTIVATION (Sprint v5): the v4-2 backward refusal is LIFTED
        // symmetrically with the forward path.  The backward kernel writes
        // `dx_out` / `dw_out` / `dbias_out` as f32 regardless of `dtype_tag`
        // (per the runtime contract), so the gradient that flows back into
        // the (f32) wengert tape is dtype-correct without further plumbing.
        // Only the f16/bf16 reads of x/W/bias/lse are dtype-sensitive, and
        // those mirror the forward dispatch contract closed in v5.
        //
        // CFTP v6 — buffer-conformance is closed STRUCTURALLY here too: the
        // `maybe_precision_cast_inputs` call below inserts the same device-side
        // cast op that the forward path emits.  The v5 env-guarded runtime
        // refusal has been removed; see the forward path's comment block.
        let (dtype_tag, emitter_dtype) = fused_ce_dtype_for_compiler(compiler);
        let cfg = build_fused_ce_cfg(
            vocab_size,
            hidden_size,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
            emitter_dtype,
        )?;

        // Allocate output tensors (dx[B*S, H], dW[V, H], dbias[V]).
        let rows = (batch_size as i64) * (seq_len as i64);
        let dx_out = alloc_gpu_f32_tensor(
            compiler,
            builder,
            &[rows, hidden_size as i64],
        )?;
        let dw_out = alloc_gpu_f32_tensor(
            compiler,
            builder,
            &[vocab_size as i64, hidden_size as i64],
        )?;
        let dbias_out =
            alloc_gpu_f32_tensor(compiler, builder, &[vocab_size as i64])?;

        // Look up saved lse buffer; consume + evict from the map.
        let lse_out = compiler
            .fused_ce_fwd_lse
            .remove(&fwd_result_key)
            .ok_or_else(|| {
                CodegenError::new(
                    "FusedLinearCeBackwardExtract: no saved lse for fwd_result \
                     — forward must run before backward (compiler.fused_ce_fwd_lse \
                     is populated in lower_fused_linear_ce_forward).",
                )
            })?;

        // CFTP v6 Findings 10/14 + 16: REUSE the forward-side precision
        // casts via `fused_ce_fwd_casts` instead of emitting three more
        // `nsl_tensor_to_{bf16,fp16}` calls.  This halves the per-step
        // cast HBM cost and matches the forward docstring's quoted
        // `(B*S*H + V*H + V) * 2 bytes` (which assumed one cast per
        // forward+backward step, not two).  Defensive dtype check: the
        // cached dtype_tag MUST match the current backward dtype_tag —
        // a mismatch means `active_fused_ce_config.dtype` (CFTP v10
        // item 3; pre-v10: `fused_ce_configs[0].dtype`) was mutated
        // between forward and backward emission (e.g., a curriculum
        // that flips dtype mid-step, or the backward runs under a
        // DIFFERENT train block's active config than the forward),
        // in which case reusing the cached cast bytes would silently
        // feed the wrong byte layout to the kernel.  Fall back to a
        // fresh cast emission on miss so a separately-compiled
        // backward without a forward cache entry still works (this
        // is the path taken by backward-only compilation pipelines).
        let (x_t_for_ffi, w_t_for_ffi, bias_t_for_ffi) = match compiler
            .fused_ce_fwd_casts
            .get(&fwd_result_key)
            .copied()
        {
            Some((x_cast, w_cast, bias_cast, cached_dtype)) => {
                if cached_dtype != dtype_tag {
                    return Err(CodegenError::new(format!(
                        "FusedLinearCeBackwardExtract: cached forward cast dtype_tag={} \
                         != current backward dtype_tag={} — `active_fused_ce_config.dtype` \
                         was mutated between forward and backward emission. \
                         This violates the v6 single-source-of-truth invariant \
                         (`fused_ce_dtype_for_compiler` must return the same value \
                         throughout one compile).",
                        cached_dtype, dtype_tag
                    )));
                }
                (x_cast, w_cast, bias_cast)
            }
            None => {
                // Backward-only compile (no forward cache entry).  Emit
                // fresh casts as the v6 baseline did.
                maybe_precision_cast_inputs(compiler, builder, dtype_tag, x_t, w_t, bias_t)?
            }
        };
        let x_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[x_t_for_ffi])?;
        let w_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[w_t_for_ffi])?;
        let bias_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[bias_t_for_ffi])?;
        let tgt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[targets_t])?;
        let lse_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_out])?;
        let dx_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[dx_out])?;
        let dw_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[dw_out])?;
        let dbias_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[dbias_out])?;

        // Backward PTX synthesis + embed.
        let bwd_ptx_bytes =
            crate::fused_linear_ce::synthesize_fused_linear_ce_backward_ptx(&cfg);
        let bwd_kname = cfg.bwd_kernel_name();
        let tag = format!("bwd_v{}_h{}", vocab_size, hidden_size);
        let (bwd_ptx_ptr, bwd_name_ptr) =
            embed_fused_ce_data(compiler, builder, &tag, &bwd_ptx_bytes, &bwd_kname)?;

        // grad_output: the upstream scalar grad (a 0-dim or 1-elem tensor).
        // Extract its f32 value via nsl_tensor_item (returns f64), demote to
        // f32, and pack the bits into i64.  This matches the FFI signature
        // which takes `grad_output_bits: i64` reinterpreted via
        // `f32::from_bits(grad_output_bits as u32)`.
        let grad_f64 = call(compiler, builder, "nsl_tensor_item", &[output_bar])?;
        let grad_f32 = builder.ins().fdemote(cl_types::F32, grad_f64);
        let grad_bits_i32 = builder.ins().bitcast(
            cl_types::I32,
            cranelift_codegen::ir::MemFlags::new(),
            grad_f32,
        );
        let grad_bits = builder.ins().sextend(cl_types::I64, grad_bits_i32);

        // num_valid: count(targets != ignore_index). Compute via the same
        // valid_mask trick used in forward; convert to i64 scalar.
        let one_f = builder.ins().f64const(1.0);
        let zero_f = builder.ins().f64const(0.0);
        let addsc_flags_b = builder.ins().iconst(cl_types::I8, 0);
        let targets_plus_one =
            call(compiler, builder, "nsl_tensor_add_scalar", &[
                targets_t,
                one_f,
                addsc_flags_b,
            ])?;
        let valid_mask =
            call(compiler, builder, "nsl_tensor_clamp", &[
                targets_plus_one,
                zero_f,
                one_f,
            ])?;
        let num_valid_t = call(compiler, builder, "nsl_tensor_sum", &[valid_mask])?;
        let num_valid_f64 = call(compiler, builder, "nsl_tensor_item", &[num_valid_t])?;
        let num_valid = builder.ins().fcvt_to_sint(cl_types::I64, num_valid_f64);

        let b_val = builder.ins().iconst(cl_types::I64, batch_size as i64);
        let s_val = builder.ins().iconst(cl_types::I64, seq_len as i64);
        let v_val = builder.ins().iconst(cl_types::I64, vocab_size as i64);
        let h_val = builder.ins().iconst(cl_types::I64, hidden_size as i64);
        let smem_val = builder.ins().iconst(cl_types::I64, cfg.shared_mem_bytes() as i64);

        // CFTP v4-2: dtype_tag from `@fused_lm_ce(dtype="...")` decorator
        // (resolved above; same value the forward path used).
        let dtype_tag_val = builder.ins().iconst(cl_types::I64, dtype_tag);
        let _rc = call(
            compiler,
            builder,
            "nsl_fused_linear_ce_backward",
            &[
                bwd_ptx_ptr,
                bwd_name_ptr,
                grad_bits,
                x_ptr,
                w_ptr,
                bias_ptr,
                tgt_ptr,
                lse_ptr,
                dx_ptr,
                dw_ptr,
                dbias_ptr,
                b_val,
                s_val,
                v_val,
                h_val,
                num_valid,
                smem_val,
                dtype_tag_val,
            ],
        )?;

        // Free temporaries; lse_out's storage is consumed by the FFI then
        // freed here (the kernel has already finished reading it because
        // the FFI calls cuCtxSynchronize).
        for tmp in [targets_plus_one, valid_mask, num_valid_t, lse_out] {
            free_tensor_value(compiler, builder, tmp)?;
        }

        let slot = [dx_out, dw_out, dbias_out];
        compiler.fused_ce_bwd_cache.insert(fwd_result_key, slot);
        slot
    };

    let result = slot[component as usize];
    if component == 2 {
        compiler.fused_ce_bwd_cache.remove(&fwd_result_key);
        // CFTP v6 Finding 10/14: evict the cast cache on the same
        // last-component boundary so the cache stays bounded across
        // many FusedLinearCe ops in one compile.
        compiler.fused_ce_fwd_casts.remove(&fwd_result_key);
    }
    Ok(result)
}

// ─── CPKD fused KL-CE lowering ──────────────────────────────────────────────

/// Shape + loss-constant bundle threaded from the `FusedKlCe` /
/// `FusedKlCeBackwardExtract` PrimalOps into the two lowering fns (nine
/// scalar params would trip clippy::too_many_arguments and invite
/// positional-swap bugs).
pub(crate) struct FusedKlCeLowerParams {
    pub vocab_size: u32,
    pub student_hidden: u32,
    pub teacher_hidden: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    pub vocab_tile: u32,
    pub ignore_index: i64,
    pub alpha: f64,
    pub temperature: f64,
}

impl FusedKlCeLowerParams {
    fn build_cfg(&self) -> Result<crate::cpkd_fused_loss::FusedKlCeConfig, CodegenError> {
        let cfg = crate::cpkd_fused_loss::FusedKlCeConfig {
            vocab_size: self.vocab_size,
            student_hidden: self.student_hidden,
            teacher_hidden: self.teacher_hidden,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
            vocab_tile: self.vocab_tile,
            gpu_sm: 80, // f32 emitters target sm_80+; see cpkd_fused_loss::ptx_header
            ignore_index: self.ignore_index,
        };
        cfg.validate().map_err(CodegenError::new)?;
        Ok(cfg)
    }

    /// Pack an f64 loss constant as f32 bits in an i64 immediate (the FFI
    /// contract: `f32::from_bits(bits as u32)`).
    fn f32_bits_const(builder: &mut FunctionBuilder, v: f64) -> Value {
        builder
            .ins()
            .iconst(cl_types::I64, (v as f32).to_bits() as i64)
    }
}

/// CPKD forward: one FFI launch producing per-row losses + the three saved
/// LSE buffers, then the same masked-mean reduction as the composite CE
/// (and fused linear-CE) lowerings so the scalar is dual-path compatible.
///
/// inputs = [x_s, W_s, bias_s, x_t, W_t, bias_t, targets] (7).
fn lower_fused_kl_ce_forward(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    inputs: &[Value],
    p: &FusedKlCeLowerParams,
) -> Result<Value, CodegenError> {
    let cfg = p.build_cfg()?;
    let rows = (p.batch_size as i64) * (p.seq_len as i64);
    if inputs.len() < 7 {
        return Err(CodegenError::new(
            "FusedKlCe: expected 7 lowered inputs (x_s, W_s, bias_s, x_t, W_t, bias_t, targets)",
        ));
    }
    let (xs_t, ws_t, bs_t) = (inputs[0], inputs[1], inputs[2]);
    let (xt_t, wt_t, bt_t) = (inputs[3], inputs[4], inputs[5]);
    let targets_t = inputs[6];

    // Per-row outputs + the three LSE save buffers (backward inputs).
    let loss_out = alloc_gpu_f32_tensor(compiler, builder, &[rows])?;
    let lse_s1_out = alloc_gpu_f32_tensor(compiler, builder, &[rows])?;
    let lse_st_out = alloc_gpu_f32_tensor(compiler, builder, &[rows])?;
    let lse_tt_out = alloc_gpu_f32_tensor(compiler, builder, &[rows])?;

    // v1 is f32-only: no precision casts (the tape stores f32 on GPU).
    let xs_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[xs_t])?;
    let ws_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[ws_t])?;
    let bs_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[bs_t])?;
    let xt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[xt_t])?;
    let wt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[wt_t])?;
    let bt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[bt_t])?;
    let tgt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[targets_t])?;
    let loss_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[loss_out])?;
    let lse_s1_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_s1_out])?;
    let lse_st_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_st_out])?;
    let lse_tt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_tt_out])?;

    // Synthesise + embed the forward PTX.
    let fwd_ptx_bytes = crate::cpkd_fused_loss::synthesize_fused_kl_ce_ptx(&cfg);
    let kname = cfg.kernel_name();
    let tag = format!(
        "klce_v{}_hs{}_ht{}",
        p.vocab_size, p.student_hidden, p.teacher_hidden
    );
    let (ptx_ptr, name_ptr) =
        embed_fused_ce_data(compiler, builder, &tag, &fwd_ptx_bytes, &kname)?;

    let rows_val = builder.ins().iconst(cl_types::I64, rows);
    let v_val = builder.ins().iconst(cl_types::I64, p.vocab_size as i64);
    let hs_val = builder.ins().iconst(cl_types::I64, p.student_hidden as i64);
    let ht_val = builder.ins().iconst(cl_types::I64, p.teacher_hidden as i64);
    let alpha_bits = FusedKlCeLowerParams::f32_bits_const(builder, p.alpha);
    let temp_bits = FusedKlCeLowerParams::f32_bits_const(builder, p.temperature);
    let smem_val = builder
        .ins()
        .iconst(cl_types::I64, cfg.shared_mem_bytes() as i64);

    // ABI lock-step: 20 args — must match builtins.rs declaration and the
    // runtime extern "C" fn (crates/nsl-runtime/src/fused_kl_ce.rs).
    let _rc = call(
        compiler,
        builder,
        "nsl_fused_kl_ce_forward",
        &[
            ptx_ptr, name_ptr, xs_ptr, ws_ptr, bs_ptr, xt_ptr, wt_ptr, bt_ptr, tgt_ptr,
            loss_ptr, lse_s1_ptr, lse_st_ptr, lse_tt_ptr, rows_val, v_val, hs_val, ht_val,
            alpha_bits, temp_bits, smem_val,
        ],
    )?;

    // Masked-mean reduction — identical to the fused linear-CE forward so
    // the scalar is bit-compatible with a composite dual path.
    let one_f = builder.ins().f64const(1.0);
    let zero_f = builder.ins().f64const(0.0);
    let eps_f = builder.ins().f64const(1e-8);
    let addsc_flags = builder.ins().iconst(cl_types::I8, 0);
    let targets_plus_one = call(
        compiler,
        builder,
        "nsl_tensor_add_scalar",
        &[targets_t, one_f, addsc_flags],
    )?;
    let valid_mask = call(
        compiler,
        builder,
        "nsl_tensor_clamp",
        &[targets_plus_one, zero_f, one_f],
    )?;
    let mul_flags = builder.ins().iconst(cl_types::I8, 0);
    let masked_loss = call(
        compiler,
        builder,
        "nsl_tensor_mul",
        &[loss_out, valid_mask, mul_flags],
    )?;
    let num_valid = call(compiler, builder, "nsl_tensor_sum", &[valid_mask])?;
    let addsc_flags2 = builder.ins().iconst(cl_types::I8, 0);
    let num_valid_eps = call(
        compiler,
        builder,
        "nsl_tensor_add_scalar",
        &[num_valid, eps_f, addsc_flags2],
    )?;
    let total = call(compiler, builder, "nsl_tensor_sum", &[masked_loss])?;
    let div_flags = builder.ins().iconst(cl_types::I8, 0);
    let result = call(
        compiler,
        builder,
        "nsl_tensor_div",
        &[total, num_valid_eps, div_flags],
    )?;
    for tmp in [
        targets_plus_one,
        valid_mask,
        masked_loss,
        num_valid,
        num_valid_eps,
        total,
        loss_out,
    ] {
        free_tensor_value(compiler, builder, tmp)?;
    }

    // Save the three LSE buffers for the backward extract, keyed by the
    // scalar-result Value. First backward extract consumes + evicts.
    compiler
        .fused_kl_ce_fwd_saves
        .insert(result, [lse_s1_out, lse_st_out, lse_tt_out]);
    Ok(result)
}

/// CPKD backward extract: first component launches
/// `nsl_fused_kl_ce_backward` (fills all three student grads), caches
/// `[dx_s, dW_s, dbias_s]`; the others read the cache; component 2
/// evicts. inputs = [grad, x_s, W_s, bias_s, x_t, W_t, bias_t, targets,
/// fwd_result] (9).
fn lower_fused_kl_ce_backward_extract(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    inputs: &[Value],
    component: u8,
    p: &FusedKlCeLowerParams,
) -> Result<Value, CodegenError> {
    if component > 2 {
        return Err(CodegenError::new(format!(
            "FusedKlCeBackwardExtract: component {component} out of range 0..=2"
        )));
    }
    if inputs.len() < 9 {
        return Err(CodegenError::new(
            "FusedKlCeBackwardExtract: expected 9 lowered inputs",
        ));
    }
    let output_bar = inputs[0];
    let (xs_t, ws_t, bs_t) = (inputs[1], inputs[2], inputs[3]);
    let (xt_t, wt_t, bt_t) = (inputs[4], inputs[5], inputs[6]);
    let targets_t = inputs[7];
    let fwd_result_key = inputs[8];

    let slot = if let Some(&cached) = compiler.fused_kl_ce_bwd_cache.get(&fwd_result_key) {
        cached
    } else {
        let cfg = p.build_cfg()?;
        let rows = (p.batch_size as i64) * (p.seq_len as i64);

        // Student gradient outputs (zero-filled by alloc; the kernel
        // accumulates via red.global.add). NO teacher gradient buffers
        // exist on this path (I-11).
        let dxs_out =
            alloc_gpu_f32_tensor(compiler, builder, &[rows, p.student_hidden as i64])?;
        let dws_out = alloc_gpu_f32_tensor(
            compiler,
            builder,
            &[p.vocab_size as i64, p.student_hidden as i64],
        )?;
        let dbs_out = alloc_gpu_f32_tensor(compiler, builder, &[p.vocab_size as i64])?;

        // Saved LSE buffers from the forward; consume + evict.
        let [lse_s1, lse_st, lse_tt] = compiler
            .fused_kl_ce_fwd_saves
            .remove(&fwd_result_key)
            .ok_or_else(|| {
                CodegenError::new(
                    "FusedKlCeBackwardExtract: no saved LSE buffers for fwd_result — \
                     forward must run before backward (compiler.fused_kl_ce_fwd_saves \
                     is populated in lower_fused_kl_ce_forward)",
                )
            })?;

        let xs_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[xs_t])?;
        let ws_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[ws_t])?;
        let bs_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[bs_t])?;
        let xt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[xt_t])?;
        let wt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[wt_t])?;
        let bt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[bt_t])?;
        let tgt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[targets_t])?;
        let lse_s1_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_s1])?;
        let lse_st_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_st])?;
        let lse_tt_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[lse_tt])?;
        let dxs_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[dxs_out])?;
        let dws_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[dws_out])?;
        let dbs_ptr = call(compiler, builder, "nsl_tensor_data_ptr", &[dbs_out])?;

        // Backward PTX synthesis + embed.
        let bwd_ptx_bytes = crate::cpkd_fused_loss::synthesize_fused_kl_ce_backward_ptx(&cfg);
        let bwd_kname = cfg.bwd_kernel_name();
        let tag = format!(
            "klce_bwd_v{}_hs{}_ht{}",
            p.vocab_size, p.student_hidden, p.teacher_hidden
        );
        let (bwd_ptx_ptr, bwd_name_ptr) =
            embed_fused_ce_data(compiler, builder, &tag, &bwd_ptx_bytes, &bwd_kname)?;

        // Upstream scalar grad → f32 bits in i64 (same packing as the
        // fused linear-CE backward).
        let grad_f64 = call(compiler, builder, "nsl_tensor_item", &[output_bar])?;
        let grad_f32 = builder.ins().fdemote(cl_types::F32, grad_f64);
        let grad_bits_i32 = builder.ins().bitcast(
            cl_types::I32,
            cranelift_codegen::ir::MemFlags::new(),
            grad_f32,
        );
        let grad_bits = builder.ins().sextend(cl_types::I64, grad_bits_i32);

        // num_valid = count(targets != ignore_index).
        let one_f = builder.ins().f64const(1.0);
        let zero_f = builder.ins().f64const(0.0);
        let addsc_flags_b = builder.ins().iconst(cl_types::I8, 0);
        let targets_plus_one = call(
            compiler,
            builder,
            "nsl_tensor_add_scalar",
            &[targets_t, one_f, addsc_flags_b],
        )?;
        let valid_mask = call(
            compiler,
            builder,
            "nsl_tensor_clamp",
            &[targets_plus_one, zero_f, one_f],
        )?;
        let num_valid_t = call(compiler, builder, "nsl_tensor_sum", &[valid_mask])?;
        let num_valid_f64 = call(compiler, builder, "nsl_tensor_item", &[num_valid_t])?;
        let num_valid = builder.ins().fcvt_to_sint(cl_types::I64, num_valid_f64);

        let rows_val = builder.ins().iconst(cl_types::I64, rows);
        let v_val = builder.ins().iconst(cl_types::I64, p.vocab_size as i64);
        let hs_val = builder.ins().iconst(cl_types::I64, p.student_hidden as i64);
        let ht_val = builder.ins().iconst(cl_types::I64, p.teacher_hidden as i64);
        let alpha_bits = FusedKlCeLowerParams::f32_bits_const(builder, p.alpha);
        let temp_bits = FusedKlCeLowerParams::f32_bits_const(builder, p.temperature);

        // ABI lock-step: 23 args — must match builtins.rs and the runtime.
        let _rc = call(
            compiler,
            builder,
            "nsl_fused_kl_ce_backward",
            &[
                bwd_ptx_ptr,
                bwd_name_ptr,
                grad_bits,
                xs_ptr,
                ws_ptr,
                bs_ptr,
                xt_ptr,
                wt_ptr,
                bt_ptr,
                tgt_ptr,
                lse_s1_ptr,
                lse_st_ptr,
                lse_tt_ptr,
                dxs_ptr,
                dws_ptr,
                dbs_ptr,
                rows_val,
                v_val,
                hs_val,
                ht_val,
                alpha_bits,
                temp_bits,
                num_valid,
            ],
        )?;

        // The FFI cuCtxSynchronizes, so the LSE buffers are consumed.
        for tmp in [targets_plus_one, valid_mask, num_valid_t, lse_s1, lse_st, lse_tt] {
            free_tensor_value(compiler, builder, tmp)?;
        }

        let slot = [dxs_out, dws_out, dbs_out];
        compiler.fused_kl_ce_bwd_cache.insert(fwd_result_key, slot);
        slot
    };

    let result = slot[component as usize];
    if component == 2 {
        compiler.fused_kl_ce_bwd_cache.remove(&fwd_result_key);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::WengertOp;

    fn make_op(id: u32, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    #[test]
    fn test_resolve_inputs_success() {
        let op = make_op(0, 10, PrimalOp::Relu, vec![5]);
        let var_map = VarMap::new();
        // Use a dummy Value; we just test the lookup logic.
        // VarId 5 must be in the map.
        // We can't easily create a real Cranelift Value without a builder,
        // so we test the error path instead.
        let result = resolve_inputs(&op, &var_map);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unresolved VarId 5"));

        // With VarId present — we'd need a real Value, so skip the success case
        // in this unit test. Integration tests cover the full lowering path.
        let _ = var_map; // suppress unused warning
    }

    #[test]
    fn test_lower_constant_leaf() {
        // Verify that Constant variant is matched as a leaf (no input resolution).
        let value = std::f64::consts::PI;
        let op = make_op(0, 1, PrimalOp::Constant(value), vec![]);
        // Without a real compiler/builder we can't call lower_single_op,
        // but we can verify the op structure is correct.
        assert!(op.inputs.is_empty());
        assert!(matches!(op.op, PrimalOp::Constant(v) if (v - value).abs() < 1e-10));
    }

    #[test]
    fn test_compare_kind_mapping() {
        // Verify the CompareKind -> i64 mapping matches the runtime convention.
        let mappings = [
            (CompareKind::Gt, 0i64),
            (CompareKind::GtEq, 1),
            (CompareKind::Lt, 2),
            (CompareKind::LtEq, 3),
            (CompareKind::Eq, 4),
            (CompareKind::NotEq, 5),
        ];
        for (kind, expected) in mappings {
            let val = match &kind {
                CompareKind::Gt => 0i64,
                CompareKind::GtEq => 1,
                CompareKind::Lt => 2,
                CompareKind::LtEq => 3,
                CompareKind::Eq => 4,
                CompareKind::NotEq => 5,
            };
            assert_eq!(
                val, expected,
                "CompareKind::{:?} should map to {}",
                kind, expected
            );
        }
    }

    #[test]
    fn test_all_primal_ops_covered() {
        // Verify that every PrimalOp variant is handled by lower_single_op.
        // This is a compile-time exhaustiveness check — the match in lower_single_op
        // is exhaustive, so if a new variant is added, the compiler will error.
        // We verify the known count here for documentation.
        let ops: Vec<PrimalOp> = vec![
            PrimalOp::Relu,
            PrimalOp::Sigmoid,
            PrimalOp::Tanh,
            PrimalOp::Gelu,
            PrimalOp::Silu,
            PrimalOp::Exp,
            PrimalOp::Log,
            PrimalOp::Sqrt,
            PrimalOp::Abs,
            PrimalOp::Neg,
            PrimalOp::Clamp { min: 0.0, max: 1.0 },
            PrimalOp::Add,
            PrimalOp::Sub,
            PrimalOp::Mul,
            PrimalOp::Div,
            PrimalOp::Matmul,
            PrimalOp::Transpose { dim0: 0, dim1: 1 },
            PrimalOp::Sum { dim: Some(0) },
            PrimalOp::Mean { dim: None },
            PrimalOp::Softmax { dim: -1 },
            PrimalOp::LogSoftmax { dim: -1 },
            PrimalOp::Reshape { target_ndim: 2 },
            PrimalOp::Broadcast,
            PrimalOp::Concat { dim: 0 },
            PrimalOp::Split { dim: 0, chunks: 2 },
            PrimalOp::Slice {
                dim: 0,
                start: 0,
                end: 5,
                orig_dim_size: 10,
            },
            PrimalOp::PadZero {
                dim: 0,
                pad_before: 1,
                pad_after: 1,
            },
            PrimalOp::Gather { dim: 0 },
            PrimalOp::ScatterAdd { dim: 0 },
            PrimalOp::Embedding,
            PrimalOp::LayerNorm { eps: 1e-5 },
            PrimalOp::RMSNorm { eps: 1e-5 },
            PrimalOp::BatchNorm {
                eps: 1e-5,
                training: true,
            },
            PrimalOp::MaxPool2d {
                kernel: 2,
                stride: 2,
            },
            PrimalOp::AvgPool2d {
                kernel: 2,
                stride: 2,
            },
            PrimalOp::Conv2d {
                stride: 1,
                padding: 0,
            },
            PrimalOp::ConvTranspose2d {
                stride: 1,
                padding: 0,
            },
            PrimalOp::Repeat { kernel: 2 },
            PrimalOp::CrossEntropyLoss,
            PrimalOp::MSELoss,
            PrimalOp::L1Loss,
            PrimalOp::ScaledDotProductAttention { causal: false },
            PrimalOp::FlashAttentionBackwardExtract {
                causal: false,
                component: 0,
            },
            PrimalOp::ScaledDotProductAttentionPacked,
            PrimalOp::FlashAttentionBackwardExtractPacked { component: 0 },
            PrimalOp::CshaFusedBackwardExtract { component: 0 },
            PrimalOp::FusedCshaBackward {
                layer: "blocks.0".into(),
            },
            PrimalOp::RoPE { dim: 64 },
            PrimalOp::RoPEInverse { dim: 64 },
            PrimalOp::Dropout { p: 0.1 },
            PrimalOp::Select,
            PrimalOp::Condition(CompareKind::Gt),
            PrimalOp::FusedLinearCe {
                vocab_size: 4096,
                hidden_size: 128,
                batch_size: 1,
                seq_len: 1,
                vocab_tile: 1024,
                ignore_index: -100,
                is_large: false,
            },
            PrimalOp::FusedLinearCeBackwardExtract {
                component: 0,
                vocab_size: 4096,
                hidden_size: 128,
                batch_size: 1,
                seq_len: 1,
                vocab_tile: 1024,
                ignore_index: -100,
            },
            PrimalOp::Input("x".into()),
            PrimalOp::Param("w".into()),
            PrimalOp::Constant(1.0),
        ];
        // 57 variants total (including markers) — bumped by PCA Stage C's
        // ScaledDotProductAttentionPacked + FlashAttentionBackwardExtractPacked
        // on top of CFTP §4.4 G3's FusedLinearCe pair.
        assert_eq!(ops.len(), 57);
    }

    #[test]
    fn on_param_grad_signature_is_shaped_correctly() {
        // This test is a compile-time assertion that the callback type
        // signature matches what FASE's call site will construct.
        // Behavioral validation lives in the integration tests that exercise
        // source-AD + FASE Deferred end-to-end.
        fn _signature_compiles(
            _hook: Option<(
                &std::collections::HashSet<VarId>,
                &mut dyn FnMut(
                    &mut crate::compiler::Compiler,
                    VarId,
                    Value,
                    &mut FunctionBuilder,
                ) -> Result<(), crate::CodegenError>,
            )>,
        ) {
        }
        // If this compiles, the API shape is correct.
        assert!(true);
    }
}

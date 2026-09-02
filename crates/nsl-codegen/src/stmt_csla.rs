//! Compiler-Scheduled Layerwise Accumulation (CSLA) — the window-region
//! emission helpers of the train block (`--layerwise-accum`).
//!
//! `stmt.rs` owns the train block: it builds the schedule
//! (`layerwise.rs`), decides per window which parameter groups allocate
//! and update, and calls in here once per group. This module only emits:
//!
//!   1. the per-group accumulator allocation at a window's start
//!      (`emit_csla_accum_alloc` — fresh `zeros_like` under the
//!      `MPartial` surface, where the baseline reuses zeroed persistent
//!      buffers),
//!   2. the per-group optimizer update once the group's layer has run its
//!      backward (`emit_csla_group_update` — the batched fused-AdamW arm,
//!      the ZeRO-3 elementwise / owner-gated arms, the Muon route, and the
//!      plain `fase_emit_final_step` fallback, each followed by the CSLA
//!      tail that frees the group's accumulators).
//!
//! Both are compile-time unrolled per group: `idxs` is the group's
//! parameter-list indices, and the only control flow emitted is the
//! ZeRO-3 owner gate around a sharded param's update. `stmt.rs` remains
//! the sole allocator of the parameter / state lists.
//!
//! Moved out of the body of `compile_train_block_inner` byte-for-byte
//! (roadmap A1, PR 2). The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the accumulator allocation, the
//! non-SR batched AdamW arm and the Muon arm (plain and `muon_state_bf16`);
//! the bf16-sr batched arm, the three ZeRO-3 arms and the offload /
//! `fase_emit_final_step` fallback have no single-GPU fixture and are
//! covered by nothing here but the byte-for-byte move.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};

use crate::compiler::Compiler;
use crate::error::CodegenError;
use crate::stmt::SURFACE_M_PARTIAL;

/// Allocate a group's window accumulators: one `zeros_like(param)` per
/// index in `idxs`, stored into `accum_val`'s matching slot.
pub(crate) fn emit_csla_accum_alloc(
    c: &mut Compiler,
    builder: &mut FunctionBuilder,
    param_list: Value,
    accum_val: Value,
    idxs: &[i64],
) -> Result<(), CodegenError> {
    if idxs.is_empty() {
        return Ok(());
    }
    // Fresh zeros_like per window under the MPartial surface —
    // identical initial bytes to the baseline's zeroed
    // persistent buffers.
    let prev =
        c.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
    let surf = builder.ins().iconst(cl_types::I8, SURFACE_M_PARTIAL);
    c.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surf])?;
    for &i in idxs {
        let iv = builder.ins().iconst(cl_types::I64, i);
        let p = c.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
        let z = c.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
        c.compile_call_by_name(builder, "nsl_list_set", &[accum_val, iv, z])?;
    }
    c.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[prev])?;
    Ok(())
}

/// P1 Muon item 11: everything the muon group-update arm needs,
/// prebuilt once per window (all Values live in the window's
/// bwd_block, which dominates every group-update call site).
pub(crate) struct MuonCslaCtx {
    pub(crate) route_list: Value,
    pub(crate) opt_fn: String,
    pub(crate) ns_steps: f64,
    pub(crate) lr: Value,
    /// AdamW-arm lr = lr x fixed ratio (see the adamw_lr knob).
    pub(crate) adamw_lr: Value,
    pub(crate) momentum: Value,
    pub(crate) dampening: Value,
    pub(crate) weight_decay: Value,
    pub(crate) nesterov: Value,
    pub(crate) beta1: Value,
    pub(crate) beta2: Value,
    pub(crate) eps: Value,
    /// (step_count + 1) as f64 — the micro-batch counter t the
    /// non-CSLA FullBuffer muon boundary step uses (historical
    /// bias-correction semantic, preserved for bit-exactness).
    pub(crate) t: Value,
    pub(crate) step_var: Variable,
}

/// Emit one parameter group's optimizer update at its window boundary,
/// then free the group's accumulators (the next window allocates fresh
/// zeros via [`emit_csla_accum_alloc`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_csla_group_update(
    c: &mut Compiler,
    builder: &mut FunctionBuilder,
    param_list: Value,
    state_list_1: Value,
    state_list_2: Value,
    two_state: bool,
    accum_val: Value,
    recipe: &crate::fase::UpdateRecipe,
    // Live learning rate at this group's update point — see
    // `fase_emit_final_step`. CSLA fires group updates at window
    // boundaries rather than at the main optimizer site, so this
    // is read from `lr_var` at the call, not threaded from there.
    lr_runtime: Value,
    bc: (Value, Value),
    wrap_precision: bool,
    wrap_offload: bool,
    muon: Option<&MuonCslaCtx>,
    // P3 ZeRO-3: Some(streamed idx set) under --zero-stage 3.
    // The group's gradient slots all-reduce FIRST (this is item
    // 14's ordering — source AD just completed this layer's
    // backward), then SHARDED (streamed) params update behind an
    // owner gate while resident/tied params update on every rank
    // from the identical reduced gradients.
    zero3: Option<&std::collections::HashSet<i64>>,
    // Item 11: the ELEMENTWISE subset of `zero3` (plan-driven).
    // These params reduce_scatter their slots (the runtime
    // dispatches on the mark), skip the owner gate, and step via
    // nsl_zero3_elem_adamw_step on EVERY rank.
    zero3_elem: Option<&std::collections::HashSet<i64>>,
    // Item 11: the fused-AdamW scalars for the elementwise step
    // (Some iff zero3_elem is non-empty — enforced at the
    // dispatcher, which refuses non-AdamW recipes).
    elem_scalars: Option<&crate::stmt_fase::FusedAdamwScalars>,
    // P4 item 17: Some(opt_step) under --param-dtype bf16-sr.
    sr_step: Option<Value>,
    // Item 8: Some(recipe scalars) admits collapsing this group
    // into one nsl_fase_fused_adamw_step_multi_idx launch. The
    // arm below re-checks the envelope booleans as a belt — a
    // future call site passing scalars alongside an envelope
    // falls through to the per-param loop instead of
    // mis-batching.
    multi: Option<&crate::stmt_fase::FusedAdamwScalars>,
    idxs: &[i64],
) -> Result<(), CodegenError> {
    if let Some(sc) = multi {
        // The envelope booleans are re-checked here as a belt: a
        // future call site passing scalars alongside an envelope
        // falls through to the per-param loop instead of
        // mis-batching. bf16-sr no longer excludes (item 8, SR
        // arm): it selects the SR twin below, whose runtime entry
        // performs the same authoritative-mirror SR step and
        // coherence widen per member that fase_emit_final_step
        // emits per param.
        if muon.is_none()
            && zero3.is_none()
            && zero3_elem.is_none()
            && !wrap_precision
            && !wrap_offload
        {
            if idxs.is_empty() {
                return Ok(());
            }
            let il = c.compile_call_by_name(builder, "nsl_list_new", &[])?;
            for &i in idxs {
                let iv = builder.ins().iconst(cl_types::I64, i);
                c.compile_call_by_name(builder, "nsl_list_push", &[il, iv])?;
            }
            // LIVE rate, not sc.lr: FusedAdamwScalars is a compile-time plan and
            // its lr is the optimizer constructor's base rate. Everything
            // else here (betas, eps, wd) genuinely is compile-time.
            let lr_v = lr_runtime;
            let b1_v = builder.ins().f64const(sc.beta1);
            let omb1_v = builder.ins().f64const(sc.one_minus_beta1);
            let b2_v = builder.ins().f64const(sc.beta2);
            let omb2_v = builder.ins().f64const(sc.one_minus_beta2);
            let eps_v = builder.ins().f64const(sc.eps);
            let wd_v = builder.ins().f64const(sc.wd);
            // CSLA has no AdamW parameter-group plumbing (the
            // group update has always applied the recipe's flat
            // λ) — 0/0 makes the runtime resolve every member to
            // `wd`, exactly the per-param loop's behavior.
            // mp_scale = 1.0: the layerwise schedule refuses
            // grad_clip, so a clip factor can never exist here.
            let zero_i = builder.ins().iconst(cl_types::I64, 0);
            if c.features.param_dtype_bf16sr {
                let step_v = sr_step.ok_or_else(|| {
                    CodegenError::new(
                        "bf16-sr group update reached without an \
                         opt_step value — dispatcher must thread \
                         sr_step",
                    )
                })?;
                c.compile_call_by_name(
                    builder,
                    "nsl_fase_fused_adamw_step_bf16sr_multi_idx",
                    &[
                        param_list,
                        state_list_1,
                        state_list_2,
                        accum_val,
                        il,
                        lr_v,
                        b1_v,
                        omb1_v,
                        b2_v,
                        omb2_v,
                        eps_v,
                        wd_v,
                        bc.0,
                        bc.1,
                        zero_i,
                        zero_i,
                        step_v,
                    ],
                )?;
            } else {
            let one_scale = builder.ins().f64const(1.0);
            c.compile_call_by_name(
                builder,
                "nsl_fase_fused_adamw_step_multi_idx",
                &[
                    param_list,
                    state_list_1,
                    state_list_2,
                    accum_val,
                    il,
                    lr_v,
                    b1_v,
                    omb1_v,
                    b2_v,
                    omb2_v,
                    eps_v,
                    wd_v,
                    bc.0,
                    bc.1,
                    zero_i,
                    zero_i,
                    one_scale,
                ],
            )?;
            }
            c.compile_call_by_name(builder, "nsl_list_free", &[il])?;
            // The CSLA tail, unchanged: free the group's
            // accumulators (fresh zeros next window). The
            // in-kernel m_partial zero is redundant before a
            // free, and harmless.
            for &i in idxs {
                let iv = builder.ins().iconst(cl_types::I64, i);
                let m_partial = c.compile_call_by_name(
                    builder,
                    "nsl_list_get",
                    &[accum_val, iv],
                )?;
                c.compile_call_by_name(
                    builder,
                    "nsl_tensor_free",
                    &[m_partial],
                )?;
                let z = builder.ins().iconst(cl_types::I64, 0);
                c.compile_call_by_name(
                    builder,
                    "nsl_list_set",
                    &[accum_val, iv, z],
                )?;
            }
            return Ok(());
        }
    }
    if zero3.is_some() {
        for &i in idxs {
            let iv = builder.ins().iconst(cl_types::I64, i);
            let rc = c.compile_call_by_name(
                builder,
                "nsl_zero3_reduce_grad_slot",
                &[accum_val, iv],
            )?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            let ok = builder.ins().icmp(IntCC::Equal, rc, z);
            let msg = "nsl: zero3 layer gradient all-reduce failed — aborting";
            c.intern_string(msg)?;
            let mp = c.compile_string_literal(builder, msg)?;
            c.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
        }
    }
    for &i in idxs {
        let iv = builder.ins().iconst(cl_types::I64, i);
        let theta =
            c.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
        let m =
            c.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, iv])?;
        let m_partial =
            c.compile_call_by_name(builder, "nsl_list_get", &[accum_val, iv])?;
        let v = if two_state {
            c.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, iv])?
        } else {
            m
        };
        // Item 11: elementwise params take their own step arm —
        // every rank updates its slice, so no owner gate.
        let is_elem = zero3_elem.is_some_and(|s| s.contains(&i));
        // P3 ZeRO-3: owner-gate the update of a SHARDED param —
        // non-owners' gathered copies are released right after
        // this group and refetched (post-update) from the owner
        // next window, so skipping their update is the sharded
        // semantic, not a divergence.
        let z3_gate: Option<(cranelift_codegen::ir::Block, cranelift_codegen::ir::Block)> =
            if !is_elem && zero3.is_some_and(|s| s.contains(&i)) {
                let owns =
                    c.compile_call_by_name(builder, "nsl_zero_owns_param", &[iv])?;
                let one = builder.ins().iconst(cl_types::I64, 1);
                let owned = builder.ins().icmp(IntCC::Equal, owns, one);
                let do_b = builder.create_block();
                let join_b = builder.create_block();
                builder.ins().brif(owned, do_b, &[], join_b, &[]);
                builder.switch_to_block(do_b);
                builder.seal_block(do_b);
                Some((do_b, join_b))
            } else {
                None
            };
        if is_elem {
            // Item 11: the elementwise fused step — every rank,
            // its own slice, the exact fused-AdamW kernel math.
            // The runtime already holds the reduce_scattered
            // gradient slice (the reduce loop above dispatched on
            // the mark), so m_partial is not an argument; codegen
            // frees it in the shared tail like every other arm.
            let sc = elem_scalars.ok_or_else(|| {
                CodegenError::new(
                    "elementwise group update reached without \
                     fused-AdamW scalars — dispatcher must thread \
                     elem_scalars",
                )
            })?;
            // LIVE rate, not sc.lr: FusedAdamwScalars is a compile-time plan and
            // its lr is the optimizer constructor's base rate. Everything
            // else here (betas, eps, wd) genuinely is compile-time.
            let lr_v = lr_runtime;
            let b1_v = builder.ins().f64const(sc.beta1);
            let omb1_v = builder.ins().f64const(sc.one_minus_beta1);
            let b2_v = builder.ins().f64const(sc.beta2);
            let omb2_v = builder.ins().f64const(sc.one_minus_beta2);
            let eps_v = builder.ins().f64const(sc.eps);
            // The recipe's flat λ — same as the per-param loop
            // and the multi arm (CSLA has no AdamW group
            // plumbing; grad_clip is refused by the schedule).
            let wd_v = builder.ins().f64const(sc.wd);
            // Item 16×11: under composed bf16-sr the slice is
            // bf16-authoritative and the step is the SR kernel —
            // a distinct entry (it needs the optimizer step for
            // the SR counter key, and the runtime belts abort on
            // a carve/dispatch mismatch in either direction).
            // One arg list, built once: the SR entry's contract
            // is "the plain entry's scalar order plus a trailing
            // step", so spelling the shared prefix twice invites
            // a future scalar landing in one copy only — which
            // type-checks and shifts every f64 on the other path.
            let mut step_args = vec![
                theta, m, v, iv, lr_v, b1_v, omb1_v, b2_v, omb2_v,
                eps_v, wd_v, bc.0, bc.1,
            ];
            let elem_callee = if c.features.param_dtype_bf16sr {
                step_args.push(sr_step.ok_or_else(|| {
                    CodegenError::new(
                        "bf16-sr elementwise group update reached \
                         without an opt_step value — dispatcher \
                         must thread sr_step",
                    )
                })?);
                "nsl_zero3_elem_sr_adamw_step"
            } else {
                "nsl_zero3_elem_adamw_step"
            };
            let rc = c.compile_call_by_name(builder, elem_callee, &step_args)?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            let ok = builder.ins().icmp(IntCC::Equal, rc, z);
            let msg =
                "nsl: zero3 elementwise step failed — aborting";
            c.intern_string(msg)?;
            let mp = c.compile_string_literal(builder, msg)?;
            c.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
        } else if let Some(mc) = muon {
            // Muon separate-accumulator mode: m_partial holds the
            // RAW window gradient sum (accum_scale forced to 1.0),
            // exactly what the non-CSLA FullBuffer path hands
            // muon_step as `gradient` — the stdlib step then does
            // its own momentum/orthogonalize (Muon route) or
            // AdamW math (routed params; v may be a null slot on
            // the Muon route, never read there). Bias correction
            // happens inside muon_step from t, so the window's
            // precomputed bc pair is unused here.
            let flag_i = c.compile_call_by_name(
                builder,
                "nsl_list_get",
                &[mc.route_list, iv],
            )?;
            let flag_f = builder.ins().fcvt_from_sint(cl_types::F64, flag_i);
            let ns_c = builder.ins().f64const(mc.ns_steps);
            // P4 item 18 rung 2: dequant the bf16 momentum into
            // an f32 working buffer for the step; SR-quant it
            // back after (nsl_muon_state_sr_store) and free the
            // working copy. m is never null (unlike v), so no
            // runtime null branch is needed.
            let m_bf16_env: Option<(Value, Value)> = if c.features.muon_state_bf16 {
                let f32_code = builder.ins().iconst(cl_types::I64, 1);
                let m_work = c.compile_call_by_name(
                    builder,
                    "nsl_tensor_cast",
                    &[m, f32_code],
                )?;
                Some((m_work, m))
            } else {
                None
            };
            let m_step = m_bf16_env.map_or(m, |(w, _)| w);
            c.emit_stdlib_optim_call(
                builder,
                "muon",
                &mc.opt_fn,
                theta,
                m_partial,
                m_step,
                v,
                mc.lr,
                mc.momentum,
                mc.dampening,
                mc.weight_decay,
                mc.nesterov,
                mc.beta1,
                mc.beta2,
                mc.eps,
                mc.step_var,
                wrap_precision,
                Some(mc.t),
                wrap_offload,
                Some((flag_f, ns_c, mc.adamw_lr)),
            )?;
            if let Some((m_work, m_bf)) = m_bf16_env {
                let step_v = sr_step.ok_or_else(|| CodegenError::new(
                    "muon-state bf16 envelope reached without an \
                     opt_step value — dispatcher must thread sr_step",
                ))?;
                c.compile_call_by_name(
                    builder,
                    "nsl_muon_state_sr_store",
                    &[m_work, m_bf, step_v, iv],
                )?;
                c.compile_call_by_name(builder, "nsl_tensor_free", &[m_work])?;
            }
        } else {
            // D2a: wrap_offload stages host-pinned m/v to θ's device
            // for the update and streams them back asynchronously —
            // the whole point of the layer-major schedule is that
            // only ONE layer's m/v are staged at a time. The
            // envelope's m_partial leg degenerates safely here (the
            // accumulator is device-resident, so its "stage" is a
            // same-device refcount bump and the tail's zero is
            // wasted-but-correct before our free below).
            c.fase_emit_final_step(
                builder,
                theta,
                m,
                m_partial,
                v,
                recipe,
                lr_runtime,
                Some(bc),
                wrap_precision,
                wrap_offload,
                sr_step,
            )?;
        }
        if let Some((_do_b, join_b)) = z3_gate {
            builder.ins().jump(join_b, &[]);
            builder.switch_to_block(join_b);
            builder.seal_block(join_b);
        }
        // The baseline zeroes m_partial for reuse; the layerwise
        // schedule frees it — next window allocates fresh zeros.
        // (muon_step borrows the gradient, so freeing here is the
        // muon arm's zeroing equivalent too. Under zero3 the free
        // runs on every rank — owner and skipped non-owner alike.)
        c.compile_call_by_name(builder, "nsl_tensor_free", &[m_partial])?;
        let z = builder.ins().iconst(cl_types::I64, 0);
        c.compile_call_by_name(builder, "nsl_list_set", &[accum_val, iv, z])?;
    }
    // D2a: the async DtoH stage-outs defer their frees to the
    // drain — one per GROUP update, so at most one layer's
    // staged tensors are ever in flight.
    if wrap_offload && !idxs.is_empty() {
        c.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
    }
    Ok(())
}

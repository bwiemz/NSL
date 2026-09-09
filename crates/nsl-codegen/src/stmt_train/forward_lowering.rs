//! The forward lowering of the train block's source-AD arm: the
//! memory-planner tape-unchanged assertion at the entry to the placement
//! consumption window, the Item 11 per-segment early-free plan (the CCR
//! segment slices and their interior free lists), and the primal lowering
//! itself — monolithic, or segment-streamed under `--weight-stream` with
//! the upload / evict (or arena-pack) calls between slices — inside the
//! in-place-suppress window.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 294 lines,
//! 6 inputs ([`ForwardLoweringInputs`]) plus the function builder
//! and state. Returns the early-free plan ([`CcrSegmentFree`], `None` when
//! it does not apply) and the [`LoweredWengert`] the adjoint lowering
//! reads through. The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin every forward this emits.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::SURFACE_WEIGHTS;
use crate::stmt_train::csla_precompute::WsForwardPlan;
use crate::wengert_lower::{LoweredWengert, VarMap};

/// Every binding of `compile_train_block_inner` the forward lowering
/// reads; names are the driver's.
pub(crate) struct ForwardLoweringInputs<'a> {
    /// The final adjoint tape (the memory-planner scan is asserted unchanged against it).
    pub(crate) adjoint: &'a crate::wengert::WengertList,
    /// The CCR plan; its segments bound the per-segment early-free slices.
    pub(crate) ccr_plan: &'a Option<crate::ccr::CcrPlan>,
    /// The forward tape to lower.
    pub(crate) effective_primal: &'a crate::wengert::WengertList,
    /// The `NslList` of model parameters (the streamed upload / evict calls index it).
    pub(crate) param_list: Value,
    /// The initial VarMap (named inputs and parameters); the lowering starts from a clone.
    pub(crate) primal_vars: &'a VarMap,
    /// The `--weight-stream` sliced-forward plan; `Some` selects the segment-streamed lowering.
    pub(crate) ws_fwd_plan: &'a Option<WsForwardPlan>,
}

pub(crate) struct CcrSegmentFree {
    /// (segment index if this slice IS a segment, start, end)
    pub(crate) slices: Vec<(Option<usize>, usize, usize)>,
    pub(crate) lists: Vec<crate::wengert::WengertList>,
}

impl Compiler<'_> {
    /// Lower the forward (see the module header).
    pub(crate) fn emit_forward_lowering(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: ForwardLoweringInputs<'_>,
    ) -> Result<(Option<CcrSegmentFree>, LoweredWengert), CodegenError> {
        let ForwardLoweringInputs {
            adjoint,
            ccr_plan,
            effective_primal,
            param_list,
            primal_vars,
            ws_fwd_plan,
        } = inputs;

        // Milestone C: the arena's positional liveness (birth/death
        // over the concatenated [forward; adjoint] timeline) was
        // captured at `transient_arena::analyze`; the placements it
        // burned into `arena_placements` are consumed from here on,
        // during lowering (`wengert_lower.rs` looks slots up per op).
        // Slot-sharing is sound only if the list being lowered is the
        // list that was analyzed — prove the adjoint has not moved
        // since the scan, at the entry to the consumption window
        // (assert at ENTRY, not after a write). Guarded: no
        // placements means nothing downstream consumes the scan.
        if !self.arena_placements.is_empty() {
            let sched = self.passes.scheduler();
            sched
                .assert_tape_unchanged_since("MemoryPlanner", adjoint)
                .map_err(CodegenError::new)?;
        }
        // Item 11: plan the per-segment forward early-free. Built
        // here — after the CCR plan's owned-tensor restriction and
        // after the weight-stream plan exists — so the eligibility
        // conditions are all decidable:
        //   * ws path: already sliced, keeps its own (post-forward)
        //     free discipline; composing the two is future work and
        //     silently changing ws lifetimes here is not it.
        //   * CSLA: buffers the adjoint's primal imports across the
        //     accumulation window; its retention contract is the
        //     bulk-free's, not this pass's. Excluded.
        //   * NSL_CCR_SEGMENT_FREE=0: kill switch for A/B under ONE
        //     binary (the runtime-archive lesson: never A/B by
        //     swapping binaries).
        let ccr_segment_free: Option<CcrSegmentFree> = match ccr_plan {
            Some(plan)
                if ws_fwd_plan.is_none()
                    && !self.compile_options.train.layerwise_accum
                    && std::env::var("NSL_CCR_SEGMENT_FREE").as_deref()
                        != Ok("0") =>
            {
                let seg_bounds: Vec<(usize, usize)> = plan
                    .segments
                    .iter()
                    .map(|seg| (seg.start, seg.end))
                    .collect();
                let slices = crate::layerwise::forward_slices(
                    &seg_bounds,
                    effective_primal.ops.len(),
                )
                .map_err(CodegenError::new)?;
                let mut tagged = Vec::with_capacity(slices.len());
                for &(s_op, e_op) in &slices {
                    let si = seg_bounds
                        .iter()
                        .position(|&(bs, be)| bs == s_op && be == e_op);
                    tagged.push((si, s_op, e_op));
                }
                Some(CcrSegmentFree {
                    slices: tagged,
                    lists: crate::ccr::build_segment_free_lists(plan),
                })
            }
            _ => None,
        };
        self.emit_inplace_suppress(builder, true)?;
        let full_lowered = if let Some(wsplan) = ws_fwd_plan {
            // Segment-streamed forward: lower the primal per CCR
            // slice with upload/evict FFI calls between slices. All
            // slices share one straight-line block chain, so SSA
            // values flow across boundaries; the fold state
            // (var_map/var_types/owned/freed) threads through
            // `compile_wengert_ops_range` so the result is
            // byte-identical to the monolithic lowering — streaming
            // only interleaves resident-set changes for θ.
            let mut var_map = primal_vars.clone();
            let mut var_types = effective_primal.var_types.clone();
            let mut owned_values = Vec::new();
            let mut hook_freed_input_vars = std::collections::HashSet::new();
            let mut explicit_freed_vars = std::collections::HashSet::new();
            // No FASE hook on this forward-streaming path — stays empty.
            let mut hook_freed_param_vars = std::collections::HashSet::new();
            let arena_mode = self.compile_options.weight_stream.arena;
            for (si, &(s, e)) in wsplan.slices.iter().enumerate() {
                // Item 10: in arena mode a whole layer pack uploads at
                // its bracket-start slice (ONE contiguous transfer);
                // otherwise the per-param first-touch set.
                let arena_uploads: Vec<Vec<i64>> = if arena_mode {
                    wsplan
                        .arena_packs
                        .iter()
                        .filter(|(f, _, _)| *f == si)
                        .map(|(_, _, idxs)| idxs.clone())
                        .collect()
                } else {
                    Vec::new()
                };
                let has_upload = if arena_mode {
                    !arena_uploads.is_empty()
                } else {
                    !wsplan.upload_per_slice[si].is_empty()
                };
                if has_upload {
                    // Upload under the Weights surface (allocation
                    // accounting).
                    let prev_surf = self.compile_call_by_name(
                        builder,
                        "nsl_gpu_get_alloc_surface",
                        &[],
                    )?;
                    let wsurf = builder.ins().iconst(cl_types::I8, SURFACE_WEIGHTS);
                    self.compile_call_by_name(
                        builder,
                        "nsl_gpu_set_alloc_surface",
                        &[wsurf],
                    )?;
                    if arena_mode {
                        for idxs in &arena_uploads {
                            self.emit_ws_pack_upload(builder, param_list, idxs)?;
                        }
                    } else {
                        for &idx in &wsplan.upload_per_slice[si] {
                            let iv = builder.ins().iconst(cl_types::I64, idx);
                            let pw = self.compile_call_by_name(
                                builder,
                                "nsl_list_get",
                                &[param_list, iv],
                            )?;
                            self.compile_call_by_name(
                                builder,
                                "nsl_weight_stream_upload",
                                &[pw],
                            )?;
                        }
                    }
                    self.compile_call_by_name(
                        builder,
                        "nsl_gpu_set_alloc_surface",
                        &[prev_surf],
                    )?;
                }
                crate::wengert_lower::compile_wengert_ops_range(
                    self,
                    builder,
                    state,
                    effective_primal,
                    s..e,
                    &mut var_map,
                    &mut var_types,
                    &mut owned_values,
                    &mut hook_freed_input_vars,
                    &mut explicit_freed_vars,
                    &mut hook_freed_param_vars,
                    None,
                )?;
                // Evict this slice's last-touch params/packs — read-only
                // (writeback=0): forwards never mutate θ, the mirror
                // is current by construction.
                if arena_mode {
                    let evicts: Vec<Vec<i64>> = wsplan
                        .arena_packs
                        .iter()
                        .filter(|(_, l, _)| *l == si)
                        .map(|(_, _, idxs)| idxs.clone())
                        .collect();
                    for idxs in &evicts {
                        self.emit_ws_pack_evict(builder, param_list, idxs, 0)?;
                    }
                } else {
                    for &idx in &wsplan.evict_per_slice[si] {
                        let iv = builder.ins().iconst(cl_types::I64, idx);
                        let pw = self.compile_call_by_name(
                            builder,
                            "nsl_list_get",
                            &[param_list, iv],
                        )?;
                        let wb = builder.ins().iconst(cl_types::I64, 0);
                        self.compile_call_by_name(
                            builder,
                            "nsl_weight_stream_evict",
                            &[pw, wb],
                        )?;
                    }
                }
            }
            // Same sdpa-extras adoption the monolithic wrapper does.
            for v in self.sdpa_extra_owned.drain(..) {
                owned_values.push((u32::MAX, v, crate::wengert::WengertType::Tensor));
            }
            crate::wengert_lower::LoweredWengert {
                var_map,
                owned_values,
                hook_freed_input_vars,
                explicit_freed_vars,
                hook_freed_param_vars,
            }
        } else if let Some(seg_free) = &ccr_segment_free {
            // Item 11: per-segment forward early-free. Same sliced
            // emission the weight-streaming branch above uses — one
            // straight-line block chain, fold state threaded through
            // `compile_wengert_ops_range`, byte-identical op stream —
            // but between slices the only thing emitted is each
            // segment's FreeTensor mini-list. With the single
            // post-forward free list, EVERY segment's interiors were
            // still live at end-of-forward, which is where the global
            // peak sits: `--checkpoint-blocks` was reducing the
            // backward's activation wall and never the forward's.
            let mut var_map = primal_vars.clone();
            let mut var_types = effective_primal.var_types.clone();
            let mut owned_values = Vec::new();
            let mut hook_freed_input_vars = std::collections::HashSet::new();
            let mut explicit_freed_vars = std::collections::HashSet::new();
            let mut hook_freed_param_vars = std::collections::HashSet::new();
            let mut freed_count = 0usize;
            let mut segments_freed = 0usize;
            for &(si, s_op, e_op) in &seg_free.slices {
                crate::wengert_lower::compile_wengert_ops_range(
                    self,
                    builder,
                    state,
                    effective_primal,
                    s_op..e_op,
                    &mut var_map,
                    &mut var_types,
                    &mut owned_values,
                    &mut hook_freed_input_vars,
                    &mut explicit_freed_vars,
                    &mut hook_freed_param_vars,
                    None,
                )?;
                if let Some(seg_idx) = si {
                    let free_list = &seg_free.lists[seg_idx];
                    if free_list.ops.is_empty() {
                        continue;
                    }
                    // Through the RANGE fold, not the wrapper: the
                    // wrapper drains `sdpa_extra_owned` into a return
                    // value, and a mid-forward call here would steal
                    // every fused-SDPA LSE pushed by the slices before
                    // it — un-owning them so the end-of-step bulk free
                    // never releases them. Review finding F1
                    // (2026-08-24), confirmed as a +8 MB/micro-step
                    // ramp at 1B (16 layers x [2,32,2048] f32) before
                    // this fix; the drain happens ONCE, after the last
                    // slice, exactly as the ws branch does.
                    let before = explicit_freed_vars.len();
                    let mut free_var_types = free_list.var_types.clone();
                    crate::wengert_lower::compile_wengert_ops_range(
                        self,
                        builder,
                        state,
                        free_list,
                        0..free_list.ops.len(),
                        &mut var_map,
                        &mut free_var_types,
                        &mut owned_values,
                        &mut hook_freed_input_vars,
                        &mut explicit_freed_vars,
                        &mut hook_freed_param_vars,
                        None,
                    )?;
                    let freed_here = explicit_freed_vars.len() - before;
                    freed_count += freed_here;
                    if freed_here > 0 {
                        segments_freed += 1;
                    }
                }
            }
            nsl_log::nsl_log!(INFO, "ccr", 
                "[ccr] per-segment early-free: {} interior value(s) freed \
                 across {} segment(s) during the forward",
                freed_count, segments_freed,
            );
            for v in self.sdpa_extra_owned.drain(..) {
                owned_values.push((u32::MAX, v, crate::wengert::WengertType::Tensor));
            }
            crate::wengert_lower::LoweredWengert {
                var_map,
                owned_values,
                hook_freed_input_vars,
                explicit_freed_vars,
                hook_freed_param_vars,
            }
        } else {
            crate::wengert_lower::compile_wengert_ops(
                self,
                builder,
                state,
                effective_primal,
                primal_vars,
                None, // FASE on_param_grad hook — wired in Task 3
            )?
        };
        self.emit_inplace_suppress(builder, false)?;

        Ok((ccr_segment_free, full_lowered))
    }
}

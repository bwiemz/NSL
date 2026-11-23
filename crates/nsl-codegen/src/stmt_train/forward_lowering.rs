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
//! reads through. `emit_tape_region_close`, which runs just before (the
//! ELTLS free of tape-held tensors, the tape-region flag, the
//! `NSL_DEBUG_WENGERT` dump), moved here from `plan_csha_prune.rs` when
//! TrainPlan step 5 split that file's plans from its emission. The
//! train-block CLIF snapshots (`tests/train_clif_snapshots.rs`) pin every
//! forward this emits.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::SURFACE_WEIGHTS;
use crate::stmt_train::csla_precompute::WsForwardPlan;
use crate::wengert::{VarId, WengertList, WengertType};
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

/// The fold state a sliced forward threads through every
/// `compile_wengert_ops_range` call — each primal slice and each segment's
/// free list — so that the slices together record exactly what one
/// monolithic `compile_wengert_ops` would have (one var_map every later slice
/// and the adjoint resolve through; owned and freed sets concatenated).
struct SlicedForward {
    var_map: VarMap,
    var_types: std::collections::HashMap<VarId, WengertType>,
    owned_values: Vec<(VarId, Value, WengertType)>,
    hook_freed_input_vars: std::collections::HashSet<VarId>,
    explicit_freed_vars: std::collections::HashSet<VarId>,
    /// No FASE hook runs on a forward; stays empty.
    hook_freed_param_vars: std::collections::HashSet<VarId>,
}

impl SlicedForward {
    fn new(primal_vars: &VarMap, primal: &WengertList) -> Self {
        Self {
            var_map: primal_vars.clone(),
            var_types: primal.var_types.clone(),
            owned_values: Vec::new(),
            hook_freed_input_vars: std::collections::HashSet::new(),
            explicit_freed_vars: std::collections::HashSet::new(),
            hook_freed_param_vars: std::collections::HashSet::new(),
        }
    }
}

/// What a forward's segment free lists freed: values, and segments that
/// freed at least one (the witness line's two counts).
#[derive(Default)]
struct FreeTally {
    freed: usize,
    segments: usize,
}

/// I5 (D1): the streamed forward and the per-segment free lists must slice
/// the forward identically — the lists are indexed by the per-segment plan's
/// slices, the loop by the streaming plan's. Both come from
/// `layerwise::forward_slices` over the same CCR segments today; a future
/// divergence would free a segment's interiors at another slice's end (too
/// early: a use after free; too late: memory) without any other symptom.
fn streamed_slices_match_segments(
    segment_slices: &[(Option<usize>, usize, usize)],
    streamed: &[(usize, usize)],
) -> Result<(), CodegenError> {
    let same = segment_slices.len() == streamed.len()
        && segment_slices
            .iter()
            .zip(streamed)
            .all(|(&(_, s, e), &(ws, we))| s == ws && e == we);
    if same {
        Ok(())
    } else {
        Err(CodegenError::new(
            "internal compiler error (CCR/weight-stream): the streamed forward's slices \
             differ from the CCR segment slices; please report this",
        ))
    }
}

/// The per-segment early-free witness: the gates' anti-vacuity anchor
/// (`ccr_segment_early_free_gate.rs`, `csla_layerwise_gate.rs` parse the
/// count), printed whenever the forward ran the per-segment free lists.
fn log_segment_free_witness(freed: usize, segments: usize) {
    nsl_log::nsl_log!(
        INFO,
        "ccr",
        "[ccr] per-segment early-free: {} interior value(s) freed \
         across {} segment(s) during the forward",
        freed,
        segments,
    );
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
        // conditions are all decidable. It applies in every mode:
        //   * CSLA (`--layerwise-accum`): the window buffers what the
        //     adjoint reads, and the adjoint reads no freed interior
        //     (`ccr::validate`) — only the shape stubs the free lists
        //     make, which the window buffers as well
        //     (`layerwise::adjoint_primal_imports`); the save phase
        //     checks both. Without this every block's interiors lived
        //     to the end of each micro-batch's forward (D1: +10 GiB at
        //     500M, OOM at 1B).
        //   * `--weight-stream`: its forward is sliced on the same CCR
        //     segments, so each segment's list runs inside the streamed
        //     loop, before that slice's evicts (checked below).
        //   * NSL_CCR_SEGMENT_FREE=0: kill switch for A/B under ONE
        //     binary (the runtime-archive lesson: never A/B by
        //     swapping binaries).
        let ccr_segment_free: Option<CcrSegmentFree> = match ccr_plan {
            Some(plan) if self.ccr_frees_per_segment() => {
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
                let tagged = crate::layerwise::segment_index_of_slices(&seg_bounds, &slices)
                    .into_iter()
                    .zip(&slices)
                    .map(|(si, &(s_op, e_op))| (si, s_op, e_op))
                    .collect();
                Some(CcrSegmentFree {
                    slices: tagged,
                    lists: crate::ccr::build_segment_free_lists(plan),
                })
            }
            _ => None,
        };
        // The adjoint was rewritten to read shape stubs, and saved tensors
        // were marked for release, on the promise that the per-segment free
        // lists run (`ccr_frees_per_segment`). Nothing else makes a stub.
        if ccr_segment_free.is_none()
            && let Some(plan) = ccr_plan
            && (!plan.shape_stubs.is_empty()
                || plan.per_segment_released.iter().any(|seg| !seg.is_empty()))
        {
            return Err(CodegenError::new(
                "internal compiler error (CCR): the backward was planned around shape \
                 stubs but the forward is not freeing per segment; please report this",
            ));
        }
        self.emit_inplace_suppress(builder, true)?;
        let add_norm_before = self.add_norm_fused.0;
        let full_lowered = if let Some(wsplan) = ws_fwd_plan {
            // Segment-streamed forward: lower the primal per CCR
            // slice with upload/evict FFI calls between slices. All
            // slices share one straight-line block chain, so SSA
            // values flow across boundaries; the fold state
            // (var_map/var_types/owned/freed) threads through
            // `compile_wengert_ops_range` so the result is
            // byte-identical to the monolithic lowering — streaming
            // only interleaves resident-set changes for θ.
            //
            // D1: each CCR segment's free list runs right after its slice
            // and BEFORE that slice's evicts, so an interior that is a
            // view of a streamed parameter is released while the
            // parameter's storage is still on the device. Before, the
            // streamed forward lowered no list at all and the driver freed
            // every interior after the whole forward.
            if let Some(seg_free) = &ccr_segment_free {
                streamed_slices_match_segments(&seg_free.slices, &wsplan.slices)?;
            }
            let mut fold = SlicedForward::new(primal_vars, effective_primal);
            let mut tally = FreeTally::default();
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
                self.emit_forward_slice(builder, state, effective_primal, s..e, &mut fold)?;
                if let Some(seg_free) = &ccr_segment_free {
                    self.emit_slice_end_frees(builder, state, seg_free, si, &mut fold, &mut tally)?;
                }
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
            if ccr_segment_free.is_some() {
                log_segment_free_witness(tally.freed, tally.segments);
            }
            self.finish_sliced_forward(fold)
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
            let mut fold = SlicedForward::new(primal_vars, effective_primal);
            let mut tally = FreeTally::default();
            // Round-2 item 6: a block ends with its residual add and the next block
            // starts with the RMSNorm of that sum, one slice later, with the segment's
            // free list between them (which frees the block's interiors, the add's
            // operand among them). So the pair is planned over the whole forward and
            // launched at the Add, before the free list. (Not under weight streaming,
            // above: the next block's gain may not be resident at the Add.)
            self.add_norm_groups = Some(crate::wengert_lower::add_rmsnorm_groups(
                effective_primal,
                0..effective_primal.ops.len(),
            ));
            for (k, &(_, s_op, e_op)) in seg_free.slices.iter().enumerate() {
                self.emit_forward_slice(builder, state, effective_primal, s_op..e_op, &mut fold)?;
                self.emit_slice_end_frees(builder, state, seg_free, k, &mut fold, &mut tally)?;
            }
            self.add_norm_groups = None;
            if !self.add_norm_stash.is_empty() {
                return Err(CodegenError::new(format!(
                    "internal: the forward's residual add + RMSNorm launches produced norm output(s) {:?} \
                     whose op was never lowered; please report this (NSL_FUSE_ADD_RMSNORM=0 avoids it)",
                    self.add_norm_stash.keys().collect::<Vec<_>>()
                )));
            }
            log_segment_free_witness(tally.freed, tally.segments);
            self.finish_sliced_forward(fold)
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
        self.add_norm_fused.1 += self.add_norm_fused.0 - add_norm_before;

        Ok((ccr_segment_free, full_lowered))
    }

    /// Lower the primal ops `range` of a sliced forward into `fold`.
    fn emit_forward_slice(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        primal: &WengertList,
        range: std::ops::Range<usize>,
        fold: &mut SlicedForward,
    ) -> Result<(), CodegenError> {
        crate::wengert_lower::compile_wengert_ops_range(
            self,
            builder,
            state,
            primal,
            range,
            &mut fold.var_map,
            &mut fold.var_types,
            &mut fold.owned_values,
            &mut fold.hook_freed_input_vars,
            &mut fold.explicit_freed_vars,
            &mut fold.hook_freed_param_vars,
            None,
        )
    }

    /// After slice `slice` of a sliced forward: when that slice is a CCR
    /// segment, lower the segment's free list and tally what it freed.
    fn emit_slice_end_frees(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        seg_free: &CcrSegmentFree,
        slice: usize,
        fold: &mut SlicedForward,
        tally: &mut FreeTally,
    ) -> Result<(), CodegenError> {
        let Some(seg_idx) = seg_free.slices[slice].0 else {
            return Ok(());
        };
        let freed = self.emit_segment_free_list(builder, state, &seg_free.lists[seg_idx], fold)?;
        tally.freed += freed;
        if freed > 0 {
            tally.segments += 1;
        }
        Ok(())
    }

    /// Lower one CCR segment's free list ([`crate::ccr::build_segment_free_lists`]:
    /// its shape stubs, then its `FreeTensor`s) at that segment's end, into
    /// the forward's fold. Returns how many values it freed.
    ///
    /// Through the RANGE fold, not the wrapper: the wrapper drains
    /// `sdpa_extra_owned` into a return value, and a mid-forward call here
    /// would steal every fused-SDPA LSE pushed by the slices before it —
    /// un-owning them so the end-of-step bulk free never releases them.
    /// Review finding F1 (2026-08-24), confirmed as a +8 MB/micro-step ramp
    /// at 1B (16 layers x [2,32,2048] f32) before this fix; the drain happens
    /// ONCE, after the last slice ([`Self::finish_sliced_forward`]).
    fn emit_segment_free_list(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        free_list: &WengertList,
        fold: &mut SlicedForward,
    ) -> Result<usize, CodegenError> {
        if free_list.ops.is_empty() {
            return Ok(0);
        }
        let before = fold.explicit_freed_vars.len();
        // The list's own type map (its stubs are Tensors); the primal's
        // map is not extended by the list's markers.
        let mut free_var_types = free_list.var_types.clone();
        crate::wengert_lower::compile_wengert_ops_range(
            self,
            builder,
            state,
            free_list,
            0..free_list.ops.len(),
            &mut fold.var_map,
            &mut free_var_types,
            &mut fold.owned_values,
            &mut fold.hook_freed_input_vars,
            &mut fold.explicit_freed_vars,
            &mut fold.hook_freed_param_vars,
            None,
        )?;
        Ok(fold.explicit_freed_vars.len() - before)
    }

    /// Close a sliced forward: adopt the fused-SDPA extras the slices
    /// accrued (the same adoption the monolithic wrapper does, once, after
    /// the last slice) and hand the fold back as the lowering result.
    fn finish_sliced_forward(&mut self, mut fold: SlicedForward) -> LoweredWengert {
        for v in self.sdpa_extra_owned.drain(..) {
            fold.owned_values.push((u32::MAX, v, WengertType::Tensor));
        }
        LoweredWengert {
            var_map: fold.var_map,
            owned_values: fold.owned_values,
            hook_freed_input_vars: fold.hook_freed_input_vars,
            explicit_freed_vars: fold.explicit_freed_vars,
            hook_freed_param_vars: fold.hook_freed_param_vars,
        }
    }

    /// Whether the forward frees each CCR segment's interiors at that
    /// segment's end (the per-segment free lists) and not once after the
    /// whole forward: in every mode (plain, `--layerwise-accum`,
    /// `--weight-stream`) unless the `NSL_CCR_SEGMENT_FREE=0` kill switch is
    /// set. `apply_to_adjoint` is told this before the forward is lowered, so
    /// both sites read it here.
    pub(crate) fn ccr_frees_per_segment(&self) -> bool {
        std::env::var("NSL_CCR_SEGMENT_FREE").as_deref() != Ok("0")
    }

    /// Close the tape region before the forward is lowered: free the
    /// tape-held tensors, clear the flag, and dump the primal list under
    /// `NSL_DEBUG_WENGERT`. `primal_vars` is read by the dump only.
    pub(crate) fn emit_tape_region_close(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        extractor: &crate::source_ad::WengertExtractor<'_>,
        primal_vars: &crate::wengert_lower::VarMap,
    ) {
        // 5. Lower PRIMAL Wengert list to Cranelift IR.
        //    This IS the forward pass — each WengertOp is compiled to
        //    its runtime FFI call, and ALL intermediate VarId → Value
        //    mappings are recorded in full_vars.
        // ELTLS: free tape-held tensors before clearing the tape flag.
        self.free_tape_held_tensors(builder, state);
        state.flags.in_tape_region = false;
        // Debug: dump primal Wengert ops
        if std::env::var("NSL_DEBUG_WENGERT").is_ok() {
            nsl_log::nsl_log!(INFO, "wengert", 
                "[wengert] primal_vars: {:?}",
                primal_vars.keys().collect::<Vec<_>>()
            );
            for op in &extractor.wengert_list().ops {
                let name = extractor
                    .wengert_list()
                    .var_names
                    .get(&op.result)
                    .cloned()
                    .unwrap_or_default();
                nsl_log::nsl_log!(INFO, "wengert", 
                    "[wengert] VarId {} '{}' = {:?} inputs={:?} in_primal={}",
                    op.result,
                    name,
                    op.op,
                    op.inputs,
                    primal_vars.contains_key(&op.result)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::streamed_slices_match_segments;

    #[test]
    fn the_streamed_slices_must_be_the_segment_slices() {
        // Prologue, two segments, epilogue — as `layerwise::forward_slices`
        // makes them for both plans.
        let tagged = [
            (None, 0, 3),
            (Some(0), 3, 7),
            (Some(1), 7, 12),
            (None, 12, 15),
        ];
        let same = [(0, 3), (3, 7), (7, 12), (12, 15)];
        assert!(streamed_slices_match_segments(&tagged, &same).is_ok());
        // A boundary moved by one op, a slice missing, a slice extra: each
        // would run a segment's frees at the wrong place.
        for streamed in [
            &[(0, 3), (3, 8), (8, 12), (12, 15)][..],
            &[(0, 3), (3, 7), (7, 15)][..],
            &[(0, 3), (3, 7), (7, 12), (12, 14), (14, 15)][..],
        ] {
            let err = streamed_slices_match_segments(&tagged, streamed).unwrap_err();
            assert!(
                err.message.contains("internal compiler error"),
                "{}",
                err.message
            );
        }
    }
}

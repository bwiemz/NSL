//! Section 7e3b of the train block: the CSLA (`--layerwise-accum`)
//! window backward. On an accumulation boundary the buffered window is
//! replayed once per micro-batch — the D1b layer-major schedule fixes the
//! replay ranges, the per-b seeding rebuilds the VarMap from the window's
//! saved imports, each range lowers the same adjoint tape and takes its
//! layer group's fused update right after, the weight-stream prefetch belt
//! double-buffers the next range's parameters, and the window cleanup
//! drops the shells for the next window.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1):
//! 1527 lines, 30 inputs ([`CslaWindowInputs`]), no escaping binding;
//! the four compile-time carriers the save phase fills ([`CslaPending`],
//! [`CslaSchedule`], [`CslaParam`], [`CslaSlotKind`]) moved with it from
//! the driver's local scope. The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the emitted window on the
//! `*_csla` fixtures and `crates/nsl-cli/tests/csla_layerwise_gate.rs`
//! runs it end to end; the refusal texts are found by the CLI composition
//! gate's wholesale sweep of `crates/nsl-codegen/src`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Value};
use cranelift_frontend::{FunctionBuilder, Variable};

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::{
    MomentFill, ParamHookEntry, SURFACE_WEIGHTS, WS_PCIE_FIXED_LAT_US, WS_PREFETCH_MIN_OPS_PER_RANGE,
};
use crate::stmt_csla::{emit_csla_accum_alloc, emit_csla_group_update, MuonCslaCtx};

/// CSLA Stage-2 (D1a): one buffered import slot of a micro-batch's
/// window save list.
pub(crate) enum CslaSlotKind {
    /// i64-typed lowered value (tensor/list pointer or integer).
    /// `owned` carries the forward lowering's ownership type: the
    /// window backward frees Tensor/List-owned slots after their
    /// micro-batch's replay (mirroring `free_wengert_owned_values`).
    Raw {
        owned: Option<crate::wengert::WengertType>,
    },
    /// f64 scalar, stored via a same-width bitcast; never freed.
    F64Bits,
}
/// One trainable tensor parameter's compile-time facts for the
/// layer-major update schedule (D1b).
pub(crate) struct CslaParam {
    pub(crate) name: String,
    /// Primal leaf VarId (for the view-of-θ hazard check).
    pub(crate) primal_vid: crate::wengert::VarId,
    /// Adjoint (gradient) VarId, when the param is read in the
    /// backward; `None` = dead/frozen — no gradient ever
    /// accumulates, but the update still fires (weight decay +
    /// moment decay mutate θ on zero gradients, exactly like the
    /// baseline's unconditional 0..num_params update loop).
    pub(crate) adj_vid: Option<crate::wengert::VarId>,
    /// param_paths index — the universal join key for
    /// param_list/state lists/accum_list.
    pub(crate) accum_idx: i64,
}
/// Compile-time context carried from the save phase (inside the
/// source-AD arm) to the window-backward emission site just before
/// the optimizer gate.
pub(crate) struct CslaPending {
    pub(crate) adjoint: crate::wengert::WengertList,
    pub(crate) slots: Vec<(crate::wengert::VarId, CslaSlotKind)>,
    pub(crate) seed_base: crate::wengert_lower::VarMap,
    pub(crate) param_adj_set: std::collections::HashSet<crate::wengert::VarId>,
    /// adjoint grad VarId → compile-time accum_list index.
    pub(crate) hook_accum_idx: std::collections::HashMap<crate::wengert::VarId, i64>,
    pub(crate) accum_scale: f64,
    /// Slot index of the loss import, when the adjoint reads the loss
    /// tensor itself: that slot's per-b free must skip the window's
    /// LAST micro-batch (the current iteration's loss — still read by
    /// on_step/on_epoch after the backward phase; it is freed by the
    /// conditional at the per-iteration loss-free site instead).
    pub(crate) loss_slot: Option<usize>,
    /// D1b: the per-param update facts (the layerwise plan itself
    /// is consumed by the pre-forward schedule derivation and no
    /// longer travels here — `schedule` below is its product).
    pub(crate) params: Vec<CslaParam>,
    /// PRIMAL-side zero-copy views of trainable params (transpose /
    /// reshape chains rooted at a param leaf): view result vid →
    /// param leaf vid. These ride the window buffer as slots but
    /// ALIAS θ's storage — a read after θ's per-layer update would
    /// see half-updated weights (review D1b-2).
    pub(crate) primal_view_of:
        std::collections::HashMap<crate::wengert::VarId, crate::wengert::VarId>,
    /// LSE tape-carry (lifts the D1a fused-SDPA refusal): one extra
    /// inner-list slot per `flash_attn_aux` entry, holding the
    /// forward-saved logsumexp (a real tensor when the fused
    /// dispatch launched, runtime 0 when it declined — the FFI's
    /// existing decline semantics). `(inner slot index, the fwd-out
    /// vids whose seeded Values must key the aux re-insert)`. The
    /// replay re-binds `flash_attn_aux[seed[vid]] = lse[b]` before
    /// each consuming range's lowering, so the emitted SDPA backward
    /// reads micro-batch b's LSE instead of missing the Value-keyed
    /// side-band.
    pub(crate) lse_slots: Vec<(usize, Vec<crate::wengert::VarId>)>,
    /// The lse Values actually pushed as window slots (tape-carry
    /// review F2): the u32::MAX-sentinel bulk-free retention is
    /// per-Value — aux entries whose out mapped to NO buffered slot
    /// were never pushed and their LSEs must still free per
    /// iteration or every fused-fired one leaks for the whole run.
    pub(crate) lse_pushed: std::collections::HashSet<Value>,
    /// Fused-CE tape-carry: one extra inner-list slot per
    /// `fused_ce_fwd_lse` entry — `(slot index, the fwd-result
    /// vids whose SEEDED Values key the replay re-bind)`. The
    /// replay re-inserts `fused_ce_fwd_lse[seed[vid]] = lse[b]`
    /// before each consuming range's lowering; the emitted fused
    /// backward then consumes AND frees the buffered tensor, so
    /// these slots take no per-b free (teardown sweeps the
    /// trailing partial window only).
    pub(crate) fce_slots: Vec<(usize, Vec<crate::wengert::VarId>)>,
    /// D2b part 2: the layer-major schedule, computed ONCE before
    /// the forward lowering so the segment-streamed forward and the
    /// window backward agree on ranges, update grouping, and the
    /// streamed-param set BY CONSTRUCTION (two independent
    /// derivations could disagree on which params the forward must
    /// re-upload — a null-data crash at best).
    pub(crate) schedule: CslaSchedule,
}
/// D2b part 2: the shared layer-major schedule (see
/// `CslaPending.schedule`).
pub(crate) struct CslaSchedule {
    /// Positional partition of the FINAL adjoint into replay
    /// ranges (prologue, one per layer in backward order; the last
    /// range swallows the embedding-backward epilogue ops).
    pub(crate) ranges: Vec<crate::layerwise::ReplayRange>,
    /// Per-range accum/param_list indices updating right after
    /// that range's replay.
    pub(crate) layer_group: Vec<Vec<i64>>,
    /// Epilogue group: every param_paths slot not claimed by a
    /// range (globals / tied / cross-layer / dead params).
    pub(crate) global_group: Vec<i64>,
    /// Weight-streamed params (`--weight-stream`): layer-grouped
    /// minus view-rooted, sorted. Empty when streaming is off.
    pub(crate) ws_streamed: Vec<i64>,
    /// Per-range Σ elements of its STREAMED params (0 where shapes
    /// were symbolic). Item 11 calibration: pack bytes = elems × 4
    /// (GPU f32) drive the prefetch overlap gate's transfer-time
    /// estimate against the target GpuSpec.
    pub(crate) range_pack_elems: Vec<u64>,
    /// Item 3: the single ParameterPlan — per-param residency,
    /// storage dtype and sharding — derived once from `ws_streamed`
    /// plus the storage flags. Both registration sites (the
    /// pre-forward belt and this window's belt) read their
    /// `nsl_sr_bf16_note_param` / `nsl_weight_stream_register`
    /// decisions from it instead of re-spelling the flag conditions,
    /// and it is baked into the binary for the runtime cross-check.
    pub(crate) plan: crate::parameter_plan::ParameterPlan,
}

/// Every binding of `compile_train_block_inner` the window backward reads;
/// names are the driver's.
pub(crate) struct CslaWindowInputs<'a> {
    /// The gradient-accumulation buffer list (`Some` whenever CSLA is on).
    pub(crate) accum_list: Option<Value>,
    /// Muon's AdamW-routed learning rate, as a ratio of `lr_value` at the step.
    pub(crate) adamw_lr_value: Option<f64>,
    pub(crate) lr_value: f64,
    pub(crate) beta1_value: f64,
    pub(crate) beta2_value: f64,
    pub(crate) dampening_value: f64,
    pub(crate) eps_value: f64,
    pub(crate) momentum_value: f64,
    pub(crate) weight_decay_value: f64,
    pub(crate) ns_steps_value: f64,
    pub(crate) nesterov_value: bool,
    /// The CPDT per-parameter dtype-code lists (m, v).
    pub(crate) cpdt_precision_dtypes: Option<(Value, Value)>,
    /// Muon's per-parameter m dtype codes (`--muon-state-dtype bf16`).
    pub(crate) muon_state_m_codes: Option<Value>,
    /// The window save-list and dict-list variables (allocated when CSLA is on).
    pub(crate) csla_buffers: Option<(Variable, Variable)>,
    /// The compile-time context the save phase left for this site (taken here).
    pub(crate) csla_pending: Option<CslaPending>,
    pub(crate) fase_plan: &'a crate::fase::FasePlan,
    pub(crate) grad_accumulation_steps: i64,
    /// The DataLoader handle, when the `data:` section declared one.
    pub(crate) has_dataloader: Option<Value>,
    pub(crate) lr_var: Variable,
    pub(crate) should_step_var: Variable,
    pub(crate) step_count_var: Variable,
    /// The stage-3 deferred-moment latch.
    pub(crate) moment_fill_latch: Option<Value>,
    /// Muon's per-parameter route flags.
    pub(crate) muon_route_list: Option<Value>,
    pub(crate) num_params_val: Value,
    pub(crate) param_list: Value,
    pub(crate) state_list_1: Value,
    pub(crate) state_list_2: Value,
    pub(crate) num_state_buffers: usize,
    pub(crate) optimizer_name: &'a str,
    pub(crate) param_paths: &'a [String],
}

impl Compiler<'_> {
    /// Emit the CSLA window backward (see the module header).
    pub(crate) fn emit_csla_window_backward(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: CslaWindowInputs<'_>,
    ) -> Result<(), CodegenError> {
        let CslaWindowInputs {
            accum_list,
            adamw_lr_value,
            lr_value,
            beta1_value,
            beta2_value,
            dampening_value,
            eps_value,
            momentum_value,
            weight_decay_value,
            ns_steps_value,
            nesterov_value,
            cpdt_precision_dtypes,
            muon_state_m_codes,
            csla_buffers,
            mut csla_pending,
            fase_plan,
            grad_accumulation_steps,
            has_dataloader,
            lr_var,
            should_step_var,
            step_count_var,
            moment_fill_latch,
            muon_route_list,
            num_params_val,
            param_list,
            state_list_1,
            state_list_2,
            num_state_buffers,
            optimizer_name,
            param_paths,
        } = inputs;

        // ── 7e3b. CSLA Stage-2: window backward phase ───────────────────
        // On accumulation boundaries, replay the adjoint once per buffered
        // micro-batch. Each replay seeds a fresh VarMap from seed_base (the
        // step iteration's forward values — loop-invariant Params/Constants
        // resolve there) overridden with that micro-batch's buffered imports,
        // then lowers the SAME adjoint tape inside a runtime b-loop (one
        // tape, N executions). The FASE hook accumulates every parameter
        // gradient into the same m_partial slot in micro-batch-ascending
        // order — exactly the baseline's per-parameter accumulation sequence,
        // so the optimizer region below consumes bit-identical m_partial.
        if let Some(pending) = csla_pending.take() {
            let (saves_outer_var, dicts_var) =
                csla_buffers.expect("csla_buffers allocated when csla_pending set");
            let accum_val = accum_list
                .ok_or_else(|| CodegenError::new("csla requires accum_list"))?;

            // ── D1b schedule (precomputed) ──────────────────────────────
            // D2b part 2: ranges / grouping / streamed set were derived
            // ONCE in the pre-forward pure pipeline (the segment-streamed
            // forward consumed them at emission time) and travel here via
            // `pending.schedule` — one derivation, both sides agree by
            // construction. The `[csla] layer-major schedule:` line prints
            // at the derivation site.
            let ranges = &pending.schedule.ranges;
            let layer_group = &pending.schedule.layer_group;
            let global_group = &pending.schedule.global_group;
            let n_ranges = ranges.len();
            let last_ri = n_ranges - 1;

            // Carry analysis: adjoint values produced in one range and read
            // in a later one (the boundary adjoints d(residual-after-L) plus
            // any straggler temporaries). Slot order sorted-by-vid.
            let mut produced_range: std::collections::HashMap<crate::wengert::VarId, usize> =
                std::collections::HashMap::new();
            for (ri, r) in ranges.iter().enumerate() {
                for op in &pending.adjoint.ops[r.start..r.end] {
                    produced_range.insert(op.result, ri);
                }
            }
            let mut carry_set: std::collections::BTreeSet<crate::wengert::VarId> =
                Default::default();
            let mut carry_last_read: std::collections::HashMap<crate::wengert::VarId, usize> =
                std::collections::HashMap::new();
            let mut carry_freed_by_marker: std::collections::HashSet<crate::wengert::VarId> =
                Default::default();
            for (ri, r) in ranges.iter().enumerate() {
                for op in &pending.adjoint.ops[r.start..r.end] {
                    for &input in &op.inputs {
                        let Some(&pr) = produced_range.get(&input) else {
                            continue;
                        };
                        if pr < ri {
                            carry_set.insert(input);
                            let e = carry_last_read.entry(input).or_insert(ri);
                            if *e < ri {
                                *e = ri;
                            }
                            if matches!(op.op, crate::wengert::PrimalOp::FreeTensor) {
                                carry_freed_by_marker.insert(input);
                            }
                        }
                    }
                }
            }
            let carry_vids: Vec<crate::wengert::VarId> = carry_set.into_iter().collect();
            let carry_slot: std::collections::HashMap<crate::wengert::VarId, usize> =
                carry_vids.iter().enumerate().map(|(i, v)| (*v, i)).collect();
            let n_carry = carry_vids.len();
            let mut exports_per_range: Vec<Vec<crate::wengert::VarId>> =
                vec![Vec::new(); n_ranges];
            for &v in &carry_vids {
                exports_per_range[produced_range[&v]].push(v);
            }
            let mut imports_per_range: Vec<Vec<crate::wengert::VarId>> =
                vec![Vec::new(); n_ranges];
            for (ri, r) in ranges.iter().enumerate() {
                let mut seen = std::collections::HashSet::new();
                for op in &pending.adjoint.ops[r.start..r.end] {
                    for &input in &op.inputs {
                        if carry_slot.contains_key(&input)
                            && produced_range[&input] < ri
                            && seen.insert(input)
                        {
                            imports_per_range[ri].push(input);
                        }
                    }
                }
                imports_per_range[ri].sort_unstable();
            }
            // Fixpoint-extended last-use POSITIONS (review D1b-3): frees must
            // key off list-membership-extended last use — a slot or carry
            // consumed by a list-building op in range R whose LIST is read in
            // range S>R must survive to S (lists hold raw, un-refcounted
            // element pointers; freeing at the direct read would dangle
            // them). Seeding stays direct-read-based (only actual op inputs
            // need values).
            let extended_last_pos: std::collections::HashMap<crate::wengert::VarId, usize> = {
                let mut lu: std::collections::HashMap<crate::wengert::VarId, usize> =
                    std::collections::HashMap::new();
                for (idx, op) in pending.adjoint.ops.iter().enumerate() {
                    for &input in &op.inputs {
                        lu.insert(input, idx);
                    }
                }
                crate::ccr::extend_last_use_through_lists(&pending.adjoint, &mut lu);
                lu
            };
            let range_of_pos = |pos: usize| -> usize {
                ranges
                    .iter()
                    .position(|r| pos >= r.start && pos < r.end)
                    .unwrap_or(last_ri)
            };

            // Carried Tensor-typed values with NO FreeTensor consumer get an
            // explicit free after their (fixpoint-extended) last consuming
            // range's replay — the belt for values the last-use-frees pass
            // protected or never saw. Skips at emission time (below) also
            // exclude hook-freed raw grads (review D1b-1: a range boundary
            // landing ON a reduce_to_shape grad op makes its raw-grad input a
            // carry that the hook's extra free already releases — freeing it
            // again here would be a double free). List-typed carries are
            // never freed (matching free_eligible's List exclusion).
            let mut carry_explicit_free: Vec<Vec<crate::wengert::VarId>> =
                vec![Vec::new(); n_ranges];
            for &v in &carry_vids {
                if carry_freed_by_marker.contains(&v) {
                    continue;
                }
                if !matches!(
                    pending.adjoint.var_types.get(&v),
                    Some(crate::wengert::WengertType::Tensor) | None
                ) {
                    continue;
                }
                let last_pos = extended_last_pos
                    .get(&v)
                    .copied()
                    .unwrap_or(pending.adjoint.ops.len().saturating_sub(1));
                let ri = range_of_pos(last_pos).max(carry_last_read[&v]);
                carry_explicit_free[ri].push(v);
            }
            // Primal-slot schedule: which ranges read each buffered slot
            // (seed only there — direct reads) and which range is a slot's
            // fixpoint-extended LAST reader (free it there, with the
            // loss-slot b<N-1 conditional).
            let mut slot_seed_per_range: Vec<Vec<usize>> = vec![Vec::new(); n_ranges];
            let mut slot_free_per_range: Vec<Vec<usize>> = vec![Vec::new(); n_ranges];
            {
                let slot_of: std::collections::HashMap<crate::wengert::VarId, usize> = pending
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(i, (v, _))| (*v, i))
                    .collect();
                for (ri, r) in ranges.iter().enumerate() {
                    let mut seen = std::collections::HashSet::new();
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        for &input in &op.inputs {
                            if let Some(&si) = slot_of.get(&input)
                                && seen.insert(si)
                            {
                                slot_seed_per_range[ri].push(si);
                            }
                        }
                    }
                    slot_seed_per_range[ri].sort_unstable();
                }
                for (vid, si) in &slot_of {
                    if let Some(&pos) = extended_last_pos.get(vid) {
                        slot_free_per_range[range_of_pos(pos)].push(*si);
                    }
                }
                for v in slot_free_per_range.iter_mut() {
                    v.sort_unstable();
                }
            }

            // Tape-carry review F3: an SDPA backward CLUSTER — the recompute
            // clone of the forward (whose Value keys the aux re-bind under
            // the Block policy) plus its dQ/dK/dV extract ops — must not be
            // split across replay ranges. The aux insert and the
            // flash_attn_bwd_cache are Value-keyed within one range's
            // lowering: a split would silently downgrade the packed backward
            // to the CPU reference (numeric divergence) and double-emit the
            // full backward triplet. Sequential-block models are
            // structurally safe (a layer's extracts sit inside its own
            // range); this guard makes any future violation loud.
            {
                let mut fwdout_range: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = std::collections::HashMap::new();
                for (ri, r) in ranges.iter().enumerate() {
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        let is_extract = matches!(
                            op.op,
                            crate::wengert::PrimalOp::FlashAttentionBackwardExtract { .. }
                                | crate::wengert::PrimalOp::FlashAttentionBackwardExtractPacked { .. }
                        );
                        if !is_extract {
                            continue;
                        }
                        let Some(&fo) = op.inputs.get(4) else { continue };
                        if let Some(&pr) = produced_range.get(&fo)
                            && pr != ri
                        {
                            return Err(CodegenError::new(format!(
                                "--layerwise-accum: an SDPA backward extract in \
                                     replay range {ri} reads a forward clone from \
                                     range {pr} — the Value-keyed attention \
                                     side-bands cannot cross ranges. This adjoint \
                                     shape is unsupported; drop --layerwise-accum",
                            )));
                        }
                        if let Some(&prev) = fwdout_range.get(&fo) {
                            if prev != ri {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: sibling SDPA backward \
                                     extracts for one attention op are split \
                                     across replay ranges {prev} and {ri}. This \
                                     adjoint shape is unsupported; drop \
                                     --layerwise-accum",
                                )));
                            }
                        } else {
                            fwdout_range.insert(fo, ri);
                        }
                    }
                }
            }

            // Fused-CE cluster guard (F3 twin): the three
            // FusedLinearCeBackwardExtract components of one op share the
            // Value-keyed `fused_ce_bwd_cache` (component 0 launches and
            // populates; 1/2 read; 2 evicts) and the LSE is consumed
            // exactly once — a split across replay ranges would relaunch
            // the backward kernel per range and hit a consumed-LSE
            // CodegenError. Structurally the extracts sit together in the
            // prologue range (the fused op is the CCR epilogue's loss
            // head); this guard makes any future violation loud.
            {
                let mut fce_range: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = std::collections::HashMap::new();
                for (ri, r) in ranges.iter().enumerate() {
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        if !matches!(
                            op.op,
                            crate::wengert::PrimalOp::FusedLinearCeBackwardExtract { .. }
                        ) {
                            continue;
                        }
                        let Some(&fr) = op.inputs.get(5) else { continue };
                        if let Some(&prev) = fce_range.get(&fr) {
                            if prev != ri {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: sibling fused-CE backward \
                                     extracts for one @fused_lm_ce op are split \
                                     across replay ranges {prev} and {ri}. This \
                                     adjoint shape is unsupported; drop \
                                     --layerwise-accum",
                                )));
                            }
                        } else {
                            fce_range.insert(fr, ri);
                        }
                    }
                }
            }

            // LSE tape-carry: which range frees each buffered logsumexp —
            // the last (fixpoint-extended) range reading ANY of its fwd-out
            // vids; that range necessarily seeds the vid, so the loaded LSE
            // value is in scope for the null-safe free.
            let lse_free_range: Vec<usize> = pending
                .lse_slots
                .iter()
                .map(|(_, vids)| {
                    vids.iter()
                        .filter_map(|v| extended_last_pos.get(v))
                        .map(|&p| range_of_pos(p))
                        .max()
                        .unwrap_or(last_ri)
                })
                .collect();

            // Review D1b-2: zero-copy VIEWS OF θ (transpose/reshape chains
            // rooted at a trainable param) alias parameter storage, so a
            // read through one AFTER that param's per-layer update would see
            // half-updated weights — invisible to the classification, which
            // tracks only the param's DIRECT reads. Refuse any read of a
            // view-of-θ in a range strictly after θ's update range. Views of
            // epilogue-group params (update range = MAX — after every
            // range) can never trip this: the tied-embedding LM head is the
            // canonical safe case. Both primal-side view chains (buffered
            // slots) and adjoint-side view chains are tracked transitively.
            {
                let idx_update_range: std::collections::HashMap<i64, usize> = layer_group
                    .iter()
                    .enumerate()
                    .flat_map(|(ri, g)| g.iter().map(move |&i| (i, ri)))
                    .collect();
                let update_range_of_vid: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = pending
                    .params
                    .iter()
                    .map(|cp| {
                        (
                            cp.primal_vid,
                            idx_update_range
                                .get(&cp.accum_idx)
                                .copied()
                                .unwrap_or(usize::MAX),
                        )
                    })
                    .collect();
                let mut view_param: std::collections::HashMap<
                    crate::wengert::VarId,
                    crate::wengert::VarId,
                > = pending.primal_view_of.clone();
                for (ri, r) in ranges.iter().enumerate() {
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        for &input in &op.inputs {
                            if let Some(&p) = view_param.get(&input) {
                                let ur = update_range_of_vid
                                    .get(&p)
                                    .copied()
                                    .unwrap_or(usize::MAX);
                                if ri > ur {
                                    return Err(CodegenError::new(format!(
                                        "--layerwise-accum: a view of parameter \
                                         VarId {p} is read in replay range {ri}, \
                                         after the param's per-layer update in \
                                         range {ur} — the view aliases θ's storage \
                                         and would see half-updated weights. This \
                                         adjoint shape is unsupported; drop \
                                         --layerwise-accum",
                                    )));
                                }
                            }
                        }
                        if crate::wengert::is_view_producing_op(&op.op) {
                            for &input in &op.inputs {
                                if update_range_of_vid.contains_key(&input) {
                                    view_param.insert(op.result, input);
                                } else if let Some(&p) = view_param.get(&input) {
                                    view_param.insert(op.result, p);
                                }
                            }
                        }
                    }
                }
            }

            // ── Window region ───────────────────────────────────────────
            let bwd_block = builder.create_block();
            let bwd_join = builder.create_block();
            let ss = builder.use_var(should_step_var);
            builder.ins().brif(ss, bwd_block, &[], bwd_join, &[]);
            builder.switch_to_block(bwd_block);
            builder.seal_block(bwd_block);
            state.current_block = Some(bwd_block);

            // Anti-vacuity mark + loud window-size assert: the modulo fires
            // every N micro-batches exactly (the counter is global, windows
            // straddle epochs, the trailing partial window never fires), so
            // the buffer MUST hold exactly N entries here.
            self.compile_call_by_name(builder, "nsl_csla_window_mark", &[])?;
            let so = builder.use_var(saves_outer_var);
            let win_len = self.compile_call_by_name(builder, "nsl_list_len", &[so])?;
            let n_val = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
            let len_ok = builder.ins().icmp(IntCC::Equal, win_len, n_val);
            let len_msg = format!(
                "csla window backward: buffered micro-batch count != \
                 grad_accumulation ({grad_accumulation_steps})"
            );
            self.intern_string(&len_msg)?;
            let len_msg_ptr = self.compile_string_literal(builder, &len_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[len_ok, len_msg_ptr])?;

            // D2b weight eviction, part 2 (whole-loop streaming): the
            // forward already registered + evicted every streamed param at
            // step-body top and re-uploads per segment, so this window
            // opens with them evicted. The register loop stays as an
            // idempotent belt (a register on an evicted registered tensor
            // is a no-op) — it keeps the window arm self-sufficient if the
            // forward emission ever changes. The streamed SET comes from
            // the shared pre-forward schedule (view-rooted params already
            // excluded there — review D2b-1); each layer re-uploads at its
            // range head and evicts+writes-back after its update; epilogue
            // params never stream.
            let ws_active = self.compile_options.weight_stream.enabled;
            let ws_streamed: std::collections::HashSet<i64> =
                pending.schedule.ws_streamed.iter().copied().collect();
            // P3 ZeRO-3: the sharded set is READ FROM THE PLAN rather than
            // re-derived as `(zero_stage == 3).then(|| ws_streamed.clone())`.
            // The two are equal by construction — `derive` marks exactly the
            // streamed set Sharded under stage 3 — but sourcing it here is
            // what makes the plan the single definition of "sharded" instead
            // of one of two independent spellings that merely agree.
            let zero3_streamed: Option<std::collections::HashSet<i64>> = self
                .features
                .zero_stage
                .filter(|&s| s == 3)
                .map(|_| {
                    pending
                        .schedule
                        .plan
                        .entries()
                        .iter()
                        .filter(|e| e.sharding.is_sharded())
                        .map(|e| e.idx)
                        .collect()
                });
            // Item 11: the ELEMENTWISE subset — these params skip the owner
            // gate (every rank steps its own slice) and their group update
            // emits nsl_zero3_elem_adamw_step instead of the stdlib call.
            // Same single-source rule as zero3_streamed: read the plan.
            let zero3_elem: Option<std::collections::HashSet<i64>> = self
                .features
                .zero_stage
                .filter(|&s| s == 3 && self.features.zero_elementwise)
                .map(|_| {
                    pending
                        .schedule
                        .plan
                        .entries()
                        .iter()
                        .filter(|e| e.is_elementwise())
                        .map(|e| e.idx)
                        .collect()
                });
            if ws_active {
                // P4 item 17: activate the bf16 mirror backend and assign
                // each streamed param its stable SR counter block BEFORE the
                // first registration (register aborts on an un-noted param).
                // Armed from the plan, for the same reason as the pre-forward
                // belt: enable and note must not be spelled from two sources.
                if pending.schedule.plan.needs_sr_backend() {
                    self.compile_call_by_name(builder, "nsl_sr_bf16_enable", &[])?;
                }
                for &idx in &pending.schedule.ws_streamed {
                    // Item 3: same plan, same decision as the pre-forward
                    // belt — the two sites can no longer disagree about who
                    // gets an SR counter block.
                    let entry = usize::try_from(idx)
                        .ok()
                        .and_then(|u| pending.schedule.plan.entries().get(u))
                        .ok_or_else(|| {
                            CodegenError::new(format!(
                                "parameter plan: window streaming schedule \
                                 registers parameter {idx}, which the plan does \
                                 not cover"
                            ))
                        })?;
                    let needs_sr_note = entry.needs_sr_note();
                    let needs_elem_mark = entry.is_elementwise();
                    let iv = builder.ins().iconst(cl_types::I64, idx);
                    let pw =
                        self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                    if needs_sr_note {
                        self.compile_call_by_name(
                            builder,
                            "nsl_sr_bf16_note_param",
                            &[pw, iv],
                        )?;
                    }
                    // Item 11: mark-before-register, from the same plan entry
                    // as the pre-forward belt; rc asserted for the same
                    // misattribution reason.
                    if needs_elem_mark {
                        // Item 16x11: storage decision from the SAME plan
                        // entry that drove the note above.
                        let srv = builder
                            .ins()
                            .iconst(cl_types::I64, i64::from(needs_sr_note));
                        let mrc = self.compile_call_by_name(
                            builder,
                            "nsl_zero3_mark_elementwise",
                            &[pw, iv, srv],
                        )?;
                        let mz = builder.ins().iconst(cl_types::I64, 0);
                        let mok = builder.ins().icmp(IntCC::Equal, mrc, mz);
                        let mmsg = "nsl: zero3 elementwise mark failed \
                                    (ZeRO context missing?) — aborting";
                        self.intern_string(mmsg)?;
                        let mmp = self.compile_string_literal(builder, mmsg)?;
                        self.compile_call_by_name(builder, "nsl_assert", &[mok, mmp])?;
                    }
                    self.compile_call_by_name(builder, "nsl_weight_stream_register", &[pw])?;
                }

                // Item C: ZeRO-3's deferred moment fill. It MUST sit here —
                // inside `ws_active` (stage 3 refuses without
                // --weight-stream, so this is unconditional under stage 3)
                // and AFTER the mark/register loop, because an elementwise
                // moment is sized from the slice the REGISTER carved. The
                // window belt, not the pre-forward belt: that one is behind
                // `if let Some(ws_fwd_plan)` and would go vacuous whenever
                // the forward schedule is absent.
                if self.features.zero_stage == Some(3) {
                    let plan_entries = pending.schedule.plan.entries();
                    // A plan-coverage miss must REFUSE, not fall back to the
                    // replicated arm: `Full` is the one shape that is not
                    // counted against `optim_elems`, so a param sliding into
                    // it would drop out of both sides of the `r0+r1 == full`
                    // identity and every gate here would stay green while the
                    // rank quietly held a full moment replica. Same rule as
                    // the register belt's positional lookup above.
                    let modes: Vec<MomentFill> = (0..param_paths.len())
                        .map(|u| {
                            let e = plan_entries.get(u).ok_or_else(|| {
                                CodegenError::new(format!(
                                    "parameter plan: the ZeRO-3 deferred moment \
                                     fill needs an entry for parameter {u}, \
                                     which the plan does not cover"
                                ))
                            })?;
                            Ok(if e.is_elementwise() {
                                MomentFill::Elementwise
                            } else if e.sharding.is_sharded() {
                                MomentFill::OwnerGated
                            } else {
                                MomentFill::Full
                            })
                        })
                        .collect::<Result<Vec<_>, CodegenError>>()?;
                    // Same precision-list projection as section 4, so a slot
                    // filled here is byte-identical to what setup would have
                    // built for it.
                    let m_codes = cpdt_precision_dtypes.map(|(m, _)| m).or(muon_state_m_codes);
                    let v_codes = cpdt_precision_dtypes.map(|(_, v)| v);
                    // Muon's v-skip condition, verbatim from section 4.
                    // `muon_resident_m` needs --optim-state-offload, which
                    // stage 3 refuses outright, so the m side has no Muon arm.
                    let muon_v_gate = if optimizer_name == "muon"
                        && !self.compile_options.optim_state_offload
                        && cpdt_precision_dtypes.is_none()
                    {
                        muon_route_list
                    } else {
                        None
                    };
                    self.emit_deferred_moment_fill(
                        builder,
                        state,
                        &modes,
                        param_list,
                        state_list_1,
                        state_list_2,
                        num_state_buffers,
                        m_codes,
                        v_codes,
                        muon_v_gate,
                        self.compile_options.optim_state_offload,
                        moment_fill_latch.expect("latch allocated under stage 3"),
                    )?;
                }
            }

            // Bias correction — the same expression the (now-bypassed)
            // optimizer site computes; step_count is untouched between here
            // and there, so every group in this window shares one pair.
            let sc_val = builder.use_var(step_count_var);
            let one_i64 = builder.ins().iconst(cl_types::I64, 1);
            let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
            let ga_const = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
            let opt_step = builder.ins().sdiv(sc_plus_one, ga_const);
            let b1c = builder.ins().f64const(fase_plan.recipe.beta1);
            let b2c = builder.ins().f64const(fase_plan.recipe.beta2);
            let bc1_inv = self.compile_call_by_name(
                builder,
                "nsl_bias_correction_inv",
                &[b1c, opt_step],
            )?;
            let bc2_inv = self.compile_call_by_name(
                builder,
                "nsl_bias_correction_inv",
                &[b2c, opt_step],
            )?;
            let wrap_precision = cpdt_precision_dtypes.is_some();
            // Review D2a-1: csla x offload x reduced-precision moments (the
            // P0.3 combined cast_from_host arm) became reachable when the
            // offload refusal narrowed. The walk-through found no defect,
            // but zero gates cover the combination — refuse until a parity
            // gate exists (deferral-must-refuse).
            if wrap_precision && self.compile_options.optim_state_offload {
                return Err(CodegenError::new(
                    "--layerwise-accum with --optim-state-offload does not yet \
                     support a CPDT reduced-precision moment plan (the P0.3 \
                     combined staging arm is ungated under the layerwise \
                     schedule). Drop the precision plan or --optim-state-offload",
                ));
            }
            let two_state = num_state_buffers >= 2;

            // P1 Muon item 11: prebuild the muon group-update context —
            // hyperparameter constants, the routing-flag list from 4a, and
            // the micro-batch counter t (the non-CSLA FullBuffer semantic).
            let muon_csla_ctx: Option<MuonCslaCtx> = if optimizer_name == "muon" {
                let route_list = muon_route_list
                    .expect("muon route list is built before state buffers (4a)");
                let mangled = "nsl_optim_muon__muon_step";
                let opt_fn = if self.registry.functions.contains_key(mangled)
                    || self.registry.runtime_fns.contains_key(mangled)
                {
                    mangled.to_string()
                } else if self.registry.functions.contains_key("muon_step") {
                    "muon_step".to_string()
                } else {
                    mangled.to_string()
                };
                let t_f = builder.ins().fcvt_from_sint(cl_types::F64, sc_plus_one);
                let lr_now = builder.use_var(lr_var);
                let ratio = adamw_lr_value.map(|a| a / lr_value).unwrap_or(1.0);
                let ratio_const = builder.ins().f64const(ratio);
                let adamw_lr_now = builder.ins().fmul(lr_now, ratio_const);
                Some(MuonCslaCtx {
                    route_list,
                    opt_fn,
                    ns_steps: ns_steps_value,
                    lr: lr_now,
                    adamw_lr: adamw_lr_now,
                    momentum: builder.ins().f64const(momentum_value),
                    dampening: builder.ins().f64const(dampening_value),
                    weight_decay: builder.ins().f64const(weight_decay_value),
                    nesterov: builder
                        .ins()
                        .iconst(cl_types::I8, if nesterov_value { 1 } else { 0 }),
                    beta1: builder.ins().f64const(beta1_value),
                    beta2: builder.ins().f64const(beta2_value),
                    eps: builder.ins().f64const(eps_value),
                    t: t_f,
                    step_var: step_count_var,
                })
            } else {
                None
            };

            // Item 8, CSLA half: when the group takes the fused AdamW step
            // with no muon routing, no ZeRO-3 owner gates, and no CPDT
            // precision or offload envelope, each layer-group update
            // collapses into ONE pointer-table launch over the group's
            // indices — nsl_fase_fused_adamw_step_multi_idx for plain f32,
            // or its bf16-sr twin (SR arm, same item) which performs the
            // identical per-member SR step and coherence widen. Both are
            // bit-identical per element to the per-param loop they replace
            // (same kernel bodies, table addressing; the SR dither is a
            // pure function of (param, element, step)).
            //
            // Fallback semantics differ per twin: the f32 runtime entry
            // demotes non-uniform members (CPU tensors, tied-θ aliases,
            // oversize params) to its sequential arm, while the SR entry
            // falls back per-param only for UN-STREAMED members (no bf16
            // mirror — which is where tied/view-rooted params land, since
            // registration refuses non-owners) and asserts the streamed
            // set's m/v/mp uniformity outright. Admission mirrors the
            // FullBuffer multi arm and shares its kill-switches.
            //
            // If AdamW parameter groups (`no_decay`) are ever threaded into
            // this batched call, the per-param loop in
            // emit_csla_group_update must gain the same group plumbing
            // FIRST — it bakes the recipe's flat λ today, and it is the
            // NSL_FASE_MULTI_STEP=0 parity reference the SR gate diffs
            // against.
            let csla_multi_scalars: Option<crate::stmt_fase::FusedAdamwScalars> =
                if muon_csla_ctx.is_none()
                    && !wrap_precision
                    && !self.compile_options.optim_state_offload
                    // bf16-sr admits (item 8, SR arm): the group update
                    // selects the SR twin of the multi_idx launch, which
                    // performs the identical per-member SR step and coherence
                    // widen. The layerwise schedule refuses grad_clip, so
                    // the SR entries' no-clip contract can never be violated
                    // from this site.
                    && zero3_streamed.is_none()
                    && two_state
                    && !self.compile_options.training_reference
                    && std::env::var("NSL_FASE_FUSED_STEP").ok().as_deref() != Some("0")
                    && std::env::var("NSL_FASE_MULTI_STEP").ok().as_deref() != Some("0")
                {
                    Self::match_adamw_program(&crate::fase_optimizer::emit_final_step(
                        &fase_plan.recipe,
                    ))
                } else {
                    None
                };

            // Item 11: the elementwise step re-runs the fused-AdamW math on a
            // slice, so the update program must BE fused-AdamW-shaped. The
            // optimizer-name refusal happened at setup; this is the belt for
            // a recipe that drifted from the name (e.g. an adam variant
            // emit_final_step no longer matches).
            let zero3_elem_scalars: Option<crate::stmt_fase::FusedAdamwScalars> =
                if zero3_elem.as_ref().is_some_and(|s| !s.is_empty()) {
                    if wrap_precision || self.compile_options.training_reference {
                        return Err(CodegenError::new(
                            "--zero-elementwise does not compose with a CPDT \
                             reduced-precision moment plan or \
                             --training-reference: the elementwise step is the \
                             fused f32 kernel on a slice, which those modes \
                             replace. Drop the conflicting option",
                        ));
                    }
                    let sc = Self::match_adamw_program(&crate::fase_optimizer::emit_final_step(
                        &fase_plan.recipe,
                    ));
                    if sc.is_none() {
                        return Err(CodegenError::new(
                            "--zero-elementwise requires the fused-AdamW update \
                             program (the elementwise step IS that math on a \
                             slice) — this train block's optimizer recipe does \
                             not match it. Use AdamW/Adam, or drop \
                             --zero-elementwise",
                        ));
                    }
                    sc
                } else {
                    None
                };

            // Global/epilogue accumulators live for the whole window (their
            // grads may come from any range).
            emit_csla_accum_alloc(self, builder, param_list, accum_val, global_group)?;

            // P0.3: open a WINDOW-scoped grad-integrity step. The CSLA
            // schedule delivers each parameter's gradient somewhere in the
            // range loop below (layer-local params in their range, epilogue/
            // cross-layer params from any range), all before the epilogue
            // update closes the bracket — so one begin/end pair per window
            // gives exactly "every optimizer step, every trainable param got
            // a usable gradient", the property the gate documents.
            //
            // The window bracket merges every (param, micro-batch) note into
            // one verdict, so present/finite/nonzero cannot tell a param that
            // got all N contributions from one that got k<N — and since
            // `fase_emit_accumulate` scales by 1/N PER NOTE, the latter's
            // gradient is k/N of its true magnitude and biased toward the
            // micro-batches that did contribute. Declaring the expected count
            // is what restores that resolution: the b-loop below replays the
            // range for each of the window's `grad_accumulation_steps`
            // buffered micro-batches (the `nsl_assert` above pins the buffer
            // to exactly that length), and each param's adjoint op lives in
            // exactly one range, so the hook must fire exactly N times per
            // param per window.
            if self.compile_options.grad_integrity {
                let expected_notes = builder
                    .ins()
                    .iconst(cl_types::I64, grad_accumulation_steps.max(1));
                self.compile_call_by_name(
                    builder,
                    "nsl_grad_integrity_step_begin",
                    &[num_params_val, expected_notes],
                )?;
            }

            // Cross-range adjoint carry: one inner list per micro-batch,
            // slot-indexed, created by the first range's replay loop.
            let carry_outer = if n_carry > 0 {
                Some(self.compile_call_by_name(builder, "nsl_list_new", &[])?)
            } else {
                None
            };

            // Exports that ghost-skipped at their producing range (compile-
            // time knowledge accumulated range by range): later ranges skip
            // seeding them so their consumers ghost-skip identically.
            let mut ghost_carries: std::collections::HashSet<crate::wengert::VarId> =
                Default::default();

            // ── Item 11: double-buffer prefetch calibration + certificate ──
            // Streamed layer groups eligible as a prefetch target.
            let streamed_range_count = (0..ranges.len())
                .filter(|&ri| layer_group[ri].iter().any(|i| ws_streamed.contains(i)))
                .count();
            // WGGO-calibrated activation, PER EDGE (issue during ri, consume
            // at ri+1): overlap pays iff range ri's compute can hide range
            // ri+1's pack transfer. Both sides in μs from the target GpuSpec:
            //   transfer(ri+1) = DMA fixed latency + pack_bytes / PCIe BW
            //   compute_lb(ri) = accum_window × max(launch floor, HBM floor)
            // where the launch floor is ops × kernel_launch_overhead (every
            // adjoint op is ≥ one launch) and the HBM floor is the range's own
            // pack read once per replay. Both compute terms are LOWER bounds
            // (no FLOP term — adjoint shapes are not static), so a discharged
            // edge is calibrated-safe while a declined edge may merely be
            // unproven — the right polarity for a perf heuristic.
            let gpu_spec = crate::gpu_specs::find_gpu(&self.compile_options.target_gpu)
                .unwrap_or_else(crate::gpu_specs::default_gpu);
            let accum_window = grad_accumulation_steps.max(1) as f64;
            let pack_bytes =
                |ri: usize| pending.schedule.range_pack_elems.get(ri).copied().unwrap_or(0) * 4;
            let transfer_us = |ri: usize| {
                WS_PCIE_FIXED_LAT_US
                    + pack_bytes(ri) as f64 / (gpu_spec.pcie_bandwidth_gbps.max(1.0) * 1e3)
            };
            let compute_lb_us = |ri: usize| {
                let launch_floor = (ranges[ri].end - ranges[ri].start) as f64
                    * gpu_spec.kernel_launch_overhead_ns as f64
                    / 1e3;
                let hbm_floor =
                    pack_bytes(ri) as f64 / (gpu_spec.peak_bandwidth_gbs.max(1.0) * 1e3);
                accum_window * launch_floor.max(hbm_floor)
            };
            // Edge ri → ri+1 activates iff ri's compute covers ri+1's
            // transfer (and both ends actually stream). Review M3: a pack
            // containing a symbolic-shape param prices at 0 bytes, which
            // would UNDERSTATE the transfer — the unsafe direction. An
            // unpriced pack instead falls back to the v1 structural
            // heuristic (avg ops/range), the behavior the GPU gates shipped
            // under; a PRICED edge uses the calibrated μs comparison and is
            // calibrated-safe.
            let avg_ops_per_range = pending.adjoint.ops.len() / ranges.len().max(1);
            let edge_on: Vec<bool> = (0..ranges.len())
                .map(|ri| {
                    let next_streams = ri + 1 < ranges.len()
                        && layer_group[ri + 1].iter().any(|i| ws_streamed.contains(i));
                    if !next_streams {
                        return false;
                    }
                    if pack_bytes(ri + 1) > 0 {
                        compute_lb_us(ri) >= transfer_us(ri + 1)
                    } else {
                        avg_ops_per_range >= WS_PREFETCH_MIN_OPS_PER_RANGE
                    }
                })
                .collect();
            let prefetch_active = self.compile_options.weight_stream.prefetch
                && self.compile_options.weight_stream.arena
                && ws_active
                && streamed_range_count >= 2
                && edge_on.iter().any(|&e| e);
            if self.compile_options.weight_stream.prefetch {
                let edges: Vec<String> = (0..ranges.len().saturating_sub(1))
                    .map(|ri| {
                        if pack_bytes(ri + 1) > 0 {
                            format!(
                                "L{ri}->L{}: compute>={:.1}us transfer~{:.1}us {}",
                                ri + 1,
                                compute_lb_us(ri),
                                transfer_us(ri + 1),
                                if edge_on[ri] { "ON" } else { "off" }
                            )
                        } else {
                            format!(
                                "L{ri}->L{}: unpriced pack, ops-heuristic (avg {} vs \
                                 min {}) {}",
                                ri + 1,
                                avg_ops_per_range,
                                WS_PREFETCH_MIN_OPS_PER_RANGE,
                                if edge_on[ri] { "ON" } else { "off" }
                            )
                        }
                    })
                    .collect();
                nsl_runtime::nsl_log!(INFO, "weight-stream", 
                    "[weight-stream] prefetch double-buffer: {} \
                     (streamed_ranges={streamed_range_count}, gpu={}, accum_window={}, \
                     edges [{}])",
                    if prefetch_active {
                        "ACTIVE — prefetch layer L+1 while computing L, event-ordered"
                    } else {
                        "DECLINED (compute too small to hide the transfer; synchronous arena)"
                    },
                    gpu_spec.name,
                    accum_window,
                    edges.join("; "),
                );
            }
            if self.compile_options.weight_stream.async_writeback {
                nsl_runtime::nsl_log!(INFO, "weight-stream", 
                    "[weight-stream] async writeback: {}",
                    if ws_active && streamed_range_count > 0 {
                        "ACTIVE — pack evict DtoH on the transfer stream, mirror \
                         scatter deferred to drain points"
                    } else {
                        "no streamed ranges (no effect)"
                    },
                );
            }
            // CADENCE-style transfer certificate: each entry is a discharged
            // (issue_range, consume_range, pack_size) obligation — a prefetch
            // issued during `issue_range`'s compute and awaited (event) at
            // `consume_range`'s head before any read.
            let mut transfer_cert: Vec<(usize, usize, usize)> = Vec::new();
            // Whether the CURRENT range's weights were prefetched by the
            // previous iteration (→ await instead of a sync upload).
            let mut ri_was_prefetched = false;

            for (ri, range) in ranges.iter().enumerate() {
                // This layer's accumulators exist only from here to its
                // update below — the m_partial surface the schedule shrinks.
                emit_csla_accum_alloc(self, builder, param_list, accum_val, &layer_group[ri])?;
                // D2b: re-upload this layer's weights for its replay range
                // (recompute clones + adjoint reads need them) under the
                // Weights surface bracket. Values dominate the b-loop.
                let ws_range: Vec<i64> = layer_group[ri]
                    .iter()
                    .copied()
                    .filter(|i| ws_streamed.contains(i))
                    .collect();
                if ws_active && !ws_range.is_empty() {
                    let prev_surf =
                        self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
                    let wsurf = builder.ins().iconst(cl_types::I8, SURFACE_WEIGHTS);
                    self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[wsurf])?;
                    if self.compile_options.weight_stream.arena {
                        if prefetch_active && ri_was_prefetched {
                            // Item 11: weights already streaming in (prefetched
                            // during the previous range's compute). Just wait
                            // on the transfer event before reading them.
                            self.emit_ws_pack_single(
                                builder,
                                param_list,
                                &ws_range,
                                "nsl_weight_stream_await_pack",
                            )?;
                        } else {
                            // Item 10: this layer group is one contiguous pack.
                            self.emit_ws_pack_upload(builder, param_list, &ws_range)?;
                        }
                    } else {
                        for &idx in &ws_range {
                            let iv = builder.ins().iconst(cl_types::I64, idx);
                            let pw = self
                                .compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                            self.compile_call_by_name(
                                builder,
                                "nsl_weight_stream_upload",
                                &[pw],
                            )?;
                        }
                    }
                    self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[prev_surf])?;
                }

                // Item 11: prefetch the NEXT streamed layer group so its HtoD
                // overlaps THIS range's compute (the b-loop below). Async on
                // the transfer stream; the next range's head awaits its event.
                // Per-edge: only where the calibration proved THIS range's
                // compute covers the NEXT pack's transfer.
                ri_was_prefetched = false;
                if prefetch_active && ri + 1 < ranges.len() && edge_on[ri] {
                    let ws_next: Vec<i64> = layer_group[ri + 1]
                        .iter()
                        .copied()
                        .filter(|i| ws_streamed.contains(i))
                        .collect();
                    if !ws_next.is_empty() {
                        let prev_surf = self
                            .compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
                        let wsurf = builder.ins().iconst(cl_types::I8, SURFACE_WEIGHTS);
                        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[wsurf])?;
                        self.emit_ws_pack_single(
                            builder,
                            param_list,
                            &ws_next,
                            "nsl_weight_stream_prefetch_pack",
                        )?;
                        self.compile_call_by_name(
                            builder,
                            "nsl_gpu_set_alloc_surface",
                            &[prev_surf],
                        )?;
                        transfer_cert.push((ri, ri + 1, ws_next.len()));
                        ri_was_prefetched = true;
                    }
                }

                let slice = crate::wengert::WengertList {
                    ops: pending.adjoint.ops[range.start..range.end].to_vec(),
                    output: pending.adjoint.output,
                    var_names: pending.adjoint.var_names.clone(),
                    var_types: pending.adjoint.var_types.clone(),
                };

                // b-loop over the buffered micro-batches, oldest first.
                let b_var = builder.declare_var(cl_types::I64);
                let b_zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(b_var, b_zero);
                let b_hdr = builder.create_block();
                let b_body = builder.create_block();
                let b_exit = builder.create_block();
                builder.ins().jump(b_hdr, &[]);
                builder.switch_to_block(b_hdr);
                let b_i = builder.use_var(b_var);
                let b_cont = builder.ins().icmp(IntCC::SignedLessThan, b_i, n_val);
                builder.ins().brif(b_cont, b_body, &[], b_exit, &[]);
                builder.switch_to_block(b_body);
                builder.seal_block(b_body);
                state.current_block = Some(b_body);

                let so_in = builder.use_var(saves_outer_var);
                let b_now = builder.use_var(b_var);
                let inner =
                    self.compile_call_by_name(builder, "nsl_list_get", &[so_in, b_now])?;
                let dict_b = if has_dataloader.is_some() {
                    let dl = builder.use_var(dicts_var);
                    let d = self.compile_call_by_name(builder, "nsl_list_get", &[dl, b_now])?;
                    // Re-install micro-batch b's packing metadata every
                    // range: the registry is thread-local per-batch state
                    // read at @flash_attention launch time, and any range
                    // may contain attention backward ops.
                    self.emit_packing_registry_stash(builder, state, d)?;
                    Some(d)
                } else {
                    None
                };
                let inner_carry = if let Some(co) = carry_outer {
                    if ri == 0 {
                        // First range: create this micro-batch's carry list,
                        // pre-filled so later nsl_list_set slots are in-bounds.
                        let ic = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        for _ in 0..n_carry {
                            self.compile_call_by_name(builder, "nsl_list_push", &[ic, zero])?;
                        }
                        self.compile_call_by_name(builder, "nsl_list_push", &[co, ic])?;
                        Some(ic)
                    } else {
                        Some(self.compile_call_by_name(builder, "nsl_list_get", &[co, b_now])?)
                    }
                } else {
                    None
                };

                // Seed: forward values (params/constants/loop-invariant SSA)
                // + this range's buffered primal imports + this range's
                // cross-range adjoint imports.
                let mut seed = pending.seed_base.clone();
                for &si in &slot_seed_per_range[ri] {
                    let (vid, kind) = &pending.slots[si];
                    let idx_val = builder.ins().iconst(cl_types::I64, si as i64);
                    let raw =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    let val = match kind {
                        CslaSlotKind::Raw { .. } => raw,
                        CslaSlotKind::F64Bits => {
                            builder.ins().bitcast(cl_types::F64, MemFlagsData::new(), raw)
                        }
                    };
                    seed.insert(*vid, val);
                }
                for &cv in &imports_per_range[ri] {
                    if ghost_carries.contains(&cv) {
                        continue;
                    }
                    let ic = inner_carry.expect("carry imports imply carry_outer");
                    let slot_val = builder
                        .ins()
                        .iconst(cl_types::I64, carry_slot[&cv] as i64);
                    let val =
                        self.compile_call_by_name(builder, "nsl_list_get", &[ic, slot_val])?;
                    seed.insert(cv, val);
                }

                // LSE tape-carry: re-bind the fused-SDPA aux side-band to
                // this micro-batch's buffered logsumexp for every fwd-out
                // vid seeded in this range — BEFORE the slice lowering emits
                // the SDPA backward that consults the Value-keyed map. The
                // loaded value is the fused forward's real LSE (or its
                // runtime-0 decline sentinel), so the emitted backward takes
                // the same kernel arm the baseline did.
                let seeded_vids_here: std::collections::HashSet<crate::wengert::VarId> =
                    slot_seed_per_range[ri]
                        .iter()
                        .map(|&si| pending.slots[si].0)
                        .collect();
                let mut lse_loaded_here: Vec<(usize, Value)> = Vec::new();
                for (k, (idx, vids)) in pending.lse_slots.iter().enumerate() {
                    let targets: Vec<crate::wengert::VarId> = vids
                        .iter()
                        .copied()
                        .filter(|v| seeded_vids_here.contains(v))
                        .collect();
                    if targets.is_empty() {
                        continue;
                    }
                    let idx_val = builder.ins().iconst(cl_types::I64, *idx as i64);
                    let lse =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    for v in targets {
                        if let Some(&out_seed) = seed.get(&v) {
                            self.flash_attn_aux.insert(out_seed, (out_seed, lse));
                        }
                    }
                    lse_loaded_here.push((k, lse));
                }

                // Fused-CE tape-carry: re-bind `fused_ce_fwd_lse` to this
                // micro-batch's buffered logsumexp, keyed by the SEEDED
                // fwd-result (loss-scalar) Value — BEFORE the slice
                // lowering emits the fused backward whose first extract
                // `.remove()`s the entry (a miss is a hard CodegenError).
                // The emitted backward frees the loaded tensor per (range,
                // b) itself; no per-b slot free exists for these slots.
                // The cast cache is deliberately NOT re-bound: the f32
                // MISS path re-emits pass-through of the extract's own
                // (seeded) inputs — byte-identical, zero extra emission.
                for (idx, vids) in pending.fce_slots.iter() {
                    let targets: Vec<crate::wengert::VarId> = vids
                        .iter()
                        .copied()
                        .filter(|v| seeded_vids_here.contains(v))
                        .collect();
                    if targets.is_empty() {
                        continue;
                    }
                    let idx_val = builder.ins().iconst(cl_types::I64, *idx as i64);
                    let lse =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    for v in targets {
                        if let Some(&res_seed) = seed.get(&v) {
                            self.fused_ce_fwd_lse.insert(res_seed, lse);
                        }
                    }
                }

                // FASE hook: identical to the baseline's fase_cb — accumulate
                // each parameter gradient into its compile-time accum_list
                // slot (allocated at window start or at this range's head)
                // and free it immediately. wrap_offload is const false BY
                // DESIGN (review D2a-2): the per-layer accumulators are
                // device-resident — that is D1's point — and the offload
                // envelope applies only at the update sites, where one
                // layer's m/v stage through at a time.
                let hook_idx_map = &pending.hook_accum_idx;
                let accum_scale = pending.accum_scale;
                let mut fase_cb = |c: &mut Compiler,
                                   var_id: crate::wengert::VarId,
                                   grad_src: crate::wengert_lower::ParamGradSource,
                                   still_needed: bool,
                                   b: &mut cranelift_frontend::FunctionBuilder|
                 -> Result<(), CodegenError> {
                    // Item 7 is refused alongside --layerwise-accum at option
                    // validation, so this arm is unreachable. It is a hard
                    // error rather than a silent fallthrough because the
                    // reason is subtle: CSLA lowers PRE-SLICED tapes, so
                    // `wgrad_fusion::plan` would see slice-local reader
                    // counts and could elide a matmul whose result a LATER
                    // slice still reads — a silently dropped gradient.
                    let grad_ptr = match grad_src {
                        crate::wengert_lower::ParamGradSource::Materialized(v) => v,
                        crate::wengert_lower::ParamGradSource::FusedWgrad { .. } => {
                            return Err(CodegenError::new(
                                "internal: --fuse-wgrad-accum fired inside the CSLA window \
                                 replay, which lowers pre-sliced tapes where the fusion's \
                                 single-reader proof does not hold. This composition is \
                                 supposed to be refused at option validation.",
                            ));
                        }
                    };
                    let Some(&accum_idx) = hook_idx_map.get(&var_id) else {
                        return Ok(());
                    };
                    let idx_val =
                        b.ins().iconst(cranelift_codegen::ir::types::I64, accum_idx);
                    let m_partial =
                        c.compile_call_by_name(b, "nsl_list_get", &[accum_val, idx_val])?;
                    // P0.3: note this parameter's gradient BEFORE accumulate
                    // consumes it — same convention as the baseline fase_cb.
                    // The bracket is WINDOW-scoped (step_begin before the
                    // range loop, step_end after the epilogue update), so
                    // note() fires once per (param, micro-batch) — each
                    // param's adjoint lands in exactly one range — and the
                    // accumulator merges repeat notes per index (finite ANDs,
                    // nonzero ORs), so the report attests every partial.
                    if c.compile_options.grad_integrity {
                        c.compile_call_by_name(
                            b,
                            "nsl_grad_integrity_note",
                            &[grad_ptr, idx_val],
                        )?;
                    }
                    c.fase_emit_accumulate(b, m_partial, grad_ptr, accum_scale, false)?;
                    // Defer the free when this param grad is a shared
                    // intermediate a later in-slice op still reads (bias-grad
                    // == d_out feeding the weight matmul). wengert_lower keeps
                    // it in var_map + owned_values; end-of-range cleanup frees
                    // it. Exports are distinct activation adjoints, never param
                    // grads, so this never strands a cross-range carry.
                    if !still_needed {
                        c.compile_call_by_name(b, "nsl_tensor_free", &[grad_ptr])?;
                    }
                    Ok(())
                };
                let grad_lowered = match crate::wengert_lower::compile_wengert_ops(
                    self,
                    builder,
                    state,
                    &slice,
                    &seed,
                    Some((&pending.param_adj_set, &mut fase_cb)),
                ) {
                    Ok(gv) => gv,
                    Err(e) => {
                        nsl_runtime::nsl_log!(ERROR, "nsl", 
                            "[nsl] csla window backward lowering failed (range {ri}: {}), \
                             rerun without --layerwise-accum",
                            e
                        );
                        return Err(e);
                    }
                };

                // Exports: store this range's boundary adjoints into the
                // carry list for later ranges (ghost-skips recorded so
                // consumers ghost-skip identically to the baseline).
                for &ev in &exports_per_range[ri] {
                    match grad_lowered.var_map.get(&ev) {
                        Some(&val) => {
                            // Review D1b-4: only i64-typed values (tensor /
                            // list pointers, integers) can ride the carry
                            // list; a scalar f64 crossing a range boundary
                            // would otherwise be a Cranelift verifier ICE.
                            // AD rules emit constants adjacent to consumers,
                            // so this is unreachable today — refuse loudly
                            // if that ever changes.
                            let vty = builder.func.dfg.value_type(val);
                            if vty != cl_types::I64 {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: cross-range adjoint value \
                                     VarId {ev} has non-i64 Cranelift type {vty}; \
                                     scalar carries are unsupported — drop \
                                     --layerwise-accum",
                                )));
                            }
                            let ic = inner_carry.expect("exports imply carry_outer");
                            let slot_val = builder
                                .ins()
                                .iconst(cl_types::I64, carry_slot[&ev] as i64);
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_set",
                                &[ic, slot_val, val],
                            )?;
                        }
                        None => {
                            ghost_carries.insert(ev);
                        }
                    }
                }

                // Per-replay cleanup: adjoint-owned intermediates minus
                // hook-consumed gradients, explicit FreeTensor victims, and
                // the exports (they outlive this range; their frees are the
                // consuming ranges' markers or the explicit list below).
                //
                // Seed from exactly the param grads the hook FREED — not the
                // whole param_adj_set. A param grad that is a shared
                // intermediate (bias-grad == d_out, read later in-slice) was
                // accumulated but its free was DEFERRED; it is absent here on
                // purpose so free_wengert_owned_values releases it once.
                let mut freed_adjoint_vars: std::collections::HashSet<crate::wengert::VarId> =
                    grad_lowered.hook_freed_param_vars.iter().copied().collect();
                freed_adjoint_vars.extend(grad_lowered.hook_freed_input_vars.iter().copied());
                freed_adjoint_vars.extend(grad_lowered.explicit_freed_vars.iter().copied());
                freed_adjoint_vars.extend(exports_per_range[ri].iter().copied());
                self.free_wengert_owned_values(
                    builder,
                    &grad_lowered.owned_values,
                    &freed_adjoint_vars,
                )?;

                // Carried values whose last consumer is this range and that
                // no FreeTensor marker covers: free their seeded value now.
                // Review D1b-1 (HIGH): skip anything the replay itself
                // already freed — the hook's reduce_to_shape extra free
                // (hook_freed_input_vars) releases a carried RAW grad when a
                // range boundary lands on the reduce op, and explicit
                // FreeTensor victims are covered by their markers. Freeing
                // either again here would be a double free.
                for &cv in &carry_explicit_free[ri] {
                    if ghost_carries.contains(&cv)
                        || grad_lowered.hook_freed_input_vars.contains(&cv)
                        || grad_lowered.explicit_freed_vars.contains(&cv)
                    {
                        continue;
                    }
                    if let Some(&val) = seed.get(&cv) {
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[val])?;
                    }
                }

                // Free the buffered primal slots whose LAST reader is this
                // range (owned tensors/lists only). The loss slot skips the
                // window's LAST entry: that is the CURRENT iteration's loss,
                // still read by on_step / on_epoch after this phase; the
                // conditional per-iteration loss-free site below owns it.
                let n_minus_1 =
                    builder.ins().iconst(cl_types::I64, grad_accumulation_steps - 1);
                for &si in &slot_free_per_range[ri] {
                    let (_vid, kind) = &pending.slots[si];
                    let CslaSlotKind::Raw { owned: Some(ty) } = kind else {
                        continue;
                    };
                    let free_fn = match ty {
                        crate::wengert::WengertType::Tensor => "nsl_tensor_free",
                        crate::wengert::WengertType::List => "nsl_list_free",
                        _ => continue,
                    };
                    let idx_val = builder.ins().iconst(cl_types::I64, si as i64);
                    let slot_val =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    if Some(si) == pending.loss_slot {
                        let b_cur = builder.use_var(b_var);
                        let is_last = builder.ins().icmp(IntCC::Equal, b_cur, n_minus_1);
                        let loss_free = builder.create_block();
                        let loss_join = builder.create_block();
                        builder.ins().brif(is_last, loss_join, &[], loss_free, &[]);
                        builder.switch_to_block(loss_free);
                        builder.seal_block(loss_free);
                        self.compile_call_by_name(builder, free_fn, &[slot_val])?;
                        builder.ins().jump(loss_join, &[]);
                        builder.switch_to_block(loss_join);
                        builder.seal_block(loss_join);
                        state.current_block = Some(loss_join);
                    } else {
                        self.compile_call_by_name(builder, free_fn, &[slot_val])?;
                    }
                }

                // LSE tape-carry: free this micro-batch's logsumexps whose
                // last consuming range is this one (null-safe — the decline
                // sentinel is a runtime 0). Re-load from the inner list when
                // this range didn't seed any of the slot's vids (review F4:
                // the fixpoint-extended free range can trail the direct
                // reads through list membership — mirror the regular slot
                // frees' unconditional re-load instead of leaking).
                for (k, (idx, _vids)) in pending.lse_slots.iter().enumerate() {
                    if lse_free_range[k] != ri {
                        continue;
                    }
                    let lse = match lse_loaded_here.iter().find(|(lk, _)| *lk == k) {
                        Some((_, v)) => *v,
                        None => {
                            let idx_val = builder.ins().iconst(cl_types::I64, *idx as i64);
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_get",
                                &[inner, idx_val],
                            )?
                        }
                    };
                    self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[lse])?;
                }

                if ri == last_ri {
                    // Batch dict values die with their micro-batch's LAST
                    // replay (the per-iteration dict free is suppressed
                    // under csla), then the shells. INVARIANT (review L2):
                    // on step iterations this destroys the CURRENT batch
                    // dict at b==N-1, i.e. BEFORE the optimizer updates and
                    // the on_step callback — nothing after this point may
                    // read step_param_var's dict (on_step receives only
                    // (step, loss)). nsl_dict_free_tensor_values destroys
                    // the WHOLE dict structure, matching the baseline's
                    // per-iteration call — popped dicts are never touched by
                    // the DataLoader teardown.
                    if let Some(d) = dict_b {
                        self.compile_call_by_name(
                            builder,
                            "nsl_dict_free_tensor_values",
                            &[d],
                        )?;
                    }
                    self.compile_call_by_name(builder, "nsl_list_free", &[inner])?;
                    if let Some(ic) = inner_carry {
                        self.compile_call_by_name(builder, "nsl_list_free", &[ic])?;
                    }
                }

                let b_cur = builder.use_var(b_var);
                let b_one = builder.ins().iconst(cl_types::I64, 1);
                let b_next = builder.ins().iadd(b_cur, b_one);
                builder.def_var(b_var, b_next);
                builder.ins().jump(b_hdr, &[]);
                builder.seal_block(b_hdr);
                builder.switch_to_block(b_exit);
                builder.seal_block(b_exit);
                state.current_block = Some(b_exit);

                // Per-layer update: this layer's accumulators are complete
                // (all N micro-batches replayed) and — by the CrossLayer
                // classification + the list-membership fixpoint — no later
                // range reads these params' OLD θ. Update, then free the
                // layer's accumulators.
                // Bound before the call: `builder` is mutably borrowed by the
                // call itself, so the live rate cannot be read inside the args.
                let csla_lr = builder.use_var(lr_var);
                emit_csla_group_update(
                    self,
                    builder,
                    param_list,
                    state_list_1,
                    state_list_2,
                    two_state,
                    accum_val,
                    &fase_plan.recipe,
                    csla_lr,
                    (bc1_inv, bc2_inv),
                    wrap_precision,
                    self.compile_options.optim_state_offload,
                    muon_csla_ctx.as_ref(),
                    zero3_streamed.as_ref(),
                    zero3_elem.as_ref(),
                    zero3_elem_scalars.as_ref(),
                    Some(opt_step),
                    csla_multi_scalars.as_ref(),
                    &layer_group[ri],
                )?;
                // D2b: this layer's θ is final for the window — write back
                // to the mirror and drop the device buffer.
                if ws_active {
                    if self.compile_options.weight_stream.async_writeback
                        && self.compile_options.weight_stream.arena
                    {
                        // Item 11 (writeback half): issue the pack's DtoH on
                        // the transfer stream and move on — the next range's
                        // compute overlaps this layer's writeback. The mirror
                        // scatter lands at the runtime's drain points (queue
                        // cap / affected re-upload / teardown).
                        self.emit_ws_pack_single(
                            builder,
                            param_list,
                            &ws_range,
                            "nsl_weight_stream_evict_pack_async",
                        )?;
                    } else if self.compile_options.weight_stream.arena {
                        // Item 10: one DtoH writeback for the whole layer pack.
                        self.emit_ws_pack_evict(builder, param_list, &ws_range, 1)?;
                    } else {
                        for &idx in &ws_range {
                            let iv = builder.ins().iconst(cl_types::I64, idx);
                            let pw = self
                                .compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                            let wb = builder.ins().iconst(cl_types::I64, 1);
                            self.compile_call_by_name(
                                builder,
                                "nsl_weight_stream_evict",
                                &[pw, wb],
                            )?;
                        }
                    }
                }
            }

            // Item 11: emit the discharged transfer certificate. Each prefetch
            // issued during range Li's compute is provably awaited (a CUDA
            // event on the compute stream) at range Li+1's head before any
            // read — the CADENCE assume/guarantee obligation, discharged.
            if prefetch_active {
                let total: usize = transfer_cert.iter().map(|(_, _, n)| n).sum();
                nsl_runtime::nsl_log!(INFO, "weight-stream", 
                    "[weight-stream] transfer certificate: {} prefetch obligations discharged \
                     ({total} params double-buffered); chain [{}]",
                    transfer_cert.len(),
                    transfer_cert
                        .iter()
                        .map(|(i, c, n)| format!("L{i}->L{c}:{n}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }

            // Epilogue: globals (embedding / final norm / LM head), tied and
            // cross-layer params, dead layers, and anything the extractor
            // never saw — after the whole backward, like the baseline.
            // Bound before the call: `builder` is mutably borrowed by the
            // call itself, so the live rate cannot be read inside the args.
            let csla_lr = builder.use_var(lr_var);
            emit_csla_group_update(
                self,
                builder,
                param_list,
                state_list_1,
                state_list_2,
                two_state,
                accum_val,
                &fase_plan.recipe,
                csla_lr,
                (bc1_inv, bc2_inv),
                wrap_precision,
                self.compile_options.optim_state_offload,
                muon_csla_ctx.as_ref(),
                zero3_streamed.as_ref(),
                zero3_elem.as_ref(),
                zero3_elem_scalars.as_ref(),
                Some(opt_step),
                csla_multi_scalars.as_ref(),
                global_group,
            )?;
            // P0.3: close the window-scoped grad-integrity step — every
            // range's hook (and the epilogue params' notes from whichever
            // range produced them) has fired by here.
            if self.compile_options.grad_integrity {
                self.compile_call_by_name(builder, "nsl_grad_integrity_step_end", &[])?;
            }
            // D2b part 2: NO post-epilogue restore. The next iterations'
            // forwards re-upload each layer right before its own segment
            // (and evict it after its last primal touch), so streamed
            // params stay off-device between their brackets for the WHOLE
            // training loop — the forward-side residency wall this part
            // removes. Teardown still restores for model_save/eval.

            // Window cleanup: drop the shells and start fresh lists for the
            // next window (carry inner shells died with the last range).
            if let Some(co) = carry_outer {
                self.compile_call_by_name(builder, "nsl_list_free", &[co])?;
            }
            let so_done = builder.use_var(saves_outer_var);
            self.compile_call_by_name(builder, "nsl_list_free", &[so_done])?;
            let so_new = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(saves_outer_var, so_new);
            let dl_done = builder.use_var(dicts_var);
            self.compile_call_by_name(builder, "nsl_list_free", &[dl_done])?;
            let dl_new = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(dicts_var, dl_new);

            builder.ins().jump(bwd_join, &[]);
            builder.switch_to_block(bwd_join);
            builder.seal_block(bwd_join);
            state.current_block = Some(bwd_join);
        }

        Ok(())
    }
}

/// The pre-forward product of the layer-major schedule derivation: the
/// per-param facts, the primal view map, the adjoint-read imports and the
/// shared schedule. Built once before the sliced forward, consumed by the
/// save phase ([`Compiler::emit_csla_window_save`]).
pub(crate) struct CslaPre {
    pub(crate) params: Vec<CslaParam>,
    pub(crate) primal_view_of: std::collections::HashMap<
        crate::wengert::VarId,
        crate::wengert::VarId,
    >,
    pub(crate) imports: Vec<crate::wengert::VarId>,
    pub(crate) schedule: CslaSchedule,
}

/// Every binding of `compile_train_block_inner` the save phase reads; names
/// are the driver's.
pub(crate) struct CslaSaveInputs<'a> {
    /// The final adjoint tape (cloned into the pending carrier).
    pub(crate) adjoint: &'a crate::wengert::WengertList,
    /// The FASE hook entries, keyed by adjoint gradient VarId.
    pub(crate) adj_vid_to_hook_entry: &'a std::collections::HashMap<crate::wengert::VarId, ParamHookEntry>,
    /// The window save-list and dict-list variables (allocated when CSLA is on).
    pub(crate) csla_buffers: Option<(Variable, Variable)>,
    /// The pre-forward schedule product (consumed here).
    pub(crate) csla_pre: Option<CslaPre>,
    pub(crate) fase_hook_active: bool,
    pub(crate) fase_plan: &'a crate::fase::FasePlan,
    /// The lowered primal (its owned-value table).
    pub(crate) full_lowered: &'a crate::wengert_lower::LoweredWengert,
    /// The primal VarMap (`&full_lowered.var_map`; cloned as the replay seed base).
    pub(crate) full_vars: &'a crate::wengert_lower::VarMap,
    /// The DataLoader handle, when the `data:` section declared one.
    pub(crate) has_dataloader: Option<Value>,
    pub(crate) loss_var_id: crate::wengert::VarId,
    pub(crate) param_adj_set: &'a std::collections::HashSet<crate::wengert::VarId>,
    pub(crate) step_param_var: Variable,
}

/// What the save phase hands back: the three window carriers the driver
/// declares above the epoch loop (names are the driver's).
pub(crate) struct CslaWindowSave {
    pub(crate) csla_pending: Option<CslaPending>,
    pub(crate) csla_loss_buffered: bool,
    pub(crate) csla_teardown_slots: Option<Vec<(i64, &'static str)>>,
}

impl Compiler<'_> {
    /// Emit the CSLA window save phase: the `csla_active` arm of the
    /// driver's adjoint-lowering site (see the module header). Instead of
    /// lowering the adjoint in place, it pushes every adjoint-read primal
    /// value into this micro-batch's slot list and leaves the pending
    /// carrier for [`Self::emit_csla_window_backward`].
    pub(crate) fn emit_csla_window_save(
        &mut self,
        builder: &mut FunctionBuilder,
        inputs: CslaSaveInputs<'_>,
    ) -> Result<CslaWindowSave, CodegenError> {
        let CslaSaveInputs {
            adjoint,
            adj_vid_to_hook_entry,
            csla_buffers,
            csla_pre,
            fase_hook_active,
            fase_plan,
            full_lowered,
            full_vars,
            has_dataloader,
            loss_var_id,
            param_adj_set,
            step_param_var,
        } = inputs;

        // === CSLA Stage-2 (D1a): window-buffered save phase ===
        //
        // Instead of lowering the adjoint here (the interleaved
        // schedule), push every adjoint-read primal value into
        // this micro-batch's slot list and defer the whole
        // window's backward to the should_step region, where a
        // runtime loop replays the adjoint once per buffered
        // micro-batch (one tape, N executions — unrolling would
        // break CCR's anchor segmentation). Param/Constant leaves
        // are loop-invariant (FASE Deferred updates θ only at
        // window boundaries, after the replay), so the replay
        // reuses the current iteration's SSA values for them via
        // seed_base.
        debug_assert!(fase_hook_active);

        // Review findings H1/H2/M1 (D1a adversarial review): three
        // compile-time SIDE CHANNELS travel by SSA Value instead
        // of the tape, so the window replay cannot see them —
        // refuse each loudly rather than replay wrong.
        //
        // H1 — CSHA-claimed fused backward: `csha_forward_saves`
        // is keyed by layer name and holds the STEP iteration's
        // save-buffer SSA values; the claimed backward consumes
        // AND frees them once — inside the b-loop that means
        // stale saves for b<N-1 plus an N-fold double-free.
        if !self.csha_forward_saves.is_empty()
            || adjoint.ops.iter().any(|op| {
                matches!(
                    op.op,
                    crate::wengert::PrimalOp::FusedCshaBackward { .. }
                        | crate::wengert::PrimalOp::CshaFusedBackwardExtract { .. }
                )
            })
        {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with CSHA-claimed \
                 fused attention backward: the claimed saves travel \
                 through a compile-time side channel \
                 (csha_forward_saves) that the window replay cannot \
                 re-bind per micro-batch. Drop @csha/@flash_attention \
                 claims or --layerwise-accum",
            ));
        }
        // H2 (RESOLVED by the LSE tape-carry): the fused SDPA
        // dispatch saves its logsumexp through the Value-keyed
        // `flash_attn_aux` side-band + a u32::MAX-sentinel owned
        // entry. The save phase below buffers every aux entry as
        // an extra window slot, and the replay re-binds the aux
        // map per micro-batch before each consuming range's
        // lowering — the emitted backward then consumes the SAME
        // per-batch LSE the baseline did (or the runtime-0
        // decline sentinel), keeping the fused phase-2 kernel on
        // the fused path.
        // M1, NARROWED by the fused-CE tape-carry: the f32
        // @fused_lm_ce path is now supported — the only real
        // side-band is `fused_ce_fwd_lse` (one [B*S] f32 tensor
        // per micro-batch), buffered as an extra window slot
        // below and re-bound per replay range exactly like the
        // flash-attention LSE; the f32 cast-cache MISS path is
        // already correct at replay (pass-through of the seeded
        // inputs, zero extra emission). Still refused:
        //
        // - distill blocks: the KL-CE carry would need THREE
        //   lse slots per micro-batch plus the frozen teacher's
        //   activation buffered per micro-batch — a large new
        //   surface with no gate; refuse until designed.
        // - @fused_lm_ce(dtype="f16"/"bf16"): the fp16/bf16
        //   shadow-cast tensors are step-scoped (freed at
        //   function-scope exit); a replay-side re-cast would
        //   pile N cast sets per window with no free (the
        //   original M1 finding), and buffering w_cast per
        //   micro-batch (V×H×2 B) would erase the memory win.
        if self.active_distill_context.is_some() {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with distill \
                 blocks: the fused KL-CE backward reads Value-keyed \
                 forward saves (three LSE buffers + the teacher \
                 activations) the window replay cannot see. Drop \
                 the distill block or --layerwise-accum",
            ));
        }
        if let Some(cfg) = &self.active_fused_ce_config {
            let non_f32 = cfg.enabled
                && !matches!(
                    cfg.dtype,
                    None | Some(crate::FusedCeDtypeHint::F32)
                );
            if non_f32 {
                return Err(CodegenError::new(
                    "--layerwise-accum supports @fused_lm_ce only \
                     with dtype=\"f32\": the fp16/bf16 forward cast \
                     tensors are step-scoped and cannot be carried \
                     through the window replay without either \
                     leaking N cast sets per window or buffering \
                     the V*H weight cast per micro-batch. Use \
                     dtype=\"f32\" or drop --layerwise-accum",
                ));
            }
        }

        let (saves_outer_var, dicts_var) =
            csla_buffers.expect("csla_buffers allocated when csla_active");
        // D2b part 2: the plan / params / view chains / imports /
        // layer-major schedule were all computed in the
        // pre-forward pure pipeline (the forward streamer needed
        // them at emission time) — consume, don't recompute.
        let pre = csla_pre.expect("csla_pre computed when csla_active");
        let imports = pre.imports;
        let owned_map: std::collections::HashMap<
            crate::wengert::VarId,
            crate::wengert::WengertType,
        > = full_lowered
            .owned_values
            .iter()
            .map(|(vid, _, ty)| (*vid, *ty))
            .collect();
        let inner = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        let mut slots: Vec<(crate::wengert::VarId, CslaSlotKind)> = Vec::new();
        let mut loss_slot: Option<usize> = None;
        for v in imports {
            // Ghost VarIds (never lowered) are skipped here AND at
            // replay seed time — the adjoint lowering ghost-skips
            // their consumers identically to the baseline.
            let Some(&val) = full_vars.get(&v) else { continue };
            let vty = builder.func.dfg.value_type(val);
            if vty == cl_types::I64 {
                self.compile_call_by_name(builder, "nsl_list_push", &[inner, val])?;
                if v == loss_var_id {
                    loss_slot = Some(slots.len());
                }
                slots.push((
                    v,
                    CslaSlotKind::Raw {
                        owned: owned_map.get(&v).copied(),
                    },
                ));
            } else if vty == cl_types::F64 {
                let bits = builder.ins().bitcast(
                    cl_types::I64,
                    MemFlagsData::new(),
                    val,
                );
                self.compile_call_by_name(builder, "nsl_list_push", &[inner, bits])?;
                slots.push((v, CslaSlotKind::F64Bits));
            } else {
                return Err(CodegenError::new(format!(
                    "--layerwise-accum: adjoint-imported primal VarId {v} \
                     lowered to unsupported Cranelift type {vty} — the \
                     window buffer stores i64 pointers/integers and \
                     bitcast f64 scalars only",
                )));
            }
        }
        // LSE tape-carry: append one slot per fused-SDPA aux
        // entry whose forward output the adjoint reads. The
        // stored value is the aux LSE (join-block param on the
        // dispatch arm — a real tensor when the fused launch
        // fired, runtime 0 when it declined; the decomposed
        // arm's compile-time iconst 0 buffers as a plain 0).
        // Sorted by min fwd-out vid so slot order is
        // deterministic.
        let mut lse_slots: Vec<(usize, Vec<crate::wengert::VarId>)> = Vec::new();
        let mut csla_lse_pushed: std::collections::HashSet<Value> =
            std::collections::HashSet::new();
        if !self.flash_attn_aux.is_empty() {
            let slot_vid_set: std::collections::HashSet<crate::wengert::VarId> =
                slots.iter().map(|(v, _)| *v).collect();
            let mut by_val: std::collections::HashMap<
                Value,
                Vec<crate::wengert::VarId>,
            > = std::collections::HashMap::new();
            for (vid, val) in full_vars.iter() {
                by_val.entry(*val).or_default().push(*vid);
            }
            let mut entries: Vec<(Vec<crate::wengert::VarId>, Value)> = self
                .flash_attn_aux
                .iter()
                .filter_map(|(out_val, (_, lse_val))| {
                    let mut vids: Vec<crate::wengert::VarId> = by_val
                        .get(out_val)?
                        .iter()
                        .copied()
                        .filter(|v| slot_vid_set.contains(v))
                        .collect();
                    if vids.is_empty() {
                        return None;
                    }
                    vids.sort_unstable();
                    Some((vids, *lse_val))
                })
                .collect();
            entries.sort_by_key(|(vids, _)| vids[0]);
            for (vids, lse_val) in entries {
                let idx = slots.len() + lse_slots.len();
                self.compile_call_by_name(
                    builder,
                    "nsl_list_push",
                    &[inner, lse_val],
                )?;
                csla_lse_pushed.insert(lse_val);
                lse_slots.push((idx, vids));
            }
        }
        // Anti-vacuity marker (tape-carry review F1): under the
        // Block checkpoint policy every in-block SDPA out is a
        // recompute victim (the clone RE-LAUNCHES the fused
        // forward during replay and re-establishes the aux
        // side-band locally), so lse_slots is 0 and the carry is
        // inert; the carry engages under --checkpoint-selective
        // (SDPA outs saved). Gates assert this line's exact slot
        // count so the tested path is named, not assumed.
        nsl_runtime::nsl_log!(INFO, "csla", "[csla] lse tape-carry: {} slots", lse_slots.len());

        // Fused-CE tape-carry: one extra slot per
        // `fused_ce_fwd_lse` entry — the [B*S] f32 logsumexp the
        // fused backward consumes, keyed by the fwd-result
        // (loss-scalar) Value. Unlike the flash-attention carry
        // this one is LIVE under BOTH checkpoint policies:
        // FusedLinearCe sits in the CCR epilogue (never a
        // recompute victim), so a replay clone can never
        // re-establish the side-band locally. Lifecycle also
        // differs: the emitted backward CONSUMES AND FREES the
        // buffered tensor per (range, b) — the per-b slot-free
        // machinery must NOT touch these slots; only the
        // trailing-partial-window teardown sweep frees
        // unreplayed entries. (The f32 cast cache needs no
        // carry: its replay MISS path re-emits pass-through
        // inputs — the seeded x/W/bias — with zero extra cost;
        // non-f32 dtypes are refused above.)
        let mut fce_slots: Vec<(usize, Vec<crate::wengert::VarId>)> = Vec::new();
        if !self.fused_ce_fwd_lse.is_empty() {
            let slot_vid_set: std::collections::HashSet<crate::wengert::VarId> =
                slots.iter().map(|(v, _)| *v).collect();
            let mut by_val: std::collections::HashMap<
                Value,
                Vec<crate::wengert::VarId>,
            > = std::collections::HashMap::new();
            for (vid, val) in full_vars.iter() {
                by_val.entry(*val).or_default().push(*vid);
            }
            let mut entries: Vec<(Vec<crate::wengert::VarId>, Value)> = self
                .fused_ce_fwd_lse
                .iter()
                .filter_map(|(res_val, lse_val)| {
                    let mut vids: Vec<crate::wengert::VarId> = by_val
                        .get(res_val)?
                        .iter()
                        .copied()
                        .filter(|v| slot_vid_set.contains(v))
                        .collect();
                    if vids.is_empty() {
                        return None;
                    }
                    vids.sort_unstable();
                    Some((vids, *lse_val))
                })
                .collect();
            entries.sort_by_key(|(vids, _)| vids[0]);
            for (vids, lse_val) in entries {
                let idx = slots.len() + lse_slots.len() + fce_slots.len();
                self.compile_call_by_name(
                    builder,
                    "nsl_list_push",
                    &[inner, lse_val],
                )?;
                fce_slots.push((idx, vids));
            }
            // The step body's adjoint is never lowered under
            // csla, so its map entries would otherwise linger
            // for the whole compile — clear them; the replay
            // re-binds fresh entries keyed by SEEDED Values
            // per consuming range. Same for the pass-through
            // cast entries (the replay's miss path is the
            // correct one).
            self.fused_ce_fwd_lse.clear();
            self.fused_ce_fwd_casts.clear();
        }
        // Anti-vacuity twin of the LSE line: asserted exactly
        // by the fused-CE gates (1 slot = the carry engaged; a
        // composite fallback shows 0 and the launch counters
        // catch it too).
        nsl_runtime::nsl_log!(INFO, "csla", 
            "[csla] fused-ce tape-carry: {} slots",
            fce_slots.len()
        );

        let so = builder.use_var(saves_outer_var);
        self.compile_call_by_name(builder, "nsl_list_push", &[so, inner])?;
        if has_dataloader.is_some() {
            let dl = builder.use_var(dicts_var);
            let batch_now = builder.use_var(step_param_var);
            self.compile_call_by_name(builder, "nsl_list_push", &[dl, batch_now])?;
        }
        let hook_accum_idx: std::collections::HashMap<crate::wengert::VarId, i64> =
            adj_vid_to_hook_entry
                .iter()
                .map(|(vid, e)| (*vid, e.accum_idx))
                .collect();
        let csla_loss_buffered = loss_slot.is_some();
        let csla_teardown_slots = Some(
            slots
                .iter()
                .enumerate()
                .filter_map(|(idx, (_, kind))| match kind {
                    CslaSlotKind::Raw {
                        owned: Some(crate::wengert::WengertType::Tensor),
                    } => Some((idx as i64, "nsl_tensor_free")),
                    CslaSlotKind::Raw {
                        owned: Some(crate::wengert::WengertType::List),
                    } => Some((idx as i64, "nsl_list_free")),
                    _ => None,
                })
                // LSE slots: owned tensors when the fused launch
                // fired, runtime 0 when it declined — the
                // null-safe free covers both.
                .chain(
                    lse_slots
                        .iter()
                        .map(|(idx, _)| (*idx as i64, "nsl_tensor_free_if_valid")),
                )
                // Fused-CE LSE slots: always real tensors (the
                // fused forward allocates unconditionally); the
                // teardown sweep is their ONLY free on the
                // trailing partial window (replayed entries are
                // consumed+freed by the emitted backward).
                .chain(
                    fce_slots
                        .iter()
                        .map(|(idx, _)| (*idx as i64, "nsl_tensor_free")),
                )
                .collect(),
        );
        let csla_pending = Some(CslaPending {
            adjoint: adjoint.clone(),
            slots,
            seed_base: full_vars.clone(),
            param_adj_set: param_adj_set.clone(),
            hook_accum_idx,
            accum_scale: fase_plan.recipe.accum_scale,
            loss_slot,
            params: pre.params,
            primal_view_of: pre.primal_view_of,
            lse_slots,
            lse_pushed: csla_lse_pushed,
            fce_slots,
            schedule: pre.schedule,
        });

        Ok(CslaWindowSave {
            csla_pending,
            csla_loss_buffered,
            csla_teardown_slots,
        })
    }
}

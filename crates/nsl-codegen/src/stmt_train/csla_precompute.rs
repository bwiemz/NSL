//! The CSLA schedule precompute (D2b part 2) of the train block's
//! source-AD arm: on the final adjoint, the layerwise plan, the per-param
//! facts, the replay ranges and the update grouping that the
//! segment-streamed forward and the window backward both consume — plus,
//! under `--weight-stream`, the sliced-forward plan (which params register,
//! upload and evict around which primal-op slice, or the per-layer arena
//! packs) and the parameter plan it is derived from.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 450 lines,
//! 8 inputs ([`CslaPrecomputeInputs`]); pure analysis — no IR is
//! emitted. Returns the [`CslaPre`] the save phase consumes and the
//! [`WsForwardPlan`] that drives the sliced forward emission, both `None`
//! when the CSLA schedule is not active. The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the `--layerwise-accum` fixtures.

use crate::compiler::Compiler;
use crate::error::CodegenError;
use crate::stmt_train::csla_window::{CslaParam, CslaPre, CslaSchedule};

/// Every binding of `compile_train_block_inner` the precompute reads;
/// names are the driver's.
pub(crate) struct CslaPrecomputeInputs<'a> {
    /// The final adjoint tape (the layerwise analysis runs over it).
    pub(crate) adjoint: &'a crate::wengert::WengertList,
    /// The CCR plan; the CSLA refusal guarantees `Some` when the schedule is active.
    pub(crate) ccr_plan: &'a Option<crate::ccr::CcrPlan>,
    /// Whether the CSLA window schedule is active (`(None, None)` otherwise).
    pub(crate) csla_active: bool,
    /// The forward tape the segments partition.
    pub(crate) effective_primal: &'a crate::wengert::WengertList,
    /// The Stage-2A element hints from the arena projection.
    pub(crate) elem_hints: &'a std::collections::HashMap<crate::wengert::VarId, u64>,
    /// The forward extractor (named parameters, symbol map).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The adjoint generator (primal → adjoint VarId map).
    pub(crate) generator: &'a crate::source_ad::AdjointGenerator,
    /// The model tensor paths in `param_list` order.
    pub(crate) param_paths: &'a [String],
}

pub(crate) struct WsForwardPlan {
    /// Half-open primal-op slices (prologue, per-segment,
    /// epilogue) — a partition of the tape.
    pub(crate) slices: Vec<(usize, usize)>,
    /// Streamed param_list indices registered (= evicted)
    /// at step-body top, every iteration (idempotent).
    pub(crate) register_idxs: Vec<i64>,
    /// Per-slice param_list indices uploaded before /
    /// evicted after that slice's ops (per-param mode).
    pub(crate) upload_per_slice: Vec<Vec<i64>>,
    pub(crate) evict_per_slice: Vec<Vec<i64>>,
    /// Item 10 (arena mode): one contiguous pack per streamed
    /// LAYER group as `(first_slice, last_slice, idxs)`. The
    /// whole group uploads at `first_slice` and evicts at
    /// `last_slice` — a matched set, so it holds exactly one
    /// arena slot for its residency (coarser than per-param
    /// touch, but batched into one HtoD / DtoH each way).
    pub(crate) arena_packs: Vec<(usize, usize, Vec<i64>)>,
}

impl Compiler<'_> {
    /// Precompute the CSLA schedule (see the module header).
    pub(crate) fn precompute_csla_schedule(
        &self,
        inputs: CslaPrecomputeInputs<'_>,
    ) -> Result<(Option<CslaPre>, Option<WsForwardPlan>), CodegenError> {
        let CslaPrecomputeInputs {
            adjoint,
            ccr_plan,
            csla_active,
            effective_primal,
            elem_hints,
            extractor,
            generator,
            param_paths,
        } = inputs;

        // ── D2b part 2: CSLA schedule precompute (pre-forward) ──
        // The layerwise plan, per-param facts, replay ranges, and
        // update grouping — computed HERE (on the final adjoint) so
        // the segment-streamed forward below and the window backward
        // consume the same schedule. `CslaPre` flows into the save
        // phase; `WsForwardPlan` drives the sliced forward emission.
        let (csla_pre, ws_fwd_plan): (Option<CslaPre>, Option<WsForwardPlan>) =
            if csla_active {
                let csla_trainable: Vec<(String, crate::wengert::VarId)> = extractor
                    .named_param_var_ids()
                    .iter()
                    .filter(|(name, _)| self.is_trainable_param_name(name))
                    .map(|(n, v)| (n.clone(), *v))
                    .collect();
                // Item 11 calibration (review M3 follow-through):
                // REAL element counts for the layerwise plan, now
                // from the SHARED Stage-2A hint map above — which
                // adds the initializer-derived field dims the pure
                // semantic map lacked (model fields are typed from
                // annotations only, so the old binding here was
                // empty for every unannotated real model and the
                // prefetch pack pricing stayed blind). Symbolic
                // shapes remain None and decline their edges.
                let vid_by_pname: std::collections::HashMap<&str, crate::wengert::VarId> =
                    csla_trainable
                        .iter()
                        .map(|(n, v)| (n.as_str(), *v))
                        .collect();
                let plan_lw =
                    crate::layerwise::analyze(adjoint, &csla_trainable, &|name| {
                        vid_by_pname
                            .get(name)
                            .and_then(|v| elem_hints.get(v))
                            .copied()
                    });
                let param_name_to_accum_idx: std::collections::HashMap<&str, i64> =
                    param_paths
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p.as_str(), i as i64))
                        .collect();
                let csla_params: Vec<CslaParam> = csla_trainable
                    .iter()
                    .filter_map(|(name, primal_vid)| {
                        let &accum_idx = param_name_to_accum_idx.get(name.as_str())?;
                        Some(CslaParam {
                            name: name.clone(),
                            primal_vid: *primal_vid,
                            adj_vid: generator.adjoint_of(*primal_vid),
                            accum_idx,
                        })
                    })
                    .collect();
                // PRIMAL-side view chains rooted at trainable params
                // (tied-head `embed.transpose(0,1)` etc.) — buffered
                // as slots but aliasing θ; the window site checks
                // their reads against each param's update range, and
                // the forward streamer keys eviction off their last
                // read too.
                let trainable_vid_set: std::collections::HashSet<
                    crate::wengert::VarId,
                > = csla_trainable.iter().map(|(_, v)| *v).collect();
                let mut primal_view_of: std::collections::HashMap<
                    crate::wengert::VarId,
                    crate::wengert::VarId,
                > = std::collections::HashMap::new();
                for op in &effective_primal.ops {
                    if crate::wengert::is_view_producing_op(&op.op) {
                        for &input in &op.inputs {
                            if trainable_vid_set.contains(&input) {
                                primal_view_of.insert(op.result, input);
                            } else if let Some(&p) = primal_view_of.get(&input) {
                                primal_view_of.insert(op.result, p);
                            }
                        }
                    }
                }
                let imports = crate::layerwise::adjoint_primal_imports(
                    effective_primal,
                    adjoint,
                );

                // ── D1b schedule derivation (moved from the window
                // site — indices are FINAL-adjoint positions) ──
                let adjoint_len = adjoint.ops.len();
                let mut ranges =
                    crate::layerwise::partition_ranges(&plan_lw, adjoint_len);
                if ranges.is_empty() {
                    // Degenerate (empty adjoint): one empty prologue
                    // so the update groups still fire.
                    ranges.push(crate::layerwise::ReplayRange {
                        start: 0,
                        end: adjoint_len,
                        layer: None,
                    });
                }
                let n_ranges = ranges.len();
                // Adjoint op position by result vid — for grad-op
                // containment.
                let adj_pos: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = adjoint
                    .ops
                    .iter()
                    .enumerate()
                    .map(|(i, op)| (op.result, i))
                    .collect();
                // Update groups. A layer's param updates right after
                // its range's replay iff its gradient op sits
                // positionally INSIDE that range (positional
                // attribution slop demotes it to the epilogue group
                // — always correct, merely later). Dead params (no
                // adjoint) update with their layer on a zero
                // accumulator. Every param_paths slot lands in
                // exactly one group.
                let mut layer_group: Vec<Vec<i64>> = vec![Vec::new(); n_ranges];
                let mut grouped: std::collections::HashSet<i64> = Default::default();
                // Item 11 calibration: static element count per grouped
                // param (0 = symbolic shape), keyed by accum_idx — the
                // prefetch gate's pack-byte source.
                let mut elems_by_accum: std::collections::HashMap<i64, u64> =
                    Default::default();
                {
                    let param_by_name: std::collections::HashMap<&str, &CslaParam> =
                        csla_params.iter().map(|p| (p.name.as_str(), p)).collect();
                    for (ri, range) in ranges.iter().enumerate() {
                        let Some(li) = range.layer else { continue };
                        for pinfo in &plan_lw.layers[li].params {
                            let Some(cp) = param_by_name.get(pinfo.name.as_str())
                            else {
                                continue;
                            };
                            let in_range =
                                match cp.adj_vid.and_then(|a| adj_pos.get(&a)) {
                                    Some(&pos) => {
                                        pos >= range.start && pos < range.end
                                    }
                                    None => true,
                                };
                            if in_range && grouped.insert(cp.accum_idx) {
                                layer_group[ri].push(cp.accum_idx);
                                elems_by_accum
                                    .insert(cp.accum_idx, pinfo.elems.unwrap_or(0));
                            }
                        }
                        layer_group[ri].sort_unstable();
                    }
                }
                let global_group: Vec<i64> = (0..param_paths.len() as i64)
                    .filter(|i| !grouped.contains(i))
                    .collect();
                // Compile-time schedule line — the gates' anti-vacuity
                // anchor for the LAYER-MAJOR shape itself (the runtime
                // window counter can't distinguish a degenerate
                // all-epilogue schedule from the real k-range one).
                nsl_runtime::nsl_log!(INFO, "csla", 
                    "[csla] layer-major schedule: {} ranges, {} layer-grouped params, \
                     {} epilogue params",
                    n_ranges,
                    grouped.len(),
                    global_group.len(),
                );

                // ── Weight-stream admission (moved from the window
                // site) + the part-2 forward streaming plan ──
                let ws_active = self.compile_options.weight_stream.enabled;
                let mut ws_streamed_sorted: Vec<i64> = Vec::new();
                let ws_plan = if ws_active {
                    // Review D2b-1 (HIGH): a buffered primal VIEW of a
                    // streamed param (e.g. transpose(w) saved for the
                    // matmul adjoint) caches a data pointer into θ's
                    // storage — eviction frees that storage and the
                    // later upload allocates a NEW buffer, so the view
                    // slot would read recycled memory: silent
                    // corruption. Any param rooting a view chain that
                    // lands in the buffered-import set stays RESIDENT
                    // (always safe, merely unstreamed). The import
                    // list is the slot superset (ghost imports never
                    // become slots but also never root tensor views).
                    // Pure helper — unit-tested in layerwise.rs
                    // (review D2b-2-3: the exclusion never fires on
                    // the gate fixtures, so the logic is pinned at
                    // the unit level).
                    let view_rooted = crate::layerwise::ws_view_rooted_params(
                        &imports,
                        &primal_view_of,
                    );
                    let unstreamable_idxs: std::collections::HashSet<i64> =
                        csla_params
                            .iter()
                            .filter(|cp| view_rooted.contains(&cp.primal_vid))
                            .map(|cp| cp.accum_idx)
                            .collect();
                    if !unstreamable_idxs.is_empty() {
                        nsl_runtime::nsl_log!(INFO, "weight-stream", 
                            "[weight-stream] {} param(s) stay resident: a buffered \
                             view of their storage rides the window slots",
                            unstreamable_idxs.len()
                        );
                    }
                    let mut ws_all: Vec<i64> = layer_group
                        .iter()
                        .flatten()
                        .copied()
                        .filter(|i| !unstreamable_idxs.contains(i))
                        .collect();
                    ws_all.sort_unstable();
                    // Part 2: slice the forward per CCR segment and
                    // key each streamed param's upload/evict off its
                    // first/last primal touch (view-closure-extended).
                    let plan_ref = ccr_plan
                        .as_ref()
                        .expect("csla refusal above guarantees a plan");
                    // Milestone C: the SECOND positional fork — the
                    // segment bounds are sliced against
                    // effective_primal ~1,200 lines after planning.
                    // Same digest, same rule as the
                    // apply_to_adjoint fork above.
                    {
                        let sched = self.passes.scheduler();
                        sched
                            .assert_tape_unchanged_since("CCR", effective_primal)
                            .map_err(CodegenError::new)?;
                    }
                    let seg_bounds: Vec<(usize, usize)> = plan_ref
                        .segments
                        .iter()
                        .map(|s| (s.start, s.end))
                        .collect();
                    let slices = crate::layerwise::forward_slices(
                        &seg_bounds,
                        effective_primal.ops.len(),
                    )
                    .map_err(|e| {
                        CodegenError::new(format!(
                            "--weight-stream: cannot slice the forward per \
                             CCR segment: {e}"
                        ))
                    })?;
                    let ws_idx_set: std::collections::HashSet<i64> =
                        ws_all.iter().copied().collect();
                    let streamed_vids: std::collections::HashSet<
                        crate::wengert::VarId,
                    > = csla_params
                        .iter()
                        .filter(|cp| ws_idx_set.contains(&cp.accum_idx))
                        .map(|cp| cp.primal_vid)
                        .collect();
                    let touch = crate::layerwise::forward_touch_slices(
                        effective_primal,
                        &slices,
                        &streamed_vids,
                        &primal_view_of,
                    );
                    let vid_to_idx: std::collections::HashMap<
                        crate::wengert::VarId,
                        i64,
                    > = csla_params
                        .iter()
                        .map(|cp| (cp.primal_vid, cp.accum_idx))
                        .collect();
                    let mut upload_per_slice: Vec<Vec<i64>> =
                        vec![Vec::new(); slices.len()];
                    let mut evict_per_slice: Vec<Vec<i64>> =
                        vec![Vec::new(); slices.len()];
                    for (vid, (first, last)) in &touch {
                        let idx = vid_to_idx[vid];
                        upload_per_slice[*first].push(idx);
                        evict_per_slice[*last].push(idx);
                    }
                    for v in upload_per_slice.iter_mut() {
                        v.sort_unstable();
                    }
                    for v in evict_per_slice.iter_mut() {
                        v.sort_unstable();
                    }
                    // Anti-vacuity: gates assert this exact line so a
                    // degenerate no-slice or no-touch plan can't pass
                    // as streaming. The per-slice vectors pin bracket
                    // PLACEMENT, not just cardinality (review D2b-2-2:
                    // a plan widened by touch over-extension — e.g. all
                    // uploads in slice 0, all evicts in the last —
                    // produces the same counts and bit-exact parity
                    // while silently reverting to full forward
                    // residency).
                    let per_slice = |v: &[Vec<i64>]| -> String {
                        v.iter()
                            .map(|s| s.len().to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    nsl_runtime::nsl_log!(INFO, "weight-stream", 
                        "[weight-stream] forward streaming: {} slices, \
                         {} streamed params ({} touched by the primal); \
                         uploads/slice [{}] evicts/slice [{}]",
                        slices.len(),
                        ws_all.len(),
                        touch.len(),
                        per_slice(&upload_per_slice),
                        per_slice(&evict_per_slice),
                    );
                    // Item 10: coarsen the per-param touch into one
                    // contiguous pack per streamed LAYER group (the
                    // group's forward bracket = [min first-touch, max
                    // last-touch] over its touched members). Matched
                    // upload/evict sets → one arena slot per pack.
                    let idx_touch: std::collections::HashMap<i64, (usize, usize)> = touch
                        .iter()
                        .map(|(vid, fl)| (vid_to_idx[vid], *fl))
                        .collect();
                    let mut arena_packs: Vec<(usize, usize, Vec<i64>)> = Vec::new();
                    for group in &layer_group {
                        let mut members: Vec<i64> = Vec::new();
                        let mut first = usize::MAX;
                        let mut last = 0usize;
                        for &idx in group {
                            if let Some(&(f, l)) = idx_touch.get(&idx) {
                                members.push(idx);
                                first = first.min(f);
                                last = last.max(l);
                            }
                        }
                        if !members.is_empty() {
                            members.sort_unstable();
                            arena_packs.push((first, last, members));
                        }
                    }
                    if self.compile_options.weight_stream.arena {
                        nsl_runtime::nsl_log!(INFO, "weight-stream", 
                            "[weight-stream] arena mode: {} contiguous layer packs \
                             (sizes [{}])",
                            arena_packs.len(),
                            arena_packs
                                .iter()
                                .map(|(_, _, m)| m.len().to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                    }
                    ws_streamed_sorted = ws_all.clone();
                    Some(WsForwardPlan {
                        slices,
                        register_idxs: ws_all,
                        upload_per_slice,
                        evict_per_slice,
                        arena_packs,
                    })
                } else {
                    None
                };
                // Item 11 calibration: Σ static elems of each range's
                // STREAMED params (the pack the gate prices). A member
                // with SYMBOLIC shape (elems recorded as 0) poisons the
                // whole range to 0 = "unpriceable" — a partial sum
                // would UNDERSTATE the transfer and wrongly activate
                // an overlap edge (review M3); the gate declines
                // unpriceable packs instead.
                let ws_set: std::collections::HashSet<i64> =
                    ws_streamed_sorted.iter().copied().collect();
                let range_pack_elems: Vec<u64> = layer_group
                    .iter()
                    .map(|g| {
                        let members: Vec<u64> = g
                            .iter()
                            .filter(|i| ws_set.contains(i))
                            .map(|i| elems_by_accum.get(i).copied().unwrap_or(0))
                            .collect();
                        if members.contains(&0) {
                            0
                        } else {
                            members.iter().sum()
                        }
                    })
                    .collect();
                // Item 3: derive the ParameterPlan ONCE, here, where
                // the streaming schedule is final. Everything
                // downstream (both registration belts, the runtime
                // cross-check) reads it rather than re-deriving
                // "streamed && bf16sr" / "zero3 ? streamed : {}" from
                // the flags — the duplication that let the three
                // residency tables be populated from three separate
                // spellings of the same intent.
                // Item 11: static element counts, aligned with
                // param_paths (0 = symbolic — derive treats it as
                // elementwise-ineligible).
                let plan_elems: Vec<u64> = (0..param_paths.len())
                    .map(|u| {
                        elems_by_accum.get(&(u as i64)).copied().unwrap_or(0)
                    })
                    .collect();
                let plan = crate::parameter_plan::ParameterPlan::derive(
                    param_paths,
                    &ws_streamed_sorted,
                    &plan_elems,
                    &crate::parameter_plan::PlanFeatures {
                        weight_stream: self.compile_options.weight_stream.enabled,
                        param_dtype_bf16sr: self.features.param_dtype_bf16sr,
                        zero_stage: self.features.zero_stage,
                        zero_elementwise: self.features.zero_elementwise,
                        world_size: self.features.world_size as u32,
                    },
                )
                .map_err(|e| {
                    CodegenError::new(format!("parameter plan: {e}"))
                })?;
                if std::env::var("NSL_PARAM_PLAN_REPORT").ok().as_deref()
                    == Some("1")
                {
                    eprint!("{}", plan.report());
                }
                (
                    Some(CslaPre {
                        params: csla_params,
                        primal_view_of,
                        imports,
                        schedule: CslaSchedule {
                            ranges,
                            layer_group,
                            global_group,
                            ws_streamed: ws_streamed_sorted,
                            range_pack_elems,
                            plan,
                        },
                    }),
                    ws_plan,
                )
            } else {
                (None, None)
            };

        Ok((csla_pre, ws_fwd_plan))
    }
}

//! Compiler-Scheduled Layerwise Accumulation (CSLA) — Stage-1 schedule analysis.
//!
//! Pure analysis over the source-AD adjoint tape: it groups trainable
//! parameters by transformer layer, computes each parameter's
//! last-backward-use, separates the layer-local parameters from the
//! global-scope ones (epilogue + tied / cross-layer weights that cannot be
//! updated layer-locally), and projects the accumulator-memory reduction of a
//! layerwise schedule versus the current full-model `m_partial` window.
//!
//! It changes no codegen. It is the oracle the Stage-2 emitter and the WGGO
//! memory model consume, and its last-backward-use graph is directly reusable
//! by the Milestone-C transient arena and the unified peak-memory scheduler.
//!
//! See `docs/research/CSLA-compiler-scheduled-layerwise-accumulation.md`.

use std::collections::{HashMap, HashSet};

use crate::wengert::{PrimalOp, VarId, WengertList};
use crate::wggo_graph::layer_prefix;

/// Why a parameter cannot be updated layer-locally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalReason {
    /// No `blocks.N` key — the embedding, final norm, or LM head (the epilogue).
    Epilogue,
    /// A block-keyed parameter whose backward reads extend into a *later*
    /// layer's adjoint range: a tied / cross-layer / adapter weight. Its update
    /// must wait for its true last-backward-use.
    CrossLayer,
}

/// A single trainable parameter's schedule facts.
#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub name: String,
    pub primal_var: VarId,
    /// Last adjoint-op index that reads this parameter's primal value; `None`
    /// if it is never read in the backward (dead / frozen).
    pub last_backward_use: Option<usize>,
    /// Element count, when a shape was available.
    pub elems: Option<u64>,
}

/// One layer's local update group.
#[derive(Debug, Clone)]
pub struct LayerSchedule {
    /// Bare layer key, e.g. `blocks.3`.
    pub layer_key: String,
    pub params: Vec<ParamInfo>,
    /// First / last adjoint index spanned by this layer's parameter reads.
    pub adjoint_first: Option<usize>,
    pub adjoint_last: Option<usize>,
    /// Σ of this layer's parameter elements (0 where unknown).
    pub elems: u64,
}

/// A parameter that must be updated at the end of the backward, not layer-locally.
#[derive(Debug, Clone)]
pub struct GlobalParam {
    pub name: String,
    pub primal_var: VarId,
    pub last_backward_use: Option<usize>,
    pub elems: Option<u64>,
    pub reason: GlobalReason,
}

/// Accumulator-memory projection: full window vs layerwise peak.
#[derive(Debug, Clone, Copy)]
pub struct MemoryProjection {
    /// Σ over every trainable parameter — the current full-model `m_partial`.
    pub full_window_elems: u64,
    /// max single layer + all global params — the CSLA peak accumulator.
    pub layerwise_peak_elems: u64,
    /// full / peak (≥ 1.0); how much smaller the accumulator surface becomes.
    /// 1.0 when there are no layer groups (nothing to gain).
    pub reduction_factor: f64,
}

/// The full CSLA schedule for one train block.
#[derive(Debug, Clone)]
pub struct LayerwisePlan {
    /// Layer groups, sorted by `adjoint_first` (reverse model order — the
    /// order the backward visits them).
    pub layers: Vec<LayerSchedule>,
    pub global_params: Vec<GlobalParam>,
    pub projection: MemoryProjection,
}

/// Build first/last adjoint-op indices for every VarId read in the backward.
///
/// `last` is extended through NslList membership (ccr's fixpoint): a param
/// feeding a list-building op (RoPE `tensor_cat`) must not be classified
/// layer-local by its direct read position — the LIST's last read is the
/// param's true last backward use, and an earlier in-place θ update would
/// corrupt it (Stage-1 originally shipped without this; Stage-2's per-layer
/// update guard made it load-bearing).
fn adjoint_use_ranges(adjoint: &WengertList) -> (HashMap<VarId, usize>, HashMap<VarId, usize>) {
    let mut first: HashMap<VarId, usize> = HashMap::new();
    let mut last: HashMap<VarId, usize> = HashMap::new();
    for (idx, op) in adjoint.ops.iter().enumerate() {
        for &input in &op.inputs {
            first.entry(input).or_insert(idx);
            last.insert(input, idx); // monotonic -> ends at the last consuming index
        }
    }
    crate::ccr::extend_last_use_through_lists(adjoint, &mut last);
    (first, last)
}

/// Analyze the layerwise-accumulation schedule.
///
/// `named_params` is the extractor's `(compound_name, primal_leaf_var)` list.
/// `param_elems(name)` returns the parameter's element count when a shape is
/// known (the caller supplies it from type info; `|_| None` is acceptable and
/// simply yields an unquantified projection).
pub fn analyze(
    adjoint: &WengertList,
    named_params: &[(String, VarId)],
    param_elems: &dyn Fn(&str) -> Option<u64>,
) -> LayerwisePlan {
    let (first_use, last_use) = adjoint_use_ranges(adjoint);

    // Bucket by layer key; unkeyed params are epilogue globals up front.
    let mut layer_map: HashMap<String, Vec<ParamInfo>> = HashMap::new();
    let mut global_params: Vec<GlobalParam> = Vec::new();

    for (name, var) in named_params {
        let info = ParamInfo {
            name: name.clone(),
            primal_var: *var,
            last_backward_use: last_use.get(var).copied(),
            elems: param_elems(name),
        };
        match layer_prefix(name) {
            Some(key) => layer_map.entry(key).or_default().push(info),
            None => global_params.push(GlobalParam {
                name: info.name,
                primal_var: info.primal_var,
                last_backward_use: info.last_backward_use,
                elems: info.elems,
                reason: GlobalReason::Epilogue,
            }),
        }
    }

    // Per-layer adjoint ranges (over the layer's parameter reads).
    let mut layers: Vec<LayerSchedule> = layer_map
        .into_iter()
        .map(|(layer_key, params)| {
            let adjoint_first = params
                .iter()
                .filter_map(|p| first_use.get(&p.primal_var).copied())
                .min();
            let adjoint_last = params
                .iter()
                .filter_map(|p| last_use.get(&p.primal_var).copied())
                .max();
            let elems = params.iter().filter_map(|p| p.elems).sum();
            LayerSchedule { layer_key, params, adjoint_first, adjoint_last, elems }
        })
        .collect();

    // Order layers by first adjoint appearance (reverse model order).
    layers.sort_by(|a, b| {
        a.adjoint_first
            .unwrap_or(usize::MAX)
            .cmp(&b.adjoint_first.unwrap_or(usize::MAX))
            .then_with(|| a.layer_key.cmp(&b.layer_key))
    });

    // Cross-layer / tied detection: a layer parameter whose last-backward-use
    // reaches into a *later* layer's adjoint range is shared beyond its layer
    // and cannot be updated layer-locally. Move it to the globals.
    let next_first: Vec<Option<usize>> = (0..layers.len())
        .map(|i| layers[i + 1..].iter().filter_map(|l| l.adjoint_first).min())
        .collect();
    for i in 0..layers.len() {
        let boundary = next_first[i];
        let (local, escaped): (Vec<ParamInfo>, Vec<ParamInfo>) =
            layers[i].params.drain(..).partition(|p| match (p.last_backward_use, boundary) {
                (Some(lu), Some(nf)) => lu < nf, // last use before the next layer starts -> local
                _ => true,                       // no next layer, or never read -> local
            });
        for p in escaped {
            global_params.push(GlobalParam {
                name: p.name,
                primal_var: p.primal_var,
                last_backward_use: p.last_backward_use,
                elems: p.elems,
                reason: GlobalReason::CrossLayer,
            });
        }
        layers[i].params = local;
        // Recompute the layer's element total after removals.
        layers[i].elems = layers[i].params.iter().filter_map(|p| p.elems).sum();
    }
    // Drop layers left empty by the moves.
    layers.retain(|l| !l.params.is_empty());

    // Projection.
    let global_elems: u64 = global_params.iter().filter_map(|g| g.elems).sum();
    let layer_elems_sum: u64 = layers.iter().map(|l| l.elems).sum();
    let full_window_elems = layer_elems_sum + global_elems;
    let max_layer_elems = layers.iter().map(|l| l.elems).max().unwrap_or(0);
    let layerwise_peak_elems = max_layer_elems + global_elems;
    let reduction_factor = if layerwise_peak_elems > 0 {
        full_window_elems as f64 / layerwise_peak_elems as f64
    } else {
        1.0
    };

    LayerwisePlan {
        layers,
        global_params,
        projection: MemoryProjection {
            full_window_elems,
            layerwise_peak_elems,
            reduction_factor,
        },
    }
}

/// CSLA Stage-2: the adjoint's imports from the primal tape — every VarId
/// produced by a primal op and read by at least one adjoint op.
///
/// These are exactly the values the window-buffered schedule must carry from
/// each micro-batch's forward to the deferred backward phase. `Param` and
/// `Constant` leaves are excluded: their lowered values are loop-invariant
/// (the same runtime parameter pointer / immediate every micro-batch — FASE
/// Deferred only updates θ at window boundaries, after the buffered backward
/// has consumed the window), so the backward loop reuses the current
/// iteration's SSA values for them. `Input` leaves (the batch dict and
/// anything else fed per step) ARE included — they change every micro-batch.
///
/// Must be computed on the FINAL adjoint (post `eliminate_dead_gradients`,
/// post CCR `apply_to_adjoint`/`splice_decompress`/`insert_adjoint_last_use_
/// frees`) so recompute-victim reads — remapped to adjoint-local clones — are
/// correctly absent, and only true boundary/saved values are buffered.
///
/// Sorted by VarId so buffer slot indices are deterministic across builds.
pub fn adjoint_primal_imports(primal: &WengertList, adjoint: &WengertList) -> Vec<VarId> {
    let mut loop_invariant: HashSet<VarId> = HashSet::new();
    let mut primal_results: HashSet<VarId> = HashSet::new();
    for op in &primal.ops {
        primal_results.insert(op.result);
        if matches!(op.op, PrimalOp::Param(_) | PrimalOp::Constant(_)) {
            loop_invariant.insert(op.result);
        }
    }
    let import_set: HashSet<VarId> = adjoint
        .ops
        .iter()
        .flat_map(|op| op.inputs.iter().copied())
        .filter(|v| primal_results.contains(v) && !loop_invariant.contains(v))
        .collect();
    let mut imports: Vec<VarId> = import_set.into_iter().collect();
    imports.sort_unstable();
    imports
}

/// CSLA Stage-2 (D1b): one contiguous replay range of the adjoint tape.
#[derive(Debug, Clone)]
pub struct ReplayRange {
    /// Half-open op-index range `[start, end)` into the FINAL adjoint.
    pub start: usize,
    pub end: usize,
    /// Index into `LayerwisePlan.layers` whose layer-local params update
    /// after this range's replay completes; `None` for the prologue (loss /
    /// LM-head / final-norm adjoints before the first layer boundary).
    pub layer: Option<usize>,
}

/// CSLA Stage-2 (D1b): positional partition of the adjoint into replay
/// ranges.
///
/// Boundaries are the layers' `adjoint_first` indices in plan order (=
/// backward visit order); range 0 is the prologue `[0, first_0)`; the LAST
/// range extends to `adjoint_len` so the epilogue ops after the final
/// layer's region (the embedding backward) replay with it — their
/// gradients belong to global-scope params whose accumulators live for the
/// whole window, so positional attribution slop is safe by construction
/// (the same slop CCR's segmentation tolerates). Layers whose
/// `adjoint_first` is `None` (all params dead — never read in the
/// backward) contribute no boundary; the caller must route their params to
/// the epilogue update group.
///
/// Every op index in `[0, adjoint_len)` lands in exactly one range. An
/// empty prologue is dropped.
pub fn partition_ranges(plan: &LayerwisePlan, adjoint_len: usize) -> Vec<ReplayRange> {
    let mut boundaries: Vec<(usize, usize)> = plan
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, l)| l.adjoint_first.map(|f| (f, li)))
        .collect();
    // plan.layers is sorted by adjoint_first ascending already; keep the
    // sort local so the contract doesn't lean on the caller.
    boundaries.sort_unstable();
    let mut ranges = Vec::with_capacity(boundaries.len() + 1);
    let first_boundary = boundaries.first().map(|&(f, _)| f).unwrap_or(adjoint_len);
    if first_boundary > 0 {
        ranges.push(ReplayRange {
            start: 0,
            end: first_boundary,
            layer: None,
        });
    }
    for (i, &(start, li)) in boundaries.iter().enumerate() {
        let end = boundaries
            .get(i + 1)
            .map(|&(next, _)| next)
            .unwrap_or(adjoint_len);
        ranges.push(ReplayRange {
            start,
            end,
            layer: Some(li),
        });
    }
    ranges
}

/// CSLA D2b part 2: forward slice boundaries from the CCR block segments.
///
/// Returns half-open op-index slices over the primal tape — an optional
/// prologue `[0, seg0.start)`, one slice per segment, and an optional
/// epilogue `[last.end, primal_len)` — that PARTITION `[0, primal_len)`.
/// The segment-streamed forward lowers each slice separately with weight
/// upload/evict calls between them, so a coverage gap or overlap would
/// skip or double-emit real forward ops: errors are hard refusals, never
/// a silent fallback.
pub fn forward_slices(
    segments: &[(usize, usize)],
    primal_len: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let mut slices = Vec::with_capacity(segments.len() + 2);
    let mut cursor = 0usize;
    for (i, &(start, end)) in segments.iter().enumerate() {
        if start < cursor || end < start || end > primal_len {
            return Err(format!(
                "segment {i} [{start}, {end}) is out of order or out of \
                 bounds (cursor {cursor}, primal len {primal_len})"
            ));
        }
        if start > cursor {
            slices.push((cursor, start));
        }
        slices.push((start, end));
        cursor = end;
    }
    if cursor < primal_len {
        slices.push((cursor, primal_len));
    }
    Ok(slices)
}

/// CSLA D2b part 2: per-parameter forward touch range over a view closure.
///
/// For each streamed param VarId, the (first, last) SLICE index whose ops
/// read the param directly OR through a transitive primal view of it
/// (`view_of`: view VarId → rooting param VarId — transpose/reshape
/// chains; a view's data pointer aliases θ's storage, so the param must
/// stay device-resident until the view's last read, not merely its own).
/// Params never read in the primal are absent (the window backward is
/// their only device consumer).
pub fn forward_touch_slices(
    primal: &WengertList,
    slices: &[(usize, usize)],
    param_vids: &HashSet<VarId>,
    view_of: &HashMap<VarId, VarId>,
) -> HashMap<VarId, (usize, usize)> {
    let slice_of_pos = |pos: usize| -> usize {
        slices
            .iter()
            .position(|&(s, e)| pos >= s && pos < e)
            .unwrap_or(slices.len().saturating_sub(1))
    };
    let mut touch: HashMap<VarId, (usize, usize)> = HashMap::new();
    for (idx, op) in primal.ops.iter().enumerate() {
        for input in &op.inputs {
            let root = if param_vids.contains(input) {
                *input
            } else {
                match view_of.get(input) {
                    Some(p) if param_vids.contains(p) => *p,
                    _ => continue,
                }
            };
            let si = slice_of_pos(idx);
            touch
                .entry(root)
                .and_modify(|(f, l)| {
                    *f = (*f).min(si);
                    *l = (*l).max(si);
                })
                .or_insert((si, si));
        }
    }
    touch
}

impl LayerwisePlan {
    /// A human-readable schedule + projection for `nsl check --training-report`.
    pub fn render_report(&self, indent: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{indent}Layerwise accumulation (CSLA) schedule:\n\
             {indent}  {} layer group(s), {} global-scope param(s)\n",
            self.layers.len(),
            self.global_params.len(),
        ));
        for l in &self.layers {
            out.push_str(&format!(
                "{indent}  {:<12} {} param(s), {} elems, adjoint [{}..{}]\n",
                l.layer_key,
                l.params.len(),
                l.elems,
                l.adjoint_first.map(|v| v as isize).unwrap_or(-1),
                l.adjoint_last.map(|v| v as isize).unwrap_or(-1),
            ));
        }
        for g in &self.global_params {
            let reason = match g.reason {
                GlobalReason::Epilogue => "epilogue",
                GlobalReason::CrossLayer => "tied/cross-layer",
            };
            out.push_str(&format!("{indent}  global: {:<24} ({reason})\n", g.name));
        }
        let p = &self.projection;
        if p.full_window_elems > 0 {
            out.push_str(&format!(
                "{indent}  m_partial window: {} elems -> layerwise peak {} elems ({:.2}x smaller)\n",
                p.full_window_elems, p.layerwise_peak_elems, p.reduction_factor,
            ));
        } else {
            out.push_str(&format!(
                "{indent}  m_partial window reduction: {} layer group(s), 1 live at a time \
                 (element counts unquantified at this stage; see the CSLA paper's per-model table)\n",
                self.layers.len(),
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::HashMap;

    /// Build a minimal adjoint op that READS `inputs` (its `op`/`result` are
    /// irrelevant to the analysis, which keys only off `inputs`).
    fn read_op(id: u32, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result: 1000 + id, // fresh result, never a param
            op: PrimalOp::Add,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn adjoint(ops: Vec<WengertOp>) -> WengertList {
        WengertList {
            ops,
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    fn elems_of(map: &[(&str, u64)]) -> impl Fn(&str) -> Option<u64> {
        let m: HashMap<String, u64> = map.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        move |name: &str| m.get(name).copied()
    }

    #[test]
    fn groups_by_layer_and_orders_reverse() {
        // Two clean stacked blocks. block.1 read early in the adjoint (reverse
        // order), block.0 later.
        let adj = adjoint(vec![
            read_op(0, vec![10]), // blocks.1.w read
            read_op(1, vec![10]),
            read_op(2, vec![11]), // blocks.0.w read
            read_op(3, vec![11]),
        ]);
        let params = vec![
            ("m.blocks.1.w".to_string(), 10u32),
            ("m.blocks.0.w".to_string(), 11u32),
        ];
        let elems = elems_of(&[("m.blocks.1.w", 100), ("m.blocks.0.w", 100)]);
        let plan = analyze(&adj, &params, &elems);

        assert_eq!(plan.layers.len(), 2);
        // Reverse model order: blocks.1 first (adjoint_first 0), then blocks.0.
        assert_eq!(plan.layers[0].layer_key, "blocks.1");
        assert_eq!(plan.layers[1].layer_key, "blocks.0");
        assert_eq!(plan.layers[0].adjoint_first, Some(0));
        assert_eq!(plan.layers[0].adjoint_last, Some(1));
        assert!(plan.global_params.is_empty());
        // full = 200 elems, peak = one layer (100) + 0 global = 100 -> 2x.
        assert_eq!(plan.projection.full_window_elems, 200);
        assert_eq!(plan.projection.layerwise_peak_elems, 100);
        assert!((plan.projection.reduction_factor - 2.0).abs() < 1e-9);
    }

    #[test]
    fn epilogue_params_are_global() {
        // embed + final norm carry no blocks.N key.
        let adj = adjoint(vec![
            read_op(0, vec![10]), // blocks.0.w
            read_op(1, vec![20]), // m.embed
            read_op(2, vec![21]), // m.norm_out
        ]);
        let params = vec![
            ("m.blocks.0.w".to_string(), 10u32),
            ("m.embed".to_string(), 20u32),
            ("m.norm_out".to_string(), 21u32),
        ];
        let elems = elems_of(&[("m.blocks.0.w", 100), ("m.embed", 500), ("m.norm_out", 4)]);
        let plan = analyze(&adj, &params, &elems);
        assert_eq!(plan.layers.len(), 1);
        let mut gnames: Vec<&str> = plan.global_params.iter().map(|g| g.name.as_str()).collect();
        gnames.sort();
        assert_eq!(gnames, vec!["m.embed", "m.norm_out"]);
        assert!(plan.global_params.iter().all(|g| g.reason == GlobalReason::Epilogue));
        // full = 100+500+4 = 604; peak = layer 100 + global 504 = 604 -> 1x
        // (the epilogue dominates — the report shows the win is bounded by it).
        assert_eq!(plan.projection.full_window_elems, 604);
        assert_eq!(plan.projection.layerwise_peak_elems, 604);
    }

    #[test]
    fn cross_layer_param_flagged_and_deferred() {
        // blocks.1.shared is read in block 1's region (op 0) AND deep in block
        // 0's region (op 5) -> last use 5 escapes into the later layer.
        let adj = adjoint(vec![
            read_op(0, vec![10, 13]), // blocks.1.w + blocks.1.shared
            read_op(1, vec![10]),
            read_op(2, vec![11]), // blocks.0.w
            read_op(3, vec![11]),
            read_op(4, vec![11]),
            read_op(5, vec![13]), // blocks.1.shared read again, inside block 0's range
        ]);
        let params = vec![
            ("m.blocks.1.w".to_string(), 10u32),
            ("m.blocks.0.w".to_string(), 11u32),
            ("m.blocks.1.shared".to_string(), 13u32),
        ];
        let elems = elems_of(&[
            ("m.blocks.1.w", 100),
            ("m.blocks.0.w", 100),
            ("m.blocks.1.shared", 50),
        ]);
        let plan = analyze(&adj, &params, &elems);
        // blocks.1.shared is moved to globals with CrossLayer reason.
        let shared = plan
            .global_params
            .iter()
            .find(|g| g.name == "m.blocks.1.shared")
            .expect("shared param should be global");
        assert_eq!(shared.reason, GlobalReason::CrossLayer);
        assert_eq!(shared.last_backward_use, Some(5));
        // blocks.1 now holds only w (100); blocks.0 holds w (100).
        let b1 = plan.layers.iter().find(|l| l.layer_key == "blocks.1").unwrap();
        assert_eq!(b1.elems, 100);
    }

    #[test]
    fn empty_is_safe() {
        let adj = adjoint(vec![]);
        let plan = analyze(&adj, &[], &|_| None);
        assert!(plan.layers.is_empty());
        assert!(plan.global_params.is_empty());
        assert_eq!(plan.projection.reduction_factor, 1.0);
    }

    fn primal_op(id: u32, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
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
    fn imports_are_adjoint_read_primal_values_minus_invariant_leaves() {
        // Primal: param(1), constant(2), input(3), act = input @ param (4),
        // loss = act + constant (5).
        let primal = adjoint(vec![
            primal_op(0, 1, PrimalOp::Param("m.w".into()), vec![]),
            primal_op(1, 2, PrimalOp::Constant(1.0), vec![]),
            primal_op(2, 3, PrimalOp::Input("batch".into()), vec![]),
            primal_op(3, 4, PrimalOp::Matmul, vec![3, 1]),
            primal_op(4, 5, PrimalOp::Add, vec![4, 2]),
        ]);
        // Adjoint reads: loss(5) for the seed, act(4) and param(1) for the
        // matmul backward, input(3) for the param-grad side, constant(2) for
        // the add backward, plus an adjoint-internal value (100).
        let adj = adjoint(vec![
            primal_op(0, 100, PrimalOp::Add, vec![5]),
            primal_op(1, 101, PrimalOp::Matmul, vec![100, 1, 4]),
            primal_op(2, 102, PrimalOp::Matmul, vec![100, 3, 2]),
            primal_op(3, 103, PrimalOp::Add, vec![101, 102]),
        ]);
        let imports = adjoint_primal_imports(&primal, &adj);
        // Param(1) and Constant(2) excluded (loop-invariant); adjoint-internal
        // 100/101/102 excluded (not primal results). Input(3), act(4), and
        // loss(5) are the window-buffered set.
        assert_eq!(imports, vec![3, 4, 5]);
    }

    #[test]
    fn imports_are_deterministically_sorted_and_deduped() {
        let primal = adjoint(vec![
            primal_op(0, 9, PrimalOp::Input("b".into()), vec![]),
            primal_op(1, 4, PrimalOp::Relu, vec![9]),
        ]);
        let adj = adjoint(vec![
            primal_op(0, 200, PrimalOp::Relu, vec![9, 4]),
            primal_op(1, 201, PrimalOp::Relu, vec![4, 9]),
        ]);
        assert_eq!(adjoint_primal_imports(&primal, &adj), vec![4, 9]);
    }

    #[test]
    fn list_membership_extends_param_last_use() {
        // Param 10 read directly at op 0, but op 0 BUILDS a list consumed at
        // op 3 — inside a later layer's range. Without the fixpoint the param
        // would classify layer-local with last use 0; with it, last use is 3
        // and it must go CrossLayer.
        let mut adj = adjoint(vec![
            read_op(0, vec![10]),      // list build reading blocks.1.w
            read_op(1, vec![11]),      // blocks.0.w read (later layer starts)
            read_op(2, vec![11]),
            read_op(3, vec![1000]),    // consumes the LIST (op 0's result)
        ]);
        adj.var_types.insert(1000, crate::wengert::WengertType::List);
        let params = vec![
            ("m.blocks.1.w".to_string(), 10u32),
            ("m.blocks.0.w".to_string(), 11u32),
        ];
        let plan = analyze(&adj, &params, &|_| None);
        let w1 = plan
            .global_params
            .iter()
            .find(|g| g.name == "m.blocks.1.w")
            .expect("list-fed param must be CrossLayer");
        assert_eq!(w1.reason, GlobalReason::CrossLayer);
        assert_eq!(w1.last_backward_use, Some(3));
    }

    #[test]
    fn partition_covers_tape_exactly_in_backward_order() {
        // Prologue ops 0-1, blocks.1 range 2-4, blocks.0 range 5-7, epilogue
        // ops 8-9 swallowed into the last range.
        let adj = adjoint(vec![
            read_op(0, vec![50]),  // loss/LM-head adjoints (no layer param)
            read_op(1, vec![50]),
            read_op(2, vec![10]),  // blocks.1
            read_op(3, vec![10]),
            read_op(4, vec![50]),
            read_op(5, vec![11]),  // blocks.0
            read_op(6, vec![11]),
            read_op(7, vec![50]),
            read_op(8, vec![50]),  // embedding backward epilogue
            read_op(9, vec![50]),
        ]);
        let params = vec![
            ("m.blocks.1.w".to_string(), 10u32),
            ("m.blocks.0.w".to_string(), 11u32),
        ];
        let plan = analyze(&adj, &params, &|_| None);
        let ranges = partition_ranges(&plan, adj.ops.len());
        assert_eq!(ranges.len(), 3);
        assert_eq!((ranges[0].start, ranges[0].end, ranges[0].layer), (0, 2, None));
        assert_eq!((ranges[1].start, ranges[1].end), (2, 5));
        assert_eq!(
            plan.layers[ranges[1].layer.unwrap()].layer_key,
            "blocks.1"
        );
        assert_eq!((ranges[2].start, ranges[2].end), (5, 10));
        assert_eq!(
            plan.layers[ranges[2].layer.unwrap()].layer_key,
            "blocks.0"
        );
        // Exact coverage, no overlap.
        let covered: usize = ranges.iter().map(|r| r.end - r.start).sum();
        assert_eq!(covered, adj.ops.len());
    }

    #[test]
    fn partition_without_layers_is_one_prologue_range() {
        let adj = adjoint(vec![read_op(0, vec![50]), read_op(1, vec![50])]);
        let plan = analyze(&adj, &[], &|_| None);
        let ranges = partition_ranges(&plan, adj.ops.len());
        assert_eq!(ranges.len(), 1);
        assert_eq!((ranges[0].start, ranges[0].end, ranges[0].layer), (0, 2, None));
    }

    // ── D2b part 2: forward slice helpers ──────────────────────────────

    #[test]
    fn forward_slices_partitions_with_prologue_and_epilogue() {
        let slices = forward_slices(&[(3, 7), (7, 12)], 15).unwrap();
        assert_eq!(slices, vec![(0, 3), (3, 7), (7, 12), (12, 15)]);
        let covered: usize = slices.iter().map(|&(s, e)| e - s).sum();
        assert_eq!(covered, 15);
    }

    #[test]
    fn forward_slices_no_prologue_no_epilogue() {
        let slices = forward_slices(&[(0, 5), (5, 9)], 9).unwrap();
        assert_eq!(slices, vec![(0, 5), (5, 9)]);
    }

    #[test]
    fn forward_slices_gap_between_segments_becomes_a_slice() {
        // Non-adjacent segments (an inter-segment gap) still partition —
        // the gap lowers as its own unstreamed slice.
        let slices = forward_slices(&[(2, 4), (6, 8)], 10).unwrap();
        assert_eq!(slices, vec![(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]);
    }

    #[test]
    fn forward_slices_refuses_overlap_and_out_of_bounds() {
        assert!(forward_slices(&[(3, 7), (5, 9)], 12).is_err());
        assert!(forward_slices(&[(0, 20)], 12).is_err());
        assert!(forward_slices(&[(4, 2)], 12).is_err());
    }

    #[test]
    fn forward_slices_empty_segments_is_one_slice() {
        assert_eq!(forward_slices(&[], 6).unwrap(), vec![(0, 6)]);
    }

    #[test]
    fn touch_slices_direct_reads_and_view_closure() {
        // Ops 0..2 = slice 0, ops 2..5 = slice 1, ops 5..6 = slice 2.
        // Param 10 read directly in slice 0 and via view 200 in slice 1;
        // param 11 read only in slice 1; param 12 never read.
        let primal = adjoint(vec![
            read_op(0, vec![10]),     // slice 0: direct read of 10
            read_op(1, vec![7]),      //          unrelated
            read_op(2, vec![11]),     // slice 1: direct read of 11
            read_op(3, vec![200]),    //          view-of-10 read
            read_op(4, vec![7]),      //          unrelated
            read_op(5, vec![7]),      // slice 2: unrelated
        ]);
        let slices = vec![(0, 2), (2, 5), (5, 6)];
        let params: HashSet<VarId> = [10, 11, 12].into_iter().collect();
        let views: HashMap<VarId, VarId> = [(200, 10)].into_iter().collect();
        let touch = forward_touch_slices(&primal, &slices, &params, &views);
        assert_eq!(touch.get(&10), Some(&(0, 1)));
        assert_eq!(touch.get(&11), Some(&(1, 1)));
        assert_eq!(touch.get(&12), None);
    }

    #[test]
    fn touch_slices_view_of_unstreamed_param_is_ignored() {
        // A view rooted at a param OUTSIDE the streamed set contributes
        // nothing (that param stays resident; no touch entry needed).
        let primal = adjoint(vec![read_op(0, vec![300])]);
        let slices = vec![(0, 1)];
        let params: HashSet<VarId> = [10].into_iter().collect();
        let views: HashMap<VarId, VarId> = [(300, 99)].into_iter().collect();
        let touch = forward_touch_slices(&primal, &slices, &params, &views);
        assert!(touch.is_empty());
    }
}

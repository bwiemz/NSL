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

use std::collections::HashMap;

use crate::wengert::{VarId, WengertList};
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
fn adjoint_use_ranges(adjoint: &WengertList) -> (HashMap<VarId, usize>, HashMap<VarId, usize>) {
    let mut first: HashMap<VarId, usize> = HashMap::new();
    let mut last: HashMap<VarId, usize> = HashMap::new();
    for (idx, op) in adjoint.ops.iter().enumerate() {
        for &input in &op.inputs {
            first.entry(input).or_insert(idx);
            last.insert(input, idx); // monotonic -> ends at the last consuming index
        }
    }
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
}

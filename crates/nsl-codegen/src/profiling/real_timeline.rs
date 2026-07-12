//! Real-liveness HBM timeline for `nsl profile --memory` (dev-tools paper
//! section 3.2), built from the CAPTURED train-block WengertList instead of
//! the fixed 2-step-lifetime approximation.
//!
//! The model: generate the adjoint with the production `AdjointGenerator`,
//! concatenate primal + adjoint ops into one program-point axis, and compute
//! exact birth/death liveness over the combined sequence. Saved-for-backward
//! activations extend naturally across the forward/backward boundary because
//! the adjoint ops consume them; parameters and parameter gradients are
//! pinned live to the end (weights are resident; grads feed the optimizer
//! step, which runs after the list).
//!
//! Byte sizes come from the capture's best-effort per-var hints (concrete
//! typed shapes only). Vars without a hint are counted separately and
//! reported — never silently billed as zero being "free".
//!
//! What-if peaks (paper section 3.2's "With FASE" / checkpointing lines) are
//! computed by re-running the same sweep with modified deaths:
//! * FASE: each parameter gradient dies right after it is produced (the
//!   fused per-layer optimizer step consumes it immediately) instead of
//!   living to the end.
//! * Full recomputation (`@checkpoint` everything): saved-for-backward
//!   activations die at their last FORWARD use; the backward pass is assumed
//!   to recompute them (recompute time cost is NOT shown — the line is a
//!   memory bound, and labeled as such).

use std::collections::{HashMap, HashSet};

use crate::profiling::types::MemoryTimelineEntry;
use crate::source_ad::AdjointGenerator;
use crate::wengert::{PrimalOp, VarId, WengertList};

/// One "with X: peak drops to Y" line for the render and the JSON report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhatIfPeak {
    pub label: String,
    pub peak_bytes: u64,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct RealTimeline {
    pub entries: Vec<MemoryTimelineEntry>,
    pub peak_bytes: u64,
    pub peak_pp: u32,
    pub forward_ops: usize,
    pub backward_ops: usize,
    /// Vars with a byte-size hint / without one. When `unsized_vars` is
    /// large the MB numbers are a lower bound and the render says so.
    pub sized_vars: usize,
    pub unsized_vars: usize,
    pub what_if: Vec<WhatIfPeak>,
}

struct Liveness {
    birth: HashMap<VarId, u32>,
    death: HashMap<VarId, u32>,
    total_pps: u32,
}

/// Build the real-liveness training timeline. Returns `None` for an empty
/// primal list (nothing to model).
pub fn build_training_timeline(
    primal: &WengertList,
    size_hints: &HashMap<VarId, u64>,
) -> Option<RealTimeline> {
    if primal.ops.is_empty() {
        return None;
    }

    // Adjoint var ids continue after the primal namespace.
    let start_var = primal
        .ops
        .iter()
        .flat_map(|op| op.inputs.iter().copied().chain(std::iter::once(op.result)))
        .max()
        .unwrap_or(0)
        + 1;
    let mut gen = AdjointGenerator::new(start_var);
    let adjoint = gen.generate(primal);

    let n_fwd = primal.ops.len();
    let n_bwd = adjoint.ops.len();
    let total = (n_fwd + n_bwd) as u32;

    let param_vars: HashSet<VarId> = primal
        .ops
        .iter()
        .filter(|op| matches!(op.op, PrimalOp::Param(_)))
        .map(|op| op.result)
        .collect();

    // Parameter gradients: the adjoint counterparts of param vars.
    let grad_vars: HashSet<VarId> = gen
        .adjoint_vars_map()
        .iter()
        .filter(|(pv, _)| param_vars.contains(pv))
        .map(|(_, av)| *av)
        .collect();

    // Adjoint vars have the shape of their primal counterpart.
    let mut sizes: HashMap<VarId, u64> = size_hints.clone();
    for (pv, av) in gen.adjoint_vars_map() {
        if let Some(&b) = size_hints.get(pv) {
            sizes.entry(*av).or_insert(b);
        }
    }

    let combined: Vec<(&VarId, &Vec<VarId>)> = primal
        .ops
        .iter()
        .chain(adjoint.ops.iter())
        .map(|op| (&op.result, &op.inputs))
        .collect();

    // Base liveness over the combined sequence.
    let mut birth: HashMap<VarId, u32> = HashMap::new();
    let mut death: HashMap<VarId, u32> = HashMap::new();
    // Last-forward-use deaths, for the recompute what-if.
    let mut fwd_death: HashMap<VarId, u32> = HashMap::new();
    for (i, (result, inputs)) in combined.iter().enumerate() {
        let pp = i as u32;
        birth.entry(**result).or_insert(pp);
        for &inp in inputs.iter() {
            death.insert(inp, pp + 1);
            if i < n_fwd {
                fwd_death.insert(inp, pp + 1);
            }
        }
    }
    // Never-consumed results are transient: die right after production.
    for (result, _) in &combined {
        death.entry(**result).or_insert(birth[result] + 1);
        fwd_death.entry(**result).or_insert(birth[result] + 1);
    }
    // Params resident for the whole step; grads live to the optimizer.
    for v in &param_vars {
        death.insert(*v, total);
    }
    for v in &grad_vars {
        death.insert(*v, total);
    }

    let base = Liveness {
        birth: birth.clone(),
        death: death.clone(),
        total_pps: total,
    };
    let (series, peak_bytes, peak_pp) = sweep(&base, &sizes);

    // What-if 1: FASE — param grads die immediately after production.
    let mut fase_death = death.clone();
    for v in &grad_vars {
        if let Some(&b) = birth.get(v) {
            fase_death.insert(*v, b + 1);
        }
    }
    let (_, fase_peak, _) = sweep(
        &Liveness {
            birth: birth.clone(),
            death: fase_death,
            total_pps: total,
        },
        &sizes,
    );

    // What-if 2: full recomputation — saved-for-backward activations die at
    // their last forward use. Params/grads keep their base lifetimes.
    let saved = crate::source_ad::analyze_saved_tensors(primal, &adjoint);
    let mut ckpt_death = death.clone();
    for info in &saved {
        if param_vars.contains(&info.var) {
            continue;
        }
        if let Some(&fd) = fwd_death.get(&info.var) {
            ckpt_death.insert(info.var, fd);
        }
    }
    let (_, ckpt_peak, _) = sweep(
        &Liveness {
            birth,
            death: ckpt_death,
            total_pps: total,
        },
        &sizes,
    );

    // Phase markers on the rendered series.
    let mut entries: Vec<MemoryTimelineEntry> = series
        .iter()
        .enumerate()
        .map(|(pp, &live_bytes)| MemoryTimelineEntry {
            program_point: pp as u32,
            live_bytes,
            phase: None,
        })
        .collect();
    mark(&mut entries, 0, "forward begins (weights resident)");
    if n_fwd >= 1 {
        mark(&mut entries, n_fwd as u32 - 1, "loss computed");
    }
    if n_bwd > 0 && (n_fwd as u32) < total {
        mark(&mut entries, n_fwd as u32, "backward begins");
    }
    if total >= 1 {
        mark(
            &mut entries,
            total - 1,
            "gradients ready (optimizer step follows)",
        );
    }
    mark(&mut entries, peak_pp, "<-- peak");

    // Partition the SAME set the timeline actually models (vars with a
    // birth in the combined sequence). `sizes` also holds adjoint-mirror
    // entries for ghost adjoint VarIds that never birth an op, so counting
    // `sizes.len()` here would inflate the note's denominator.
    let all_vars: HashSet<VarId> = base.birth.keys().copied().collect();
    let sized_vars = all_vars.iter().filter(|v| sizes.contains_key(v)).count();
    let unsized_vars = all_vars.len() - sized_vars;

    let mut what_if = Vec::new();
    if !grad_vars.is_empty() && fase_peak < peak_bytes {
        what_if.push(WhatIfPeak {
            label: "With FASE".to_string(),
            peak_bytes: fase_peak,
            note: "parameter-gradient buffers freed per layer".to_string(),
        });
    }
    if !saved.is_empty() && ckpt_peak < peak_bytes {
        what_if.push(WhatIfPeak {
            label: "With full recompute (@checkpoint)".to_string(),
            peak_bytes: ckpt_peak,
            note: "memory bound only; recompute time cost not shown".to_string(),
        });
    }

    Some(RealTimeline {
        entries,
        peak_bytes,
        peak_pp,
        forward_ops: n_fwd,
        backward_ops: n_bwd,
        sized_vars,
        unsized_vars,
        what_if,
    })
}

fn mark(entries: &mut [MemoryTimelineEntry], pp: u32, label: &str) {
    if let Some(e) = entries.get_mut(pp as usize) {
        match &mut e.phase {
            Some(existing) => {
                existing.push_str("; ");
                existing.push_str(label);
            }
            None => e.phase = Some(label.to_string()),
        }
    }
}

/// Event sweep: +size at birth, -size at death; prefix-sum to a per-pp
/// live-bytes series. Returns (series, peak_bytes, peak_pp).
fn sweep(live: &Liveness, sizes: &HashMap<VarId, u64>) -> (Vec<u64>, u64, u32) {
    let n = live.total_pps as usize;
    let mut delta = vec![0i64; n + 1];
    for (v, &b) in &live.birth {
        let size = sizes.get(v).copied().unwrap_or(0) as i64;
        if size == 0 {
            continue;
        }
        let d = live.death.get(v).copied().unwrap_or(b + 1).min(live.total_pps);
        if d <= b {
            continue;
        }
        delta[b as usize] += size;
        delta[d as usize] -= size;
    }
    let mut series = Vec::with_capacity(n);
    let mut acc: i64 = 0;
    let (mut peak, mut peak_pp) = (0u64, 0u32);
    for (pp, d) in delta.iter().take(n).enumerate() {
        acc += d;
        let bytes = acc.max(0) as u64;
        if bytes > peak {
            peak = bytes;
            peak_pp = pp as u32;
        }
        series.push(bytes);
    }
    (series, peak, peak_pp)
}

/// Render the real timeline in the paper's section 3.2 shape: sampled bar
/// rows (phase-marker rows always kept), peak annotation, what-if lines,
/// and an honesty note when unsized vars exist.
pub fn render(rt: &RealTimeline, max_rows: usize) -> String {
    const BAR_WIDTH: usize = 20;
    let mut out = String::from("\n=== Memory Timeline (real liveness) ===\n\n");
    out.push_str(&format!(
        "Forward ops: {}   Backward ops: {}\n\nTime (pp)  HBM Usage\n",
        rt.forward_ops, rt.backward_ops
    ));

    let n = rt.entries.len();
    let stride = n.div_ceil(max_rows.max(1)).max(1);
    for (i, e) in rt.entries.iter().enumerate() {
        if e.phase.is_none() && i % stride != 0 && i != n - 1 {
            continue;
        }
        let filled = if rt.peak_bytes == 0 {
            0
        } else {
            (e.live_bytes * BAR_WIDTH as u64 / rt.peak_bytes) as usize
        };
        let bar = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(BAR_WIDTH - filled);
        let mb = e.live_bytes as f64 / (1024.0 * 1024.0);
        let phase = e.phase.as_deref().unwrap_or("");
        out.push_str(&format!(
            "{:>5}      {} {:>8.2} MB  {}\n",
            e.program_point, bar, mb, phase
        ));
    }

    out.push_str(&format!(
        "\nPeak: {:.2} MB at pp={}\n",
        rt.peak_bytes as f64 / (1024.0 * 1024.0),
        rt.peak_pp
    ));
    for w in &rt.what_if {
        out.push_str(&format!(
            "{}: peak drops to {:.2} MB ({})\n",
            w.label,
            w.peak_bytes as f64 / (1024.0 * 1024.0),
            w.note
        ));
    }
    if rt.unsized_vars > 0 {
        out.push_str(&format!(
            "\n  NOTE: {} of {} vars have no concrete compile-time shape; their\n  \
             bytes are NOT included — treat the MB figures as a lower bound.\n",
            rt.unsized_vars,
            rt.unsized_vars + rt.sized_vars
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{WengertOp, WengertType};
    use std::collections::HashMap as Map;

    fn op(id: u32, result: VarId, o: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    /// x -> h = matmul(x, w1) -> y = matmul(h, w2); loss = y.
    fn two_matmul_primal() -> WengertList {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("m.w1".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            op(3, 3, PrimalOp::Param("m.w2".into()), vec![]),
            op(4, 4, PrimalOp::Matmul, vec![2, 3]),
        ];
        let mut var_types = Map::new();
        for v in 0..5u32 {
            var_types.insert(v, WengertType::Tensor);
        }
        WengertList {
            ops,
            output: 4,
            var_names: Map::new(),
            var_types,
        }
    }

    fn hints() -> HashMap<VarId, u64> {
        // x: 1 MB, w1: 4 MB, h: 2 MB, w2: 4 MB, y: 1 MB
        let mut h = HashMap::new();
        h.insert(0u32, 1 << 20);
        h.insert(1u32, 4 << 20);
        h.insert(2u32, 2 << 20);
        h.insert(3u32, 4 << 20);
        h.insert(4u32, 1 << 20);
        h
    }

    #[test]
    fn timeline_covers_forward_and_backward() {
        let rt = build_training_timeline(&two_matmul_primal(), &hints()).unwrap();
        assert_eq!(rt.forward_ops, 5);
        assert!(rt.backward_ops > 0, "adjoint must be non-empty");
        assert_eq!(rt.entries.len(), rt.forward_ops + rt.backward_ops);
        let marked: Vec<&str> = rt
            .entries
            .iter()
            .filter_map(|e| e.phase.as_deref())
            .collect();
        assert!(marked.iter().any(|p| p.contains("backward begins")));
        assert!(marked.iter().any(|p| p.contains("forward begins")));
    }

    #[test]
    fn params_resident_throughout() {
        let rt = build_training_timeline(&two_matmul_primal(), &hints()).unwrap();
        // Weights = 8 MB total; every pp must carry at least the weights
        // that have been registered by that point, and the last pp still
        // holds both params + their grads.
        let last = rt.entries.last().unwrap();
        assert!(
            last.live_bytes >= 8 << 20,
            "params (8 MB) must be live at the end, got {}",
            last.live_bytes
        );
        assert!(rt.peak_bytes >= last.live_bytes);
    }

    #[test]
    fn fase_what_if_reduces_peak() {
        let rt = build_training_timeline(&two_matmul_primal(), &hints()).unwrap();
        let fase = rt.what_if.iter().find(|w| w.label.contains("FASE"));
        // Param grads (4+4 MB) live to the end in the base model, so FASE
        // must strictly reduce the peak here.
        let fase = fase.expect("FASE what-if expected for a model with param grads");
        assert!(fase.peak_bytes < rt.peak_bytes);
    }

    #[test]
    fn empty_primal_returns_none() {
        let empty = WengertList {
            ops: vec![],
            output: 0,
            var_names: Map::new(),
            var_types: Map::new(),
        };
        assert!(build_training_timeline(&empty, &HashMap::new()).is_none());
    }

    #[test]
    fn render_contains_peak_and_what_if_lines() {
        let rt = build_training_timeline(&two_matmul_primal(), &hints()).unwrap();
        let text = render(&rt, 48);
        assert!(text.contains("Peak:"));
        assert!(text.contains("backward begins"));
        if !rt.what_if.is_empty() {
            assert!(text.contains("peak drops to"));
        }
    }

    #[test]
    fn unsized_vars_produce_lower_bound_note() {
        // No hints at all: every var unsized, series all zeros, note shown.
        let rt = build_training_timeline(&two_matmul_primal(), &HashMap::new()).unwrap();
        assert_eq!(rt.peak_bytes, 0);
        assert!(rt.unsized_vars > 0);
        let text = render(&rt, 48);
        assert!(text.contains("lower bound"));
    }
}

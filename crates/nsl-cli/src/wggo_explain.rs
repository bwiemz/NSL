//! Renders WGGO decision explanations (PDF §3.3) from a [`WggoPlan`].
//!
//! Pure rendering — takes a plan and returns the human-readable
//! explanation text. The CLI wires this to `--explain-wggo` in Task 4.

use nsl_codegen::wggo::WggoPlan;
use nsl_codegen::wggo_conflicts::Resolution;
use nsl_codegen::wggo_ilp::{DecisionKind, DecisionTrace, LayerIlpSolution};

/// Render a plan's per-layer decisions and conflict resolutions as a
/// PDF §3.3-style explanation.
pub fn render_explain(plan: &WggoPlan) -> String {
    let mut out = String::from("=== WGGO Decision Explanation ===\n\n");
    if plan.per_layer.is_empty() {
        out.push_str("(no layers analyzed — was --wggo full active?)\n");
        return out;
    }
    for (idx, layer) in plan.per_layer.iter().enumerate() {
        render_layer(&mut out, idx, layer);
        for r in &plan.resolutions {
            if resolution_layer_idx(r) == Some(idx) {
                render_resolution(&mut out, r);
            }
        }
        out.push('\n');
    }
    out
}

fn render_layer(out: &mut String, layer_idx: usize, layer: &LayerIlpSolution) {
    out.push_str(&format!("Layer {} decisions:\n", layer_idx));
    if layer.decision_trace.is_empty() {
        out.push_str("  (no decisions traced — wggo mode not Full)\n");
        return;
    }
    for trace in &layer.decision_trace {
        render_trace(out, trace);
    }
}

fn render_trace(out: &mut String, t: &DecisionTrace) {
    let label = match t.kind {
        DecisionKind::CepHeadPrune => "CEP",
        DecisionKind::CshaLevel => "CSHA",
        DecisionKind::WrgaAdapter => "WRGA",
        DecisionKind::CpdtPrecision => "CPDT",
        DecisionKind::FaseStep => "FASE",
        DecisionKind::PcaPacking => "PCA",
    };
    out.push_str(&format!("  {}: {}\n", label, t.chosen));
    out.push_str(&format!("    Reason: {}\n", t.metric_summary));
    if let Some(c) = &t.binding_constraint {
        out.push_str(&format!("    Constraint: {}\n", c));
    }
    if let Some(note) = &t.cross_decision_note {
        out.push_str(&format!("    BUT: {}\n", note));
    }
    if let Some(ru) = &t.runner_up {
        out.push_str(&format!("    Runner-up: {}\n", ru));
    }
}

fn render_resolution(out: &mut String, r: &Resolution) {
    let (header, detail) = match r {
        Resolution::DowngradeCsha { layer: _, to_level } => {
            ("CSHA downgrade", format!("→ Level {}", to_level))
        }
        Resolution::RemoveWrgaAdapter { layer: _ } => ("WRGA removed", String::new()),
        Resolution::DeferFaseStep { layer: _ } => ("FASE deferred", String::new()),
        Resolution::AcceptNonUniformShard { layer: _ } => {
            ("Non-uniform shard accepted", String::new())
        }
        Resolution::NoChange => ("No conflict", String::new()),
    };
    if detail.is_empty() {
        out.push_str(&format!("  Conflict resolved: {}\n", header));
    } else {
        out.push_str(&format!("  Conflict resolved: {} {}\n", header, detail));
    }
}

fn resolution_layer_idx(r: &Resolution) -> Option<usize> {
    match r {
        Resolution::DowngradeCsha { layer, .. }
        | Resolution::RemoveWrgaAdapter { layer }
        | Resolution::DeferFaseStep { layer }
        | Resolution::AcceptNonUniformShard { layer } => Some(*layer as usize),
        Resolution::NoChange => None,
    }
}

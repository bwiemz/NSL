//! WGGO — user-forced layer prune: `--wggo-prune-layers` and
//! `--wggo-layer-prune-fraction`.
//!
//! The Level-1 DP only offers `CoarseDecision::Prune` for a layer whose
//! importance is below `prune_floor`, and production planning never fills
//! per-layer importance — so before these flags no real program could reach
//! the prune rewrite (`wggo_prune.rs`). This module turns the user's request
//! into a set of layer indices the DP must prune ([`DpConfig::forced_prune`]),
//! leaving `importance.per_layer` (and therefore every plan built without the
//! flags) untouched.
//!
//! Every way the request can fail to mean something is a hard error — an
//! unknown layer name, a layer the rewrite cannot prune, a fraction with no
//! weights to rank by, a fraction that rounds down to no layer at all. A
//! request that prunes nothing must never look like one that succeeded.
//!
//! [`DpConfig::forced_prune`]: crate::wggo_dp::DpConfig::forced_prune

use std::collections::{BTreeMap, BTreeSet};

use crate::wggo_graph::{layer_prefix, LayerRole, OptGraph};

/// One block layer's weight-magnitude importance, as ranked by
/// `--wggo-layer-prune-fraction`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerImportance {
    pub layer_index: u32,
    pub layer_name: String,
    /// RMS over every element of every weight mapped to the layer, divided
    /// by the largest such RMS across the ranked layers (so the most
    /// important layer reads 1.0).
    pub importance: f64,
}

/// The `--wggo-layer-prune-fraction` selection: the layer indices to
/// force-prune, and the ranking they were chosen from (for the `[wggo]`
/// report line).
#[derive(Debug, Clone, PartialEq)]
pub struct MagnitudeSelection {
    pub layers: BTreeSet<u32>,
    pub ranking: Vec<LayerImportance>,
}

/// Roles the prune rewrite executes: whole blocks (v2 chain-collapse) and
/// the v1 sub-block roles. Embeddings, the LM head and the synthetic
/// `other` bucket have no residual identity to collapse to.
fn prunable(role: LayerRole) -> bool {
    matches!(role, LayerRole::Block | LayerRole::Attention | LayerRole::Ffn)
}

fn known_layers(graph: &OptGraph) -> String {
    graph
        .layers
        .iter()
        .map(|l| format!("{} ({:?})", l.name, l.role))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolve `--wggo-prune-layers` names against the train block's layer graph.
///
/// `Err` carries the full user-facing message.
pub fn resolve_named(graph: &OptGraph, names: &[String]) -> Result<BTreeSet<u32>, String> {
    let mut out = BTreeSet::new();
    for raw in names {
        let name = raw.trim();
        let Some(layer) = graph.layers.iter().find(|l| l.name == name) else {
            return Err(format!(
                "--wggo-prune-layers: unknown layer `{name}`.\n  \
                 requested:  prune {name}\n  \
                 expected:   the name of a layer in this train block's WGGO layer graph\n  \
                 found:      layers {}",
                known_layers(graph)
            ));
        };
        if !prunable(layer.role) {
            return Err(format!(
                "--wggo-prune-layers: layer `{name}` cannot be pruned.\n  \
                 requested:  prune {name}  (role={:?})\n  \
                 expected:   a residual block (role Block: blocks.N / layers.N / h.N) \
                 or sub-block (role Attention / Ffn)\n  \
                 found:      role {:?}, which has no residual identity to collapse to; \
                 layers {}",
                layer.role,
                layer.role,
                known_layers(graph)
            ));
        }
        out.insert(layer.index);
    }
    Ok(out)
}

/// Rank the graph's whole-block layers by weight magnitude and select the
/// `floor(fraction * n)` least important for `--wggo-layer-prune-fraction`.
///
/// `weights` lists `(name, data)` for every tensor in the `.nslweights`
/// file; a tensor belongs to the layer [`layer_prefix`] maps its name to
/// (`blocks.2.wq` and `m.blocks.2.wq` both belong to `blocks.2`). `None`
/// means no `--wggo-weights` was given. `weights_label` names the file in
/// messages.
///
/// Importance is the RMS over all elements of the layer's weights, divided
/// by the max across block layers; sums accumulate in f64 over the tensors
/// in name order so the ranking is deterministic. Ties go to the lower layer
/// index. At most `n - 1` layers are selected.
pub fn select_by_magnitude(
    graph: &OptGraph,
    fraction: f64,
    weights: Option<&[(&str, &[f32])]>,
    weights_label: &str,
) -> Result<MagnitudeSelection, String> {
    if !(fraction > 0.0 && fraction < 1.0) {
        return Err(format!(
            "--wggo-layer-prune-fraction must be in (0, 1), got {fraction}"
        ));
    }
    let Some(weights) = weights else {
        return Err(format!(
            "--wggo-layer-prune-fraction requires --wggo-weights <file.nslweights>: the \
             layers to prune are the ones with the smallest weight magnitude, and \
             without a weights file there is nothing to rank them by (fraction \
             {fraction} would prune nothing). Pass --wggo-weights, or name the layers \
             with --wggo-prune-layers"
        ));
    };
    let blocks: Vec<(u32, &str)> = graph
        .layers
        .iter()
        .filter(|l| l.role == LayerRole::Block)
        .map(|l| (l.index, l.name.as_str()))
        .collect();
    if blocks.is_empty() {
        return Err(format!(
            "--wggo-layer-prune-fraction: this train block's WGGO layer graph has no \
             whole-block layers (blocks.N / layers.N / h.N) to rank; layers {}",
            known_layers(graph)
        ));
    }

    // Per-layer Σx² and element count, tensors in name order.
    let mut sorted: Vec<&(&str, &[f32])> = weights.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for (name, data) in sorted {
        let Some(layer) = layer_prefix(name) else { continue };
        if let Some((bad_i, bad)) = data.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(format!(
                "--wggo-layer-prune-fraction: tensor `{name}` in {weights_label} holds a \
                 non-finite value ({bad} at element {bad_i}); its layer's magnitude cannot \
                 be ranked"
            ));
        }
        let e = acc.entry(layer).or_insert((0.0, 0));
        e.0 += data.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
        e.1 += data.len();
    }

    let missing: Vec<&str> = blocks
        .iter()
        .filter(|(_, name)| acc.get(*name).is_none_or(|(_, n)| *n == 0))
        .map(|(_, name)| *name)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "--wggo-layer-prune-fraction: {weights_label} has no weights for {} of the \
             {} block layer(s): {}. Every block must be ranked, or the selection would \
             silently skip the unranked ones. Weight names are mapped to layers the way \
             WGGO names them (`blocks.2.wq` or `m.blocks.2.wq` belong to `blocks.2`)",
            missing.len(),
            blocks.len(),
            missing.join(", ")
        ));
    }

    let rms: Vec<(u32, &str, f64)> = blocks
        .iter()
        .map(|(idx, name)| {
            let (sumsq, n) = acc[*name];
            (*idx, *name, (sumsq / n as f64).sqrt())
        })
        .collect();
    let max = rms.iter().map(|r| r.2).fold(0.0f64, f64::max);
    if max <= 0.0 {
        return Err(format!(
            "--wggo-layer-prune-fraction: every block layer's weights in {weights_label} \
             are all zero, so weight magnitude cannot rank them"
        ));
    }
    let ranking: Vec<LayerImportance> = rms
        .iter()
        .map(|(idx, name, r)| LayerImportance {
            layer_index: *idx,
            layer_name: (*name).to_string(),
            importance: r / max,
        })
        .collect();

    let n = blocks.len();
    let k = ((fraction * n as f64).floor() as usize).min(n - 1);
    if k == 0 {
        return Err(format!(
            "--wggo-layer-prune-fraction {fraction}: floor({fraction} x {n} block \
             layers) = 0, so nothing would be pruned. Use a fraction of at least {:.4} \
             to prune one layer",
            1.0 / n as f64
        ));
    }
    let mut order: Vec<&LayerImportance> = ranking.iter().collect();
    order.sort_by(|a, b| {
        a.importance
            .total_cmp(&b.importance)
            .then(a.layer_index.cmp(&b.layer_index))
    });
    let layers: BTreeSet<u32> = order.iter().take(k).map(|l| l.layer_index).collect();
    Ok(MagnitudeSelection { layers, ranking })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_graph::Layer;

    fn layer(index: u32, name: &str, role: LayerRole) -> Layer {
        Layer {
            index,
            name: name.to_string(),
            role,
            op_indices: vec![index],
            param_names: Vec::new(),
            depends_on: Vec::new(),
        }
    }

    /// `other` first (the Input leaves land there in a real extraction),
    /// then four blocks — the shape `wggo_graph::build` gives the gate
    /// fixture, so block `blocks.N` has layer index N + 1.
    fn graph() -> OptGraph {
        let mut layers = vec![layer(0, "other", LayerRole::Other)];
        for b in 0..4u32 {
            layers.push(layer(b + 1, &format!("blocks.{b}"), LayerRole::Block));
        }
        OptGraph { layers, total_ops: 5 }
    }

    #[test]
    fn named_layers_resolve_to_their_graph_indices() {
        let got = resolve_named(&graph(), &["blocks.1".into(), " blocks.3".into()]).unwrap();
        assert_eq!(got, [2u32, 4].into_iter().collect());
    }

    #[test]
    fn unknown_name_lists_the_known_layers() {
        let err = resolve_named(&graph(), &["blocks.9".into()]).unwrap_err();
        assert!(err.contains("unknown layer `blocks.9`"), "{err}");
        assert!(err.contains("blocks.0 (Block)") && err.contains("other (Other)"), "{err}");
    }

    #[test]
    fn a_layer_without_residual_identity_is_refused() {
        let err = resolve_named(&graph(), &["other".into()]).unwrap_err();
        assert!(err.contains("`other` cannot be pruned"), "{err}");
    }

    fn weights(scale_of: impl Fn(u32) -> f32) -> Vec<(String, Vec<f32>)> {
        let mut w = Vec::new();
        for b in 0..4u32 {
            for p in ["wa", "wb"] {
                w.push((format!("blocks.{b}.{p}"), vec![scale_of(b); 16]));
            }
        }
        // A non-block tensor is ignored, not an error.
        w.push(("head".to_string(), vec![9.0; 4]));
        w
    }

    fn as_refs(w: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
        w.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect()
    }

    #[test]
    fn fraction_prunes_the_lowest_magnitude_blocks() {
        let w = weights(|b| if b == 2 { 0.001 } else { 0.05 });
        let got = select_by_magnitude(&graph(), 0.25, Some(&as_refs(&w)), "w.nslweights").unwrap();
        assert_eq!(got.layers, [3u32].into_iter().collect(), "blocks.2 is layer index 3");
        let b2 = got.ranking.iter().find(|r| r.layer_name == "blocks.2").unwrap();
        assert!((b2.importance - 0.02).abs() < 1e-6, "{b2:?}");
        assert!(got.ranking.iter().any(|r| r.importance == 1.0));

        // 0.5 x 4 = 2 layers; ties among the 0.05 blocks go to the lower index.
        let got = select_by_magnitude(&graph(), 0.5, Some(&as_refs(&w)), "w.nslweights").unwrap();
        assert_eq!(got.layers, [1u32, 3].into_iter().collect());
    }

    #[test]
    fn model_variable_prefixed_weight_names_map_to_layers() {
        let w: Vec<(String, Vec<f32>)> = weights(|b| if b == 0 { 0.001 } else { 0.05 })
            .into_iter()
            .map(|(n, d)| (format!("m.{n}"), d))
            .collect();
        let got = select_by_magnitude(&graph(), 0.25, Some(&as_refs(&w)), "w").unwrap();
        assert_eq!(got.layers, [1u32].into_iter().collect());
    }

    #[test]
    fn fraction_refusals_never_prune_nothing_silently() {
        let w = weights(|_| 0.05);
        let r = as_refs(&w);
        let err = select_by_magnitude(&graph(), 0.25, None, "-").unwrap_err();
        assert!(err.contains("requires --wggo-weights"), "{err}");
        let err = select_by_magnitude(&graph(), 0.1, Some(&r), "w").unwrap_err();
        assert!(err.contains("= 0, so nothing would be pruned"), "{err}");
        for bad in [0.0, 1.0, -0.5, f64::NAN] {
            let err = select_by_magnitude(&graph(), bad, Some(&r), "w").unwrap_err();
            assert!(err.contains("must be in (0, 1)"), "{bad}: {err}");
        }
        // A block with no weights in the file.
        let partial: Vec<(&str, &[f32])> =
            r.iter().copied().filter(|(n, _)| !n.starts_with("blocks.3.")).collect();
        let err = select_by_magnitude(&graph(), 0.25, Some(&partial), "w").unwrap_err();
        assert!(err.contains("no weights for 1 of the 4 block layer(s): blocks.3"), "{err}");
        // All-zero weights cannot be ranked.
        let zeros = weights(|_| 0.0);
        let err = select_by_magnitude(&graph(), 0.25, Some(&as_refs(&zeros)), "w").unwrap_err();
        assert!(err.contains("all zero"), "{err}");
        // A non-finite value.
        let mut nan = weights(|_| 0.05);
        nan[3].1[5] = f32::NAN;
        let err = select_by_magnitude(&graph(), 0.25, Some(&as_refs(&nan)), "w").unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
    }

    #[test]
    fn never_prunes_every_block() {
        let w = weights(|b| 0.01 * (b + 1) as f32);
        let got = select_by_magnitude(&graph(), 0.99, Some(&as_refs(&w)), "w").unwrap();
        assert_eq!(got.layers.len(), 3, "floor(0.99 x 4) = 3 = n - 1");
        assert!(!got.layers.contains(&4), "the largest block survives");
    }
}

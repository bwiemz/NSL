//! CEP — pruned architecture rewriting.
//!
//! Rather than mutating an AST in place, CEP works on a compact
//! [`ModelSpec`] (`cep_oracle`) and emits a [`ModelSpec`] with the
//! pruning decisions applied.  Rewriting at the spec level keeps the
//! search loop fast — we never touch the full AST until CEP hands its
//! final decision back to the caller.

use serde::Serialize;

use crate::cep_oracle::{Activation, ModelSpec, NormType};

/// Per-layer delta: which heads to drop and what new FFN width to use.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LayerDelta {
    pub layer: u32,
    /// Head indices to prune (0-based, relative to the current spec).
    pub pruned_heads: Vec<u32>,
    /// New FFN inner dimension, or `None` to leave unchanged.
    pub new_d_ff: Option<u32>,
    /// Whether to drop the whole layer.
    pub drop_layer: bool,
}

/// Aggregate pruning delta.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PruneDelta {
    pub per_layer: Vec<LayerDelta>,
}

impl PruneDelta {
    /// Build an empty delta matching the layer count of `spec`.
    pub fn for_spec(spec: &ModelSpec) -> Self {
        Self {
            per_layer: (0..spec.n_layers)
                .map(|i| LayerDelta {
                    layer: i,
                    ..Default::default()
                })
                .collect(),
        }
    }

    /// Set pruned heads for a specific layer, replacing any prior set.
    pub fn with_pruned_heads(mut self, layer: u32, heads: Vec<u32>) -> Self {
        if let Some(d) = self.per_layer.iter_mut().find(|d| d.layer == layer) {
            d.pruned_heads = heads;
        }
        self
    }

    /// Shrink the FFN width on a specific layer.
    pub fn with_ffn(mut self, layer: u32, new_d_ff: u32) -> Self {
        if let Some(d) = self.per_layer.iter_mut().find(|d| d.layer == layer) {
            d.new_d_ff = Some(new_d_ff);
        }
        self
    }

    /// Drop a whole layer.
    pub fn dropping(mut self, layer: u32) -> Self {
        if let Some(d) = self.per_layer.iter_mut().find(|d| d.layer == layer) {
            d.drop_layer = true;
        }
        self
    }

    /// Number of heads pruned across all layers.
    pub fn total_pruned_heads(&self) -> u32 {
        self.per_layer.iter().map(|d| d.pruned_heads.len() as u32).sum()
    }
}

/// Apply a [`PruneDelta`] to a [`ModelSpec`] and return a new spec.
///
/// Rules:
///   * Dropped heads reduce `n_heads[layer]` and `n_kv_heads[layer]`.
///     When heads are pruned inside a GQA group, the KV-head count is
///     reduced proportionally (at least 1 KV head is retained).
///   * Dropped layers are removed from all per-layer vectors and
///     `n_layers` is decremented.
///   * FFN width is clamped to at least `head_dim` (avoids zero-width
///     FFNs the backend can't lower).
pub fn apply_delta(spec: &ModelSpec, delta: &PruneDelta) -> ModelSpec {
    let mut out = spec.clone();
    let mut keep = vec![true; spec.n_layers as usize];

    for ld in &delta.per_layer {
        let l = ld.layer as usize;
        if l >= spec.n_layers as usize {
            continue;
        }
        if ld.drop_layer {
            keep[l] = false;
            continue;
        }
        // Prune heads.  We must preserve GQA group integrity: the kept
        // count is snapped to the nearest lower multiple of the group
        // size, and KV heads shrink in lockstep so `n_heads %
        // n_kv_heads == 0` remains invariant.
        if !ld.pruned_heads.is_empty() {
            let original = spec.n_heads[l];
            let original_kv = spec.n_kv_heads[l].max(1);
            let group = (original / original_kv).max(1);
            let requested_kept = original.saturating_sub(ld.pruned_heads.len() as u32);
            // Snap down to a multiple of `group`, clamped to at least
            // `group` (so at least one KV group survives).
            let snapped = ((requested_kept / group) * group).max(group);
            out.n_heads[l] = snapped;
            let new_kv = (snapped / group).max(1);
            out.n_kv_heads[l] = new_kv;
        }
        // Adjust FFN.
        if let Some(ff) = ld.new_d_ff {
            let floor = spec.head_dim.get(l).copied().unwrap_or(1);
            out.d_ff[l] = ff.max(floor).max(1);
        }
    }

    if keep.iter().any(|k| !*k) {
        let surviving: Vec<usize> = (0..spec.n_layers as usize).filter(|i| keep[*i]).collect();
        out.n_layers = surviving.len() as u32;
        out.n_heads = surviving.iter().map(|i| out.n_heads[*i]).collect();
        out.n_kv_heads = surviving.iter().map(|i| out.n_kv_heads[*i]).collect();
        out.head_dim = surviving.iter().map(|i| out.head_dim[*i]).collect();
        out.d_ff = surviving.iter().map(|i| out.d_ff[*i]).collect();
    }

    out
}

/// Candidate architecture spanned by a search-space description.
#[derive(Debug, Clone)]
pub struct SearchAxes {
    pub d_model: Vec<u32>,
    pub n_layers: Vec<u32>,
    pub n_heads: Vec<u32>,
    pub n_kv_heads: Vec<u32>,
    pub d_ff: Vec<u32>,
    pub activation: Vec<Activation>,
    pub norm: Vec<NormType>,
    /// Other fields fixed across the search.
    pub vocab: u32,
    pub head_dim: u32,
    pub max_seq: u32,
    pub batch: u32,
    pub dtype_bytes: u32,
}

impl SearchAxes {
    pub fn num_candidates(&self) -> usize {
        self.d_model.len()
            * self.n_layers.len()
            * self.n_heads.len()
            * self.n_kv_heads.len()
            * self.d_ff.len()
            * self.activation.len()
            * self.norm.len()
    }

    /// Enumerate all combinations as [`ModelSpec`]s.  Candidates whose
    /// `n_heads % n_kv_heads != 0` are filtered out (shape algebra
    /// rejection).
    pub fn enumerate(&self) -> Vec<ModelSpec> {
        let mut out = Vec::new();
        for &d in &self.d_model {
            for &nl in &self.n_layers {
                for &nh in &self.n_heads {
                    for &nkv in &self.n_kv_heads {
                        if nkv == 0 || nh % nkv != 0 {
                            continue;
                        }
                        if d % nh != 0 {
                            continue;
                        }
                        for &ff in &self.d_ff {
                            for &act in &self.activation {
                                for &norm in &self.norm {
                                    out.push(ModelSpec {
                                        d_model: d,
                                        n_layers: nl,
                                        n_heads: vec![nh; nl as usize],
                                        n_kv_heads: vec![nkv; nl as usize],
                                        head_dim: vec![self.head_dim; nl as usize],
                                        d_ff: vec![ff; nl as usize],
                                        vocab: self.vocab,
                                        max_seq: self.max_seq,
                                        batch: self.batch,
                                        activation: act,
                                        norm,
                                        dtype_bytes: self.dtype_bytes,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ModelSpec {
        ModelSpec::uniform(512, 4, 8, 4, 64, 1024, 32000)
    }

    #[test]
    fn empty_delta_is_identity() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec);
        let pruned = apply_delta(&spec, &delta);
        assert_eq!(pruned, spec);
    }

    #[test]
    fn pruned_heads_reduce_n_heads() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec).with_pruned_heads(1, vec![3, 5]);
        let pruned = apply_delta(&spec, &delta);
        assert_eq!(pruned.n_heads[1], 6);
        // GQA group is 8/4 = 2; 6/2 = 3 kv heads remain.
        assert_eq!(pruned.n_kv_heads[1], 3);
    }

    #[test]
    fn dropping_layer_removes_it_entirely() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec).dropping(2);
        let pruned = apply_delta(&spec, &delta);
        assert_eq!(pruned.n_layers, 3);
        assert_eq!(pruned.n_heads.len(), 3);
    }

    #[test]
    fn ffn_shrink_honours_head_dim_floor() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec).with_ffn(0, 32);
        // head_dim=64 → floor prevents FFN going below 64.
        let pruned = apply_delta(&spec, &delta);
        assert_eq!(pruned.d_ff[0], 64);
    }

    #[test]
    fn delta_total_pruned_heads() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec)
            .with_pruned_heads(0, vec![1, 2])
            .with_pruned_heads(3, vec![7]);
        assert_eq!(delta.total_pruned_heads(), 3);
    }

    #[test]
    fn pruned_spec_still_validates() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec).with_pruned_heads(0, vec![3, 4]);
        let pruned = apply_delta(&spec, &delta);
        pruned.validate().expect("pruned spec valid");
    }

    #[test]
    fn search_axes_enumerate_rejects_bad_ratios() {
        let axes = SearchAxes {
            d_model: vec![256, 512],
            n_layers: vec![4],
            n_heads: vec![4, 8],
            n_kv_heads: vec![2, 3], // 3 doesn't divide 4
            d_ff: vec![1024],
            activation: vec![Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 32000,
            head_dim: 64,
            max_seq: 1024,
            batch: 1,
            dtype_bytes: 2,
        };
        let candidates = axes.enumerate();
        // d=256 × {nh=4,nkv=2} ok, {nh=8,nkv=2} ok: 2
        // d=512 × {nh=4,nkv=2} ok, {nh=8,nkv=2} ok: 2 → 4 total
        assert_eq!(candidates.len(), 4);
        for c in &candidates {
            c.validate().expect("enumerated candidate valid");
        }
    }

    #[test]
    fn search_axes_candidate_count_matches_num_candidates_before_filter() {
        let axes = SearchAxes {
            d_model: vec![512],
            n_layers: vec![4],
            n_heads: vec![8],
            n_kv_heads: vec![4],
            d_ff: vec![1024, 2048],
            activation: vec![Activation::Gelu, Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 32000,
            head_dim: 64,
            max_seq: 1024,
            batch: 1,
            dtype_bytes: 2,
        };
        assert_eq!(axes.num_candidates(), 4); // 2 ffn × 2 activation
        assert_eq!(axes.enumerate().len(), 4); // nothing filtered
    }

    #[test]
    fn dropping_layer_zero_leaves_other_layers_intact() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec).dropping(0);
        let pruned = apply_delta(&spec, &delta);
        assert_eq!(pruned.n_layers, 3);
        assert_eq!(pruned.n_heads.len(), 3);
        // All surviving layers keep their original n_heads value.
        for nh in &pruned.n_heads {
            assert_eq!(*nh, spec.n_heads[0]);
        }
    }

    #[test]
    fn pruning_never_yields_zero_heads() {
        let spec = base();
        let delta = PruneDelta::for_spec(&spec)
            .with_pruned_heads(0, vec![0, 1, 2, 3, 4, 5, 6, 7]); // try to drop all
        let pruned = apply_delta(&spec, &delta);
        assert!(pruned.n_heads[0] >= 1);
        assert!(pruned.n_kv_heads[0] >= 1);
    }
}

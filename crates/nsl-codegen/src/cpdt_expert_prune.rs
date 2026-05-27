//! CPDT Part III — MoE dead-expert pruning (v1).
//!
//! Executes the dead-expert decision as a real weight-drop transform: slice the
//! router columns + drop the expert blocks for low-affinity experts, driven by
//! an `index_remap` single source of truth. A non-WGGO compile pass makes it
//! reachable with `--cpdt --weights` (no `--wggo`), sidestepping Part II's
//! WGGO/source-AD activation blocker. See
//! `docs/superpowers/specs/2026-05-27-cpdt-moe-dead-expert-pruning-v1-design.md`.

use crate::weight_aware::WeightDType;

/// Raw-byte MoE weight bundle for pruning. dtype-agnostic — slicing is by byte.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeWeightBundle {
    /// Router weights `[d_model, n_experts]` row-major (expert `e` = column `e`).
    pub router: Vec<u8>,
    /// Expert weights `[n_experts, expert_block_elems]` row-major (packed).
    pub experts: Vec<u8>,
    pub d_model: usize,
    pub n_experts: usize,
    /// Elements per expert block (e.g. hidden * intermediate).
    pub expert_block_elems: usize,
    pub dtype: WeightDType,
    pub top_k: usize,
}

/// Result of a successful prune. All three outputs derive from `index_remap`.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneResult {
    pub sliced_router: Vec<u8>,
    pub kept_experts: Vec<u8>,
    /// new contiguous slot `j` -> original expert id (ascending). Single source of truth.
    pub index_remap: Vec<u32>,
    pub n_live: usize,
    /// dead expert ids, sorted unique.
    pub dead_experts: Vec<u32>,
}

/// Precondition failures. On any refusal, nothing is produced (input untouched).
#[derive(Debug, Clone, PartialEq)]
pub enum ExpertPruneRefusal {
    RouterShapeMismatch { expected_elems: usize, actual_elems: usize },
    BundleInconsistent { reason: String },
    DeadIndexOutOfRange { index: u32, n_experts: usize },
    AllExpertsDead,
    InsufficientLiveExperts { n_live: usize, top_k: usize },
}

/// Prune dead experts from a MoE weight bundle: slice router columns + drop
/// expert blocks, both keyed by the `index_remap` single source of truth.
pub fn prune_dead_experts(
    bundle: &MoeWeightBundle,
    dead_experts: &[u32],
) -> Result<PruneResult, ExpertPruneRefusal> {
    let bw = bundle.dtype.byte_width();
    let n = bundle.n_experts;

    // Bundle consistency (byte-length checks).
    let expected_router = bundle.d_model * n * bw;
    if bundle.router.len() != expected_router {
        return Err(ExpertPruneRefusal::RouterShapeMismatch {
            expected_elems: bundle.d_model * n,
            actual_elems: bundle.router.len() / bw.max(1),
        });
    }
    let expected_experts = n * bundle.expert_block_elems * bw;
    if bundle.experts.len() != expected_experts {
        return Err(ExpertPruneRefusal::BundleInconsistent {
            reason: format!(
                "experts {} bytes != n_experts {} * block_elems {} * bw {}",
                bundle.experts.len(),
                n,
                bundle.expert_block_elems,
                bw
            ),
        });
    }

    // Dead-index range check.
    for &d in dead_experts {
        if d as usize >= n {
            return Err(ExpertPruneRefusal::DeadIndexOutOfRange { index: d, n_experts: n });
        }
    }

    // index_remap: survivors in ascending order (the single source of truth).
    let dead_set: std::collections::BTreeSet<u32> = dead_experts.iter().copied().collect();
    let index_remap: Vec<u32> = (0..n as u32).filter(|e| !dead_set.contains(e)).collect();
    let n_live = index_remap.len();
    if n_live == 0 {
        return Err(ExpertPruneRefusal::AllExpertsDead);
    }
    if n_live < bundle.top_k {
        return Err(ExpertPruneRefusal::InsufficientLiveExperts { n_live, top_k: bundle.top_k });
    }

    // Slice router: [d_model, n] -> [d_model, n_live], keep columns index_remap.
    let mut sliced_router = vec![0u8; bundle.d_model * n_live * bw];
    for r in 0..bundle.d_model {
        for (j, &orig) in index_remap.iter().enumerate() {
            let src = (r * n + orig as usize) * bw;
            let dst = (r * n_live + j) * bw;
            sliced_router[dst..dst + bw].copy_from_slice(&bundle.router[src..src + bw]);
        }
    }

    // Drop expert blocks: keep block index_remap[j].
    let blk = bundle.expert_block_elems * bw;
    let mut kept_experts = vec![0u8; n_live * blk];
    for (j, &orig) in index_remap.iter().enumerate() {
        let src = orig as usize * blk;
        let dst = j * blk;
        kept_experts[dst..dst + blk].copy_from_slice(&bundle.experts[src..src + blk]);
    }

    Ok(PruneResult {
        sliced_router,
        kept_experts,
        index_remap,
        n_live,
        dead_experts: dead_set.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::WeightDType;

    /// Build little-endian f32 bytes.
    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// d_model=2, n_experts=4, 1 elem/expert. Distinct router columns + expert blocks.
    fn toy_bundle() -> MoeWeightBundle {
        MoeWeightBundle {
            router: f32s(&[
                10.0, 20.0, 30.0, 40.0, // row 0: e0 e1 e2 e3
                11.0, 21.0, 31.0, 41.0, // row 1
            ]),
            experts: f32s(&[100.0, 200.0, 300.0, 400.0]), // 4 experts × 1 elem
            d_model: 2,
            n_experts: 4,
            expert_block_elems: 1,
            dtype: WeightDType::F32,
            top_k: 1,
        }
    }

    #[test]
    fn prune_by_identity() {
        let b = toy_bundle();
        let res = prune_dead_experts(&b, &[1]).unwrap();
        assert_eq!(res.index_remap, vec![0, 2, 3]);
        assert_eq!(res.n_live, 3);
        assert_eq!(res.dead_experts, vec![1]);
    }

    #[test]
    fn internal_consistency() {
        let b = toy_bundle();
        let res = prune_dead_experts(&b, &[1]).unwrap();
        let bw = 4usize;
        // Router: j-th live column bytes == original column index_remap[j] bytes.
        for r in 0..b.d_model {
            for (j, &orig) in res.index_remap.iter().enumerate() {
                let got = &res.sliced_router[(r * res.n_live + j) * bw..(r * res.n_live + j) * bw + bw];
                let want = &b.router[(r * b.n_experts + orig as usize) * bw..(r * b.n_experts + orig as usize) * bw + bw];
                assert_eq!(got, want, "router col j={j} (orig {orig}) row {r}");
            }
        }
        // Experts: block j bytes == original block index_remap[j] bytes.
        let blk = b.expert_block_elems * bw;
        for (j, &orig) in res.index_remap.iter().enumerate() {
            let got = &res.kept_experts[j * blk..j * blk + blk];
            let want = &b.experts[orig as usize * blk..orig as usize * blk + blk];
            assert_eq!(got, want, "expert block j={j} (orig {orig})");
        }
    }
}

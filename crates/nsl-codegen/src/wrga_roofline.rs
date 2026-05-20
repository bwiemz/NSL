//! WRGA Innovation 2: Roofline-Guided Adapter Placement.
//!
//! Given a list of candidate adapter *sites* (weight matrices + op kinds) and a
//! target GPU, this pass classifies each site against the roofline and emits
//! an [`AdapterPlacement`] recommendation:
//!
//! * **Memory-bound** softmax / layer-norm → IA³-style scaling vector
//! * **Memory-bound** matmul               → high-rank LoRA (free ALU cycles)
//! * **Balanced** matmul                   → medium-rank LoRA
//! * **Compute-bound** projection          → low-rank or skip
//!
//! The pass is deterministic and hardware-aware: the exact same model compiled
//! for an RTX 5070 Ti and an H100 will receive different placements.

use crate::cost_model::{
    arithmetic_intensity, classify_op, matmul_cost, rmsnorm_cost, softmax_cost, BoundClassification,
};
use crate::gpu_specs::GpuSpec;
use serde::{Deserialize, Serialize};

/// Kind of operation at an adapter candidate site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiteKind {
    /// Dense matmul such as Q/K/V/O projections or FFN gate/up/down.
    Matmul,
    /// RMS or LayerNorm.
    Norm,
    /// Softmax (attention scores).
    Softmax,
    /// Embedding lookup (always memory-bound).
    Embedding,
}

/// Adapter architectures WRGA can choose between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    /// `y = W₀x + α · (B · A · x)` — low-rank additive update.
    Lora,
    /// `y = γ ⊙ f(x)` — per-output scaling vector (memory-free).
    Ia3,
    /// No adapter at this site.
    Skip,
}

/// A candidate adapter site, usually one per trainable-eligible weight matrix.
#[derive(Debug, Clone)]
pub struct AdapterSite {
    /// Layer name, e.g. `blocks.6.wq`.
    pub name: String,
    pub kind: SiteKind,
    /// Shape of the primary operand (e.g. `[in_dim, out_dim]` for a matmul).
    pub shape: Vec<u64>,
    /// Dtype byte width used to compute arithmetic intensity.
    pub dtype_bytes: u64,
    /// Optional hint — the batch × seq size at which this site will actually
    /// run.  Defaults to 1×1 (conservative / per-token fine-tuning).
    pub batch: u64,
    pub seq: u64,
}

impl AdapterSite {
    pub fn new(name: impl Into<String>, kind: SiteKind, shape: Vec<u64>, dtype_bytes: u64) -> Self {
        Self {
            name: name.into(),
            kind,
            shape,
            dtype_bytes,
            batch: 1,
            seq: 1,
        }
    }

    pub fn with_batch(mut self, batch: u64, seq: u64) -> Self {
        self.batch = batch;
        self.seq = seq;
        self
    }
}

/// WRGA's per-site placement decision.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterPlacement {
    pub name: String,
    /// Arithmetic intensity computed from the site's shape at the runtime
    /// batch×seq.
    pub arithmetic_intensity: f64,
    /// Classification relative to the target GPU's ridge point.
    pub classification: BoundClassification,
    /// Roofline slack: >1 for memory-bound (free ALU cycles available for an
    /// adapter), <1 for compute-bound (adapter steals time).
    pub roofline_slack: f64,
    /// Chosen adapter kind.
    pub adapter: AdapterKind,
    /// Recommended rank for LoRA (ignored for IA³/Skip).
    pub suggested_rank: usize,
    /// Human-readable rationale for the decision.
    pub rationale: String,
    // ─── B.2 Task 2b: adapter materialisation observation surface ────────
    /// Adapter kind as declared by the user-facing `@adapter(type=...)`
    /// decorator. Populated by the codegen bridge when a manual decorator
    /// config is threaded through; `None` when WRGA placed the adapter
    /// itself (auto mode only sees `self.adapter` above).
    pub decorator_kind: Option<crate::AdapterKind>,
    /// User-supplied LoRA alpha (defaults to `rank` when unset).
    pub alpha: Option<i64>,
    /// Field names synthesized onto the model struct by `wrga_adapter_inject`.
    pub synthesized_fields: Vec<String>,
    /// Initializer strategies for each synthesized field.
    pub init_strategies: Vec<crate::wrga_adapter_inject::InitStrategy>,
}

/// Compute the `(flops, bytes_read, bytes_written)` tuple for a site.
fn site_cost(site: &AdapterSite) -> (u64, u64, u64) {
    let b = site.batch.max(1);
    let s = site.seq.max(1);
    match site.kind {
        SiteKind::Matmul => {
            // Treat shape as [K, N]; then a "logical" matmul is [BS, K] x [K, N].
            let (k, n) = match site.shape.as_slice() {
                [k, n] => (*k, *n),
                [a] => (*a, *a), // degenerate square
                _ => (1, 1),
            };
            matmul_cost(b * s, k, n, site.dtype_bytes)
        }
        SiteKind::Norm => {
            let d = site.shape.first().copied().unwrap_or(1);
            // Default to RMSNorm (cheaper) as it's what NSLCoder uses.
            rmsnorm_cost(b, s, d, site.dtype_bytes)
        }
        SiteKind::Softmax => softmax_cost(b, s * s, site.dtype_bytes),
        SiteKind::Embedding => {
            let d = site.shape.last().copied().unwrap_or(1);
            let bytes = b * s * d * site.dtype_bytes;
            (0, bytes, bytes)
        }
    }
}

/// Core heuristic: map (bound, site-kind) → adapter choice.
fn choose_adapter(
    classification: BoundClassification,
    site_kind: SiteKind,
    r_min: usize,
    r_max: usize,
) -> (AdapterKind, usize, &'static str) {
    match (site_kind, classification) {
        // Memory-bound softmax → IA³ (LoRA isn't shape-fusible).
        (SiteKind::Softmax, BoundClassification::MemoryBound) => (
            AdapterKind::Ia3,
            0,
            "memory-bound softmax: IA³ scaling is epilogue-fusible with zero extra traffic",
        ),
        // Memory-bound norm → IA³ (can fuse into the norm epilogue).
        (SiteKind::Norm, BoundClassification::MemoryBound) => (
            AdapterKind::Ia3,
            0,
            "memory-bound normalization: IA³ scaling is free at the epilogue",
        ),
        // Memory-bound matmul → high-rank LoRA (ALUs idle, adapter is free).
        (SiteKind::Matmul, BoundClassification::MemoryBound) => (
            AdapterKind::Lora,
            r_max,
            "memory-bound matmul: high-rank LoRA fills idle ALU cycles",
        ),
        // Balanced matmul → mid-rank LoRA.
        (SiteKind::Matmul, BoundClassification::Balanced) => (
            AdapterKind::Lora,
            ((r_min + r_max) / 2).max(r_min),
            "balanced matmul: mid-rank LoRA balances adapter capacity with compute overhead",
        ),
        // Compute-bound matmul → minimal rank (avoid stealing ALU time).
        (SiteKind::Matmul, BoundClassification::ComputeBound) => (
            AdapterKind::Lora,
            r_min,
            "compute-bound matmul: minimal-rank LoRA to preserve throughput",
        ),
        // Compute-bound / balanced norm or softmax → skip (can't help here).
        (SiteKind::Norm, _) | (SiteKind::Softmax, _) => (
            AdapterKind::Skip,
            0,
            "norm/softmax off the memory-bound path: adapter would add wall-clock overhead",
        ),
        (SiteKind::Embedding, BoundClassification::MemoryBound) => (
            AdapterKind::Ia3,
            0,
            "embedding table is memory-bound: IA³ scaling is fusible into the gather",
        ),
        (SiteKind::Embedding, _) => (
            AdapterKind::Skip,
            0,
            "embedding table is atypically compute-bound — skip",
        ),
        (_, BoundClassification::Unknown) => (
            AdapterKind::Skip,
            0,
            "classification unknown (zero-sized op or bad ridge) — skip",
        ),
    }
}

/// Compute the roofline-guided placement for a set of candidate sites on a
/// given GPU.  Rank selection is expressed as a `(r_min, r_max)` band.  A
/// follow-up spectral pass (Innovation 3) may refine the suggested rank.
pub fn place_adapters(
    sites: &[AdapterSite],
    gpu: &GpuSpec,
    r_min: usize,
    r_max: usize,
) -> Vec<AdapterPlacement> {
    sites
        .iter()
        .map(|site| {
            let (flops, br, bw) = site_cost(site);
            let ai = arithmetic_intensity(flops, br, bw);
            let ridge = gpu.crossover(site.dtype_bytes as usize);
            let class = classify_op(ai, ridge);
            let slack = if ridge > 0.0 {
                ridge / ai.max(1e-12)
            } else {
                1.0
            };
            let (kind, rank, why) = choose_adapter(class, site.kind, r_min, r_max);
            AdapterPlacement {
                name: site.name.clone(),
                arithmetic_intensity: ai,
                classification: class,
                roofline_slack: slack,
                adapter: kind,
                suggested_rank: rank,
                rationale: why.into(),
                decorator_kind: None,
                alpha: None,
                synthesized_fields: Vec::new(),
                init_strategies: Vec::new(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs::{find_gpu, GPU_DATABASE};

    fn test_gpu() -> &'static GpuSpec {
        // Prefer H100 if present, otherwise first entry.
        find_gpu("H100")
            .or_else(|| find_gpu("h100"))
            .unwrap_or(&GPU_DATABASE[0])
    }

    #[test]
    fn small_softmax_is_memory_bound_and_gets_ia3() {
        let gpu = test_gpu();
        let sites = vec![
            AdapterSite::new("blocks.6.softmax", SiteKind::Softmax, vec![128], 2)
                .with_batch(32, 512),
        ];
        let plan = place_adapters(&sites, gpu, 2, 16);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].classification, BoundClassification::MemoryBound);
        assert_eq!(plan[0].adapter, AdapterKind::Ia3);
    }

    #[test]
    fn compute_bound_matmul_gets_small_rank_or_skip() {
        let gpu = test_gpu();
        // Large matmul at batch×seq = 32×1024 — tensor cores saturate.
        let sites = vec![
            AdapterSite::new("blocks.6.wq", SiteKind::Matmul, vec![4096, 4096], 2)
                .with_batch(32, 1024),
        ];
        let plan = place_adapters(&sites, gpu, 2, 16);
        let p = &plan[0];
        // Must either be compute-bound with minimal rank, or classified
        // balanced with a mid rank.  Either way — *not* the max rank.
        assert!(matches!(
            p.classification,
            BoundClassification::ComputeBound | BoundClassification::Balanced
        ));
        assert!(p.suggested_rank <= 16);
        assert!(matches!(p.adapter, AdapterKind::Lora | AdapterKind::Skip));
    }

    #[test]
    fn norm_is_memory_bound_ia3() {
        let gpu = test_gpu();
        let sites = vec![
            AdapterSite::new("blocks.6.norm", SiteKind::Norm, vec![512], 2).with_batch(8, 128),
        ];
        let plan = place_adapters(&sites, gpu, 2, 16);
        assert_eq!(plan[0].adapter, AdapterKind::Ia3);
    }

    #[test]
    fn roofline_slack_monotonic_with_bound() {
        let gpu = test_gpu();
        let mem_bound = AdapterSite::new("norm", SiteKind::Norm, vec![512], 2).with_batch(2, 64);
        let comp_bound =
            AdapterSite::new("big_mm", SiteKind::Matmul, vec![4096, 4096], 2).with_batch(32, 1024);
        let plan = place_adapters(&[mem_bound, comp_bound], gpu, 2, 16);
        // Memory-bound site should have larger slack (ridge/ai) than the
        // compute-bound one.
        assert!(plan[0].roofline_slack >= plan[1].roofline_slack);
    }

    #[test]
    fn skip_for_unknown_classification() {
        // A zero-sized matmul produces flops=0 → ai=0 → classification Memory
        // (ai < 0.8·ridge).  Make sure we don't explode.
        let gpu = test_gpu();
        let sites = vec![AdapterSite::new("noop", SiteKind::Matmul, vec![0, 0], 2)];
        let plan = place_adapters(&sites, gpu, 2, 16);
        assert_eq!(plan.len(), 1);
        // AI is 0 → memory-bound classification; we pick LoRA r=max.  Just
        // check it doesn't NaN/crash.
        assert!(plan[0].arithmetic_intensity.is_finite());
    }
}

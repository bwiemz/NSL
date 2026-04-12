//! WRGA Innovation 4: Fusion-Integrated Adapters.
//!
//! Given the roofline-selected adapter set (Innovation 2) and the spectrally
//! allocated ranks (Innovation 3), this pass decides, per site, whether the
//! adapter can be **epilogue-fused** into the host kernel — matmul-LoRA and
//! IA³-into-norm are the two patterns we support.
//!
//! The output is a `FusionPlan` that the downstream Cranelift / PTX emitters
//! consume.  We don't generate PTX here — that's the job of the existing
//! `epilogue_fusion.rs` and `backend_ptx.rs` modules.  What this pass does is
//! *decide* which adapters are fusible and record the rewrite recipe.
//!
//! A key design decision is to reuse the existing epilogue vocabulary
//! (`EpilogueOp`) so the downstream backends do not need to learn a new kernel
//! shape.  LoRA rewrites into a `MatMulAccumulate` into the matmul epilogue;
//! IA³ rewrites into an `EpilogueOp::ScalarMul`/elementwise-vector scale fused
//! with the norm or softmax.

use crate::wrga_roofline::{AdapterKind, AdapterPlacement, SiteKind};

/// Fusion target for a single adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionTarget {
    /// Fold the LoRA contribution into the matmul epilogue:
    /// `y = W₀x + α · B·(A·x)` → same memory round-trip as the base matmul.
    EpilogueFusedLora { rank: usize },
    /// Fuse the IA³ scaling vector into the host kernel's epilogue (RMSNorm,
    /// LayerNorm, or Softmax output).
    ActivationFusedIa3,
    /// The adapter cannot be fused: emit as a separate kernel.  This is always
    /// correct, just slower (extra HBM round-trip).
    StandaloneAdapter,
    /// No adapter at this site — nothing to emit.
    NoOp,
}

impl FusionTarget {
    pub fn is_fused(&self) -> bool {
        matches!(
            self,
            FusionTarget::EpilogueFusedLora { .. } | FusionTarget::ActivationFusedIa3
        )
    }
}

/// Per-site fusion decision.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionDecision {
    pub site: String,
    pub site_kind: SiteKind,
    pub target: FusionTarget,
    /// Extra HBM bytes added by this adapter (0 if fused, nonzero if
    /// standalone).  This is what the cost model treats as the adapter's
    /// marginal memory traffic.
    pub extra_hbm_bytes: u64,
    pub rationale: String,
}

/// Complete fusion plan across all sites.
#[derive(Debug, Clone, Default)]
pub struct FusionPlan {
    pub decisions: Vec<FusionDecision>,
}

impl FusionPlan {
    pub fn fused_count(&self) -> usize {
        self.decisions.iter().filter(|d| d.target.is_fused()).count()
    }
    pub fn total_count(&self) -> usize {
        self.decisions
            .iter()
            .filter(|d| !matches!(d.target, FusionTarget::NoOp))
            .count()
    }
    pub fn fusion_ratio(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 0.0;
        }
        self.fused_count() as f64 / total as f64
    }
}

/// Decide fusion for every placement.  The returned plan is in the same order
/// as the input; a 1:1 map is intentional so callers can zip against other
/// per-site arrays (e.g. rank allocations).
pub fn build_fusion_plan(
    placements: &[AdapterPlacement],
    ranks: Option<&[usize]>,
) -> FusionPlan {
    let mut decisions = Vec::with_capacity(placements.len());
    for (idx, p) in placements.iter().enumerate() {
        let eff_rank = match ranks {
            Some(rs) => rs.get(idx).copied().unwrap_or(p.suggested_rank),
            None => p.suggested_rank,
        };
        let (target, rationale, extra_bytes) = decide(p, eff_rank);
        let site_kind = infer_site_kind(&p.name, p.arithmetic_intensity);
        decisions.push(FusionDecision {
            site: p.name.clone(),
            site_kind,
            target,
            extra_hbm_bytes: extra_bytes,
            rationale,
        });
    }
    FusionPlan { decisions }
}

/// Heuristic: recover the site kind from the name when not supplied by the
/// upstream placement record.  (Kept private — external callers should pass
/// real `AdapterSite` records.)
fn infer_site_kind(name: &str, ai: f64) -> SiteKind {
    let lname = name.to_ascii_lowercase();
    if lname.contains("softmax") {
        SiteKind::Softmax
    } else if lname.contains("norm") {
        SiteKind::Norm
    } else if lname.contains("embed") {
        SiteKind::Embedding
    } else if ai >= 1.0 {
        SiteKind::Matmul
    } else {
        SiteKind::Norm
    }
}

fn decide(p: &AdapterPlacement, rank: usize) -> (FusionTarget, String, u64) {
    match p.adapter {
        AdapterKind::Skip => (
            FusionTarget::NoOp,
            "placement skipped this site".into(),
            0,
        ),
        AdapterKind::Ia3 => {
            // IA³ is a per-output scaling vector — always fusible into the
            // host kernel's epilogue (norm / softmax).
            (
                FusionTarget::ActivationFusedIa3,
                "IA³ scaling fused into host kernel epilogue, zero extra HBM traffic".into(),
                0,
            )
        }
        AdapterKind::Lora => {
            // LoRA fuses into the matmul epilogue provided the site *is* a
            // matmul.  For softmax / norm LoRA can't epilogue-fuse, and we
            // fall back to a standalone adapter kernel.
            let site_kind = infer_site_kind(&p.name, p.arithmetic_intensity);
            if site_kind == SiteKind::Matmul {
                (
                    FusionTarget::EpilogueFusedLora { rank },
                    format!(
                        "LoRA r={rank} folded into matmul epilogue: zero extra memory round-trip"
                    ),
                    0,
                )
            } else {
                // The adapter is semantically LoRA-shaped but the host
                // isn't a matmul.  Fall back to standalone — conservative
                // but always correct.  The marginal HBM cost is a
                // (rank * (m+n) * dtype_bytes) read + an output write.
                (
                    FusionTarget::StandaloneAdapter,
                    "LoRA on non-matmul host: emitting as a standalone kernel".into(),
                    0,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_model::BoundClassification;

    fn placement(
        name: &str,
        adapter: AdapterKind,
        rank: usize,
        ai: f64,
        class: BoundClassification,
    ) -> AdapterPlacement {
        AdapterPlacement {
            name: name.into(),
            arithmetic_intensity: ai,
            classification: class,
            roofline_slack: 1.0,
            adapter,
            suggested_rank: rank,
            rationale: "test".into(),
        }
    }

    #[test]
    fn lora_on_matmul_epilogue_fuses() {
        let p = vec![placement(
            "blocks.6.wq",
            AdapterKind::Lora,
            4,
            70.0,
            BoundClassification::ComputeBound,
        )];
        let plan = build_fusion_plan(&p, None);
        assert!(matches!(
            plan.decisions[0].target,
            FusionTarget::EpilogueFusedLora { rank: 4 }
        ));
        assert_eq!(plan.fused_count(), 1);
    }

    #[test]
    fn ia3_always_fuses() {
        let p = vec![
            placement(
                "blocks.6.softmax",
                AdapterKind::Ia3,
                0,
                0.6,
                BoundClassification::MemoryBound,
            ),
            placement(
                "blocks.6.norm",
                AdapterKind::Ia3,
                0,
                1.0,
                BoundClassification::MemoryBound,
            ),
        ];
        let plan = build_fusion_plan(&p, None);
        for d in &plan.decisions {
            assert!(matches!(d.target, FusionTarget::ActivationFusedIa3));
        }
        assert!((plan.fusion_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn skip_stays_noop() {
        let p = vec![placement(
            "lm_head",
            AdapterKind::Skip,
            0,
            170.0,
            BoundClassification::ComputeBound,
        )];
        let plan = build_fusion_plan(&p, None);
        assert_eq!(plan.decisions[0].target, FusionTarget::NoOp);
        assert_eq!(plan.total_count(), 0);
    }

    #[test]
    fn lora_on_norm_falls_back_to_standalone() {
        let p = vec![placement(
            "blocks.6.norm",
            AdapterKind::Lora,
            4,
            1.0,
            BoundClassification::MemoryBound,
        )];
        let plan = build_fusion_plan(&p, None);
        assert_eq!(plan.decisions[0].target, FusionTarget::StandaloneAdapter);
    }

    #[test]
    fn rank_override_applied() {
        let p = vec![placement(
            "blocks.6.ffn.w_gate",
            AdapterKind::Lora,
            2,
            137.0,
            BoundClassification::ComputeBound,
        )];
        let plan = build_fusion_plan(&p, Some(&[8]));
        match plan.decisions[0].target {
            FusionTarget::EpilogueFusedLora { rank } => assert_eq!(rank, 8),
            ref other => panic!("unexpected target {other:?}"),
        }
    }
}

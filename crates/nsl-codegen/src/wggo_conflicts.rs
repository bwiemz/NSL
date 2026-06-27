//! WGGO — inter-technique conflict detection & priority-ordered resolution.
//!
//! Independent WRGA/CSHA/FASE/PCA/CEP/CPDT passes can produce
//! locally-optimal but mutually-incompatible per-layer decisions.  This
//! module detects those conflicts and resolves them using the priority
//! order the CFTP/WGGO paper prescribes:
//!
//! ```text
//! CEP > CPDT > CSHA > WRGA > FASE > PCA
//! ```
//!
//! Rationale (paper §3): structural decisions (which heads exist, how
//! layers are sharded) must override optimisation decisions (which fused
//! kernel to emit, what rank the adapter has).

use serde::Serialize;

/// Conflict category flagged by the detector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ConflictKind {
    /// CSHA fusion level was chosen before head pruning decided how many
    /// heads exist; the fused kernel's SMEM/shape must be revalidated.
    CshaVsCep {
        layer: u32,
        csha_level: u8,
        pruned_heads: u32,
    },
    /// WRGA placed an adapter on a layer that CPDT shards across devices;
    /// the adapter's gradient-allreduce cost may exceed the allowed
    /// budget.
    WrgaVsCpdt {
        layer: u32,
        adapter_rank: u64,
        shard_factor: u32,
    },
    /// FASE wants to fuse the optimizer step during backward, but CPDT
    /// needs reduce-scatter before the step can run.  The two must be
    /// co-scheduled (fused step → after reduce-scatter).
    FaseVsCpdt {
        layer: u32,
        shard_factor: u32,
    },
    /// CEP produced heterogeneous layer dimensions, which defeats CPDT's
    /// bucketed allreduce.  Either accept a bigger allreduce or
    /// re-prune to uniform shapes.
    CepVsCpdt {
        layer: u32,
        head_count: u32,
    },
}

/// Resolution verdict emitted by the resolver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Resolution {
    /// Downgrade CSHA to a feasible lower level.
    DowngradeCsha { layer: u32, to_level: u8 },
    /// Drop the WRGA adapter at this layer (re-allocate budget to other
    /// layers).
    RemoveWrgaAdapter { layer: u32 },
    /// FASE step runs *after* reduce-scatter instead of fused into the
    /// backward.
    DeferFaseStep { layer: u32 },
    /// Accept a non-uniform allreduce — no change to CEP, but CPDT's
    /// communication cost increases.
    AcceptNonUniformShard { layer: u32 },
    /// No change required (conflict was a false positive).
    NoChange,
}

impl Resolution {
    /// The layer this resolution applies to, if any (`NoChange` has none).
    pub fn layer(&self) -> Option<u32> {
        match *self {
            Resolution::DowngradeCsha { layer, .. }
            | Resolution::RemoveWrgaAdapter { layer }
            | Resolution::DeferFaseStep { layer }
            | Resolution::AcceptNonUniformShard { layer } => Some(layer),
            Resolution::NoChange => None,
        }
    }
}

/// The optimization techniques WGGO coordinates, ordered by resolution
/// priority.  `priority()` returns a rank where **lower wins** — so when two
/// techniques conflict, the one with the larger rank yields.  This encodes the
/// paper's order `CEP > CPDT > CSHA > WRGA > FASE > PCA` as an explicit
/// comparator instead of per-conflict hardcoded winners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Technique {
    Cep,
    Cpdt,
    Csha,
    Wrga,
    Fase,
    Pca,
}

impl Technique {
    /// Resolution rank — lower wins.  `CEP` is highest priority (0), `PCA`
    /// lowest (5).
    pub fn priority(self) -> u8 {
        match self {
            Technique::Cep => 0,
            Technique::Cpdt => 1,
            Technique::Csha => 2,
            Technique::Wrga => 3,
            Technique::Fase => 4,
            Technique::Pca => 5,
        }
    }
}

impl ConflictKind {
    /// The two techniques in tension for this conflict.
    fn techniques(&self) -> (Technique, Technique) {
        match self {
            ConflictKind::CshaVsCep { .. } => (Technique::Csha, Technique::Cep),
            ConflictKind::WrgaVsCpdt { .. } => (Technique::Wrga, Technique::Cpdt),
            ConflictKind::FaseVsCpdt { .. } => (Technique::Fase, Technique::Cpdt),
            ConflictKind::CepVsCpdt { .. } => (Technique::Cep, Technique::Cpdt),
        }
    }

    /// The technique that must yield, decided purely by priority comparison
    /// (the higher-rank technique loses).
    pub fn loser(&self) -> Technique {
        let (a, b) = self.techniques();
        if a.priority() < b.priority() {
            b
        } else {
            a
        }
    }
}

/// Compact per-layer decisions used by the detector.  Callers populate
/// this from the individual WRGA/CSHA/FASE/PCA plans.
#[derive(Debug, Clone, Default)]
pub struct LayerDecisions {
    pub layer: u32,
    pub csha_level: u8,
    pub head_count: u32,
    pub pruned_heads: u32,
    pub adapter_rank: u64,
    pub shard_factor: u32,
    pub fase_fused: bool,
    pub adapter_comm_cost: f64,
    pub adapter_comm_budget: f64,
}

/// Detect conflicts across the full per-layer decision table.
pub fn detect(decisions: &[LayerDecisions]) -> Vec<ConflictKind> {
    let mut out = Vec::new();
    for d in decisions {
        if d.csha_level >= 2 && d.pruned_heads > 0 {
            out.push(ConflictKind::CshaVsCep {
                layer: d.layer,
                csha_level: d.csha_level,
                pruned_heads: d.pruned_heads,
            });
        }
        if d.adapter_rank > 0
            && d.shard_factor > 1
            && d.adapter_comm_cost > d.adapter_comm_budget
        {
            out.push(ConflictKind::WrgaVsCpdt {
                layer: d.layer,
                adapter_rank: d.adapter_rank,
                shard_factor: d.shard_factor,
            });
        }
        if d.fase_fused && d.shard_factor > 1 {
            out.push(ConflictKind::FaseVsCpdt {
                layer: d.layer,
                shard_factor: d.shard_factor,
            });
        }
    }
    // Global CEP-vs-CPDT: any layer whose head count differs from the
    // others' on a multi-GPU run triggers a non-uniform-shard conflict.
    if decisions.len() > 1 {
        let shards_used = decisions.iter().any(|d| d.shard_factor > 1);
        if shards_used {
            let first = decisions[0].head_count;
            for d in decisions.iter().skip(1) {
                if d.head_count != first {
                    out.push(ConflictKind::CepVsCpdt {
                        layer: d.layer,
                        head_count: d.head_count,
                    });
                }
            }
        }
    }
    out
}

/// Apply priority-ordered resolution (CEP > CPDT > CSHA > WRGA > FASE > PCA).
///
/// Returns one [`Resolution`] per input conflict (same order).
pub fn resolve(conflicts: &[ConflictKind]) -> Vec<Resolution> {
    conflicts
        .iter()
        .map(|c| {
            // The loser is decided by the explicit priority comparator; the
            // resolution then drops/downgrades that technique.
            let loser = c.loser();
            match (c, loser) {
                (ConflictKind::CshaVsCep { layer, csha_level, .. }, Technique::Csha) => {
                    // CEP wins over CSHA → downgrade.  Level 2/3 → Level 1.
                    let to = csha_level.saturating_sub(1).min(1);
                    Resolution::DowngradeCsha {
                        layer: *layer,
                        to_level: to,
                    }
                }
                (ConflictKind::WrgaVsCpdt { layer, .. }, Technique::Wrga) => {
                    // CPDT wins over WRGA → drop the adapter.
                    Resolution::RemoveWrgaAdapter { layer: *layer }
                }
                (ConflictKind::FaseVsCpdt { layer, .. }, Technique::Fase) => {
                    // CPDT wins over FASE → defer the step.
                    Resolution::DeferFaseStep { layer: *layer }
                }
                (ConflictKind::CepVsCpdt { layer, .. }, Technique::Cpdt) => {
                    // CEP wins over CPDT → accept a heterogeneous shard.
                    Resolution::AcceptNonUniformShard { layer: *layer }
                }
                // Defensive: any (conflict, loser) pairing the priority order
                // doesn't recognise is a no-op rather than a silent mis-edit.
                _ => Resolution::NoChange,
            }
        })
        .collect()
}

/// Greedy mode (paper §5.3): run each technique independently, detect
/// conflicts, and resolve with priority ordering.  Returns the resolved
/// decisions and how many changes were applied.
pub fn greedy_resolve(mut decisions: Vec<LayerDecisions>) -> (Vec<LayerDecisions>, Vec<Resolution>) {
    let conflicts = detect(&decisions);
    let resolutions = resolve(&conflicts);
    for res in &resolutions {
        apply_resolution(&mut decisions, res);
    }
    (decisions, resolutions)
}

fn apply_resolution(decisions: &mut [LayerDecisions], res: &Resolution) {
    match *res {
        Resolution::DowngradeCsha { layer, to_level } => {
            if let Some(d) = decisions.iter_mut().find(|d| d.layer == layer) {
                d.csha_level = to_level;
            }
        }
        Resolution::RemoveWrgaAdapter { layer } => {
            if let Some(d) = decisions.iter_mut().find(|d| d.layer == layer) {
                d.adapter_rank = 0;
            }
        }
        Resolution::DeferFaseStep { layer } => {
            if let Some(d) = decisions.iter_mut().find(|d| d.layer == layer) {
                d.fase_fused = false;
            }
        }
        Resolution::AcceptNonUniformShard { .. } | Resolution::NoChange => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base(layer: u32) -> LayerDecisions {
        LayerDecisions {
            layer,
            csha_level: 0,
            head_count: 8,
            pruned_heads: 0,
            adapter_rank: 0,
            shard_factor: 1,
            fase_fused: false,
            adapter_comm_cost: 0.0,
            adapter_comm_budget: 1e9,
        }
    }

    #[test]
    fn csha_vs_cep_detected_when_heads_pruned() {
        let d = vec![LayerDecisions {
            csha_level: 2,
            pruned_heads: 2,
            ..base(4)
        }];
        let c = detect(&d);
        assert_eq!(c.len(), 1);
        assert!(matches!(c[0], ConflictKind::CshaVsCep { .. }));
    }

    #[test]
    fn wrga_vs_cpdt_fires_only_on_communication_overage() {
        // Under budget → no conflict.
        let ok = vec![LayerDecisions {
            adapter_rank: 4,
            shard_factor: 8,
            adapter_comm_cost: 100.0,
            adapter_comm_budget: 1000.0,
            ..base(0)
        }];
        assert!(detect(&ok).is_empty());

        // Over budget → conflict.
        let bad = vec![LayerDecisions {
            adapter_rank: 4,
            shard_factor: 8,
            adapter_comm_cost: 2000.0,
            adapter_comm_budget: 1000.0,
            ..base(0)
        }];
        assert_eq!(detect(&bad).len(), 1);
    }

    #[test]
    fn fase_vs_cpdt_detected_when_sharded_and_fused() {
        let d = vec![LayerDecisions {
            fase_fused: true,
            shard_factor: 8,
            ..base(0)
        }];
        assert_eq!(detect(&d).len(), 1);
    }

    #[test]
    fn cep_vs_cpdt_detected_when_heads_differ_across_sharded_layers() {
        let d = vec![
            LayerDecisions {
                head_count: 8,
                shard_factor: 4,
                ..base(0)
            },
            LayerDecisions {
                head_count: 6,
                shard_factor: 4,
                ..base(1)
            },
        ];
        let conflicts = detect(&d);
        assert!(conflicts.iter().any(|c| matches!(c, ConflictKind::CepVsCpdt { .. })));
    }

    #[test]
    fn resolve_downgrades_csha() {
        let c = vec![ConflictKind::CshaVsCep {
            layer: 5,
            csha_level: 2,
            pruned_heads: 2,
        }];
        let r = resolve(&c);
        assert_eq!(r[0], Resolution::DowngradeCsha { layer: 5, to_level: 1 });
    }

    #[test]
    fn resolve_removes_wrga_adapter() {
        let c = vec![ConflictKind::WrgaVsCpdt {
            layer: 2,
            adapter_rank: 8,
            shard_factor: 4,
        }];
        assert_eq!(resolve(&c)[0], Resolution::RemoveWrgaAdapter { layer: 2 });
    }

    #[test]
    fn resolve_defers_fase_step() {
        let c = vec![ConflictKind::FaseVsCpdt { layer: 3, shard_factor: 4 }];
        assert_eq!(resolve(&c)[0], Resolution::DeferFaseStep { layer: 3 });
    }

    #[test]
    fn greedy_applies_resolutions_inplace() {
        let initial = vec![
            LayerDecisions {
                csha_level: 2,
                pruned_heads: 2,
                ..base(0)
            },
            LayerDecisions {
                fase_fused: true,
                shard_factor: 4,
                ..base(1)
            },
        ];
        let (resolved, res) = greedy_resolve(initial);
        assert_eq!(res.len(), 2);
        assert!(resolved[0].csha_level < 2); // downgraded
        assert!(!resolved[1].fase_fused); // deferred
    }

    #[test]
    fn no_conflicts_means_no_resolutions() {
        let d = vec![base(0), base(1)];
        assert!(detect(&d).is_empty());
        let (kept, res) = greedy_resolve(d.clone());
        assert!(res.is_empty());
        assert_eq!(kept.len(), d.len());
    }

    #[test]
    fn priority_orders_techniques_cep_highest_pca_lowest() {
        // Strictly increasing rank: CEP > CPDT > CSHA > WRGA > FASE > PCA.
        let order = [
            Technique::Cep,
            Technique::Cpdt,
            Technique::Csha,
            Technique::Wrga,
            Technique::Fase,
            Technique::Pca,
        ];
        for pair in order.windows(2) {
            assert!(
                pair[0].priority() < pair[1].priority(),
                "{:?} should outrank {:?}",
                pair[0],
                pair[1]
            );
        }
        assert_eq!(Technique::Cep.priority(), 0);
        assert_eq!(Technique::Pca.priority(), 5);
    }

    #[test]
    fn loser_is_the_lower_priority_technique() {
        // The yielding technique in each conflict is decided by the comparator.
        assert_eq!(
            ConflictKind::CshaVsCep {
                layer: 0,
                csha_level: 3,
                pruned_heads: 2
            }
            .loser(),
            Technique::Csha
        );
        assert_eq!(
            ConflictKind::WrgaVsCpdt {
                layer: 0,
                adapter_rank: 8,
                shard_factor: 4
            }
            .loser(),
            Technique::Wrga
        );
        assert_eq!(
            ConflictKind::FaseVsCpdt {
                layer: 0,
                shard_factor: 4
            }
            .loser(),
            Technique::Fase
        );
        assert_eq!(
            ConflictKind::CepVsCpdt {
                layer: 0,
                head_count: 6
            }
            .loser(),
            Technique::Cpdt
        );
    }
}

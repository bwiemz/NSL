//! Gate / sensitivity / parity fixture registry for the
//! `nsl-codegen-bench` binary.
//!
//! Spec §4 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`
//! defines the three-tier matrix:
//!
//! * **Gate** (this file, today): single load-bearing config used for the
//!   keep/revert decision. §4.1 pins: segment-masked causal, seq_len=4096,
//!   head_dim=64, batch=4, block 64×64, target_sparsity=50%, sm_120.
//! * **Sensitivity** (B1.5-4): three fixtures at sparsities {10%, 50%, 90%}
//!   sharing the gate's other dims.
//! * **Parity** (B1.5-3): six fixtures drawn from the existing
//!   `PackingFixture` matrix; the M3 bit-identical correctness assertion.

use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

/// A bench fixture — one FA config plus the launch-time scalars
/// (`seq_len`, `batch`) and the input-generation target (`target_sparsity`).
///
/// `seq_len` is the runtime sequence length passed as the kernel's
/// `seq_len` arg. It is NOT inferred from `config` because
/// `FlashAttentionConfig` only carries per-tile dims (block_q / block_kv /
/// head_dim); the sequence length is a launch-time scalar.
///
/// `batch` mirrors the runtime `batch` arg. Heads=1 throughout the matrix
/// (single attention head suffices to measure tile-skip behaviour).
///
/// `target_sparsity` is the fraction of kv-tiles expected to be empty under
/// the segment mask generated for this fixture; the segment-mask generator
/// in `launch::generate_segment_mask` arranges segment boundaries so the
/// expected ratio matches.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: &'static str,
    pub config: FlashAttentionConfig,
    pub seq_len: u32,
    pub batch: u32,
    pub target_sparsity: f64,
}

/// Lookup a fixture by name. Returns `None` if unknown — main() maps that
/// to exit code 1.
pub fn lookup(name: &str) -> Option<Fixture> {
    registry().into_iter().find(|f| f.name == name)
}

/// Static registry of available fixtures. Function rather than `static`
/// so each call sees a fresh `Clone`-able `FlashAttentionConfig` (the
/// struct contains `Option<CshaExtras>` which doesn't implement `Copy`).
///
/// Order pins:
///   1. `gate_4096` — the single decision-driving fixture (§4.1).
///
/// Sensitivity (§4.3) and parity (§4.2) fixtures are added in B1.5-4 /
/// B1.5-3 respectively.
fn registry() -> Vec<Fixture> {
    vec![Fixture {
        name: "gate_4096",
        config: FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: true,
            csha: None,
        },
        seq_len: 4096,
        batch: 4,
        target_sparsity: 0.5,
    }]
    // Sensitivity (§4.3) and parity (§4.2) fixtures are added in B1.5-4 /
    // B1.5-3 respectively.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_returns_gate_fixture() {
        let f = lookup("gate_4096").expect("gate_4096 registered");
        assert_eq!(f.name, "gate_4096");
    }

    #[test]
    fn lookup_returns_none_for_unknown() {
        assert!(lookup("does_not_exist").is_none());
    }
}

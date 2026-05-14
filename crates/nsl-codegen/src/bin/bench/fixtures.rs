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
    // Helper to build a parity fixture; shares the gate fixture's tile
    // dims (64×64×64, segment_masked causal, sm_120) and varies only
    // seq_len and target_sparsity per the six existing `PackingFixture`
    // configurations in `tests/fixtures/mod.rs`. Each `parity_N` fixture
    // exercises the Tier B skip predicate on a different segment-id
    // packing pattern; the M3 parity test runs each twice (`--tier-b on`
    // and `--tier-b off`) and asserts byte-identical O outputs.
    fn parity(name: &'static str, seq_len: u32, target_sparsity: f64) -> Fixture {
        Fixture {
            name,
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
            seq_len,
            batch: 1,
            target_sparsity,
        }
    }

    vec![
        Fixture {
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
        },
        // Parity tier (§4.2 / spec §6.1). Six fixtures drawn from the
        // existing `PackingFixture` matrix in `tests/fixtures/mod.rs`.
        // Names map 1:1 to the M3 parity test functions in
        // `tests/pca_tier_b_m3_parity.rs`.
        //
        // The launch-time `target_sparsity` here only steers the
        // segment-mask generator; the M3 assertion is byte-equality of
        // Tier-B-on vs Tier-B-off outputs, which is correctness-preserving
        // for ANY segment-mask shape per design §6.1.
        parity("parity_1", 4096, 0.3),  // standard_3doc — 3 docs ≈ 30% empty kv-tiles
        parity("parity_2", 16_384, 0.3), // long_seq_5doc — 5 docs, larger seq
        parity("parity_3", 4096, 0.5),  // skewed_packing — uneven doc lengths
        parity("parity_4", 4096, 0.8),  // boundary_dense — 16 small docs, lots of empties
        parity("parity_5", 4096, 0.0),  // single_doc — no empties, predicate always false
        parity("parity_6", 4096, 0.5),  // tail_padding — 2 docs + padding sentinel
        // Sensitivity tier (§4.3 / spec §4.3.1). Three fixtures sharing the
        // gate fixture's dims (block 64×64×64, segment-masked causal, sm_120,
        // seq_len=4096, batch=4, head_dim=64) and varying only `target_sparsity`
        // to characterize the sparsity → benefit curve:
        //
        //   - sensitivity_10 (sub-threshold candidate): few skippable tiles;
        //     answers "does Tier B actively hurt at low sparsity?"
        //   - sensitivity_50 (gate redundancy): structurally identical to
        //     `gate_4096` — cross-check that two independent measurements
        //     of the same configuration agree within ±10% (§4.3.1).
        //   - sensitivity_90 (saturation regime): most tiles skippable;
        //     establishes the upper bound of Tier B's benefit.
        //
        // The triple feeds the §4.3.3 "keep with sparsity gate" outcome
        // when the curve shows a threshold shape.
        sensitivity("sensitivity_10", 0.1),
        sensitivity("sensitivity_50", 0.5),
        sensitivity("sensitivity_90", 0.9),
    ]
}

/// Helper to build a sensitivity fixture; shares the gate fixture's
/// tile dims, seq_len, batch, and head_dim, varying only `target_sparsity`.
/// Kept as a free fn (not a closure inside `registry()`) so the gate
/// fixture's literal block above stays the single source-of-truth for the
/// non-sparsity dims while still permitting per-fixture construction.
fn sensitivity(name: &'static str, target_sparsity: f64) -> Fixture {
    Fixture {
        name,
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
        target_sparsity,
    }
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

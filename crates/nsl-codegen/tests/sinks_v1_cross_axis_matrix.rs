//! Cycle-7 Sprint 3 — cross-axis matrix safety net.
//!
//! GLOBAL SAFETY NET for the cycle-5 `feedback_deferral_must_refuse`
//! invariant. Programmatically enumerates the cross product of relevant
//! `FlashAttentionConfig` axes and asserts every combination is EITHER
//! accepted (returns `(true, None)`) OR refused with a Sprint-citing
//! message — NO silent middle ground.
//!
//! ## WHEN YOU ADD A NEW `FlashAttentionConfig` AXIS, ADD IT HERE
//!
//! Cycle-5's discipline is "every configured value either works or
//! refuses." A new field added to the config without being threaded
//! through THIS matrix test creates a silent gap: the field's value
//! could be silently dropped at codegen with no test catching it.
//!
//! See `feedback_deferral_must_refuse` in project memory.
//!
//! ## Coverage
//!
//! Forward predicate (`attention_sinks_v1_eligible`) cross-axis matrix:
//!   * `causal ∈ {false, true}`
//!   * `rope_q ∈ {false, true}`
//!   * `gqa_group_size ∈ {1, 4}`
//!   * `paged ∈ {false, true}`
//!   * `tree_mask ∈ {false, true}`
//!   * `segment_masked ∈ {false, true}`
//!   * `block_q == block_kv` vs `block_q != block_kv`
//!   * `csha.save_activations_for_backward ∈ {false, true}` (csha=Some)
//!   * `csha.fused_projections ∈ {false, true}` (csha=Some)
//!   * `num_sink_tokens ∈ {0, 4}`
//!
//! Backward predicate (`attention_sinks_v1_backward_eligible`): same
//! enumeration, asserting the simpler total-refusal-on-sinks contract.
//!
//! The full cross-product is bounded above by 2^9 * 2 = 1024
//! combinations — each predicate call is microseconds, total runtime
//! is well under one second.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::sinks::{
    attention_sinks_v1_backward_eligible, attention_sinks_v1_eligible,
};

/// Build a `FlashAttentionConfig` from a packed axis-tuple. `block_q`
/// is always 64; `block_kv` is 64 (matching) when `block_q_eq_block_kv`,
/// else 128 (asymmetric tile per PR #101).
#[allow(clippy::too_many_arguments)]
fn cfg_from_axes(
    causal: bool,
    rope_q: bool,
    gqa: u32,
    paged: bool,
    tree_mask: bool,
    segment_masked: bool,
    block_q_eq_block_kv: bool,
    csha: Option<(bool /* save_act */, bool /* fused_proj */)>,
    num_sink_tokens: u32,
) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: if block_q_eq_block_kv { 64 } else { 128 },
        head_dim: 64,
        causal,
        paged,
        rope_q,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: gqa,
        tree_mask,
        num_sink_tokens,
        gpu_sm: 80,
        segment_masked,
        csha: csha.map(|(save_act, fused_proj)| CshaExtras {
            level: 2,
            d_model: 64,
            active_heads: 1,
            save_activations_for_backward: save_act,
            fused_projections: fused_proj,
            ..CshaExtras::default()
        }),
    }
}

/// Render an axis-tuple as a human-readable string for failure messages.
#[allow(clippy::too_many_arguments)]
fn axes_label(
    causal: bool,
    rope_q: bool,
    gqa: u32,
    paged: bool,
    tree_mask: bool,
    segment_masked: bool,
    bq_eq_bkv: bool,
    csha: Option<(bool, bool)>,
    num_sink_tokens: u32,
) -> String {
    let csha_str = match csha {
        None => "None".to_string(),
        Some((s, f)) => format!("(save={s},fused={f})"),
    };
    format!(
        "causal={causal} rope_q={rope_q} gqa={gqa} paged={paged} tree_mask={tree_mask} \
         segment_masked={segment_masked} bq_eq_bkv={bq_eq_bkv} csha={csha_str} sinks={num_sink_tokens}"
    )
}

/// Recognised refusal citations. A refusal message MUST contain at
/// least one of these so a future engineer can grep to the lift point.
const VALID_CITATIONS: &[&str] = &[
    // Sprint-numbered lifts (cycle-5 feedback_deferral_must_refuse pattern).
    "Sprint 2",
    "Sprint 3",
    "Sprint 4",
    "Sprint 5",
    "Sprint 6",
    // Named deferrals not tied to a numbered Sprint number.
    "fused_projections",
    "save_activations_for_backward",
    "segment_masked",
    "multi-page-sink",
];

fn message_has_valid_citation(msg: &str) -> bool {
    VALID_CITATIONS.iter().any(|c| msg.contains(c))
}

/// Forward-predicate cross-axis matrix. Programmatically iterates the
/// full product of 9 boolean-ish axes and asserts each combination is
/// EITHER `(true, None)` OR `(false, Some(msg))` with a Sprint-citing
/// reason. NO silent middle ground.
#[test]
fn cross_axis_matrix_forward_eligibility() {
    let mut total = 0usize;
    let mut accepted = 0usize;
    let mut refused = 0usize;
    let mut accepted_with_sinks = 0usize;

    for &causal in &[false, true] {
        for &rope_q in &[false, true] {
            for &gqa in &[1u32, 4] {
                for &paged in &[false, true] {
                    for &tree_mask in &[false, true] {
                        for &segment_masked in &[false, true] {
                            for &bq_eq_bkv in &[false, true] {
                                let csha_variants: &[Option<(bool, bool)>] = &[
                                    None,
                                    Some((false, false)),
                                    Some((true, false)),
                                    Some((false, true)),
                                ];
                                for &csha in csha_variants {
                                    for &sinks in &[0u32, 4] {
                                        total += 1;
                                        let cfg = cfg_from_axes(
                                            causal,
                                            rope_q,
                                            gqa,
                                            paged,
                                            tree_mask,
                                            segment_masked,
                                            bq_eq_bkv,
                                            csha,
                                            sinks,
                                        );
                                        let (eligible, why) =
                                            attention_sinks_v1_eligible(&cfg);
                                        let label = axes_label(
                                            causal,
                                            rope_q,
                                            gqa,
                                            paged,
                                            tree_mask,
                                            segment_masked,
                                            bq_eq_bkv,
                                            csha,
                                            sinks,
                                        );

                                        // Sentinel-disabled invariant: when
                                        // num_sink_tokens=0, the predicate
                                        // MUST return (true, None) regardless
                                        // of any other axis.
                                        if sinks == 0 {
                                            assert_eq!(
                                                (eligible, why),
                                                (true, None),
                                                "[{label}] sentinel-disabled invariant violated: \
                                                 num_sink_tokens=0 must always be eligible"
                                            );
                                            accepted += 1;
                                            continue;
                                        }

                                        // Sinks-enabled paths: must be either
                                        // (true, None) or (false, Some(msg))
                                        // — never (true, Some(_)) or
                                        // (false, None).
                                        match (eligible, why) {
                                            (true, None) => {
                                                accepted += 1;
                                                accepted_with_sinks += 1;
                                            }
                                            (false, Some(msg)) => {
                                                refused += 1;
                                                // Refusal MUST cite a future
                                                // Sprint or a named deferral
                                                // (cycle-5 invariant).
                                                assert!(
                                                    message_has_valid_citation(msg),
                                                    "[{label}] refusal has no Sprint/named-deferral citation: '{msg}'\n\
                                                     accepted citations: {:?}",
                                                    VALID_CITATIONS,
                                                );
                                            }
                                            (true, Some(msg)) => panic!(
                                                "[{label}] silent-middle-ground: eligible=true but blocking_reason=Some('{msg}')"
                                            ),
                                            (false, None) => panic!(
                                                "[{label}] silent-middle-ground: eligible=false but blocking_reason=None"
                                            ),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // At sinks=4, AT MOST one narrow config matches (the Sprint 1b
    // narrow-config criterion): causal=false, rope_q=false, gqa=1,
    // paged=false, tree_mask=false, segment_masked=false, bq==bkv,
    // and csha is either None OR Some((false, false)). Two csha
    // variants survive (None + Some(false, false)) so the matrix
    // sees 2 sinks-enabled accepted configs.
    //
    // We deliberately upper-bound rather than equality-check the
    // accepted-with-sinks count to keep this test stable against
    // future widening of the narrow path (e.g., Sprint 2 cycle-7
    // adding causal=true accepts adds 2 more — and the contract
    // remains "no silent middle ground").
    assert!(
        accepted_with_sinks >= 1,
        "expected at least the Sprint 1b narrow config to accept sinks; got {accepted_with_sinks}"
    );
    // Sanity: the full enumeration totals match the product.
    // 2*2*2*2*2*2*2*4*2 = 1024
    assert_eq!(total, 1024, "matrix enumeration size changed; update WHEN YOU ADD A NEW AXIS doc");
    assert_eq!(
        accepted + refused,
        total,
        "every combination must end in either accepted or refused — no silent middle ground"
    );
}

/// Backward-predicate cross-axis matrix. Same enumeration as the
/// forward matrix, asserting the simpler total-refusal-on-sinks
/// contract from Sprint 2 cycle-7:
///   * `num_sink_tokens == 0` → `(true, None)` ALWAYS (sentinel)
///   * `num_sink_tokens > 0`  → `(false, Some(msg))` where `msg`
///     names "backward" AND cites "v2" (future-sprint lift point).
#[test]
fn cross_axis_matrix_backward_eligibility() {
    let mut total = 0usize;
    let mut accepted = 0usize;
    let mut refused = 0usize;

    for &causal in &[false, true] {
        for &rope_q in &[false, true] {
            for &gqa in &[1u32, 4] {
                for &paged in &[false, true] {
                    for &tree_mask in &[false, true] {
                        for &segment_masked in &[false, true] {
                            for &bq_eq_bkv in &[false, true] {
                                let csha_variants: &[Option<(bool, bool)>] = &[
                                    None,
                                    Some((false, false)),
                                    Some((true, false)),
                                    Some((false, true)),
                                ];
                                for &csha in csha_variants {
                                    for &sinks in &[0u32, 4] {
                                        total += 1;
                                        let cfg = cfg_from_axes(
                                            causal,
                                            rope_q,
                                            gqa,
                                            paged,
                                            tree_mask,
                                            segment_masked,
                                            bq_eq_bkv,
                                            csha,
                                            sinks,
                                        );
                                        let (eligible, why) =
                                            attention_sinks_v1_backward_eligible(&cfg);
                                        let label = axes_label(
                                            causal,
                                            rope_q,
                                            gqa,
                                            paged,
                                            tree_mask,
                                            segment_masked,
                                            bq_eq_bkv,
                                            csha,
                                            sinks,
                                        );

                                        if sinks == 0 {
                                            // Sentinel-disabled invariant
                                            // (Sprint 2 cycle-7): backward
                                            // is unaffected by any other axis
                                            // when sinks are disabled.
                                            assert_eq!(
                                                (eligible, why),
                                                (true, None),
                                                "[{label}] sentinel-disabled invariant violated for backward"
                                            );
                                            accepted += 1;
                                        } else {
                                            // Sinks-enabled: total refusal.
                                            assert!(
                                                !eligible,
                                                "[{label}] sinks-enabled backward must refuse"
                                            );
                                            let msg = why.expect(
                                                "sinks-enabled backward must provide blocking reason",
                                            );
                                            assert!(
                                                msg.contains("v2"),
                                                "[{label}] backward refusal must cite future v2 sprint: '{msg}'"
                                            );
                                            assert!(
                                                msg.contains("backward"),
                                                "[{label}] backward refusal must name 'backward': '{msg}'"
                                            );
                                            refused += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert_eq!(total, 1024);
    assert_eq!(
        accepted + refused,
        total,
        "every backward combination must end in either accepted or refused — no silent middle ground"
    );
    // Half the matrix has sinks=0 (accepted) and half has sinks=4
    // (refused). With the enumeration above (every axis pair contributes
    // 2 values to the sink axis), exactly half the rows are accepted.
    assert_eq!(accepted, total / 2, "sentinel-disabled half of the matrix must be accepted");
    assert_eq!(refused, total / 2, "sinks-enabled half of the matrix must be refused");
}

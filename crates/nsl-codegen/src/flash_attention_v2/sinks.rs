//! §4.3 attention sinks — Sprint 1a cycle-7 precursor: effective_block_kv
//! indirection so multiple consumer sites in the v2 forward emitter agree
//! on the same row count.
//!
//! Today (`num_sink_tokens == 0` is the only flowing value post-cycle-5
//! refusal): `effective_block_kv(config) == config.block_kv`. The indirection
//! is byte-identical in emitted PTX vs pre-Sprint-1a.
//!
//! Sprint 1b lifts the cycle-5 refusal for the narrow Tier A forward
//! single-tile config and `effective_block_kv` then returns
//! `block_kv + num_sink_tokens`. Subsequent sprints (multi-tile + causal,
//! GQA, RoPE) build on this foundation.

use crate::flash_attention::FlashAttentionConfig;

/// The effective number of KV rows present in the SMEM KV slab. Equal to
/// `config.block_kv` when sinks are disabled (`num_sink_tokens == 0` — the
/// only currently-flowing value post-cycle-5 refusal). When sinks are
/// enabled in Sprint 1b: `block_kv + num_sink_tokens` for the v1 single-tile
/// case. Consumers MUST read this helper rather than the raw `block_kv`
/// to keep the load bound, S-store stride, softmax chunks, PV loop bound,
/// and SP scratch bytes in lockstep.
#[inline]
pub fn effective_block_kv(config: &FlashAttentionConfig) -> i64 {
    config.block_kv + (config.num_sink_tokens as i64)
}

/// Byte cost of the persistent sink slab at the front of the KV region.
/// Returns 0 when sinks are disabled.
#[inline]
pub fn sink_slab_bytes(config: &FlashAttentionConfig) -> usize {
    (config.num_sink_tokens as usize) * (config.head_dim as usize) * 2 /* f16 bytes */
}

/// Sprint 1b cycle-7: attention sinks v1 narrow-config eligibility.
/// Returns `(eligible, blocking_reason)`. `blocking_reason` is `Some` iff
/// `num_sink_tokens > 0` and the config violates v1 narrowness. The
/// string MUST name the failing axis AND cite the future Sprint number
/// that lifts it (cycle-5 feedback_deferral_must_refuse invariant).
///
/// At `num_sink_tokens == 0` the function returns `(true, None)`
/// unconditionally — this is the sentinel "sinks disabled" path and the
/// other axes (causal/rope/gqa/...) are not consulted.
///
/// v1 supports: Tier A scalar forward, single-tile
/// (`block_q==block_kv==seq_len`), `causal=false`, `rope_q=false`,
/// `gqa_group_size=1`, `paged=false`, `segment_masked=false`,
/// `tree_mask=false`, no `csha.save_activations_for_backward` (no
/// backward), no `csha.fused_projections`, and
/// `num_sink_tokens <= block_kv`.
pub fn attention_sinks_v1_eligible(
    config: &FlashAttentionConfig,
) -> (bool, Option<&'static str>) {
    if config.num_sink_tokens == 0 {
        return (true, None); // sentinel disabled
    }
    if config.causal {
        return (
            false,
            Some(
                "causal=true (deferred to cycle-7 Sprint 2: multi-tile + causal sink-bypass predicate)",
            ),
        );
    }
    if config.rope_q {
        return (
            false,
            Some(
                "rope_q=true (deferred to cycle-7 Sprint 4: StreamingLLM no-rotation-for-sinks policy)",
            ),
        );
    }
    if config.gqa_group_size != 1 {
        return (
            false,
            Some("gqa_group_size > 1 (deferred to cycle-7 Sprint 3: GQA + sinks stride)"),
        );
    }
    if config.paged {
        return (
            false,
            Some("paged=true (deferred to cycle-7 Sprint 6: paged + sinks design pending)"),
        );
    }
    if config.segment_masked {
        return (
            false,
            Some("segment_masked=true (deferred to cycle-7 Sprint 6: PCA Tier A + sinks not designed)"),
        );
    }
    if config.tree_mask {
        return (
            false,
            Some("tree_mask=true (deferred to cycle-7 Sprint 6: tree_mask + sinks refusal)"),
        );
    }
    if let Some(csha) = config.csha.as_ref() {
        if csha.save_activations_for_backward {
            return (
                false,
                Some(
                    "save_activations_for_backward=true (deferred to cycle-7 Sprint 2: backward sinks refusal)",
                ),
            );
        }
        if csha.fused_projections {
            return (
                false,
                Some(
                    "csha.fused_projections=true (deferred to cycle-7 Sprint 5: fused projections + sinks v2)",
                ),
            );
        }
    }
    if config.block_q != config.block_kv {
        return (
            false,
            Some("block_q != block_kv (deferred to cycle-7 Sprint 2: multi-tile sink layout)"),
        );
    }
    if config.num_sink_tokens as i64 > config.block_kv {
        return (
            false,
            Some(
                "num_sink_tokens > block_kv (deferred to cycle-7 Sprint 6: multi-page-sink design)",
            ),
        );
    }
    (true, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

    fn cfg(block_kv: i64, num_sink_tokens: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv, head_dim: 64,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::Adjacent, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens,
            gpu_sm: 80, segment_masked: false, csha: None,
        }
    }

    #[test]
    fn effective_block_kv_byte_identity_at_zero_sinks() {
        // Sprint 1a invariant: at num_sink_tokens=0 (the only flowing value
        // post-cycle-5 refusal), effective_block_kv MUST equal block_kv.
        for bkv in [16, 32, 64, 128, 256_i64] {
            assert_eq!(effective_block_kv(&cfg(bkv, 0)), bkv);
        }
    }

    #[test]
    fn effective_block_kv_extends_with_sinks() {
        // Sprint 1b will lift the cycle-5 refusal for narrow configs and
        // this is the value those sites will see.
        assert_eq!(effective_block_kv(&cfg(64, 4)), 68);
        assert_eq!(effective_block_kv(&cfg(128, 8)), 136);
    }

    #[test]
    fn sink_slab_bytes_zero_when_disabled() {
        assert_eq!(sink_slab_bytes(&cfg(64, 0)), 0);
    }

    #[test]
    fn sink_slab_bytes_is_f16_per_row_per_dim() {
        // num_sink × head_dim × 2 (f16)
        assert_eq!(sink_slab_bytes(&cfg(64, 4)), 4 * 64 * 2);
        assert_eq!(sink_slab_bytes(&cfg(64, 8)), 8 * 64 * 2);
    }
}

#[cfg(test)]
mod eligibility_tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn narrow_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 64,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::Adjacent, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 4,
            gpu_sm: 80, segment_masked: false, csha: None,
        }
    }

    #[test]
    fn zero_sinks_always_eligible() {
        // num_sink_tokens=0 is the sentinel: short-circuits before any
        // other axis is consulted. Even with otherwise-blocking axes set,
        // the disabled-sinks path must return (true, None).
        let mut cfg = narrow_cfg();
        cfg.num_sink_tokens = 0;
        cfg.causal = true;
        cfg.rope_q = true;
        cfg.gqa_group_size = 8;
        assert_eq!(attention_sinks_v1_eligible(&cfg), (true, None));
    }

    #[test]
    fn narrow_config_eligible() {
        assert_eq!(attention_sinks_v1_eligible(&narrow_cfg()), (true, None));
    }

    // Per-axis refusal helper. Each test must assert the message contains
    // (a) the axis name, AND (b) a future Sprint citation.
    macro_rules! per_axis_refusal_test {
        ($name:ident, $mutate:expr, $expect_substr:literal) => {
            #[test]
            fn $name() {
                let mut cfg = narrow_cfg();
                let mutate: fn(&mut FlashAttentionConfig) = $mutate;
                mutate(&mut cfg);
                let (eligible, why) = attention_sinks_v1_eligible(&cfg);
                assert!(!eligible, "expected refusal for mutated narrow_cfg");
                let msg = why.expect("blocking reason must be Some for ineligible");
                assert!(
                    msg.contains($expect_substr),
                    "expected '{}' in '{}'",
                    $expect_substr,
                    msg
                );
                assert!(
                    msg.contains("Sprint"),
                    "blocking reason must cite future Sprint: '{}'",
                    msg
                );
            }
        };
    }

    per_axis_refusal_test!(
        causal_refused,
        |c: &mut FlashAttentionConfig| c.causal = true,
        "causal=true"
    );
    per_axis_refusal_test!(
        rope_q_refused,
        |c: &mut FlashAttentionConfig| c.rope_q = true,
        "rope_q=true"
    );
    per_axis_refusal_test!(
        gqa_refused,
        |c: &mut FlashAttentionConfig| c.gqa_group_size = 8,
        "gqa_group_size > 1"
    );
    per_axis_refusal_test!(
        paged_refused,
        |c: &mut FlashAttentionConfig| c.paged = true,
        "paged=true"
    );
    per_axis_refusal_test!(
        segment_masked_refused,
        |c: &mut FlashAttentionConfig| c.segment_masked = true,
        "segment_masked=true"
    );
    per_axis_refusal_test!(
        tree_mask_refused,
        |c: &mut FlashAttentionConfig| c.tree_mask = true,
        "tree_mask=true"
    );
    per_axis_refusal_test!(
        multi_tile_refused,
        |c: &mut FlashAttentionConfig| {
            c.block_q = 64;
            c.block_kv = 128;
        },
        "block_q != block_kv"
    );
    per_axis_refusal_test!(
        too_many_sinks_refused,
        |c: &mut FlashAttentionConfig| {
            c.block_q = 4;
            c.block_kv = 4;
            c.num_sink_tokens = 16;
        },
        "num_sink_tokens > block_kv"
    );

    #[test]
    fn save_activations_refused_via_csha_extras() {
        let mut cfg = narrow_cfg();
        cfg.csha = Some(CshaExtras {
            save_activations_for_backward: true,
            ..CshaExtras::default()
        });
        let (eligible, why) = attention_sinks_v1_eligible(&cfg);
        assert!(!eligible);
        let msg = why.unwrap();
        assert!(msg.contains("save_activations_for_backward"));
        assert!(msg.contains("Sprint"));
    }

    #[test]
    fn fused_projections_refused_via_csha_extras() {
        let mut cfg = narrow_cfg();
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            ..CshaExtras::default()
        });
        let (eligible, why) = attention_sinks_v1_eligible(&cfg);
        assert!(!eligible);
        let msg = why.unwrap();
        assert!(msg.contains("fused_projections"));
        assert!(msg.contains("Sprint"));
    }
}

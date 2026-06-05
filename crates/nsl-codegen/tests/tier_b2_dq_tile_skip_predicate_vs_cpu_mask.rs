//! Spec §4.4 D1 verification: tile_skip predicate matches CPU causal mask.
//! D1 is non-lane-sensitive, so structural ptxas-clean + CPU-mask-comparison
//! test instead of the §5.5 byte-pattern lane-mapping discipline.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn causal_cfg(bq: i64, hd: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq, block_kv: bq, head_dim: hd,
        causal: true, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
        gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn tile_skip_predicate_matches_cpu_causal_mask() {
    let cfg = causal_cfg(64, 32);
    let bq = 64u32;
    let bkv = 64u32;

    // CPU truth table for predicate at known (q_iter, kv_iter) pairs:
    let cases = [
        (0u32, 0u32, 1u32, "kv_tile precedes q_tile_end (active)"),
        (0u32, 1u32, 0u32, "kv_tile follows q_tile_end (causal skip)"),
        (1u32, 1u32, 1u32, "kv_tile coincides with q_tile range"),
        (2u32, 1u32, 1u32, "kv_tile fully before q_tile (active)"),
        (1u32, 2u32, 0u32, "kv_tile_start > q_tile_end (skip)"),
    ];
    for (q_iter, kv_iter, expected, label) in cases {
        let q_tile_end = q_iter * bq + (bq - 1);
        let kv_tile_start = kv_iter * bkv;
        let cpu_pred = if kv_tile_start <= q_tile_end { 1u32 } else { 0u32 };
        assert_eq!(cpu_pred, expected,
            "CPU causal mask wrong for ({q_iter}, {kv_iter}): {label}");
    }
    // Structural verification: the kernel's emitted PTX contains the right
    // comparison operator + operands. Behavioral verification is at H4 GPU run.
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    assert!(ptx.contains("setp.le.u32 %p_causal_active, %kv_tile_start, %q_tile_end"),
        "expected causal predicate setp.le on (%kv_tile_start, %q_tile_end)");
}

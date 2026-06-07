//! Sprint 1b cycle-7 paper §4.3: SMEM budget validator pin.
//!
//! When `num_sink_tokens > 0`, the KV SMEM slab grows from
//! `block_kv * head_dim * 2` to `(block_kv + num_sink_tokens) * head_dim * 2`
//! bytes (the sink rows are pinned at the front of the slab; see
//! Sprint 1b cycle-7 `sinks::effective_block_kv` + the updated
//! `smem_layout::sp_offset`). For configs near the 99 KB dynamic-SMEM
//! ceiling, the additional sink rows can push the kernel over budget.
//!
//! This test pins that the validator rejects such configs at codegen
//! with a diagnostic that includes the actual byte count, rather than
//! silently emitting a kernel that fails at launch with
//! `CUDA_ERROR_OUT_OF_MEMORY` or stomps onto neighboring SMEM regions.

#![cfg(feature = "test-helpers")]

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::{
    total_bytes, validate_scalar_v2_config, Direction, SMEM_DYNAMIC_BUDGET_BYTES,
};

fn cfg(block_q: i64, block_kv: i64, head_dim: i64, num_sink_tokens: u32) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q,
        block_kv,
        head_dim,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens,
        gpu_sm: 80,
        segment_masked: false,
        csha: None,
    }
}

#[test]
fn sinks_extending_kv_slab_can_overflow_99kb_cap() {
    // Fixture: a config that PASSES the validator at num_sink_tokens=0
    // but FAILS once the sink slab is folded in. This proves the
    // validator sees `effective_block_kv` (via sp_offset → total_bytes),
    // not raw `block_kv`.
    //
    // At hd=128, bq=128, bkv=128:
    //   Q   = 128*128*2 = 32768 B
    //   KV  = 128*128*2 = 32768 B
    //   SP  = 4*128*4 = 2048 B
    //   total ≈ 67584 B  (fits 99 KB)
    //
    // With num_sink_tokens=128 the KV slab doubles:
    //   KV  = (128+128)*128*2 = 65536 B
    //   SP  = 4*256*4 = 4096 B
    //   total ≈ 102400 B = 100 KB  (exceeds 99 KB cap)
    let baseline = cfg(128, 128, 128, 0);
    assert!(
        validate_scalar_v2_config(&baseline, Direction::Forward).is_ok(),
        "baseline (sinks disabled) must fit the dynamic SMEM cap; \
         if this regresses the fixture must be retuned"
    );
    let baseline_total = total_bytes(&baseline);
    assert!(
        baseline_total <= SMEM_DYNAMIC_BUDGET_BYTES,
        "baseline total {baseline_total} must be <= cap {SMEM_DYNAMIC_BUDGET_BYTES}"
    );

    let with_sinks = cfg(128, 128, 128, 128);
    let with_sinks_total = total_bytes(&with_sinks);
    // Confirm the sink slab actually grows the total past the cap.
    assert!(
        with_sinks_total > SMEM_DYNAMIC_BUDGET_BYTES,
        "fixture must overflow once sinks are folded in (got {with_sinks_total} <= {SMEM_DYNAMIC_BUDGET_BYTES}); \
         retune the fixture or the validator regressed"
    );

    // The validator must surface the overflow with a diagnostic naming
    // the byte total and the cap (load-bearing for users debugging
    // sink-induced launch failures).
    let err = validate_scalar_v2_config(&with_sinks, Direction::Forward)
        .expect_err("sinks-extended config must fail validation");
    let msg = format!("{err}");
    assert!(
        msg.contains(&with_sinks_total.to_string()),
        "validator error must include the actual byte total {with_sinks_total}: {msg}"
    );
    assert!(
        msg.contains("99 KB") || msg.contains("99"),
        "validator error must cite the 99 KB cap: {msg}"
    );
}

#[test]
fn sinks_at_zero_match_baseline_total() {
    // Direct byte-identity check for the sp_offset Sprint 1b edit: at
    // num_sink_tokens=0, the SP region must land at the SAME byte offset
    // it did pre-Sprint-1b (effective_block_kv == block_kv).
    let cfg_zero = cfg(64, 64, 64, 0);
    let cfg_zero_total = total_bytes(&cfg_zero);
    // Hand-computed: Q=64*64*2=8192, KV=64*64*2=8192, SP=4*64*4=1024 → 17408.
    assert_eq!(
        cfg_zero_total, 17408,
        "Sprint 1a/1b byte-identity invariant: at num_sink_tokens=0, \
         total_bytes(bq=64, bkv=64, hd=64) MUST equal 17408 (the \
         pre-Sprint-1a value). A change here means sp_offset's \
         effective_block_kv routing regressed."
    );
}

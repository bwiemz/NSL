//! Guardrail tests for the dK/dV-kernel scaffold.
//!
//! Verifies:
//!   - No atomic or red.* instructions emitted (dK/dV uses register-resident
//!     accumulators, not atomicAdd).
//!   - PTX is 7-bit ASCII (feedback_ptx_comment_ascii_only invariant).

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;

fn cfg(hd: i64, bq: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq, block_kv: bq, head_dim: hd, causal: true, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn dkdv_kernel_has_no_atomic_or_red() {
    for (hd, bq) in [(32, 64), (64, 64), (128, 32)] {
        let ptx = synthesize_dkdv_kernel(&cfg(hd, bq)).expect("synth ok");
        assert!(!ptx.contains("atom."), "dK/dV must not use atomicAdd (hd={hd})");
        // Check for PTX reduction instructions (red.global.* / red.shared.*).
        // A PTX `red.*` instruction always appears as a standalone instruction on its
        // own line starting with whitespace then "red.". We check line-by-line to avoid
        // false-positives from "cp.async.ca.shared.global" which contains "red.global"
        // as a substring (sha"red.global").
        for line in ptx.lines() {
            let trimmed = line.trim_start();
            assert!(
                !trimmed.starts_with("red."),
                "dK/dV must not use red.* reduction instruction (hd={hd}): found line: {line}"
            );
        }
    }
}

#[test]
fn dkdv_kernel_ptx_is_ascii_only() {
    let ptx = synthesize_dkdv_kernel(&cfg(128, 32)).expect("synth ok");
    assert!(ptx.is_ascii(), "dK/dV PTX must be 7-bit ASCII (feedback_ptx_comment_ascii_only)");
}

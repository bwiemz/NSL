//! Tier B.2 backward emitter ASCII-only invariant.
//!
//! Catches Unicode characters (em-dashes, multiplication signs, box-drawing
//! glyphs) being pushed into emitted PTX. The cudarc-backed `ptxas` JIT
//! aborts with `CUDA_ERROR_INVALID_PTX` ("Unexpected non-ASCII character
//! encountered") when any non-ASCII byte appears in the PTX stream — even
//! inside a `//` comment. This guardrail asserts every byte of the emitted
//! PTX is in the 7-bit ASCII range.
//!
//! Real-incident origin: 2026-05-20 — first attempted GPU launch of the
//! D pre-pass kernel failed with ptxas exit code 218 because emitter
//! comments contained `—` (em-dash) and `×` (multiplication sign). Memory
//! note: `feedback_ptx_comment_ascii_only.md`.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass;
use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn cfg(hd: i64) -> FlashAttentionConfig {
    let bq = if hd >= 128 { 32 } else { 64 };
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bq,
        head_dim: hd,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
        checkpoint: None,
    }
}

fn assert_ascii_only(label: &str, ptx: &str) {
    for (line_idx, line) in ptx.lines().enumerate() {
        for (col, byte) in line.bytes().enumerate() {
            assert!(
                byte.is_ascii(),
                "{label}: non-ASCII byte 0x{byte:02x} at line {} col {}:\n  {line}",
                line_idx + 1,
                col + 1,
            );
        }
    }
}

#[test]
fn tier_b2_d_prepass_emits_ascii_only_ptx() {
    for hd in [32, 64, 128, 256] {
        let ptx = synthesize_d_prepass(&cfg(hd)).unwrap();
        assert_ascii_only(&format!("d_prepass hd={hd}"), &ptx);
    }
}

#[test]
fn tier_b2_dq_kernel_emits_ascii_only_ptx() {
    for hd in [32, 64, 128, 256] {
        let ptx = synthesize_dq_kernel(&cfg(hd)).unwrap();
        assert_ascii_only(&format!("dq_kernel hd={hd}"), &ptx);
    }
}

#[test]
fn tier_b2_dkdv_kernel_emits_ascii_only_ptx() {
    for hd in [32, 64, 128, 256] {
        let ptx = synthesize_dkdv_kernel(&cfg(hd)).unwrap();
        assert_ascii_only(&format!("dkdv_kernel hd={hd}"), &ptx);
    }
}

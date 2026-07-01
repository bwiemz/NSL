//! CSHA Cycle 19 T3 — q_load RoPE numerical GREEN witness (scaffold).
//!
//! Purpose: prove that after the partner-shuffle fix in q_load.rs, the
//! rotated Q produced by the v2 kernel (rope_q=true, fused_projections
//! DISABLED so the inline RoPE branch is exercised) matches a CPU
//! reference RoPE oracle to `max_rel_err < 1e-3`.
//!
//! Status: gated behind `#[cfg(feature = "cuda")]` AND `#[ignore]`. The
//! non-CSHA v2 launcher for a standalone rope_q=true forward is not
//! currently wired at the Rust FFI level (existing `nsl_flash_attention`
//! entry points go through the CSHA fused path). The CPU oracle IS
//! implemented below so that once the launcher wiring lands, only the
//! `dispatch_forward_kernel` helper needs to be filled in.
//!
//! The structural GREEN witness for the fix lives in
//! `csha_cycle19_qload_rope_partner.rs` — that test guards the emitter
//! and turns RED-then-GREEN on the fix. This file is scoped to the
//! numerical follow-on.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

fn base_config(style: RopeStyle) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: true,
        rope_style: style,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: None,
        checkpoint: None,
    }
}

/// CPU-reference RoPE oracle for one head-dim vector.
///
/// Adjacent style: pairs are (x[2k], x[2k+1]) rotated by (cos[k], sin[k]).
/// HalfSplit style: pairs are (x[k], x[k + head_dim/2]) rotated by
/// (cos[k], sin[k]).
///
///   new_x0 = x0*cos − x1*sin
///   new_x1 = x0*sin + x1*cos
#[allow(dead_code)]
fn cpu_rope_reference(
    q: &[f32],
    head_dim: usize,
    seq: usize,
    style: RopeStyle,
    rope_freq_base: f32,
) -> Vec<f32> {
    let mut out = q.to_vec();
    for s in 0..seq {
        let row = &mut out[s * head_dim..(s + 1) * head_dim];
        let pos = s as f32;
        let half = head_dim / 2;
        for k in 0..half {
            let inv_freq =
                1.0f32 / rope_freq_base.powf(2.0 * (k as f32) / (head_dim as f32));
            let theta = pos * inv_freq;
            let (sin_t, cos_t) = theta.sin_cos();
            let (i0, i1) = match style {
                RopeStyle::Adjacent => (2 * k, 2 * k + 1),
                RopeStyle::HalfSplit => (k, k + half),
            };
            let x0 = row[i0];
            let x1 = row[i1];
            row[i0] = x0 * cos_t - x1 * sin_t;
            row[i1] = x0 * sin_t + x1 * cos_t;
        }
    }
    out
}

#[allow(dead_code)]
fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() { return false; }
    // Runtime CUDA driver init check would go here; without it the test
    // is `#[ignore]`-gated so this is fine.
    true
}

/// PLACEHOLDER: dispatches the v2 forward kernel for the given config
/// and returns the rotated Q as a Vec<f32> (seq * head_dim).
///
/// The launcher wiring for the non-CSHA rope_q=true path is not yet in
/// place at the Rust FFI level. When this is wired (see the flash_attention
/// runtime FFI in `crates/nsl-runtime/src/flash_attention.rs`), replace
/// this stub with the real dispatch and drop `#[ignore]` from the tests
/// below.
#[allow(dead_code)]
fn dispatch_forward_kernel(
    _cfg: &FlashAttentionConfig,
    _q_in: &[f32],
    _cos: &[f32],
    _sin: &[f32],
    _seq: usize,
) -> Vec<f32> {
    unimplemented!(
        "non-CSHA v2 rope_q forward launcher not yet wired at Rust FFI \
         level; see CSHA cycle 19 T3 follow-on notes"
    )
}

#[allow(dead_code)]
fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let denom = x.abs().max(y.abs()).max(1e-6);
            (x - y).abs() / denom
        })
        .fold(0f32, f32::max)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "non-CSHA v2 rope_q forward launcher not yet wired at FFI level \
            (structural fix guarded by csha_cycle19_qload_rope_partner.rs)"]
fn qload_rope_adjacent_matches_cpu_reference() {
    if !cuda_available() {
        eprintln!("skipping — CUDA not available in this environment");
        return;
    }
    let cfg = base_config(RopeStyle::Adjacent);
    // Sanity: PTX synthesizes cleanly.
    let ptx = synthesize_flash_attention_ptx_v2(&cfg);
    assert!(!ptx.is_empty(), "PTX must synthesize");

    let head_dim = cfg.head_dim as usize;
    let seq = 8usize;
    let rope_freq_base = 10_000.0f32;

    // Deterministic input.
    let q: Vec<f32> = (0..seq * head_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    // Precompute cos/sin tables in the same layout the kernel reads
    // (row-major [seq, head_dim/2] with duplicated bins for Adjacent).
    //
    // Cos/sin table has cos[2k] == cos[2k+1] and sin[2k] == sin[2k+1]
    // because both lanes of a rotation pair share the same angle. The
    // sign flip happens in the FMA (partner*sin negated on the x0 branch,
    // positive on the x1 branch), not in the table.
    let half = head_dim / 2;
    let mut cos_tab = vec![0.0f32; seq * head_dim];
    let mut sin_tab = vec![0.0f32; seq * head_dim];
    for s in 0..seq {
        for k in 0..half {
            let inv_freq =
                1.0f32 / rope_freq_base.powf(2.0 * (k as f32) / (head_dim as f32));
            let theta = (s as f32) * inv_freq;
            let (sin_t, cos_t) = theta.sin_cos();
            cos_tab[s * head_dim + 2 * k] = cos_t;
            cos_tab[s * head_dim + 2 * k + 1] = cos_t;
            sin_tab[s * head_dim + 2 * k] = sin_t;
            sin_tab[s * head_dim + 2 * k + 1] = sin_t;
        }
    }

    let cpu_ref = cpu_rope_reference(&q, head_dim, seq, RopeStyle::Adjacent, rope_freq_base);
    let gpu_out = dispatch_forward_kernel(&cfg, &q, &cos_tab, &sin_tab, seq);
    let mre = max_rel_err(&gpu_out, &cpu_ref);
    assert!(
        mre < 1e-3,
        "Adjacent RoPE: max_rel_err = {mre:.3e} exceeds 1e-3 threshold"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "non-CSHA v2 rope_q forward launcher not yet wired at FFI level \
            (structural fix guarded by csha_cycle19_qload_rope_partner.rs)"]
fn qload_rope_halfsplit_matches_cpu_reference() {
    if !cuda_available() {
        eprintln!("skipping — CUDA not available in this environment");
        return;
    }
    let cfg = base_config(RopeStyle::HalfSplit);
    let ptx = synthesize_flash_attention_ptx_v2(&cfg);
    assert!(!ptx.is_empty(), "PTX must synthesize");

    let head_dim = cfg.head_dim as usize;
    let seq = 8usize;
    let rope_freq_base = 10_000.0f32;

    let q: Vec<f32> = (0..seq * head_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    // HalfSplit cos/sin layout: [seq, head_dim] where cos[s, k] and
    // cos[s, k+half] both hold cos(theta_k(s)).
    let half = head_dim / 2;
    let mut cos_tab = vec![0.0f32; seq * head_dim];
    let mut sin_tab = vec![0.0f32; seq * head_dim];
    for s in 0..seq {
        for k in 0..half {
            let inv_freq =
                1.0f32 / rope_freq_base.powf(2.0 * (k as f32) / (head_dim as f32));
            let theta = (s as f32) * inv_freq;
            let (sin_t, cos_t) = theta.sin_cos();
            cos_tab[s * head_dim + k] = cos_t;
            cos_tab[s * head_dim + k + half] = cos_t;
            sin_tab[s * head_dim + k] = sin_t;
            sin_tab[s * head_dim + k + half] = sin_t;
        }
    }

    let cpu_ref = cpu_rope_reference(&q, head_dim, seq, RopeStyle::HalfSplit, rope_freq_base);
    let gpu_out = dispatch_forward_kernel(&cfg, &q, &cos_tab, &sin_tab, seq);
    let mre = max_rel_err(&gpu_out, &cpu_ref);
    assert!(
        mre < 1e-3,
        "HalfSplit RoPE: max_rel_err = {mre:.3e} exceeds 1e-3 threshold"
    );
}

#[test]
fn qload_rope_ptx_synthesizes_both_styles() {
    // Non-ignored guard: at minimum PTX must synthesize cleanly for both
    // styles under (rope_q=true, csha=None). This exercises the changed
    // emit path without requiring a GPU launcher.
    for style in [RopeStyle::Adjacent, RopeStyle::HalfSplit] {
        let cfg = base_config(style);
        let ptx = synthesize_flash_attention_ptx_v2(&cfg);
        assert!(
            !ptx.is_empty(),
            "PTX must synthesize for rope_style={style:?}"
        );
        // Sanity: the RoPE block appears.
        let s = String::from_utf8_lossy(&ptx);
        let marker = match style {
            RopeStyle::Adjacent => "// rope adjacent slice 0",
            RopeStyle::HalfSplit => "// rope halfsplit slice 0",
        };
        assert!(
            s.contains(marker),
            "PTX must contain RoPE marker {marker:?}"
        );
    }
}

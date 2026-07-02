//! CSHA cycle 20 T4 — G16-3 forward +0xB30 triage probe harness.
//!
//! # Purpose
//!
//! Cycle 18 T5 identified 1024 sm_120 compute-sanitizer violations at
//! FORWARD kernel offset `+0xB30` under the config quartet
//!   `(save_activations=true, !fused_projections, d_model=0, head_dim=64)`.
//!
//! The violation pattern was `0x5 + 0xA * lane` — odd byte offsets that
//! suggest misaligned f16 shared reads. Cycle 18 SPECULATED the culprit
//! was `emit_save_activations_subset` at csha_hooks.rs:953
//! (`ld.shared.b16 %h_save_v, [%rd_save_smem]`).
//!
//! Phase A I3 REFUTED that hypothesis for the backward analogue
//! (3-way alias affects backward only). Static-analysis audit of the
//! forward save-activation address math shows that for
//! `head_dim=64`, `slices_per_lane=2`:
//!
//!   `%rd_save_smem = %<qkv>_smem_base
//!                     + wrow * 128        (head_dim*2, EVEN)
//!                     + col  * 2          (per-lane f16 stride, EVEN)`
//!
//! where `%<qkv>_smem_base` derives from `cvta.shared.u64 %shmem_base,
//! shmem` and the `.shared .align 16 .b8 shmem[]` declaration guarantees
//! the base is 16-byte aligned. Every addend is even, so the composed
//! address is even, so `ld.shared.b16` cannot be misaligned from this
//! emitter.
//!
//! The `0x5 + 0xA * lane` pattern (10 bytes per lane, offset 5) does
//! NOT match the per-lane stride of the save-activation SMEM read (2
//! bytes per lane column stride, since col = lane * slices_per_lane +
//! slice = lane*2 + slice; SMEM byte stride = col*2 = lane*4 + slice*2).
//!
//! # Classification (see triage report in commit message)
//!
//! Cycle 20 T4 concludes this is most likely **outcome (d) — c18 primary
//! candidate was wrong; +0xB30 maps to a different emitter**.  The probe
//! scaffolding here lets a future cycle prove that on-device by dumping
//! the actual `%rd_save_smem` values for slice=0 sample sites.
//!
//! # Feature-gated compile
//!
//! This test is compiled only when both `--features csha_cycle20_g163_probe`
//! is passed (unlocks the emitter marker) AND `--ignored` is set (so CI
//! never runs a probe test by default). Feature-off runs of the codegen
//! test suite MUST leave `fa_v2_snapshots` at 25/25 byte-identical.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

/// Reproduce the G16-3 config: `save_activations=true`,
/// `fused_projections=false`, `d_model=0`, `head_dim=64`,
/// `block_q=64`, `block_kv=64`, `heads=1` (minimal). All other CSHA
/// knobs left at their defaults.
fn g16_3_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            fused_projections: false,
            save_activations_for_backward: true,
            d_model: 0,
            ..CshaExtras::default()
        }),
        checkpoint: None,
    }
}

/// Feature-off structural witness: the emitter marker MUST NOT appear in
/// the synthesized PTX when the probe feature is disabled. This locks
/// `fa_v2_snapshots` byte identity across the c20 T4 landing.
#[test]
fn c20_t4_probe_marker_absent_when_feature_off() {
    // Build the forward PTX by calling into the v2 emitter for the
    // G16-3 config. When the `csha_cycle20_g163_probe` feature is off,
    // no `CSHA_C20_G163_PROBE` marker should appear in the output.
    let cfg = g16_3_config();
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&cfg);
    // Trailing NUL from cuModuleLoadData contract — strip for str-search.
    let ptx = String::from_utf8_lossy(&ptx_bytes).into_owned();
    let probe_marker_count = ptx.matches("CSHA_C20_G163_PROBE").count();

    #[cfg(not(feature = "csha_cycle20_g163_probe"))]
    assert_eq!(
        probe_marker_count, 0,
        "feature-off build must not emit probe marker (fa_v2_snapshots byte-identity gate); \
         found {probe_marker_count} marker occurrences",
    );

    // When the feature is on, we do NOT assert here — a separate
    // `#[ignore]`d test below asserts the on-shape (so `cargo test`
    // default-run stays no-op under the feature).
    #[cfg(feature = "csha_cycle20_g163_probe")]
    {
        // NOTE: this branch is silent so that flipping the feature on
        // does not fail the fast structural test; the presence check
        // is on the `#[ignore]`d test.
        let _ = probe_marker_count;
    }
}

/// Feature-on witness: when the probe feature is enabled, exactly one
/// marker per (Q, K, V) label must appear at slice=0, q_tile_iter=0.
/// Only runs under `cargo test --features csha_cycle20_g163_probe -- --ignored`.
#[test]
#[ignore = "requires --features csha_cycle20_g163_probe and manual invocation"]
fn c20_t4_probe_marker_present_when_feature_on() {
    let cfg = g16_3_config();
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&cfg);
    // Trailing NUL from cuModuleLoadData contract — strip for str-search.
    let ptx = String::from_utf8_lossy(&ptx_bytes).into_owned();

    // We expect one marker per label × (q_tile_iter == 0) × slice == 0.
    // Under the standard non-fused path, `SaveSet::QK` fires per q_iter
    // (2 markers on q_iter=0: Q + K) and `SaveSet::V` fires per q_iter
    // (1 marker: V). Total on iter=0: 3.  Later q_iters do not emit
    // (probe gates on iter==0). Total: 3 markers.
    let count = ptx.matches("CSHA_C20_G163_PROBE").count();
    assert_eq!(
        count, 3,
        "feature-on build must emit exactly 3 probe markers for G16-3 \
         (Q@iter0, K@iter0, V@iter0 all at slice=0); got {count}\n\
         emit context:\n{}\n",
        ptx.lines()
            .filter(|l| l.contains("CSHA_C20_G163_PROBE"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Confirm each label appears once at slice=0/iter=0.
    for label in ["label=Q", "label=K", "label=V"] {
        assert!(
            ptx.contains(label),
            "missing probe marker for {label}",
        );
    }
}

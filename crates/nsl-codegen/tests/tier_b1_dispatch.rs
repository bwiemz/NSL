//! End-to-end dispatch test: confirms `synthesize_flash_attention_ptx_v2`
//! routes to `tier_b1::synthesize` when `csha.level >= 2` AND `gpu_sm >= 80`,
//! and falls through to Tier A otherwise.
//!
//! At B1.5 Task 5.3 stage, `chunk_config::select` performs a real descending
//! search over `{128, 64, 32, FLOOR}`, gated on SMEM (spec section 3.4) and
//! register budget (spec section 5.4). Eligible level=2 + sm>=80 configs whose
//! tile shape fits the 99 KB SMEM budget route into `tier_b1::synthesize`,
//! which now emits the full single-iteration scaffold (prelude + active_heads +
//! RMSNorm + Q projection + cp.async prologue kicks + Phase A/B/C + finalize)
//! and the "Tier B.1 single-iter scaffold complete" sentinel comment. B1.6
//! lands the multi-iter loop alongside the register-hoist refactor.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

/// Construct a Tier B.1 eligible config: `csha.level = 2` (Pipeline),
/// `gpu_sm = 120` (Blackwell), small tile sizes so the SMEM budget is
/// satisfied at multiple chunk candidates. `d_model = 2048` is the
/// smallest dm in the V3 supported-matrix CSV; selectors require a
/// non-zero d_model to size chunk staging.
fn tier_b1_eligible_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 2048,
            ..CshaExtras::default()
        }),
    }
}

#[test]
fn tier_b1_eligible_routes_to_synthesize_stub() {
    let config = tier_b1_eligible_config();
    let ptx = synthesize_flash_attention_ptx_v2(&config);
    let s = String::from_utf8_lossy(&ptx);
    // B1.5 Task 5.3 replaced the "Tier B.1 stub" placeholder with the full
    // single-iteration scaffold (prelude + active_heads + RMSNorm + Q proj +
    // cp.async prologue + Phase A/B/C + finalize). The sentinel comment at the
    // tail of synthesize() is the load-bearing marker for this test.
    // FLIP-POINT B1.6: when the multi-iter loop lands (with the register-hoist
    // refactor in attention_mma.rs Phase-B deferral #4), update this assertion
    // to match the new sentinel or the loop-control label.
    assert!(
        s.contains("Tier B.1 single-iter scaffold complete"),
        "B1.5 Task 5.3 ships the single-iter scaffold marker; B1.6 will replace this with the multi-iter form. Got: {}",
        &s[..s.len().min(200)]
    );
}

#[test]
fn level_1_config_does_not_route_to_tier_b1() {
    let mut config = tier_b1_eligible_config();
    // Drop to CSHA Level 1 (Boundary).  Dispatch predicate requires
    // `csha.level >= 2`, so level=1 must skip the tier_b1 branch
    // outright (no `chunk_config::select` call) and emit Tier A.
    config.csha = Some(CshaExtras {
        level: 1,
        ..CshaExtras::default()
    });
    let ptx = synthesize_flash_attention_ptx_v2(&config);
    let s = String::from_utf8_lossy(&ptx);
    // This assertion stays the same — level=1 will never route to tier_b1
    // regardless of milestone. Checked against the B1.5 scaffold sentinel.
    assert!(
        !s.contains("Tier B.1 single-iter scaffold complete"),
        "csha.level=1 must NOT route to tier_b1; got: {}",
        &s[..s.len().min(200)]
    );
}

#[test]
fn old_sm_does_not_route_to_tier_b1() {
    let mut config = tier_b1_eligible_config();
    // sm_75 is below the sm_80 cp.async/MMA gate.  Dispatch predicate
    // requires `gpu_sm >= 80`, so this must skip the tier_b1 branch
    // outright and emit Tier A.
    config.gpu_sm = 75;
    let ptx = synthesize_flash_attention_ptx_v2(&config);
    let s = String::from_utf8_lossy(&ptx);
    // This assertion stays the same — sm<80 will never route to tier_b1
    // regardless of milestone. Checked against the B1.5 scaffold sentinel.
    assert!(
        !s.contains("Tier B.1 single-iter scaffold complete"),
        "gpu_sm=75 must NOT route to tier_b1; got: {}",
        &s[..s.len().min(200)]
    );
}

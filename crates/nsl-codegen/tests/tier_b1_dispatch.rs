//! End-to-end dispatch test: confirms `synthesize_flash_attention_ptx_v2`
//! routes to `tier_b1::synthesize` when `csha.level >= 2` AND `gpu_sm >= 80`,
//! and falls through to Tier A otherwise.
//!
//! At B1.3 stage, `chunk_config::select` performs a real descending search
//! over `{128, 64, 32, FLOOR}`, gated on SMEM (spec §3.4) and register
//! budget (spec §5.4). Eligible level=2 + sm>=80 configs whose tile shape
//! fits the 99 KB SMEM budget now route into `tier_b1::synthesize`, which
//! at this stage emits a placeholder ("Tier B.1 stub") marker. B1.4+ lands
//! the real PTX emission.

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
    // B1.3 landed the real `chunk_config::select` (SMEM + register-budget
    // gates per spec §3.4 + §5.4). For this small 32x32x32 config it
    // succeeds, so dispatch routes into `tier_b1::synthesize`, which at
    // B1.3 emits the "Tier B.1 stub" placeholder marker. B1.4+ replaces
    // that placeholder with real PTX emission.
    // FLIP-POINT B1.4: when real emission lands the marker string will
    // change — update this assertion to match the new emitter output
    // (likely a kernel name like ".visible .entry flash_attn_tier_b1_*"
    // or a sentinel block header).
    assert!(
        s.contains("Tier B.1 stub"),
        "B1.3 chunk_config::select should now succeed and route to tier_b1::synthesize stub; got: {}",
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
    // FLIP-POINT B1.3: this assertion stays the same — level=1 will
    // never route to tier_b1 regardless of `chunk_config::select`'s state.
    assert!(
        !s.contains("Tier B.1 stub"),
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
    // FLIP-POINT B1.3: this assertion stays the same — sm<80 will
    // never route to tier_b1 regardless of `chunk_config::select`'s state.
    assert!(
        !s.contains("Tier B.1 stub"),
        "gpu_sm=75 must NOT route to tier_b1; got: {}",
        &s[..s.len().min(200)]
    );
}

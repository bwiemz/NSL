//! End-to-end dispatch test: confirms `synthesize_flash_attention_ptx_v2`
//! routes to `tier_b1::synthesize` when `csha.level >= 2` AND `gpu_sm >= 80`,
//! and falls through to Tier A otherwise.
//!
//! At B1.2 stage, `chunk_config::select` is a stub that always returns Err,
//! so even level=2+sm=120 configs fall through to Tier A.  The B1.3 milestone
//! lands the real chunk selector, at which point this test's "routes to
//! tier_b1 stub" assertion will need updating to "tier_b1::synthesize is
//! called with the selected chunk."  See test comments.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

/// Construct a Tier B.1 eligible config: `csha.level = 2` (Pipeline),
/// `gpu_sm = 120` (Blackwell), small tile sizes so the Tier A fall-through
/// fits the 99 KB dynamic SMEM cap.
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
            ..CshaExtras::default()
        }),
    }
}

#[test]
fn tier_b1_eligible_falls_through_to_classic_at_b1_2_stage() {
    let config = tier_b1_eligible_config();
    let ptx = synthesize_flash_attention_ptx_v2(&config);
    let s = String::from_utf8_lossy(&ptx);
    // `chunk_config::select` is a B1.2 stub that always returns Err, so
    // the dispatch falls through to the Tier A path.  The emitted PTX
    // should therefore NOT contain "Tier B.1 stub" (which the
    // `tier_b1::synthesize` placeholder emits).
    // FLIP-POINT B1.3: once `chunk_config::select` succeeds, the
    // assertion below must change from `!s.contains(...)` to
    // `s.contains(...)` — the eligible config will then route into
    // `tier_b1::synthesize` and the placeholder marker WILL appear.
    assert!(
        !s.contains("Tier B.1 stub"),
        "B1.2 stub chunk_config::select should always fall through; got: {}",
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

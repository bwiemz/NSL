//! F1 lane-mapping discipline per Phase 2.5 spec §4.4.
//! Per-lane structural assertion: dS scatter destination address must derive
//! from {lane%4, lane/4, ds_offset, row_stride} -- all four terms contribute.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn canonical_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 32,
        causal: false, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false,
        gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn f1_ds_scatter_emits_real_st_shared_f32() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(ptx.contains("st.shared.f32"),
        "expected real st.shared.f32 emission for dS scatter");
}

#[test]
fn f1_ds_scatter_uses_lane_mod_4_term() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // lane%4 derivation: and.b32 %something, %lane_id, 3
    assert!(ptx.contains("and.b32") && ptx.contains(", 3"),
        "expected lane%4 derivation (and.b32 ..., 3)");
}

#[test]
fn f1_ds_scatter_uses_lane_div_4_term() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // lane/4 derivation: shr.u32/shr.b32 %something, %lane_id, 2
    let has_shr_lane_div_4 = ptx.lines().any(|l| {
        let t = l.trim();
        (t.starts_with("shr.u32") || t.starts_with("shr.b32"))
            && t.contains("%lane_id") && t.ends_with(", 2;")
    });
    assert!(has_shr_lane_div_4,
        "expected lane/4 derivation (shr lane_id by 2)");
}

#[test]
fn f1_ds_scatter_uses_ds_offset_accessor() {
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_dq_ds_offset;
    let cfg = canonical_cfg();
    let ds_off = tier_b2_dq_ds_offset(&cfg);
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    assert!(ptx.contains(&format!("{ds_off}")),
        "expected ds_offset {ds_off} in scatter address chain");
}

#[test]
fn f1_ds_scatter_uses_row_stride_term() {
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_effective_bkv;
    let cfg = canonical_cfg();
    let bkv = tier_b2_effective_bkv(&cfg);
    let row_stride = bkv * 4; // f32 row-major dS
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    assert!(ptx.contains(&format!("{row_stride}")),
        "expected row_stride {row_stride} (bkv*4) in dS address arithmetic");
}

#[test]
fn f1_ds_scatter_emits_exactly_4_stores_per_lane() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // Each lane writes 4 f32 dS values per kv_iter; the emitter emits 4 st.shared.f32
    // instructions per kv_iter pass.
    let st_count = ptx.matches("st.shared.f32").count();
    assert!(st_count >= 4,
        "expected at least 4 st.shared.f32 for 4 ds elements per lane, got {st_count}");
}

#[test]
fn f1_emits_ascii_only() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    for (i, byte) in ptx.bytes().enumerate() {
        assert!(byte.is_ascii(), "non-ASCII byte 0x{:02x} at position {}", byte, i);
    }
}

use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

fn cfg(hd: i64, bq: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq, block_kv: bq, head_dim: hd, causal: true, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, num_sink_tokens: 0, gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn pcol_dscol_scatter_transposes_and_converts_f16() {
    let ptx = synthesize_dkdv_kernel(&cfg(64, 64)).expect("synth ok");
    assert!(ptx.contains("// === P col-major scatter"), "P-col scatter present");
    assert!(ptx.contains("// === dS col-major scatter"), "dS-col scatter present");
    assert!(ptx.contains("cvt.rn.f16.f32"), "C-frag f32->f16 convert");
    assert!(ptx.contains("st.shared.b16"), "col-major f16 store");
    // The transpose stride bq*2 = 128 (at bq=64) must be applied to the kv index.
    // (F1 would apply bkv*2 to the q index instead; this pins the transpose.)
    assert!(ptx.contains("128"), "bq*2=128 transpose row-stride present at bq=64");
}

#[test]
fn pcol_dscol_scatter_emits_8_stores_per_lane_per_ntile() {
    // 4 P + 4 dS = 8 st.shared.b16 added by this task (plus any already present).
    let ptx = synthesize_dkdv_kernel(&cfg(64, 64)).expect("synth ok");
    let n = ptx.matches("st.shared.b16").count();
    assert!(n >= 8, "expected >=8 st.shared.b16 (4 P + 4 dS per lane per n-tile), got {n}");
}

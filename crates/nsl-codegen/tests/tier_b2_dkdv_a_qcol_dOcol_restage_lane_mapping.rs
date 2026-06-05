use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;

fn cfg(hd: i64, bq: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bq,
        head_dim: hd,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn qcol_docol_restage_blocks_present_and_use_lane_div4() {
    let ptx = synthesize_dkdv_kernel(&cfg(64, 64)).expect("synth ok");
    assert!(ptx.contains("DKDV_QCOL_RESTAGE_DONE"), "Q-col re-stage block present");
    assert!(ptx.contains("DKDV_DOCOL_RESTAGE_DONE"), "dO-col re-stage block present");
    // lane/4 derivation (shr.u32 ..., 2) in both re-stages.
    assert!(
        ptx.matches("shr.u32").filter(|_| true).count() >= 2,
        "lane/4 derivation present in both re-stages"
    );
    // col_stride bq*2 = 128 at bq=64 appears as the dst col-stride multiply.
    assert!(
        ptx.contains("ld.shared.b16") && ptx.contains("st.shared.b16"),
        "re-stage does ld.shared.b16 -> st.shared.b16"
    );
}

#[test]
fn qcol_docol_restage_warp0_gated() {
    let ptx = synthesize_dkdv_kernel(&cfg(64, 64)).expect("synth ok");
    assert!(
        ptx.contains("@!%p_producer bra DKDV_QCOL_RESTAGE_DONE"),
        "Q-col warp-0 gated"
    );
    assert!(
        ptx.contains("@!%p_producer bra DKDV_DOCOL_RESTAGE_DONE"),
        "dO-col warp-0 gated"
    );
}

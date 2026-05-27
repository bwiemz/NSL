use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::proj_backward::synthesize_proj_backward;

fn smoke_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 64,
            active_heads: 1,
            ..Default::default()
        }),
    }
}

#[test]
fn proj_backward_has_entry_phases_and_no_grad_store() {
    let ptx = synthesize_proj_backward(&smoke_cfg()).expect("synth ok");
    assert!(
        ptx.contains(".visible .entry tier_b2_proj_backward"),
        "entry name"
    );
    assert!(ptx.contains("V2_BWD_DPROJ_WQ"), "emit_dproj reused");
    assert!(ptx.contains("V2_BWD_DRMSNORM"), "emit_drmsnorm reused");
    assert!(
        ptx.contains("V2_BWD_XNORM_RECOMPUTE"),
        "emit_xnorm_recompute reused"
    );
    assert!(ptx.contains("st.shared.f32"), "dQ/dK/dV staged (T2 load)");
    assert!(ptx.is_ascii(), "ASCII-only");
    // Read-only on dQ/dK/dV: this kernel must NOT store dQ/dK/dV back to HBM.
    // (emit_dproj/emit_drmsnorm store dW*/dx with st.global.b16/f32; that's expected.)
    // The dQ/dK/dV input pointer regs are %rd_bwd_dq / %rd_bwd_dk / %rd_bwd_dv.
    // Assert no st.global targets those pointer regs.
    for grad_reg in ["%rd_bwd_dq", "%rd_bwd_dk", "%rd_bwd_dv"] {
        for line in ptx.lines() {
            let t = line.trim_start();
            if t.starts_with("st.global") || t.starts_with("@%") && t.contains("st.global") {
                assert!(
                    !t.contains(grad_reg),
                    "proj-backward must be read-only on dQ/dK/dV: found store to {grad_reg}: {line}"
                );
            }
        }
    }
}

#[test]
fn proj_backward_rejects_non_multiple_of_32_head_dim() {
    let mut cfg = smoke_cfg();
    cfg.head_dim = 48;
    cfg.csha = Some(CshaExtras {
        level: 2,
        d_model: 48,
        active_heads: 1,
        ..Default::default()
    });
    let err = synthesize_proj_backward(&cfg).unwrap_err();
    use nsl_codegen::flash_attention_v2::tier_b2::backward::BackwardSynthError;
    assert_eq!(err, BackwardSynthError::UnsupportedHeadDim(48));
}

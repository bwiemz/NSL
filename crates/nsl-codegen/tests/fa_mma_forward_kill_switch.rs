//! Kill-switch coverage for the tensor-core SDPA forward, isolated in
//! its own test binary: `NSL_FA_FWD_MMA` is process-global state, and a
//! sibling test synthesizing PTX concurrently would race the override
//! (the exact failure mode the matmul math-mode suites solve the same way).

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2,
};

fn sdpa_config() -> FlashAttentionConfig {
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
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: None,
        checkpoint: None,
    }
}

#[test]
fn kill_switch_restores_scalar_body() {
    let prior = std::env::var("NSL_FA_FWD_MMA").ok();
    std::env::set_var("NSL_FA_FWD_MMA", "0");
    let cfg = sdpa_config();
    let name = flash_attention_kernel_name_v2(&cfg);
    let mut bytes = synthesize_flash_attention_ptx_v2(&cfg);
    bytes.pop();
    let ptx = String::from_utf8(bytes).unwrap();
    match prior {
        Some(v) => std::env::set_var("NSL_FA_FWD_MMA", v),
        None => std::env::remove_var("NSL_FA_FWD_MMA"),
    }
    assert_eq!(
        name, "flash_attn_p0_r0_hs_g1_c1_t0_q64_kv64_v2",
        "NSL_FA_FWD_MMA=0 must drop the _mma suffix"
    );
    assert_eq!(
        ptx.matches("mma.sync").count(),
        0,
        "NSL_FA_FWD_MMA=0 must restore the scalar body"
    );
}


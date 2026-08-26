//! NSL_FA_BWD_MULTIWARP=0 kill switch — in its OWN test binary because the
//! env var races with any parallel test that synthesizes backward PTX (the
//! same isolation the NSL_FA_FWD_MMA switch needed: 3 tests failed before it
//! got its own binary).

use nsl_codegen::flash_attention::{
    backward_select_blocks, flash_attention_bwd_main_kernel_name,
    synthesize_flash_attention_backward_ptx, FlashAttentionBackwardConfig,
};

/// With the switch off, the emission must revert to the historical
/// single-warp contract: unsuffixed kernel name (=> the runtime launches
/// block_q threads), no partition scaffolding, warp-0 accumulate gates back
/// in place, row-per-thread P section.
#[test]
fn killswitch_restores_single_warp_emission() {
    // SAFETY: this is the only test in this binary; nothing else reads the
    // var concurrently.
    unsafe { std::env::set_var("NSL_FA_BWD_MULTIWARP", "0") };

    let (block_q, block_kv) = backward_select_blocks(64);
    let config = FlashAttentionBackwardConfig {
        block_q,
        block_kv,
        head_dim: 64,
        causal: true,
        gpu_sm: 80,
        segment_masked: false,
    };
    assert_eq!(
        flash_attention_bwd_main_kernel_name(&config),
        "flash_attn_bwd_main_c1_q32_kv32",
        "kill switch must drop the _w4 launch suffix"
    );
    let (_p1, p2) = synthesize_flash_attention_backward_ptx(&config);
    let text = String::from_utf8_lossy(&p2);
    assert!(!text.contains("_w4"), "kill switch left _w4 in the module");
    assert!(
        !text.contains("_ARM0_END"),
        "kill switch left partition arms in the body"
    );
    assert!(
        text.contains("BWD_MAIN_MMA_P_NJ"),
        "kill switch must restore the row-per-thread P section"
    );
    assert!(
        text.contains("setp.ne.u32 %bwd_mma_pw, %bwd_mma_wid, 0;")
            || text.contains("shr.u32 %bwd_mma_m_tile, %bwd_mma_wid, 0;"),
        "single-warp body should keep the degenerate wid-derived init \
         (warps=1 emits shr-by-0), proving the shared code path"
    );

    unsafe { std::env::remove_var("NSL_FA_BWD_MULTIWARP") };
}

//! CSHA Tier B.1 — Level 2 pipelined attention forward kernel.
//!
//! Design spec: docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md
//!
//! Module map:
//!   * `pipeline`        - cp.async ping-pong helpers (commit_group, wait_group, slot mgmt)
//!   * `projection_mma`  - Q/K/V projection chunk-streamed MMA emitter
//!   * `attention_mma`   - QK^T + softmax + PV emitter
//!   * `chunk_config`    - d_model_chunk selection per variant
//!   * `register_budget` - static spill analysis pre-emission

pub mod pipeline;
pub mod projection_mma;
pub mod attention_mma;
pub mod chunk_config;
pub mod register_budget;

use crate::flash_attention::FlashAttentionConfig;

/// Top-level Tier B.1 PTX emitter. Called from
/// `flash_attention_v2::synthesize_flash_attention_ptx_v2` when the
/// dispatch criteria are met. Returns PTX bytes terminated with NL+NUL.
///
/// B1.3 status: emits prelude + active_heads guard + RMSNorm pre-pass +
/// Q projection (chunk-streamed via MMA m16n8k16). K/V projection,
/// cp.async ping-pong, Phase B attention (QK^T + softmax + PV), and the
/// final HBM write land in B1.4 + B1.5. Until then `synthesize` produces
/// a kernel body that ends in the `Tier B.1 stub` placeholder marker —
/// the marker is load-bearing for `tier_b1_eligible_routes_to_synthesize_stub`
/// in `tests/tier_b1_dispatch.rs`.
pub fn synthesize(config: &FlashAttentionConfig, chunk: u32) -> Vec<u8> {
    let mut ptx = String::new();

    // 1. PTX header (target, version, param block, register decls).
    //    Reused 1:1 from Tier A per V1 findings — the 37-slot param
    //    block is ABI-compatible across both tiers.
    crate::flash_attention_v2::phases::forward::prelude::emit(&mut ptx, config);

    // 2. CSHA A.4 active_heads guard (reused from Tier A unchanged).
    crate::flash_attention_v2::phases::forward::csha_hooks::emit_active_heads_guard(
        &mut ptx, config,
    );

    // 3. RMSNorm pre-pass. Each q_tile_iter normalises its rows into
    //    csha_x_ptr. 8-warp partition extension (vs Tier A's 4-warp)
    //    is deferred to B1.5; for B1.3 we reuse the 4-warp impl.
    let iters = (config.block_q as u32).div_ceil(4);
    for q_iter in 0..iters {
        crate::flash_attention_v2::phases::forward::csha_hooks::emit_prologue(
            &mut ptx, config, q_iter,
        );
    }

    // 4. Q projection (B1.3 Task 3.3): chunk-streamed MMA m16n8k16
    //    accumulating into per-warp Q registers, then packed + scattered
    //    to the Q SMEM tile at tier_b1_q_offset.
    projection_mma::emit_q_projection(&mut ptx, config, chunk);

    // 5. Placeholder for B1.4 (cp.async ping-pong pipeline) + B1.5
    //    (K/V projection + Phase B attention + finalize). The "Tier B.1
    //    stub" marker remains load-bearing for the dispatch test until
    //    B1.5 replaces it with the kernel finalisation tail.
    ptx.push_str("    // Tier B.1 stub - K/V projection + cp.async ping-pong + attention land in B1.4 + B1.5\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

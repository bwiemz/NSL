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
/// Stub through B1.3: returns a placeholder PTX string so dispatch
/// routing is testable. B1.3 wired the real `chunk_config::select` +
/// `register_budget::analyze` upstream — eligible configs now reach
/// this function — but the PTX body itself remains a placeholder. Real
/// emission lands in B1.4 (cp.async pipeline helpers) and B1.5 (main
/// FSM orchestration).
pub fn synthesize(config: &FlashAttentionConfig, chunk: u32) -> Vec<u8> {
    let _ = (config, chunk);
    let mut ptx = String::from("// Tier B.1 stub - real emission lands in B1.4 + B1.5\n");
    ptx.push('\n');
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

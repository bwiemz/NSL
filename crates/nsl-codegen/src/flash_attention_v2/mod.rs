//! FlashAttention-2 scalar-path emitter v2.
//!
//! Replaces the structurally incorrect v1 scalar forward path with a
//! warp-per-row thread-mapping contract. See
//! `docs/superpowers/specs/2026-04-14-fa-scalar-emitter-rewrite-design.md`
//! for the phase-level algorithm and constraints.
//!
//! Routed via `flash_attention_selector::select_emitter` when
//! `NSL_FA_EMITTER=v2` and `gpu_sm < 80`. The MMA path (sm>=80) stays on
//! v1 until a separate spec covers MMA correctness.

pub mod smem_layout;
pub mod register_budget;
pub mod phases;

use crate::flash_attention::FlashAttentionConfig;

/// v2 entry point. Returns a byte vector ending with a single trailing
/// newline followed by a NUL terminator so `cuModuleLoadData` accepts it.
pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig) -> Vec<u8> {
    smem_layout::validate_scalar_v2_config(config)
        .expect("v2 emitter called with unsupported config — selector must gate this");
    // Populated by later tasks. For Task 1 the function is unreachable
    // because the selector defaults to v1.
    unimplemented!("v2 orchestrator lands in Task 11");
}

/// Kernel entry-point name for v2. Same format as v1 with a `_v2` suffix
/// so module caches never collide between versions.
pub fn flash_attention_kernel_name_v2(config: &FlashAttentionConfig) -> String {
    format!("{}_v2", crate::flash_attention::flash_attention_kernel_name(config))
}

/// SMEM byte count for a v2 kernel. Computed from the layout module so
/// static-shmem declaration and launch-arg stay in sync.
pub fn shared_mem_bytes_v2(config: &FlashAttentionConfig) -> u32 {
    smem_layout::total_bytes(config)
}

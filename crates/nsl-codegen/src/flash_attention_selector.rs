//! Selects the FlashAttention-2 scalar emitter.
//!
//! v1 was deleted in Task C5. v2 is the sole scalar emitter.  Out-of-matrix
//! configs are rejected at compile time (panic) instead of silently falling
//! back to v1.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2, synthesize_flash_attention_ptx_v2,
    smem_layout,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emitter { V2 }

/// Select the emitter, validating the config against the v2 scalar matrix.
///
/// Panics if the config is rejected by `validate_scalar_v2_config` — there is
/// no fallback now that v1 is deleted.  Out-of-matrix configs must be fixed at
/// the call site (reduce block sizes, head_dim, or target sm>=80 MMA path).
///
/// `diagnostics` is accepted for API stability with existing call sites; it
/// will always be empty on success (panic on failure means it never gets
/// populated).
pub fn select_emitter_with_diag(
    config: &FlashAttentionConfig,
    diagnostics: &mut Vec<String>,
) -> Emitter {
    let _ = diagnostics; // v1 fallback diagnostics no longer apply
    if let Err(e) = smem_layout::validate_scalar_v2_config(config) {
        panic!(
            "v2 rejected config (block_q={}, block_kv={}, head_dim={}): {}; \
             v1 is deleted — fix the config or use a supported matrix entry",
            config.block_q, config.block_kv, config.head_dim, e
        );
    }
    Emitter::V2
}

/// Legacy entry point — routes through `select_emitter_with_diag`.
pub fn select_emitter(config: &FlashAttentionConfig) -> Emitter {
    select_emitter_with_diag(config, &mut Vec::new())
}

/// Synthesise PTX.  Validates the config; panics on out-of-matrix configs.
/// `diagnostics` retained for call-site API stability.
pub fn synthesize_flash_attention_ptx_selected_with_diag(
    config: &FlashAttentionConfig,
    diagnostics: &mut Vec<String>,
) -> Vec<u8> {
    let _ = select_emitter_with_diag(config, diagnostics); // panics on reject
    synthesize_flash_attention_ptx_v2(config)
}

/// Return the kernel name for any config (no SMEM budget check — planning can
/// compute names for configs outside the 48 KB budget so the planner can
/// reason about them).  `diagnostics` retained for API stability.
pub fn flash_attention_kernel_name_selected_with_diag(
    config: &FlashAttentionConfig,
    _diagnostics: &mut Vec<String>,
) -> String {
    flash_attention_kernel_name_v2(config)
}

/// Return shared-memory bytes for any config (no SMEM budget check — planning
/// functions need sizes for configs outside budget to make tile-size decisions).
/// `diagnostics` retained for API stability.
pub fn shared_mem_bytes_selected_with_diag(
    config: &FlashAttentionConfig,
    _diagnostics: &mut Vec<String>,
) -> u32 {
    shared_mem_bytes_v2(config)
}

/// Back-compat shim.
pub fn synthesize_flash_attention_ptx_selected(config: &FlashAttentionConfig) -> Vec<u8> {
    synthesize_flash_attention_ptx_selected_with_diag(config, &mut Vec::new())
}

/// Back-compat shim.
pub fn flash_attention_kernel_name_selected(config: &FlashAttentionConfig) -> String {
    flash_attention_kernel_name_selected_with_diag(config, &mut Vec::new())
}

/// Back-compat shim.
pub fn shared_mem_bytes_selected(config: &FlashAttentionConfig) -> u32 {
    shared_mem_bytes_selected_with_diag(config, &mut Vec::new())
}

#[cfg(test)]
mod selector_tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};
    use std::sync::{Mutex, MutexGuard};

    // Serialise all tests that mutate process state to avoid cross-test interference.
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cfg(block_q: i64, block_kv: i64, head_dim: i64, gpu_sm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q,
            block_kv,
            head_dim,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm,
            csha: None,
        }
    }

    // ---- surviving tests ----

    #[test]
    fn a2_in_matrix_config_routes_to_v2() {
        let _guard = lock_env();
        // block_q=32, block_kv=32, head_dim=32: SMEM fits easily.
        let config = cfg(32, 32, 32, 75);
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V2);
        assert!(diagnostics.is_empty(), "no diagnostic expected on valid config");
    }

    #[test]
    fn c4_default_is_v2_when_env_unset_and_config_valid() {
        let _guard = lock_env();
        let config = cfg(32, 32, 32, 75);
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V2, "selector must return V2 on valid config");
        assert!(diagnostics.is_empty(), "no diagnostic on valid config");
    }

    // sm >= 80 now also routes to v2 scalar (MMA forward spec pending).
    #[test]
    fn sm80_config_routes_to_v2_scalar() {
        let _guard = lock_env();
        let config = cfg(32, 32, 32, 80);
        let result = select_emitter(&config);
        assert_eq!(result, Emitter::V2, "sm>=80 routes to v2 scalar until MMA forward spec lands");
    }

    // ---- C5 new test ----

    #[test]
    #[should_panic(expected = "v2 rejected config")]
    fn c5_out_of_matrix_config_panics_after_v1_deletion() {
        let _guard = lock_env();
        // block_q=128, block_kv=128, head_dim=128 — SMEM budget exceeded;
        // v2 rejects it, and v1 no longer exists as a fallback.
        let config = cfg(128, 128, 128, 75);
        let mut diagnostics = vec![];
        let _ = select_emitter_with_diag(&config, &mut diagnostics);
    }
}

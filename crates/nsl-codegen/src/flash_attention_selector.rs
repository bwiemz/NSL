//! Selects between v1 and v2 FlashAttention-2 emitters.

use crate::flash_attention::{
    FlashAttentionConfig, flash_attention_kernel_name as v1_kernel_name,
    shared_mem_bytes as v1_shared_mem, synthesize_flash_attention_ptx as v1_synth,
    use_mma_path,
};
use crate::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2, synthesize_flash_attention_ptx_v2,
    smem_layout,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emitter { V1, V2 }

/// Per-compilation-unit dedup state for v2 fallback warnings.
///
/// Keyed by `(block_q, block_kv, head_dim)` so each unique config triple
/// emits at most one diagnostic per compile.  Lives on the `Compiler`
/// struct (see `compiler/mod.rs`) — never static or module-global.
#[derive(Default)]
pub struct FallbackSeen {
    keys: std::collections::HashSet<(i64, i64, i64)>,
}

impl FallbackSeen {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Select v1 or v2, pre-checking `validate_scalar_v2_config` before
/// routing to v2.  On validator rejection, falls back to v1 and appends
/// a single diagnostic to `diagnostics` (one per unique triple, no dedup).
///
/// Callers that need dedup across multiple invocations should use
/// `select_emitter_with_dedup` with a `FallbackSeen` kept on the compiler.
pub fn select_emitter_with_diag(
    config: &FlashAttentionConfig,
    diagnostics: &mut Vec<String>,
) -> Emitter {
    let mut seen = FallbackSeen::new();
    select_emitter_with_dedup(config, diagnostics, &mut seen)
}

/// Select v1 or v2 with dedup: at most one warning per unique
/// `(block_q, block_kv, head_dim)` triple per `seen` lifetime.
///
/// `seen` should be owned by the `Compiler` so dedup spans the whole
/// compilation unit.
pub fn select_emitter_with_dedup(
    config: &FlashAttentionConfig,
    diagnostics: &mut Vec<String>,
    seen: &mut FallbackSeen,
) -> Emitter {
    // MMA path is not this spec's concern — stays on v1 until MMA spec lands.
    if use_mma_path(config.gpu_sm) {
        return Emitter::V1;
    }
    let env_v2 = std::env::var("NSL_FA_EMITTER").ok().as_deref() == Some("v2");
    if !env_v2 {
        return Emitter::V1; // default = v1 until Task 15 flips it
    }
    // Pre-check validator before routing to v2.
    match smem_layout::validate_scalar_v2_config(config) {
        Ok(()) => Emitter::V2,
        Err(e) => {
            let key = (config.block_q, config.block_kv, config.head_dim);
            if seen.keys.insert(key) {
                diagnostics.push(format!(
                    "fa emitter: v2 rejected config (block_q={}, block_kv={}, head_dim={}): {}; falling back to v1",
                    config.block_q, config.block_kv, config.head_dim, e
                ));
            }
            Emitter::V1
        }
    }
}

/// Legacy entry point — routes through `select_emitter_with_diag` with a
/// throwaway diagnostic buffer.  Retained for call sites that have no
/// compiler context; prefer the `_with_dedup` variant where possible.
pub fn select_emitter(config: &FlashAttentionConfig) -> Emitter {
    select_emitter_with_diag(config, &mut Vec::new())
}

pub fn synthesize_flash_attention_ptx_selected(config: &FlashAttentionConfig) -> Vec<u8> {
    match select_emitter(config) {
        Emitter::V1 => v1_synth(config),
        Emitter::V2 => synthesize_flash_attention_ptx_v2(config),
    }
}

pub fn flash_attention_kernel_name_selected(config: &FlashAttentionConfig) -> String {
    match select_emitter(config) {
        Emitter::V1 => v1_kernel_name(config),
        Emitter::V2 => flash_attention_kernel_name_v2(config),
    }
}

pub fn shared_mem_bytes_selected(config: &FlashAttentionConfig) -> u32 {
    match select_emitter(config) {
        Emitter::V1 => v1_shared_mem(config),
        Emitter::V2 => shared_mem_bytes_v2(config),
    }
}

#[cfg(test)]
mod selector_tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

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

    #[test]
    fn a2_out_of_matrix_config_falls_back_to_v1_with_diagnostic() {
        // head_dim=128, block_q=128, block_kv=128 is in the allowed lists —
        // but the SMEM total (128*128*2 + 128*128*2 + 4*128*4 = 67584) exceeds
        // the 48 KB budget, so validate_scalar_v2_config rejects it.
        std::env::set_var("NSL_FA_EMITTER", "v2");
        let config = cfg(128, 128, 128, 75);
        let mut diagnostics = Vec::<String>::new();
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V1, "expected fallback to v1");
        assert!(
            diagnostics.iter().any(|d|
                d.contains("v2 rejected") &&
                d.contains("block_q=128") &&
                d.contains("block_kv=128") &&
                d.contains("head_dim=128")
            ),
            "diagnostic should name the triple, got {diagnostics:?}"
        );
    }

    #[test]
    fn a2_diagnostic_deduped_per_config() {
        std::env::set_var("NSL_FA_EMITTER", "v2");
        let config = cfg(128, 128, 128, 75);
        let mut diagnostics = Vec::<String>::new();
        let mut seen = FallbackSeen::new();
        for _ in 0..10 {
            select_emitter_with_dedup(&config, &mut diagnostics, &mut seen);
        }
        assert_eq!(diagnostics.len(), 1, "expected dedup to emit only one warning");
    }

    #[test]
    fn a2_in_matrix_config_routes_to_v2() {
        // block_q=32, block_kv=32, head_dim=32: SMEM = 32*32*2 + 32*32*2 + 4*32*4 = 4608 bytes → fits
        std::env::set_var("NSL_FA_EMITTER", "v2");
        let config = cfg(32, 32, 32, 75);
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V2);
        assert!(diagnostics.is_empty(), "no diagnostic expected on valid config");
    }

    #[test]
    fn a2_env_unset_keeps_current_default() {
        // Today's default is v1; C4 will flip this. A2 must not change the default.
        std::env::remove_var("NSL_FA_EMITTER");
        let config = cfg(32, 32, 32, 75);
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V1, "default (env unset) is still v1 before C4");
    }
}

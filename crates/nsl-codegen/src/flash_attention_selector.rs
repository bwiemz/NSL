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
/// a single diagnostic to `diagnostics`.
///
/// NOTE: this creates a fresh `FallbackSeen` per call, so dedup only
/// applies within a single invocation.  For cross-call dedup (e.g., across
/// a full compilation unit) use `select_emitter_with_dedup` with a
/// compiler-owned `FallbackSeen`.
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
    // v2 is the default; v1 reachable only via explicit opt-out.
    let env_choice = std::env::var("NSL_FA_EMITTER").ok();
    let env_v1 = env_choice.as_deref() == Some("v1");
    let want_v2 = !env_v1;
    // Keep gpu_sm >= 80 → v1 routing (MMA path) until that check is removed separately.
    if !want_v2 {
        return Emitter::V1;
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

/// Synthesise PTX, threading diagnostics through a compiler-owned `FallbackSeen`
/// for cross-call dedup.  Prefer this over `synthesize_flash_attention_ptx_selected`
/// at any call site that has a `Compiler` (or equivalent) in scope.
pub fn synthesize_flash_attention_ptx_selected_with_diag(
    config: &FlashAttentionConfig,
    seen: &mut FallbackSeen,
    diagnostics: &mut Vec<String>,
) -> Vec<u8> {
    match select_emitter_with_dedup(config, diagnostics, seen) {
        Emitter::V1 => v1_synth(config),
        Emitter::V2 => synthesize_flash_attention_ptx_v2(config),
    }
}

/// Return the kernel name, threading diagnostics through a compiler-owned
/// `FallbackSeen` for cross-call dedup.
pub fn flash_attention_kernel_name_selected_with_diag(
    config: &FlashAttentionConfig,
    seen: &mut FallbackSeen,
    diagnostics: &mut Vec<String>,
) -> String {
    match select_emitter_with_dedup(config, diagnostics, seen) {
        Emitter::V1 => v1_kernel_name(config),
        Emitter::V2 => flash_attention_kernel_name_v2(config),
    }
}

/// Return shared-memory bytes, threading diagnostics through a compiler-owned
/// `FallbackSeen` for cross-call dedup.
pub fn shared_mem_bytes_selected_with_diag(
    config: &FlashAttentionConfig,
    seen: &mut FallbackSeen,
    diagnostics: &mut Vec<String>,
) -> u32 {
    match select_emitter_with_dedup(config, diagnostics, seen) {
        Emitter::V1 => v1_shared_mem(config),
        Emitter::V2 => shared_mem_bytes_v2(config),
    }
}

/// Back-compat shim — routes through a throwaway diagnostic buffer.
/// Retained for test helpers and call sites without compiler context;
/// prefer `*_with_diag` variants in production code.
pub fn synthesize_flash_attention_ptx_selected(config: &FlashAttentionConfig) -> Vec<u8> {
    synthesize_flash_attention_ptx_selected_with_diag(config, &mut FallbackSeen::new(), &mut Vec::new())
}

/// Back-compat shim — routes through a throwaway diagnostic buffer.
pub fn flash_attention_kernel_name_selected(config: &FlashAttentionConfig) -> String {
    flash_attention_kernel_name_selected_with_diag(config, &mut FallbackSeen::new(), &mut Vec::new())
}

/// Back-compat shim — routes through a throwaway diagnostic buffer.
pub fn shared_mem_bytes_selected(config: &FlashAttentionConfig) -> u32 {
    shared_mem_bytes_selected_with_diag(config, &mut FallbackSeen::new(), &mut Vec::new())
}

#[cfg(test)]
mod selector_tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};
    use std::sync::{Mutex, MutexGuard};

    // Serialise all tests that touch NSL_FA_EMITTER to avoid cross-test env interference.
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

    #[test]
    fn a2_out_of_matrix_config_falls_back_to_v1_with_diagnostic() {
        let _guard = lock_env();
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
        let _guard = lock_env();
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
        let _guard = lock_env();
        // block_q=32, block_kv=32, head_dim=32: SMEM = 32*32*2 + 32*32*2 + 4*32*4 = 4608 bytes → fits
        std::env::set_var("NSL_FA_EMITTER", "v2");
        let config = cfg(32, 32, 32, 75);
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V2);
        assert!(diagnostics.is_empty(), "no diagnostic expected on valid config");
    }

    #[test]
    fn a2_wired_call_site_emits_eprintln_via_compiler_state() {
        let _guard = lock_env();
        // Verify the new _with_diag wrappers thread FallbackSeen correctly:
        // calling the same out-of-matrix config twice must produce exactly one
        // diagnostic across both calls (cross-call dedup via shared FallbackSeen).
        std::env::set_var("NSL_FA_EMITTER", "v2");
        let config = cfg(128, 128, 128, 75);
        let mut seen = FallbackSeen::new();
        let mut diags = Vec::<String>::new();
        // First call — should produce one diagnostic.
        let _ = synthesize_flash_attention_ptx_selected_with_diag(&config, &mut seen, &mut diags);
        // Second call — dedup should suppress the second warning.
        let _ = synthesize_flash_attention_ptx_selected_with_diag(&config, &mut seen, &mut diags);
        assert_eq!(diags.len(), 1, "cross-call dedup via shared FallbackSeen should collapse to one warning");
        assert!(
            diags[0].contains("block_q=128") && diags[0].contains("block_kv=128") && diags[0].contains("head_dim=128"),
            "diagnostic should include the config triple, got: {:?}", diags[0]
        );
    }

    #[test]
    fn a2_env_unset_keeps_current_default() {
        let _guard = lock_env();
        // C4 flipped the default to v2. Env-unset now routes to v2 on valid configs.
        std::env::remove_var("NSL_FA_EMITTER");
        let config = cfg(32, 32, 32, 75);
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&config, &mut diagnostics);
        assert_eq!(result, Emitter::V2, "default (env unset) is v2 after C4");
    }

    #[test]
    fn c4_default_is_v2_when_env_unset_and_config_valid() {
        let _guard = lock_env();
        std::env::remove_var("NSL_FA_EMITTER");
        let mut cfg = cfg(32, 32, 32, 75);
        cfg.block_q = 32; cfg.block_kv = 32; cfg.head_dim = 32; cfg.gpu_sm = 75;
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&cfg, &mut diagnostics);
        assert_eq!(result, Emitter::V2, "env-unset default should be v2 post-C4");
        assert!(diagnostics.is_empty(), "no diagnostic expected on valid config");
    }

    #[test]
    fn c4_explicit_v1_opt_out_still_works() {
        let _guard = lock_env();
        std::env::set_var("NSL_FA_EMITTER", "v1");
        let mut cfg = cfg(32, 32, 32, 75);
        cfg.block_q = 32; cfg.block_kv = 32; cfg.head_dim = 32; cfg.gpu_sm = 75;
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&cfg, &mut diagnostics);
        assert_eq!(result, Emitter::V1, "explicit NSL_FA_EMITTER=v1 must still route to v1");
        std::env::remove_var("NSL_FA_EMITTER");
    }

    #[test]
    fn c4_default_still_falls_back_to_v1_on_v2_reject() {
        let _guard = lock_env();
        std::env::remove_var("NSL_FA_EMITTER");
        let mut cfg = cfg(128, 128, 128, 75);
        cfg.block_q = 128; cfg.block_kv = 128; cfg.head_dim = 128; cfg.gpu_sm = 75;
        // block_q=128, block_kv=128, head_dim=128 exceeds SMEM budget → v2 rejected.
        let mut diagnostics = vec![];
        let result = select_emitter_with_diag(&cfg, &mut diagnostics);
        assert_eq!(result, Emitter::V1, "v2-rejected config should still fall back to v1");
        assert_eq!(diagnostics.len(), 1, "fallback should emit one diagnostic");
        assert!(diagnostics[0].contains("block_q=128") && diagnostics[0].contains("head_dim=128"));
    }
}

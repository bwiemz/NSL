//! PCA Tier B — codegen-side dispatch helpers + sentinel construction.
//!
//! Per planner spec `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §5
//! and (α) commitment from P-2 (runtime is the source of truth for the two empirical
//! constants since `nsl-codegen` already depends on `nsl-runtime`).
//!
//! ## Codegen-side gate (§5)
//!
//! Decides whether to emit a Tier-B-on PTX variant alongside the base PTX for each
//! FA-2 kernel emission. The case-(β-ii) collapse from the dispatch spec's §14
//! amendment: at codegen, the heuristic reduces to `config.segment_masked`; the
//! seq_len floor + max-baked gate is applied at runtime by the launch wrapper
//! (per planner spec §6 + `nsl_runtime::pca_tier_b_runtime`).
//!
//! ## Cranelift-side sentinels (§4.2)
//!
//! Production call sites in `compiler/kernel.rs` emit the Tier B variant pointer
//! pair (`tier_b_ptx_ptr`, `tier_b_name_ptr`) via [`tier_b_disabled_sentinel`] /
//! [`tier_b_enabled`], NEVER inline `0, 0` literals. The runtime-side
//! `assert_tier_b_sentinels` panic enforces this structurally at FFI entry.
//!
//! Production call-site migrations at `kernel.rs:750, 1050, 1173` are P-3c scope;
//! this module only declares the helpers.
//!
//! ## Naming discrepancy: `SegmentResidency::Tiled` vs `::Shared`
//!
//! The dispatch spec (§7) and planner spec (§5.4) reference `SegmentResidency::Tiled`
//! as the default residency for the Tier-B-on emission. The current enum (in
//! `pca_segment.rs:19`) only has `Shared` and `Streamed`. The bench harness and all
//! existing Tier-B-on callers use `SegmentResidency::Shared` (see
//! `src/bin/bench/launch.rs:335,553,815,1103`). Per the dispatch spec §7.1 — "the
//! heuristic emits what was measured" — `Shared` IS what M2/M6 verified, so we use
//! that variant here. The eventual rename to `Tiled` is a spec-§9 v2 trigger
//! (per-config residency selection).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2_with_tier_b,
};
use crate::pca_segment::SegmentResidency;

/// Re-exported from `nsl_runtime::pca_tier_b_runtime` (the (α) single source of truth).
/// See the runtime crate for the findings-doc citations and const assertions.
pub use nsl_runtime::pca_tier_b_runtime::{TIER_B_MAX_BAKED_SEQ_LEN, TIER_B_SEQ_LEN_FLOOR};

/// Codegen-time gate: should this config's emission include a Tier-B-on PTX variant?
///
/// Returns `true` iff `config.segment_masked`. The seq_len floor + max-baked gates
/// are applied at runtime by the launch wrapper (per planner spec §6.1); this
/// codegen-side toggle is the case-(β-ii) collapse from the dispatch spec's §14
/// amendment.
pub fn should_emit_tier_b_at_codegen(config: &FlashAttentionConfig) -> bool {
    config.segment_masked
}

/// Result of codegen-side Tier B variant emission for a single FA-2 config.
///
/// The base PTX path is always populated (unchanged from PR #168/#169 behavior;
/// equivalent to `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)`).
/// The Tier-B-on path is populated iff [`should_emit_tier_b_at_codegen`].
pub struct TierBEmissionResult {
    pub base_ptx: Vec<u8>,
    pub base_kernel_name: String,
    /// `Some` iff [`should_emit_tier_b_at_codegen`].
    pub tier_b_on_ptx: Option<Vec<u8>>,
    pub tier_b_on_kernel_name: Option<String>,
}

/// Emit base + (optional) Tier-B-on PTX variants for a config.
///
/// Single edit point for the codegen emission policy (planner spec §5.4).
/// Production callers in `crates/nsl-codegen/src/compiler/kernel.rs:750, 1050, 1173`
/// migrate to this helper at P-3c (planner spec's P-3.5 task).
pub fn emit_tier_b_variants_for_config(config: &FlashAttentionConfig) -> TierBEmissionResult {
    let base_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(config, None);
    let base_kernel_name = flash_attention_kernel_name_v2(config);

    let (tier_b_on_ptx, tier_b_on_kernel_name) = if should_emit_tier_b_at_codegen(config) {
        // Per planner spec §5.4: `SegmentResidency::Shared` is the M2/M6-measured
        // default (spec calls this `Tiled` — see module-level discrepancy note).
        let tier_b_args = Some((TIER_B_MAX_BAKED_SEQ_LEN, SegmentResidency::Shared));
        let on_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b_args);
        let on_name = flash_attention_kernel_name_v2_tier_b_on(config);
        (Some(on_ptx), Some(on_name))
    } else {
        (None, None)
    };

    TierBEmissionResult {
        base_ptx,
        base_kernel_name,
        tier_b_on_ptx,
        tier_b_on_kernel_name,
    }
}

/// Kernel-name for the Tier-B-on variant.
///
/// Encodes `TIER_B_MAX_BAKED_SEQ_LEN` per planner spec §5.5 to eliminate cross-PR,
/// cross-architecture, and debug-vs-release kernel-name-collision class. Example:
/// `flash_attn_p0_r0_hs_g1_c1_t0_q64_kv64_v2_tier_b_max16384`.
pub fn flash_attention_kernel_name_v2_tier_b_on(config: &FlashAttentionConfig) -> String {
    format!(
        "{}_tier_b_max{}",
        flash_attention_kernel_name_v2(config),
        TIER_B_MAX_BAKED_SEQ_LEN,
    )
}

// ── Cranelift-side sentinel helpers ──────────────────────────────────────────
//
// Per planner spec §4.2, production call sites in `compiler/kernel.rs` emit the
// Tier B sentinel pair via these helpers, NEVER inline `0, 0` literals. The
// runtime-side `assert_tier_b_sentinels` panic enforces the discipline
// structurally at FFI entry.
//
// The Cranelift idiom mirrors `compiler/main_entry.rs:456-459` and
// `compiler/declaration.rs:916-917` exactly:
//   - `module.declare_data_in_func(data_id, builder.func)` -> `GlobalValue`
//   - `builder.ins().symbol_value(cl_types::I64, gv)` -> `Value` (the address)
//   - `builder.ins().iconst(cl_types::I64, 0)` for null sentinels.

use cranelift_codegen::ir::{types as cl_types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataId, Module};

/// Emit sentinel pair `(0, 0)` indicating "no Tier B variant available."
///
/// Called by Cranelift-emitted production call sites whose codegen path didn't
/// produce a Tier-B-on PTX blob (non-`segment_masked` configs, or any path that
/// pre-dates this spec). The runtime-side `assert_tier_b_sentinels` panic
/// validates that both elements are zero when the first is.
pub fn tier_b_disabled_sentinel(builder: &mut FunctionBuilder<'_>) -> [Value; 2] {
    [
        builder.ins().iconst(cl_types::I64, 0),
        builder.ins().iconst(cl_types::I64, 0),
    ]
}

/// Emit sentinel pair `(ptx_addr, name_addr)` indicating Tier-B-on variant available.
///
/// Called by Cranelift-emitted production call sites whose codegen path produced
/// a Tier-B-on PTX blob, with `tier_b_ptx_data` / `tier_b_name_data` referencing
/// the embedded data sections (see `embed_raw_data` in `compiler/kernel.rs:1285`).
///
/// Generic over `M: Module` so this helper is callable from both `ObjectModule`
/// (production path) and any test-double module — matching the broader idiom in
/// `cranelift_module::Module` consumers.
pub fn tier_b_enabled<M: Module>(
    builder: &mut FunctionBuilder<'_>,
    module: &mut M,
    tier_b_ptx_data: DataId,
    tier_b_name_data: DataId,
) -> [Value; 2] {
    let ptx_gv = module.declare_data_in_func(tier_b_ptx_data, builder.func);
    let name_gv = module.declare_data_in_func(tier_b_name_data, builder.func);
    [
        builder.ins().symbol_value(cl_types::I64, ptx_gv),
        builder.ins().symbol_value(cl_types::I64, name_gv),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

    fn segment_masked_cfg() -> FlashAttentionConfig {
        // Mirrors the hand-built FlashAttentionConfig pattern used throughout
        // `flash_attention.rs` tests (e.g. `test_kernel_name_encoding`).
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
            segment_masked: true,
            csha: None,
        }
    }

    #[test]
    fn tier_b_on_kernel_name_has_max_baked_suffix() {
        let cfg = segment_masked_cfg();
        let base = flash_attention_kernel_name_v2(&cfg);
        let tier_b_on = flash_attention_kernel_name_v2_tier_b_on(&cfg);
        let expected_suffix = format!("_tier_b_max{}", TIER_B_MAX_BAKED_SEQ_LEN);
        assert!(
            tier_b_on.ends_with(&expected_suffix),
            "expected suffix {expected_suffix:?} on name {tier_b_on:?}"
        );
        assert_ne!(base, tier_b_on, "Tier-B-on must differ from base");
    }
}

//! Tier B.2 backward kernel emitter — Phase 3 implementation.
//!
//! Phase 3 implements the four-kernel plan:
//!   - D pre-pass (`d_prepass` module, Task 6)
//!   - dQ-kernel (Tasks 8-16)
//!   - dK/dV-kernel (Phase 3)
//!   - proj-backward (`proj_backward` module, Phase 3 T4)
//!
//! Contract: `synthesize_tier_b2_backward` concatenates the PTX output of
//! all four kernels into a single module.

pub mod d_prepass;
pub mod dkdv;
pub mod dq;
pub mod hbm_addr;
pub mod proj_backward;

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackwardSynthError {
    /// Phase 1 placeholder. Kept for the `NotImplemented` variant so that
    /// downstream match arms compile; unreachable in Phase 2+.
    NotImplemented,
    /// head_dim must be divisible by 32 for the warp-shfl reduction.
    UnsupportedHeadDim(u32),
    /// A config invariant the emitter relies on is violated (e.g. the dK/dV
    /// kernel requires effective_bq == effective_bkv so the warp-band register
    /// is a valid base for BOTH the q-indexed inner work and the kv-indexed
    /// accumulator). Carries a human-readable explanation.
    UnsupportedConfig(String),
}

impl std::fmt::Display for BackwardSynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "Tier B.2 backward emitter path not yet implemented"
            ),
            Self::UnsupportedHeadDim(hd) => write!(
                f,
                "Tier B.2 backward requires head_dim divisible by 32, got {}",
                hd
            ),
            Self::UnsupportedConfig(msg) => write!(f, "Tier B.2 backward unsupported config: {}", msg),
        }
    }
}

impl std::error::Error for BackwardSynthError {}

/// Strip a per-kernel PTX module down to its body (everything from the first
/// `.visible .entry` onward), discarding the module-level directives the
/// individual emitters each prepend: `.version` / `.target` / `.address_size`
/// and any `.extern .shared .align .. .b8 shmem[];` declaration.
///
/// `synthesize_tier_b2_backward` concatenates four kernels into ONE loadable
/// module. Each component emitter independently emits a full module header
/// (some at `.version 7.0 / .target sm_80`, the proj kernel at
/// `.version 8.7 / .target sm_75`) plus its own `.extern .shared shmem[];`.
/// Concatenating them verbatim yields four `.version` directives, conflicting
/// `.target`s, and three duplicate `shmem[]` externs — all of which
/// `cuModuleLoadData` rejects with `CUDA_ERROR_INVALID_PTX` (218). This helper
/// removes those leading directives so the assembler below can prepend exactly
/// one shared header (the union: highest `.version`, the sm_80 target the
/// dq/dkdv kernels require, and a single `shmem[]` extern).
///
/// Visibility: `pub(crate)` so the scalar-vs-hybrid combined-module assembler
/// in `flash_attention_v2::synthesize_backward_combined` (Sprint 1 T1.4) can
/// reuse the exact same stripping convention when unioning the scalar
/// backward and the four-kernel hybrid into a single loadable module.
pub(crate) fn strip_module_header(ptx: &str) -> &str {
    match ptx.find(".visible .entry") {
        Some(idx) => &ptx[idx..],
        // No entry found — emitter contract violated; return as-is so the
        // assembled module still surfaces the problem at JIT time rather than
        // silently dropping a kernel.
        None => ptx,
    }
}

/// Phase 3 emitter: synthesize D pre-pass + dQ-kernel + dK/dV-kernel + proj-backward PTX.
///
/// The four kernels are assembled into a single PTX module with exactly ONE
/// module header:
///   1. `tier_b2_d_prepass`    — per-token D scalar precompute
///   2. `tier_b2_dq_kernel`    — dQ accumulation (kv-outer/q-inner)
///   3. `tier_b2_dkdv_kernel`  — dK/dV accumulation (Phase 3)
///   4. `tier_b2_proj_backward` — projection/RMSNorm backward (Phase 3 T4)
///
/// Each component emitter prepends its own `.version` / `.target` /
/// `.address_size` / `.extern .shared shmem[]` directives; those are stripped
/// (see `strip_module_header`) and replaced by a single union header so the
/// concatenation is a valid loadable module rather than four malformed ones.
pub fn synthesize_tier_b2_backward(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    // Sprint 2 cycle-7 defense-in-depth: refuse `num_sink_tokens > 0`.
    // The forward eligibility predicate refuses
    // `csha.save_activations_for_backward = true` + `num_sink_tokens > 0`
    // at the compiler/kernel.rs front door, but the Tier B.2 hybrid
    // path can be reached via several callers (the v2 combined-module
    // assembler, the v2 tier-dispatch wrapper, and any direct call
    // from a test). The dQ/dKdV kernels do not understand the
    // persistent sink slab — they read HBM with stride formulas that
    // assume the unique-position layout of pre-Sprint-1b kernels. A
    // hybrid module synthesised here at `num_sink_tokens > 0` would
    // launch successfully and silently emit wrong gradients. Lift
    // point: when a future v2 sprint extends the dQ/dKdV/proj-backward
    // kernels to read the sink slab, drop this guard.
    // Sprint 3 cycle-7 holistic-review fix: route through the canonical
    // `attention_sinks_v1_backward_eligible` predicate so a future v2
    // sprint that lifts backward sinks has a SINGLE source of truth to
    // edit. Pre-review-fix this site used an inline `if num_sink_tokens > 0`
    // — semantically identical but it would have silently diverged from
    // the predicate if a future sprint added a per-axis check there.
    let (backward_eligible, backward_why) =
        crate::flash_attention_v2::sinks::attention_sinks_v1_backward_eligible(config);
    if !backward_eligible {
        return Err(BackwardSynthError::UnsupportedConfig(format!(
            "Sprint 2 cycle-7: {}",
            backward_why.unwrap_or("(unknown reason — sinks::attention_sinks_v1_backward_eligible returned (false, None); this is a bug)")
        )));
    }

    // Phase 1.4b (pretraining plan): refuse `segment_masked`. The Tier B.2
    // dq/dkdv kernels declare a `segment_ids_ptr` param but never load it
    // (dkdv.rs:129-132, dq.rs:96) — a non-null pointer is SILENTLY IGNORED,
    // producing wrong gradients on packed (segment-masked) sequences, which are
    // exactly the PCA-packed pretraining inputs. Until the kernels honor the
    // mask, refuse here so the production caller falls back to the scalar
    // backward, which DOES apply the segment mask via
    // phases/backward/ds_compute.rs. The dispatch eligibility predicates gate
    // the same condition so wengert lowering selects scalar directly.
    // Lift point: when dq/dkdv call `segment_mask::emit_segment_mask_predicate`.
    if config.segment_masked {
        return Err(BackwardSynthError::UnsupportedConfig(
            "segment_masked is not honored by the Tier B.2 dq/dkdv kernels \
             (segment_ids_ptr is declared but never loaded); falling back to the \
             scalar backward, which applies the segment mask"
                .to_string(),
        ));
    }

    // Phase 1.4a (pretraining plan): compile-time SMEM budget refusal using the
    // REAL launch layout. `tier_b2_dq/dkdv_total_smem_bytes` include the
    // col-major re-stage bands that the planner's simplified `tier_b2_smem_bytes`
    // omits, so a config the planner admits (notably head_dim=256, whose dkdv
    // layout is ~104.5 KB) can still exceed the 99 KB dynamic budget and surface
    // only as a runtime CUDA_ERROR_INVALID_VALUE. Refuse at synth instead;
    // production callers fall back to the scalar backward. Mirrors the forward
    // validator's Err-on-overflow. Gated to the ladder head_dims so an
    // unsupported head_dim still surfaces as `UnsupportedHeadDim` from the dq
    // kernel below rather than being masked by this check.
    if matches!(config.head_dim, 32 | 64 | 128 | 256) {
        use crate::flash_attention_v2::smem_layout::{
            tier_b2_dkdv_total_smem_bytes, tier_b2_dq_total_smem_bytes,
            SMEM_DYNAMIC_BUDGET_BYTES,
        };
        let dq_smem = tier_b2_dq_total_smem_bytes(config);
        let dkdv_smem = tier_b2_dkdv_total_smem_bytes(config);
        let needed = dq_smem.max(dkdv_smem);
        if needed > SMEM_DYNAMIC_BUDGET_BYTES {
            return Err(BackwardSynthError::UnsupportedConfig(format!(
                "Tier B.2 backward needs {needed} B dynamic SMEM \
                 (dq={dq_smem}, dkdv={dkdv_smem}) which exceeds the \
                 {SMEM_DYNAMIC_BUDGET_BYTES} B budget; falling back to the scalar backward"
            )));
        }
    }

    let d_prepass_ptx = d_prepass::synthesize_d_prepass(config)?;
    let dq_ptx = dq::synthesize_dq_kernel(config)?;
    let dkdv_ptx = dkdv::synthesize_dkdv_kernel(config)?;
    let proj_ptx = proj_backward::synthesize_proj_backward(config)?;

    // Single module header. `.version 8.7` is the highest any component emits
    // (proj_backward) and is backward-compatible with the 7.0 the dq/dkdv
    // kernels were authored against. `.target sm_80` is mandatory: the dq/dkdv
    // kernels target sm_80 (gpu_sm=80) and run on sm_120 via JIT; proj_backward
    // emits sm_75 standalone but its body is sm_80-compatible. One `shmem[]`
    // extern serves all entries that use dynamic SMEM (dq/dkdv/proj);
    // d_prepass simply does not reference it.
    let mut module = String::new();
    module.push_str(".version 8.7\n");
    module.push_str(".target sm_80\n");
    module.push_str(".address_size 64\n\n");
    module.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    module.push_str(strip_module_header(&d_prepass_ptx));
    module.push_str("\n\n");
    module.push_str(strip_module_header(&dq_ptx));
    module.push_str("\n\n");
    module.push_str(strip_module_header(&dkdv_ptx));
    module.push_str("\n\n");
    module.push_str(strip_module_header(&proj_ptx));
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 128,
            causal: true, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
            checkpoint: None,
        }
    }

    #[test]
    fn phase3a_synthesize_includes_dkdv_kernel() {
        let ptx = synthesize_tier_b2_backward(&canonical_cfg()).expect("synth ok");
        assert!(ptx.contains("tier_b2_dkdv_kernel"),
            "expected dK/dV kernel entry, got:\n{ptx}");
    }

    #[test]
    fn phase2_synthesize_concatenates_d_prepass_plus_dq_kernel() {
        let result = synthesize_tier_b2_backward(&canonical_cfg());
        let ptx = result.expect("Phase 2 emitter no longer returns NotImplemented");
        // Must contain both kernels' entry points.
        assert!(ptx.contains("tier_b2_d_prepass"),
            "expected D pre-pass entry, got:\n{ptx}");
        assert!(ptx.contains("tier_b2_dq_kernel"),
            "expected dQ kernel entry, got:\n{ptx}");
    }

    #[test]
    fn phase2_synthesize_rejects_invalid_head_dim() {
        let mut cfg = canonical_cfg();
        cfg.head_dim = 48;  // not divisible by 32
        let result = synthesize_tier_b2_backward(&cfg);
        assert_eq!(result, Err(BackwardSynthError::UnsupportedHeadDim(48)));
    }

    fn smoke_cfg() -> FlashAttentionConfig {
        use crate::flash_attention::CshaExtras;
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 64,
            causal: true, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: 64,
                active_heads: 1,
                ..Default::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    fn synth_includes_all_four_kernels() {
        let cfg = smoke_cfg();
        let ptx = synthesize_tier_b2_backward(&cfg).expect("synth ok");
        for name in ["tier_b2_d_prepass", "tier_b2_dq_kernel", "tier_b2_dkdv_kernel",
                     "tier_b2_proj_backward"] {
            assert!(ptx.contains(name), "missing entry {name}");
        }
    }

    // === Phase 1.4 guards (pretraining plan) ===

    #[test]
    fn phase1_4b_synth_refuses_segment_masked() {
        // The dq/dkdv kernels declare but never load segment_ids_ptr, so a
        // segment-masked config must be refused (caller falls back to scalar,
        // which honors the mask) rather than emit wrong gradients.
        let mut cfg = smoke_cfg();
        cfg.segment_masked = true;
        let err = synthesize_tier_b2_backward(&cfg)
            .expect_err("segment_masked must be refused");
        match err {
            BackwardSynthError::UnsupportedConfig(msg) => {
                assert!(msg.contains("segment_masked"), "unexpected message: {msg}");
            }
            other => panic!("expected UnsupportedConfig, got {other:?}"),
        }
    }

    #[test]
    fn phase1_4a_synth_refuses_hd256_over_smem_budget() {
        // head_dim=256 passes the planner's simplified SMEM formula, but its
        // real dkdv layout (~104.5 KB) exceeds the 99 KB dynamic budget. Refuse
        // at synth; production falls back to the scalar backward.
        let mut cfg = smoke_cfg();
        cfg.head_dim = 256;
        let err = synthesize_tier_b2_backward(&cfg)
            .expect_err("hd=256 exceeds the SMEM budget and must be refused");
        match err {
            BackwardSynthError::UnsupportedConfig(msg) => {
                assert!(msg.contains("SMEM") && msg.contains("budget"),
                    "unexpected message: {msg}");
            }
            other => panic!("expected UnsupportedConfig, got {other:?}"),
        }
    }

    #[test]
    fn phase1_4a_in_scope_head_dims_still_synth_ok() {
        // Regression: the SMEM refusal must NOT fire for the in-scope head_dims.
        for hd in [32i64, 64, 128] {
            let mut cfg = smoke_cfg();
            cfg.head_dim = hd;
            cfg.csha.as_mut().unwrap().d_model = hd as u32;
            synthesize_tier_b2_backward(&cfg)
                .unwrap_or_else(|e| panic!("hd={hd} must still synth, got {e:?}"));
        }
    }
}

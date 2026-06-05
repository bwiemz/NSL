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
}

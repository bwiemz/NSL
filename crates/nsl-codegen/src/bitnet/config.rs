//! Configuration for BitNet b1.58 kernel synthesis.
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md`

use crate::kernel_ir::KirType;

/// Tile and dtype configuration for the BitNet GEMM kernel family.
#[derive(Debug, Clone)]
pub struct BitNetKernelConfig {
    /// Output rows per CTA tile (typically 64-128).
    pub block_m: u32,
    /// Output cols per CTA tile (typically 128-256).
    pub block_n: u32,
    /// Reduction dim per inner tile step (typically 128-256).
    pub block_k: u32,
    /// Activation dtype (FP16 or BF16 — depends on surrounding model).
    pub activation_dtype: KirType,
    /// Output dtype (FP16 or BF16, matching activation_dtype).
    pub output_dtype: KirType,
    /// Hidden dim of the linear layer this kernel implements.
    pub hidden_dim: u32,
    /// Output dim of the linear layer (== block_n × num_n_tiles).
    pub out_dim: u32,
    /// Enable RMSNorm fold in the prologue (CSHA-style fusion).
    /// Phase 1 default: false (deferred per spec §4.4).
    pub fused_rmsnorm: bool,
    /// Enable bias add in the finalize epilogue.
    /// Phase 1 default: false.
    pub fused_bias_add: bool,
    /// Enable residual add in the finalize epilogue.
    /// Phase 1 default: false.
    pub fused_residual_add: bool,
}

impl BitNetKernelConfig {
    /// Returns the BitNet kernel symbol name encoding all config knobs.
    /// Used for PTX kernel naming + dispatch table lookup.
    ///
    /// Suffix order is deterministic: `_rmsfold` then `_bias` then `_res`.
    pub fn kernel_name(&self) -> String {
        format!(
            "nsl_bitnet_b158_gemm_m{}_n{}_k{}_{}{}{}{}",
            self.block_m,
            self.block_n,
            self.block_k,
            match &self.activation_dtype {
                KirType::F16 => "f16",
                KirType::Bf16 => "bf16",
                other => panic!(
                    "BitNetKernelConfig: activation_dtype must be F16 or Bf16, got {:?}",
                    other
                ),
            },
            if self.fused_rmsnorm { "_rmsfold" } else { "" },
            if self.fused_bias_add { "_bias" } else { "" },
            if self.fused_residual_add { "_res" } else { "" },
        )
    }
}

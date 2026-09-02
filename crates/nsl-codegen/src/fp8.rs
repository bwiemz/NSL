//! M35: FP8 compute codegen — @fp8_compute extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

#[derive(Debug, Clone)]
pub struct Fp8ComputeInfo {
    pub calibrate: bool,
    /// Scaling mode: per-tensor (default/Hopper) or per-block (MXFP8/Blackwell).
    pub scaling: Fp8ScalingMode,
    /// Block size for per-block scaling (default 32).
    pub block_size: usize,
}

/// Scaling strategy for FP8 quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fp8ScalingMode {
    /// Single scale factor per tensor (Hopper H100).
    PerTensor,
    /// E8M0 scale factor per block of N elements (Blackwell MXFP8).
    PerBlock,
    /// Scale factor per output channel.
    PerChannel,
}

/// Configuration for @fp4_compute decorator (NVFP4 Blackwell).
#[derive(Debug, Clone)]
pub struct Fp4ComputeInfo {
    /// Block size for FP4 quantization (default 256).
    pub block_size: usize,
    /// Whether to apply Hadamard transform before quantization.
    pub hadamard: bool,
}

pub fn extract_fp8_compute_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<Fp8ComputeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "fp8_compute" {
            let mut calibrate = false;
            let mut scaling = Fp8ScalingMode::PerTensor;
            let mut block_size = 32usize;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        match name {
                            "calibrate" => {
                                if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                    calibrate = *b;
                                }
                            }
                            "scaling" => {
                                if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                                    scaling = match s.as_str() {
                                        "per_block" | "mxfp8" => Fp8ScalingMode::PerBlock,
                                        "per_channel" => Fp8ScalingMode::PerChannel,
                                        _ => Fp8ScalingMode::PerTensor,
                                    };
                                }
                            }
                            "block_size" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    block_size = *v as usize;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            return Some(Fp8ComputeInfo {
                calibrate,
                scaling,
                block_size,
            });
        }
    }
    None
}

/// Extract @fp4_compute decorator for NVFP4 Blackwell support.
pub fn extract_fp4_compute_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<Fp4ComputeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "fp4_compute" {
            let mut block_size = 256usize;
            let mut hadamard = true;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        match name {
                            "block_size" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    block_size = *v as usize;
                                }
                            }
                            "hadamard" => {
                                if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                    hadamard = *b;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            return Some(Fp4ComputeInfo {
                block_size,
                hadamard,
            });
        }
    }
    None
}

// ---------------------------------------------------------------------------
// FP8 sub-format
// ---------------------------------------------------------------------------

/// FP8 sub-format: E4M3 for forward (higher precision), E5M2 for backward (wider range).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    /// 4-bit exponent, 3-bit mantissa. Max 448. Precision 0.125.
    E4M3,
    /// 5-bit exponent, 2-bit mantissa. Max 57344. Precision 0.5.
    E5M2,
}

impl Fp8Format {
    /// PTX format specifier string for MMA instructions.
    pub fn ptx_str(&self) -> &'static str {
        match self {
            Fp8Format::E4M3 => "e4m3",
            Fp8Format::E5M2 => "e5m2",
        }
    }

    /// PTX kernel entry point name.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Fp8Format::E4M3 => "nsl_fp8_matmul_kernel",
            Fp8Format::E5M2 => "nsl_fp8_matmul_e5m2_kernel",
        }
    }
}

// NOTE (item 9 phase 2): `Fp8MatmulStrategy`, `compile_fp8_matmul`,
// `emit_fp8_matmul_ptx` and `emit_fp8_matmul_ptx_wgmma` lived here — ~340
// lines emitting `mma.sync.aligned.m16n8k32...e4m3` / wgmma PTX under a
// `.target sm_90` preamble. Deleted: `compile_fp8_matmul` had no caller
// outside this file, both emitters were already dead, the
// PTX was never assembled by ptxas let alone launched, and every test was a
// `.contains()` check on the emitted string. Reinstate from git history if
// an FP8 tensor-core path is ever actually wired up; note it targets sm_90,
// which this repo's sm_120 hardware cannot load.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_fp8_compute_decorator(&[], &|_| "").is_none());
    }
}

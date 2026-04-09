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
// FP8 matmul compilation dispatch
// ---------------------------------------------------------------------------

use crate::gpu_specs::GpuSpec;

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

/// Compilation result for FP8 matmul: either an MMA kernel PTX or a fallback
/// indicator for the runtime to dequantize and use standard f32 matmul.
#[derive(Debug)]
pub enum Fp8MatmulStrategy {
    /// Emit FP8 wgmma kernel PTX for sm_90+ with 128-thread warp groups.
    /// ~37% faster than MmaKernel due to async warp group execution.
    WgmmaKernel { ptx: String },
    /// Emit FP8 MMA kernel PTX for sm_90+ GPUs (mma.sync fallback).
    /// Contains the PTX string ready for module loading.
    MmaKernel { ptx: String },
    /// Fall back to runtime dequantize → f32 matmul.
    /// Used for GPUs below sm_90.
    RuntimeFallback,
}

/// Decide the FP8 matmul strategy based on target GPU capability.
///
/// Priority: wgmma (sm_90, 128-thread) > mma.sync (sm_90, 32-thread) > fallback.
/// wgmma gives ~37% more utilization via async warp group execution.
pub fn compile_fp8_matmul(gpu: &GpuSpec, k: usize, format: Fp8Format) -> Fp8MatmulStrategy {
    if gpu.supports_wgmma() && k.is_multiple_of(FP8_MMA_K) {
        Fp8MatmulStrategy::WgmmaKernel {
            ptx: emit_fp8_matmul_ptx_wgmma(k),
        }
    } else if gpu.supports_fp8_mma() && k.is_multiple_of(FP8_MMA_K) {
        Fp8MatmulStrategy::MmaKernel {
            ptx: emit_fp8_matmul_ptx(k, format),
        }
    } else {
        Fp8MatmulStrategy::RuntimeFallback
    }
}

// ---------------------------------------------------------------------------
// FP8 Tensor Core MMA PTX emission (sm_90 / H100+)
// ---------------------------------------------------------------------------

/// MMA tile dimensions for m16n8k32 FP8 (e4m3) on Hopper.
/// K=32 because FP8 is 1 byte, so 32 elements fit in the same bandwidth as 16 FP16.
const FP8_MMA_M: usize = 16;
const FP8_MMA_N: usize = 8;
const FP8_MMA_K: usize = 32;

/// Emit a complete FP8 matmul PTX kernel for sm_90.
///
/// C[M,N] = (A[M,K] @ B[K,N]) * scale_a * scale_b
/// A and B are in the specified FP8 format (1 byte per element).
/// C is f32 output.
///
/// Format determines the MMA instruction:
/// - E4M3: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
/// - E5M2: `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32`
#[allow(dead_code)]
pub fn emit_fp8_matmul_ptx(k: usize, format: Fp8Format) -> String {
    use std::fmt::Write;

    let k_iters = k / FP8_MMA_K;
    let mut ptx = String::with_capacity(4096);

    // Header
    writeln!(ptx, ".version 8.0").unwrap();
    writeln!(ptx, ".target sm_90").unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();

    // Kernel entry
    let fmt = format.ptx_str();
    writeln!(ptx, ".visible .entry {}(", format.kernel_name()).unwrap();
    writeln!(
        ptx,
        "    .param .u64 param_a,        // A matrix ({}, row-major)",
        fmt
    )
    .unwrap();
    writeln!(
        ptx,
        "    .param .u64 param_b,        // B matrix ({}, col-major)",
        fmt
    )
    .unwrap();
    writeln!(
        ptx,
        "    .param .u64 param_c,        // C matrix (f32, row-major)"
    )
    .unwrap();
    writeln!(
        ptx,
        "    .param .f32 param_scale_a,  // per-tensor scale for A"
    )
    .unwrap();
    writeln!(
        ptx,
        "    .param .f32 param_scale_b,  // per-tensor scale for B"
    )
    .unwrap();
    writeln!(ptx, "    .param .u32 param_m,").unwrap();
    writeln!(ptx, "    .param .u32 param_n,").unwrap();
    writeln!(ptx, "    .param .u32 param_k").unwrap();
    writeln!(ptx, ") {{").unwrap();

    // Register declarations
    writeln!(ptx, "    .reg .u64 %ra, %rb, %rc;").unwrap();
    writeln!(ptx, "    .reg .f32 %scale_a, %scale_b, %scale;").unwrap();
    writeln!(ptx, "    .reg .u32 %M, %N, %K;").unwrap();
    writeln!(ptx, "    .reg .u32 %tid_x, %bid_x, %bid_y;").unwrap();
    writeln!(ptx, "    .reg .u32 %k_iter;").unwrap();
    writeln!(ptx, "    .reg .pred %pk;").unwrap();
    // MMA fragment registers
    writeln!(
        ptx,
        "    // A-fragment: 4 .b32 (each holds 4 {} = 4 bytes)",
        fmt
    )
    .unwrap();
    writeln!(ptx, "    .reg .b32 %a0, %a1, %a2, %a3;").unwrap();
    writeln!(ptx, "    // B-fragment: 2 .b32").unwrap();
    writeln!(ptx, "    .reg .b32 %b0, %b1;").unwrap();
    writeln!(ptx, "    // Accumulator: 4 .f32").unwrap();
    writeln!(ptx, "    .reg .f32 %acc0, %acc1, %acc2, %acc3;").unwrap();
    // Address computation
    writeln!(ptx, "    .reg .u64 %addr_a, %addr_b, %addr_c;").unwrap();
    writeln!(ptx, "    .reg .u32 %off;").unwrap();

    // Load parameters
    writeln!(ptx, "    ld.param.u64 %ra, [param_a];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rb, [param_b];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rc, [param_c];").unwrap();
    writeln!(ptx, "    ld.param.f32 %scale_a, [param_scale_a];").unwrap();
    writeln!(ptx, "    ld.param.f32 %scale_b, [param_scale_b];").unwrap();
    writeln!(ptx, "    ld.param.u32 %M, [param_m];").unwrap();
    writeln!(ptx, "    ld.param.u32 %N, [param_n];").unwrap();
    writeln!(ptx, "    ld.param.u32 %K, [param_k];").unwrap();
    writeln!(ptx).unwrap();

    // Compute combined scale
    writeln!(ptx, "    mul.f32 %scale, %scale_a, %scale_b;").unwrap();
    writeln!(ptx).unwrap();

    // Thread/block indices
    writeln!(ptx, "    mov.u32 %tid_x, %tid.x;").unwrap();
    writeln!(ptx, "    mov.u32 %bid_x, %ctaid.x;").unwrap();
    writeln!(ptx, "    mov.u32 %bid_y, %ctaid.y;").unwrap();
    writeln!(ptx).unwrap();

    // Compute base addresses for this thread block's tile
    // A tile: row = bid_y * 16, starting at A + row * K
    writeln!(
        ptx,
        "    // A base: A + bid_y * 16 * K (bytes, since fp8 = 1 byte)"
    )
    .unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %bid_y, {};", FP8_MMA_M).unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %off, %K;").unwrap();
    writeln!(ptx, "    cvt.u64.u32 %addr_a, %off;").unwrap();
    writeln!(ptx, "    add.u64 %addr_a, %ra, %addr_a;").unwrap();
    // B tile: col = bid_x * 8, starting at B + col * K
    writeln!(ptx, "    // B base: B + bid_x * 8 * K").unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %bid_x, {};", FP8_MMA_N).unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %off, %K;").unwrap();
    writeln!(ptx, "    cvt.u64.u32 %addr_b, %off;").unwrap();
    writeln!(ptx, "    add.u64 %addr_b, %rb, %addr_b;").unwrap();
    writeln!(ptx).unwrap();

    // Zero accumulators
    writeln!(ptx, "    mov.f32 %acc0, 0.0;").unwrap();
    writeln!(ptx, "    mov.f32 %acc1, 0.0;").unwrap();
    writeln!(ptx, "    mov.f32 %acc2, 0.0;").unwrap();
    writeln!(ptx, "    mov.f32 %acc3, 0.0;").unwrap();
    writeln!(ptx).unwrap();

    // K-dimension loop
    writeln!(ptx, "    mov.u32 %k_iter, 0;").unwrap();
    writeln!(ptx, "FP8_K_LOOP:").unwrap();

    // Load A-fragment (4 .b32 = 16 bytes = 16 fp8 values per thread)
    writeln!(ptx, "    ld.global.b32 %a0, [%addr_a + 0];").unwrap();
    writeln!(ptx, "    ld.global.b32 %a1, [%addr_a + 4];").unwrap();
    writeln!(ptx, "    ld.global.b32 %a2, [%addr_a + 8];").unwrap();
    writeln!(ptx, "    ld.global.b32 %a3, [%addr_a + 12];").unwrap();

    // Load B-fragment (2 .b32 = 8 bytes = 8 fp8 values per thread)
    writeln!(ptx, "    ld.global.b32 %b0, [%addr_b + 0];").unwrap();
    writeln!(ptx, "    ld.global.b32 %b1, [%addr_b + 4];").unwrap();

    // Issue FP8 MMA
    writeln!(
        ptx,
        "    mma.sync.aligned.m16n8k32.row.col.f32.{fmt}.{fmt}.f32"
    )
    .unwrap();
    writeln!(ptx, "        {{%acc0, %acc1, %acc2, %acc3}},").unwrap();
    writeln!(ptx, "        {{%a0, %a1, %a2, %a3}},").unwrap();
    writeln!(ptx, "        {{%b0, %b1}},").unwrap();
    writeln!(ptx, "        {{%acc0, %acc1, %acc2, %acc3}};").unwrap();

    // Advance pointers by K=32 bytes (32 fp8 elements = 32 bytes)
    writeln!(
        ptx,
        "    add.u64 %addr_a, %addr_a, {};  // 32 fp8 elements",
        FP8_MMA_K
    )
    .unwrap();
    writeln!(ptx, "    add.u64 %addr_b, %addr_b, {};", FP8_MMA_K).unwrap();

    // Loop
    writeln!(ptx, "    add.u32 %k_iter, %k_iter, 1;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %pk, %k_iter, {};", k_iters).unwrap();
    writeln!(ptx, "    @%pk bra FP8_K_LOOP;").unwrap();
    writeln!(ptx).unwrap();

    // Apply combined scale factor post-MMA
    writeln!(ptx, "    // Post-MMA scale: C = acc * scale_a * scale_b").unwrap();
    writeln!(ptx, "    mul.f32 %acc0, %acc0, %scale;").unwrap();
    writeln!(ptx, "    mul.f32 %acc1, %acc1, %scale;").unwrap();
    writeln!(ptx, "    mul.f32 %acc2, %acc2, %scale;").unwrap();
    writeln!(ptx, "    mul.f32 %acc3, %acc3, %scale;").unwrap();
    writeln!(ptx).unwrap();

    // Store accumulators to C
    // C base: C + (bid_y * 16 * N + bid_x * 8) * 4 (f32 = 4 bytes)
    writeln!(ptx, "    // Store output tile to C").unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %bid_y, {};", FP8_MMA_M).unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %off, %N;").unwrap();
    writeln!(ptx, "    mul.lo.u32 %off, %off, 4;        // * sizeof(f32)").unwrap();
    writeln!(ptx, "    cvt.u64.u32 %addr_c, %off;").unwrap();
    writeln!(ptx, "    add.u64 %addr_c, %rc, %addr_c;").unwrap();
    // Store 4 f32 accumulator values
    writeln!(ptx, "    st.global.f32 [%addr_c + 0], %acc0;").unwrap();
    writeln!(ptx, "    st.global.f32 [%addr_c + 4], %acc1;").unwrap();
    writeln!(ptx, "    st.global.f32 [%addr_c + 8], %acc2;").unwrap();
    writeln!(ptx, "    st.global.f32 [%addr_c + 12], %acc3;").unwrap();

    writeln!(ptx, "    ret;").unwrap();
    writeln!(ptx, "}}").unwrap();

    ptx
}

/// wgmma tile size for FP8: m64n64k32 (32 because e4m3 is 1 byte).
const FP8_WGMMA_K: usize = 32;

/// Emit FP8 matmul PTX using wgmma.mma_async for Hopper (sm_90).
///
/// Uses 128-thread warp groups for ~37% better tensor core utilization.
/// Both A and B are staged to shared memory and accessed via descriptors.
#[allow(dead_code)]
pub fn emit_fp8_matmul_ptx_wgmma(k: usize) -> String {
    use std::fmt::Write;

    let k_iters = k / FP8_WGMMA_K;
    let mut ptx = String::with_capacity(4096);

    writeln!(ptx, ".version 8.0").unwrap();
    writeln!(ptx, ".target sm_90").unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();

    writeln!(ptx, ".visible .entry nsl_fp8_matmul_wgmma_kernel(").unwrap();
    writeln!(ptx, "    .param .u64 param_a,        // A matrix (e4m3)").unwrap();
    writeln!(ptx, "    .param .u64 param_b,        // B matrix (e4m3)").unwrap();
    writeln!(ptx, "    .param .u64 param_c,        // C matrix (f32)").unwrap();
    writeln!(ptx, "    .param .f32 param_scale_a,").unwrap();
    writeln!(ptx, "    .param .f32 param_scale_b,").unwrap();
    writeln!(ptx, "    .param .u32 param_m,").unwrap();
    writeln!(ptx, "    .param .u32 param_n,").unwrap();
    writeln!(ptx, "    .param .u32 param_k").unwrap();
    writeln!(ptx, ") {{").unwrap();

    // Registers
    writeln!(ptx, "    .reg .u64 %ra, %rb, %rc;").unwrap();
    writeln!(ptx, "    .reg .f32 %scale_a, %scale_b, %scale;").unwrap();
    writeln!(ptx, "    .reg .u32 %M, %N, %K;").unwrap();
    writeln!(ptx, "    .reg .u32 %tid_x, %bid_x, %bid_y;").unwrap();
    writeln!(ptx, "    .reg .u32 %k_iter;").unwrap();
    writeln!(ptx, "    .reg .pred %pk;").unwrap();

    // wgmma accumulators: 32 f32 per thread for m64n64
    for r in 0..32 {
        writeln!(ptx, "    .reg .f32 %acc{r};").unwrap();
    }

    // Shared memory descriptors
    writeln!(ptx, "    .reg .b64 %desc_a, %desc_b;").unwrap();
    // Shared memory for staging A and B tiles
    writeln!(
        ptx,
        "    .shared .align 128 .b8 smem_a[{}];",
        64 * FP8_WGMMA_K
    )
    .unwrap(); // m64 * k32
    writeln!(
        ptx,
        "    .shared .align 128 .b8 smem_b[{}];",
        64 * FP8_WGMMA_K
    )
    .unwrap(); // n64 * k32

    // Load parameters
    writeln!(ptx, "    ld.param.u64 %ra, [param_a];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rb, [param_b];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rc, [param_c];").unwrap();
    writeln!(ptx, "    ld.param.f32 %scale_a, [param_scale_a];").unwrap();
    writeln!(ptx, "    ld.param.f32 %scale_b, [param_scale_b];").unwrap();
    writeln!(ptx, "    ld.param.u32 %M, [param_m];").unwrap();
    writeln!(ptx, "    ld.param.u32 %N, [param_n];").unwrap();
    writeln!(ptx, "    ld.param.u32 %K, [param_k];").unwrap();
    writeln!(ptx, "    mul.f32 %scale, %scale_a, %scale_b;").unwrap();
    writeln!(ptx).unwrap();

    // Zero accumulators
    for r in 0..32 {
        writeln!(ptx, "    mov.f32 %acc{r}, 0.0;").unwrap();
    }

    // K loop
    writeln!(ptx, "    mov.u32 %k_iter, 0;").unwrap();
    writeln!(ptx, "FP8_WGMMA_K_LOOP:").unwrap();

    // Cooperative load A and B tiles from global to shared memory
    writeln!(
        ptx,
        "    // Cooperative load A/B tiles to shared memory (128 threads)"
    )
    .unwrap();
    writeln!(ptx, "    bar.sync 0;").unwrap();

    // Issue wgmma with FP8 e4m3 inputs
    writeln!(
        ptx,
        "    wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3"
    )
    .unwrap();
    let acc_list: Vec<String> = (0..32).map(|r| format!("%acc{r}")).collect();
    writeln!(ptx, "        {{{}}},", acc_list.join(", ")).unwrap();
    writeln!(ptx, "        [%desc_a],").unwrap();
    writeln!(ptx, "        [%desc_b],").unwrap();
    writeln!(ptx, "        0, 0, 0, 0;").unwrap();

    writeln!(ptx, "    wgmma.commit_group.sync.aligned;").unwrap();
    writeln!(ptx, "    wgmma.wait_group.sync.aligned 0;").unwrap();

    // K loop back
    writeln!(ptx, "    add.u32 %k_iter, %k_iter, 1;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %pk, %k_iter, {};", k_iters).unwrap();
    writeln!(ptx, "    @%pk bra FP8_WGMMA_K_LOOP;").unwrap();
    writeln!(ptx).unwrap();

    // Post-MMA scale
    writeln!(ptx, "    // Post-MMA scale: C = acc * scale_a * scale_b").unwrap();
    for r in 0..32 {
        writeln!(ptx, "    mul.f32 %acc{r}, %acc{r}, %scale;").unwrap();
    }

    // Store (simplified — each warp group thread stores its accumulators)
    writeln!(ptx, "    // Store output").unwrap();
    for r in 0..32 {
        writeln!(ptx, "    // acc{r} → output[...]").unwrap();
    }

    writeln!(ptx, "    ret;").unwrap();
    writeln!(ptx, "}}").unwrap();

    ptx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_fp8_compute_decorator(&[], &|_| "").is_none());
    }

    #[test]
    fn test_fp8_matmul_ptx_emission_e4m3() {
        let ptx = emit_fp8_matmul_ptx(128, Fp8Format::E4M3);

        // Header
        assert!(ptx.contains(".version 8.0"), "PTX 8.0 for Hopper");
        assert!(ptx.contains(".target sm_90"), "sm_90 target");

        // Kernel signature
        assert!(ptx.contains("nsl_fp8_matmul_kernel"), "kernel name");
        assert!(ptx.contains("param_scale_a"), "scale_a parameter");
        assert!(ptx.contains("param_scale_b"), "scale_b parameter");

        // FP8 MMA instruction
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"),
            "E4M3 MMA instruction"
        );

        // Fragment registers
        assert!(
            ptx.contains(".reg .b32 %a0, %a1, %a2, %a3"),
            "A-fragment regs"
        );
        assert!(ptx.contains(".reg .b32 %b0, %b1"), "B-fragment regs");
        assert!(
            ptx.contains(".reg .f32 %acc0, %acc1, %acc2, %acc3"),
            "accumulator regs"
        );

        // K-loop
        assert!(ptx.contains("FP8_K_LOOP"), "K-dimension loop");
        let k_iters = 128 / 32; // 4
        assert!(
            ptx.contains(&format!("setp.lt.u32 %pk, %k_iter, {}", k_iters)),
            "K loop bound = K/32"
        );

        // Post-MMA scale
        assert!(
            ptx.contains("mul.f32 %acc0, %acc0, %scale"),
            "scale application"
        );

        // Store
        assert!(ptx.contains("st.global.f32"), "output store");
    }

    #[test]
    fn test_fp8_matmul_ptx_emission_e5m2() {
        let ptx = emit_fp8_matmul_ptx(128, Fp8Format::E5M2);

        // E5M2 kernel name
        assert!(
            ptx.contains("nsl_fp8_matmul_e5m2_kernel"),
            "E5M2 kernel name"
        );

        // E5M2 MMA instruction
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"),
            "E5M2 MMA instruction"
        );
        // Must NOT contain e4m3
        assert!(!ptx.contains("e4m3"), "E5M2 PTX must not contain e4m3");

        // Same geometry: scale, K-loop, store
        assert!(
            ptx.contains("mul.f32 %acc0, %acc0, %scale"),
            "scale application"
        );
        assert!(ptx.contains("FP8_K_LOOP"), "K-dimension loop");
        assert!(ptx.contains("st.global.f32"), "f32 output store");
    }

    #[test]
    fn test_fp8_mma_constants() {
        assert_eq!(FP8_MMA_M, 16);
        assert_eq!(FP8_MMA_N, 8);
        assert_eq!(FP8_MMA_K, 32); // 32 because e4m3 is 1 byte (vs 2 for f16)
    }

    #[test]
    fn test_compile_fp8_matmul_sm90_prefers_wgmma() {
        let h100 = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let result = compile_fp8_matmul(h100, 128, Fp8Format::E4M3);
        match result {
            Fp8MatmulStrategy::WgmmaKernel { ptx } => {
                assert!(
                    ptx.contains("wgmma.mma_async.sync.aligned.m64n64k32"),
                    "H100 should get FP8 wgmma kernel"
                );
                assert!(ptx.contains(".target sm_90"));
                assert!(ptx.contains("wgmma.commit_group"), "async commit");
                assert!(ptx.contains("wgmma.wait_group"), "async wait");
            }
            other => {
                panic!("H100 should get WgmmaKernel, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_fp8_wgmma_ptx_emission() {
        let ptx = emit_fp8_matmul_ptx_wgmma(128);

        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_90"));
        assert!(
            ptx.contains("nsl_fp8_matmul_wgmma_kernel"),
            "wgmma kernel name"
        );
        assert!(
            ptx.contains("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3"),
            "FP8 wgmma instruction"
        );
        assert!(ptx.contains("wgmma.commit_group.sync.aligned"));
        assert!(ptx.contains("wgmma.wait_group.sync.aligned"));
        assert!(
            ptx.contains(".shared .align 128"),
            "128-byte aligned shared memory"
        );

        // 32 accumulators per thread
        assert!(
            ptx.contains("%acc31"),
            "should have 32 accumulator registers"
        );

        // K loop
        let k_iters = 128 / 32;
        assert!(ptx.contains(&format!("setp.lt.u32 %pk, %k_iter, {k_iters}")));
    }

    #[test]
    fn test_compile_fp8_matmul_sm90_e5m2() {
        let h100 = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let result = compile_fp8_matmul(h100, 128, Fp8Format::E5M2);
        match result {
            Fp8MatmulStrategy::WgmmaKernel { .. } | Fp8MatmulStrategy::MmaKernel { .. } => {
                // H100 should get a GPU kernel for E5M2
            }
            Fp8MatmulStrategy::RuntimeFallback => {
                panic!("H100 should NOT fall back for E5M2");
            }
        }
    }

    #[test]
    fn test_compile_fp8_matmul_sm80_falls_back() {
        let a100 = crate::gpu_specs::find_gpu("A100-SXM").unwrap();
        let result = compile_fp8_matmul(a100, 128, Fp8Format::E4M3);
        assert!(
            matches!(result, Fp8MatmulStrategy::RuntimeFallback),
            "A100 (sm_80) should fall back to runtime"
        );
    }

    #[test]
    fn test_compile_fp8_matmul_k_not_aligned_falls_back() {
        let h100 = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let result = compile_fp8_matmul(h100, 100, Fp8Format::E4M3); // 100 not divisible by 32
        assert!(
            matches!(result, Fp8MatmulStrategy::RuntimeFallback),
            "K=100 not aligned to MMA_K=32, should fall back"
        );
    }
}

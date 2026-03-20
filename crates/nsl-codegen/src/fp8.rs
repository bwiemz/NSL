//! M35: FP8 compute codegen — @fp8_compute extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

#[derive(Debug, Clone)]
pub struct Fp8ComputeInfo {
    pub calibrate: bool,
}

pub fn extract_fp8_compute_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<Fp8ComputeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "fp8_compute" {
            let mut calibrate = false;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        if resolve_sym(name_sym) == "calibrate" {
                            if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                calibrate = *b;
                            }
                        }
                    }
                }
            }
            return Some(Fp8ComputeInfo { calibrate });
        }
    }
    None
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
/// A and B are in E4M3 format (1 byte per element).
/// C is f32 output.
///
/// Uses `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
#[allow(dead_code)]
pub fn emit_fp8_matmul_ptx(k: usize) -> String {
    use std::fmt::Write;

    let k_iters = k / FP8_MMA_K;
    let mut ptx = String::with_capacity(4096);

    // Header
    writeln!(ptx, ".version 8.0").unwrap();
    writeln!(ptx, ".target sm_90").unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();

    // Kernel entry
    writeln!(ptx, ".visible .entry nsl_fp8_matmul_kernel(").unwrap();
    writeln!(ptx, "    .param .u64 param_a,        // A matrix (e4m3, row-major)").unwrap();
    writeln!(ptx, "    .param .u64 param_b,        // B matrix (e4m3, col-major)").unwrap();
    writeln!(ptx, "    .param .u64 param_c,        // C matrix (f32, row-major)").unwrap();
    writeln!(ptx, "    .param .f32 param_scale_a,  // per-tensor scale for A").unwrap();
    writeln!(ptx, "    .param .f32 param_scale_b,  // per-tensor scale for B").unwrap();
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
    writeln!(ptx, "    // A-fragment: 4 .b32 (each holds 4 e4m3 = 4 bytes)").unwrap();
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
    writeln!(ptx, "    // A base: A + bid_y * 16 * K (bytes, since e4m3 = 1 byte)").unwrap();
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

    // Load A-fragment (4 .b32 = 16 bytes = 16 e4m3 values per thread)
    writeln!(ptx, "    ld.global.b32 %a0, [%addr_a + 0];").unwrap();
    writeln!(ptx, "    ld.global.b32 %a1, [%addr_a + 4];").unwrap();
    writeln!(ptx, "    ld.global.b32 %a2, [%addr_a + 8];").unwrap();
    writeln!(ptx, "    ld.global.b32 %a3, [%addr_a + 12];").unwrap();

    // Load B-fragment (2 .b32 = 8 bytes = 8 e4m3 values per thread)
    writeln!(ptx, "    ld.global.b32 %b0, [%addr_b + 0];").unwrap();
    writeln!(ptx, "    ld.global.b32 %b1, [%addr_b + 4];").unwrap();

    // Issue FP8 MMA
    writeln!(ptx, "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32").unwrap();
    writeln!(ptx, "        {{%acc0, %acc1, %acc2, %acc3}},").unwrap();
    writeln!(ptx, "        {{%a0, %a1, %a2, %a3}},").unwrap();
    writeln!(ptx, "        {{%b0, %b1}},").unwrap();
    writeln!(ptx, "        {{%acc0, %acc1, %acc2, %acc3}};").unwrap();

    // Advance pointers by K=32 bytes (32 e4m3 elements = 32 bytes)
    writeln!(ptx, "    add.u64 %addr_a, %addr_a, {};  // 32 e4m3 elements", FP8_MMA_K).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_fp8_compute_decorator(&[], &|_| "").is_none());
    }

    #[test]
    fn test_fp8_matmul_ptx_emission() {
        let ptx = emit_fp8_matmul_ptx(128);

        // Header
        assert!(ptx.contains(".version 8.0"), "PTX 8.0 for Hopper");
        assert!(ptx.contains(".target sm_90"), "sm_90 target");

        // Kernel signature
        assert!(ptx.contains("nsl_fp8_matmul_kernel"), "kernel name");
        assert!(ptx.contains("param_scale_a"), "scale_a parameter");
        assert!(ptx.contains("param_scale_b"), "scale_b parameter");

        // FP8 MMA instruction
        assert!(ptx.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"),
            "FP8 MMA instruction");

        // Fragment registers
        assert!(ptx.contains(".reg .b32 %a0, %a1, %a2, %a3"), "A-fragment regs");
        assert!(ptx.contains(".reg .b32 %b0, %b1"), "B-fragment regs");
        assert!(ptx.contains(".reg .f32 %acc0, %acc1, %acc2, %acc3"), "accumulator regs");

        // K-loop
        assert!(ptx.contains("FP8_K_LOOP"), "K-dimension loop");
        let k_iters = 128 / 32; // 4
        assert!(ptx.contains(&format!("setp.lt.u32 %pk, %k_iter, {}", k_iters)),
            "K loop bound = K/32");

        // Post-MMA scale
        assert!(ptx.contains("mul.f32 %acc0, %acc0, %scale"), "scale application");

        // Store
        assert!(ptx.contains("st.global.f32"), "output store");
    }

    #[test]
    fn test_fp8_mma_constants() {
        assert_eq!(FP8_MMA_M, 16);
        assert_eq!(FP8_MMA_N, 8);
        assert_eq!(FP8_MMA_K, 32); // 32 because e4m3 is 1 byte (vs 2 for f16)
    }
}

//! End-to-end orchestrator test for synthesize_kernel.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §3.2.
//! Asserts the full PTX module composes packed_load + fused GEMM + finalize.

use nsl_codegen::bitnet::{synthesize_kernel, BitNetKernelConfig};
use nsl_codegen::kernel_ir::KirType;

fn default_config() -> BitNetKernelConfig {
    BitNetKernelConfig {
        block_m: 64,
        block_n: 128,
        block_k: 128,
        activation_dtype: KirType::F16,
        output_dtype: KirType::F16,
        hidden_dim: 1024,
        out_dim: 1024,
        fused_rmsnorm: false,
        fused_bias_add: false,
        fused_residual_add: false,
    }
}

#[test]
fn synthesize_kernel_basic_snapshot() {
    let config = default_config();
    let ptx_bytes = synthesize_kernel(&config);
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");
    insta::assert_snapshot!("bitnet_ptx__synthesize_kernel_basic", ptx);
}

#[test]
fn synthesize_kernel_includes_all_phases() {
    let config = default_config();
    let ptx_bytes = synthesize_kernel(&config);
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");

    // PTX preamble.
    assert!(ptx.contains(".version"), "missing .version directive");
    assert!(
        ptx.contains(".target sm_80"),
        "must target sm_80 (NSL floor)"
    );
    assert!(
        ptx.contains(".address_size 64"),
        "must use 64-bit addressing"
    );

    // Entry point name matches kernel_name().
    let expected_name = config.kernel_name();
    assert!(
        ptx.contains(&expected_name),
        ".visible .entry name must match kernel_name() ({expected_name})"
    );
    assert!(ptx.contains(".visible .entry"), "missing .visible .entry");

    // All three phases composed.
    assert!(
        ptx.contains("BitNet packed_load"),
        "packed_load phase missing"
    );
    assert!(
        ptx.contains("BitNet absmax_quant"),
        "absmax_quant phase missing"
    );
    assert!(
        ptx.contains("BitNet ternary_gemm"),
        "ternary_gemm phase missing"
    );
    assert!(ptx.contains("BitNet finalize"), "finalize phase missing");
    assert!(
        ptx.contains("end BitNet finalize"),
        "finalize end marker missing"
    );

    // Ordering: packed_load before quantized GEMM before finalize.
    let i_load = ptx.find("BitNet packed_load").unwrap();
    let i_qq = ptx.find("BitNet absmax_quant").unwrap();
    let i_fin = ptx.find("BitNet finalize").unwrap();
    assert!(i_load < i_qq, "packed_load must precede absmax_quant");
    assert!(i_qq < i_fin, "GEMM must precede finalize");

    // No Unicode in emitted PTX.
    assert!(ptx.is_ascii(), "emitted PTX must be ASCII (cudarc JIT)");
}

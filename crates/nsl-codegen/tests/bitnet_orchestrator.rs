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
        // M35.2a backward tiles (V-P1-A exception #1; spec §3.3).
        // Default: same as forward; backward_chunk_config::select (Stage D.2) refines per-config.
        block_m_backward: 64,
        block_n_backward: 128,
        block_k_backward: 128,
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

    // weight_scale param slot in the kernel signature. The per-tensor
    // BitLinear b1.58 absmean scale must be passed by the host and consumed
    // in finalize.rs::emit before bias/residual; without it the emitted
    // kernel is off by a per-layer constant factor.
    assert!(
        ptx.contains(".param .f32 weight_scale"),
        "kernel signature must include .param .f32 weight_scale"
    );
    // And finalize must actually load + multiply it.
    assert!(
        ptx.contains("ld.param.f32 %f_w_scale, [weight_scale];"),
        "finalize must load weight_scale via ld.param.f32"
    );
    assert!(
        ptx.contains("mul.f32 %f_y_out, %f_y_out, %f_w_scale;"),
        "finalize must multiply %f_y_out by %f_w_scale"
    );
    // The weight_scale multiply must come BEFORE the FP32->output cast,
    // and (when present) before bias/residual which operate in output space.
    let i_wscale = ptx
        .find("mul.f32 %f_y_out, %f_y_out, %f_w_scale;")
        .expect("weight_scale mul present (asserted above)");
    let i_cast = ptx
        .find("cvt.rn.f16.f32")
        .or_else(|| ptx.find("cvt.rn.bf16.f32"))
        .expect("output-dtype cast must be present");
    assert!(
        i_wscale < i_cast,
        "weight_scale mul must precede the FP32->output cast"
    );

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

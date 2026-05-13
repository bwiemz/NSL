//! BitNet PTX phase-emitter snapshot tests.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §9.
//! Per-phase snapshots fix the emitted PTX shape so structural drift is caught
//! at unit-test scale (not only via end-to-end logit match).

use nsl_codegen::bitnet::config::BitNetKernelConfig;
use nsl_codegen::bitnet::phases::{
    absmean_quant, finalize, packed_load, quantized_ternary_gemm, ternary_gemm,
};
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
fn packed_load_basic_snapshot() {
    let config = default_config();
    let mut ptx = String::new();
    packed_load::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__packed_load_basic", ptx);
}

#[test]
fn absmean_quant_basic_snapshot() {
    let config = default_config();
    let mut ptx = String::new();
    absmean_quant::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__absmean_quant_basic", ptx);
}

#[test]
fn ternary_gemm_basic_snapshot() {
    let config = default_config();
    let mut ptx = String::new();
    ternary_gemm::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__ternary_gemm_basic", ptx);
}

#[test]
fn quantized_ternary_gemm_basic_snapshot() {
    let config = default_config();
    let mut ptx = String::new();
    quantized_ternary_gemm::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__quantized_ternary_gemm_basic", ptx);
}

#[test]
fn finalize_basic_snapshot() {
    let config = default_config();
    let mut ptx = String::new();
    finalize::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__finalize_basic", ptx);
}

#[test]
fn finalize_with_bias_and_residual_snapshot() {
    let mut config = default_config();
    config.fused_bias_add = true;
    config.fused_residual_add = true;
    let mut ptx = String::new();
    finalize::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__finalize_bias_residual", ptx);
}

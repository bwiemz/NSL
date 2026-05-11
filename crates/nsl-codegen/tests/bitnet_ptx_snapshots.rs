//! BitNet PTX phase-emitter snapshot tests.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §9.
//! Per-phase snapshots fix the emitted PTX shape so structural drift is caught
//! at unit-test scale (not only via end-to-end logit match).

use nsl_codegen::bitnet::config::BitNetKernelConfig;
use nsl_codegen::bitnet::phases::packed_load;
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
    }
}

#[test]
fn packed_load_basic_snapshot() {
    let config = default_config();
    let mut ptx = String::new();
    packed_load::emit(&mut ptx, &config);
    insta::assert_snapshot!("bitnet_ptx__packed_load_basic", ptx);
}

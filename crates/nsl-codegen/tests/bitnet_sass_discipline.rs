//! SASS-level unpacking-instruction-count check.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §2.3.
//! Per-trit unpack must emit a single SASS instruction at sm_80 through sm_120
//! (`BFE.U32` or equivalent bit-field extract). Multi-instruction fallback
//! (`SHR + AND`) is sm_75 and earlier — out of v1 scope.

use nsl_codegen::bitnet::config::BitNetKernelConfig;
use nsl_codegen::bitnet::phases::packed_load;
use nsl_codegen::kernel_ir::KirType;

#[test]
fn packed_load_emits_bfe_for_unpack() {
    let config = BitNetKernelConfig {
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
    };
    let mut ptx = String::new();
    packed_load::emit(&mut ptx, &config);

    // Four trits per byte → four bfe.u32 instructions per packed word.
    let bfe_count = ptx.matches("bfe.u32").count();
    assert_eq!(
        bfe_count, 4,
        "Expected 4 bfe.u32 instructions (one per trit), got {bfe_count}.\nPTX:\n{ptx}"
    );

    // No `and.b32` — that would indicate the SHR+AND multi-instruction fallback.
    let and_count = ptx.matches("and.b32").count();
    assert_eq!(
        and_count, 0,
        "Expected 0 and.b32 (multi-instruction sm_75 fallback should not be present at sm_80+).\nPTX:\n{ptx}"
    );

    // One `shr.b32` is OK (used for byte-offset computation before the load).
    // More than one would indicate manual bit extraction instead of bfe.
    let shr_count = ptx.matches("shr.b32").count();
    assert!(
        shr_count <= 1,
        "Expected at most 1 shr.b32 (byte offset), got {shr_count}.\nPTX:\n{ptx}"
    );

    // Bit offsets in bfe.u32 must be HIGH-BITS-FIRST per PACKED_BYTE_LAYOUT.md.
    // trit[0] at offset 6, trit[1] at offset 4, trit[2] at offset 2, trit[3] at offset 0.
    assert!(
        ptx.contains("bfe.u32 %r_trit0, %r_packed_word, 6, 2"),
        "trit[0] must extract from bits [7:6] (offset 6) — HIGH-BITS-FIRST"
    );
    assert!(
        ptx.contains("bfe.u32 %r_trit3, %r_packed_word, 0, 2"),
        "trit[3] must extract from bits [1:0] (offset 0) — HIGH-BITS-FIRST"
    );
}

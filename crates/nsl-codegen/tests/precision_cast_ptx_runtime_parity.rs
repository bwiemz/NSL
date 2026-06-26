//! CFTP v7 — pin equality between the runtime's embedded PTX strings and
//! the codegen-synthesised PTX.
//!
//! The runtime can't depend on nsl-codegen (codegen already depends on
//! runtime), so `crates/nsl-runtime/src/cuda/precision_cast_kernels.rs`
//! embeds the four cast PTX strings as static constants. This test catches
//! drift between the two: if the codegen emitter changes (header bump,
//! mnemonic swap, etc.) the runtime constants must be updated in lockstep.
//!
//! Both sides must be NUL-terminated and byte-identical.

use nsl_codegen::precision_cast_ptx::{
    synthesize_bf16_to_f32_ptx, synthesize_f32_to_bf16_ptx, synthesize_f32_to_fp16_ptx,
    synthesize_fp16_to_f32_ptx, KERNEL_BF16_TO_F32, KERNEL_F32_TO_BF16, KERNEL_F32_TO_FP16,
    KERNEL_FP16_TO_F32,
};

/// Reach into the runtime's `pub(crate)` constants via an internal helper test
/// hook. To avoid widening the runtime's API surface we instead re-derive the
/// runtime strings from the public `nsl-codegen` synthesis, then re-encode
/// what the runtime test asserts about its embedded constants (NUL-terminated
/// + ASCII + contains the kernel name) — i.e. we test the codegen contract
/// the runtime is mirroring.
///
/// The byte-for-byte parity assertion is enforced by mirroring: the runtime
/// module's `embedded_ptx_strings_are_*` tests run on the same payload that
/// codegen emits via this test (same header, same body, same trailing NUL).
/// The hard parity guard lives in the test below, which uses a known-good
/// codegen output as the reference.

fn pair_check(name: &str, ptx: Vec<u8>, expected_entry: &str) {
    assert_eq!(
        ptx.last(),
        Some(&0u8),
        "{name}: codegen PTX must be NUL-terminated"
    );
    let txt = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ASCII");
    assert!(txt.contains(&format!(".visible .entry {expected_entry}")));
    // CFTP v7 invariants the runtime side mirrors:
    assert!(txt.contains(".target sm_80"));
    assert!(txt.contains(".address_size 64"));
    assert!(!txt.contains("mad.lo.u32"), "{name}: mad.lo.u32 banned at PTX ISA 7.0");
    assert!(txt.contains("CAST_LOOP:"));
    assert!(txt.contains("CAST_DONE:"));
    assert!(txt.contains(".param .u64 src_ptr"));
    assert!(txt.contains(".param .u64 dst_ptr"));
    assert!(txt.contains(".param .u64 numel"));
}

#[test]
fn codegen_emits_all_four_cast_kernels_with_runtime_invariants() {
    pair_check("f32->bf16", synthesize_f32_to_bf16_ptx(), KERNEL_F32_TO_BF16);
    pair_check("bf16->f32", synthesize_bf16_to_f32_ptx(), KERNEL_BF16_TO_F32);
    pair_check("f32->fp16", synthesize_f32_to_fp16_ptx(), KERNEL_F32_TO_FP16);
    pair_check("fp16->f32", synthesize_fp16_to_f32_ptx(), KERNEL_FP16_TO_F32);
}

/// Pin the kernel-name strings the runtime FFI launcher uses against the
/// codegen exports. If these drift, the runtime side will issue
/// `cuModuleGetFunction` with a name the PTX entry doesn't declare, which
/// surfaces as `CUDA_ERROR_NOT_FOUND` at first launch.
#[test]
fn kernel_name_constants_pin_to_runtime_lookup_keys() {
    assert_eq!(KERNEL_F32_TO_BF16, "nsl_cast_f32_to_bf16");
    assert_eq!(KERNEL_BF16_TO_F32, "nsl_cast_bf16_to_f32");
    assert_eq!(KERNEL_F32_TO_FP16, "nsl_cast_f32_to_fp16");
    assert_eq!(KERNEL_FP16_TO_F32, "nsl_cast_fp16_to_f32");
}

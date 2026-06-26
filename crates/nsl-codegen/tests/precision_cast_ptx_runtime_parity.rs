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
//!
//! # CFTP v7 follow-on — real byte-for-byte parity (findings 3/11/13)
//!
//! The original `codegen_emits_all_four_cast_kernels_with_runtime_invariants`
//! only checked structural substrings on the CODEGEN side and never
//! compared the two byte arrays. The runtime docstring claimed a
//! `cast_ptx_runtime_matches_codegen` test enforced byte parity — no such
//! test existed. This file now ships:
//!
//! * `runtime_embedded_ptx_matches_codegen_byte_for_byte` — compares the
//!   four embedded runtime constants against `synthesize_*_ptx()` byte
//!   by byte, including the trailing NUL.
//! * `runtime_embedded_kernel_names_match_codegen_byte_for_byte` — same
//!   discipline for the four kernel-name C strings.
//!
//! Both consume `__test_runtime_ptx_strings()` /
//! `__test_runtime_kernel_names()` hooks exposed by the runtime module.
//! Any future divergence (mnemonic swap, `.version` bump, reg-decl
//! reorder, blank-line shift) trips the test loudly instead of going
//! silently stale.

use nsl_codegen::precision_cast_ptx::{
    synthesize_bf16_to_f32_ptx, synthesize_f32_to_bf16_ptx, synthesize_f32_to_fp16_ptx,
    synthesize_fp16_to_f32_ptx, KERNEL_BF16_TO_F32, KERNEL_F32_TO_BF16, KERNEL_F32_TO_FP16,
    KERNEL_FP16_TO_F32,
};
use nsl_runtime::precision_cast_kernels::{
    __test_runtime_kernel_names, __test_runtime_ptx_strings,
};

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

/// CFTP v7 follow-on (findings 3/11/13): true byte-for-byte parity guard.
///
/// The runtime-embedded constants must equal the codegen synthesizer's
/// output exactly. Pinned ORDER: (F32->BF16, BF16->F32, F32->FP16, FP16->F32).
#[test]
fn runtime_embedded_ptx_matches_codegen_byte_for_byte() {
    let runtime = __test_runtime_ptx_strings();
    let codegen: [Vec<u8>; 4] = [
        synthesize_f32_to_bf16_ptx(),
        synthesize_bf16_to_f32_ptx(),
        synthesize_f32_to_fp16_ptx(),
        synthesize_fp16_to_f32_ptx(),
    ];
    let labels = ["f32->bf16", "bf16->f32", "f32->fp16", "fp16->f32"];
    for (i, label) in labels.iter().enumerate() {
        let runtime_bytes = runtime[i].as_bytes();
        let codegen_bytes = codegen[i].as_slice();
        assert_eq!(
            runtime_bytes.len(),
            codegen_bytes.len(),
            "[{label}] PTX byte-length differs: runtime={} codegen={} \
             — runtime embedded constant is stale relative to codegen",
            runtime_bytes.len(),
            codegen_bytes.len(),
        );
        // Index-by-index comparison so a divergence reports the offset.
        for (j, (&r, &c)) in runtime_bytes.iter().zip(codegen_bytes.iter()).enumerate() {
            assert_eq!(
                r, c,
                "[{label}] PTX byte differs at offset {j}: runtime=0x{r:02x} \
                 codegen=0x{c:02x} — update PTX_* constants in \
                 crates/nsl-runtime/src/cuda/precision_cast_kernels.rs to \
                 match the synthesizer output",
            );
        }
    }
}

/// Same byte-for-byte discipline for the kernel-name C strings.
#[test]
fn runtime_embedded_kernel_names_match_codegen_byte_for_byte() {
    let runtime = __test_runtime_kernel_names();
    // Codegen-side names are NOT NUL-terminated; append NUL to match the
    // runtime form (which embeds C strings for cuModuleGetFunction).
    let codegen: [String; 4] = [
        format!("{KERNEL_F32_TO_BF16}\0"),
        format!("{KERNEL_BF16_TO_F32}\0"),
        format!("{KERNEL_F32_TO_FP16}\0"),
        format!("{KERNEL_FP16_TO_F32}\0"),
    ];
    let labels = ["f32->bf16", "bf16->f32", "f32->fp16", "fp16->f32"];
    for (i, label) in labels.iter().enumerate() {
        assert_eq!(
            runtime[i], codegen[i].as_str(),
            "[{label}] kernel name C string differs (runtime={:?}, codegen+NUL={:?})",
            runtime[i], codegen[i],
        );
    }
}

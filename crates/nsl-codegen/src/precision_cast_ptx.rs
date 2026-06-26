//! CFTP v7 — element-wise precision-cast PTX kernels.
//!
//! Synthesises four standalone PTX modules, one per cast pair:
//!
//!   * `nsl_cast_f32_to_bf16` — RTE narrowing  f32 -> bf16  (PTX .version 8.0)
//!   * `nsl_cast_bf16_to_f32` — exact widening bf16 -> f32  (PTX .version 8.0)
//!   * `nsl_cast_f32_to_fp16` — RTE narrowing  f32 -> f16   (PTX .version 7.0)
//!   * `nsl_cast_fp16_to_f32` — exact widening f16  -> f32  (PTX .version 7.0)
//!
//! ## Design (see Discovery notes for full rationale)
//!
//! * Four separate monolithic kernels — one emitter per cast pair. Mirrors
//!   the `Dtype::F32|F16|Bf16` match-arm pattern in
//!   `fused_linear_ce::ptx_header()` (no runtime branching in PTX, no header
//!   conflation between the f16 path's `.version 7.0` and bf16's
//!   `.version 8.0`).
//! * Grid topology: 1D `block=(256,1,1)`, grid sized by the caller. Each
//!   kernel uses a grid-stride loop so a caller that clamps gridDim.x to
//!   65535 still covers arbitrarily-large `numel` correctly.
//! * Zero shared memory — pure element-wise, no inter-thread comms.
//! * FFI signature: `(src_ptr: .u64, dst_ptr: .u64, numel: .u64)`.
//! * PTX bytes are NUL-terminated to satisfy `cuModuleLoadData`'s C-string
//!   contract (same invariant as `synthesize_fused_linear_ce_ptx`).
//! * ASCII-only — Unicode in PTX triggers `CUDA_ERROR_INVALID_PTX` under
//!   the cudarc JIT (same gate the F32/F16/Bf16 fused-LCE paths enforce).
//! * `mad.lo.u32` is BANNED per the project-wide PTX ISA 7.0 invariant
//!   (see MEMORY.md). The grid-stride index computation uses
//!   `mul.lo.u32` + `add.u32` instead.

/// Block dim used by every cast kernel. Matches the codebase-standard CTA
/// width for memory-bound element-wise work; the caller MUST launch with
/// `block = (256, 1, 1)`.
pub const CAST_BLOCK_DIM_X: u32 = 256;

/// Kernel-name constants. Match the FFI launcher's lookup keys; do NOT
/// rename without updating the runtime side simultaneously.
pub const KERNEL_F32_TO_BF16: &str = "nsl_cast_f32_to_bf16";
pub const KERNEL_BF16_TO_F32: &str = "nsl_cast_bf16_to_f32";
pub const KERNEL_F32_TO_FP16: &str = "nsl_cast_f32_to_fp16";
pub const KERNEL_FP16_TO_F32: &str = "nsl_cast_fp16_to_f32";

/// Emit the standard PTX header for a cast kernel.
///
/// * `version` is `"7.0"` for f16 pairs (`cvt.{f32.f16, rn.f16.f32}` are
///   available in PTX 6.0+) and `"8.0"` for bf16 pairs (`cvt.f32.bf16` /
///   `cvt.rn.bf16.f32` require PTX ISA 7.8+ and `.target sm_80+`).
/// * `.target sm_80` is the floor; mirrors `FusedLinearCEConfig::sm_tag()`'s
///   `self.gpu_sm.max(80)` clamp.
fn emit_header(ptx: &mut String, version: &str) {
    ptx.push_str(&format!(".version {version}\n"));
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");
}

/// Emit the shared kernel preamble: param-load + grid-stride index and
/// stride registers.
///
/// On exit, the following are live:
///   * `%rd_src`  — source base pointer
///   * `%rd_dst`  — destination base pointer
///   * `%r_numel` — element count (u32; cast from u64 param)
///   * `%r_idx`   — current element index (grid-stride init)
///   * `%r_stride` — grid stride (`gridDim.x * blockDim.x`)
///   * `%p_done`  — predicate register reserved for the bounds check
fn emit_entry_preamble(ptx: &mut String, kernel_name: &str) {
    ptx.push_str(&format!(".visible .entry {kernel_name} (\n"));
    ptx.push_str("    .param .u64 src_ptr,\n");
    ptx.push_str("    .param .u64 dst_ptr,\n");
    ptx.push_str("    .param .u64 numel\n");
    ptx.push_str(")\n");
    ptx.push_str("{\n");

    // Register declarations.
    ptx.push_str("    .reg .pred %p_done;\n");
    ptx.push_str("    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;\n");
    ptx.push_str("    .reg .u32 %r_numel, %r_idx, %r_stride, %r_tmp;\n");
    ptx.push_str("    .reg .u64 %rd_src, %rd_dst, %rd_numel64;\n");
    ptx.push_str("    .reg .u64 %rd_off, %rd_off_bytes, %rd_addr;\n");
    ptx.push_str("    .reg .f32 %f_val;\n");
    ptx.push_str("    .reg .b16 %h_val;\n\n");

    // Load src / dst base pointers.
    ptx.push_str("    ld.param.u64 %rd_src, [src_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_dst, [dst_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_numel64, [numel];\n");
    ptx.push_str("    cvt.u32.u64 %r_numel, %rd_numel64;\n\n");

    // tid / cta / ntid / nctaid.
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    mov.u32 %ctaid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %ntid_x, %ntid.x;\n");
    ptx.push_str("    mov.u32 %nctaid_x, %nctaid.x;\n\n");

    // idx = ctaid.x * ntid.x + tid.x. mad.lo.u32 is banned at PTX ISA 7.0;
    // emit mul.lo + add (see MEMORY.md global GPU invariants).
    ptx.push_str("    mul.lo.u32 %r_tmp, %ctaid_x, %ntid_x;\n");
    ptx.push_str("    add.u32 %r_idx, %r_tmp, %tid_x;\n");

    // stride = nctaid.x * ntid.x.
    ptx.push_str("    mul.lo.u32 %r_stride, %nctaid_x, %ntid_x;\n\n");
}

/// Common kernel body suffix: closes the grid-stride loop and `ret`s.
fn emit_loop_tail(ptx: &mut String) {
    ptx.push_str("    add.u32 %r_idx, %r_idx, %r_stride;\n");
    ptx.push_str("    bra CAST_LOOP;\n");
    ptx.push_str("CAST_DONE:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

/// Build the f32 -> {bf16|f16} narrowing kernel body.
///
/// `cvt_mnemonic` is the full mnemonic the per-pair caller supplies, e.g.
/// `"cvt.rn.bf16.f32"` or `"cvt.rn.f16.f32"`. The `.rn.` round-to-nearest-
/// even modifier is the established pattern (see `fused_linear_ce.rs:1028`).
fn emit_narrow_body(ptx: &mut String, cvt_mnemonic: &str) {
    ptx.push_str("CAST_LOOP:\n");
    ptx.push_str("    setp.ge.u32 %p_done, %r_idx, %r_numel;\n");
    ptx.push_str("    @%p_done bra CAST_DONE;\n");

    // Source offset (f32 = 4 bytes).
    ptx.push_str("    cvt.u64.u32 %rd_off, %r_idx;\n");
    ptx.push_str("    shl.b64 %rd_off_bytes, %rd_off, 2;\n");
    ptx.push_str("    add.u64 %rd_addr, %rd_src, %rd_off_bytes;\n");
    ptx.push_str("    ld.global.f32 %f_val, [%rd_addr];\n");

    // Narrowing conversion: f32 -> b16 (bf16 or f16).
    ptx.push_str(&format!("    {cvt_mnemonic} %h_val, %f_val;\n"));

    // Destination offset (b16 = 2 bytes).
    ptx.push_str("    shl.b64 %rd_off_bytes, %rd_off, 1;\n");
    ptx.push_str("    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;\n");
    ptx.push_str("    st.global.b16 [%rd_addr], %h_val;\n");

    emit_loop_tail(ptx);
}

/// Build the {bf16|f16} -> f32 widening kernel body.
///
/// `cvt_mnemonic` is the widening cvt mnemonic — no rounding modifier is
/// allowed because the widening is exact (every bf16/f16 value fits in f32).
/// Example values: `"cvt.f32.bf16"` or `"cvt.f32.f16"`.
fn emit_widen_body(ptx: &mut String, cvt_mnemonic: &str) {
    ptx.push_str("CAST_LOOP:\n");
    ptx.push_str("    setp.ge.u32 %p_done, %r_idx, %r_numel;\n");
    ptx.push_str("    @%p_done bra CAST_DONE;\n");

    // Source offset (b16 = 2 bytes).
    ptx.push_str("    cvt.u64.u32 %rd_off, %r_idx;\n");
    ptx.push_str("    shl.b64 %rd_off_bytes, %rd_off, 1;\n");
    ptx.push_str("    add.u64 %rd_addr, %rd_src, %rd_off_bytes;\n");
    ptx.push_str("    ld.global.b16 %h_val, [%rd_addr];\n");

    // Widening conversion: b16 (bf16 or f16) -> f32.
    ptx.push_str(&format!("    {cvt_mnemonic} %f_val, %h_val;\n"));

    // Destination offset (f32 = 4 bytes).
    ptx.push_str("    shl.b64 %rd_off_bytes, %rd_off, 2;\n");
    ptx.push_str("    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;\n");
    ptx.push_str("    st.global.f32 [%rd_addr], %f_val;\n");

    emit_loop_tail(ptx);
}

/// Synthesize the f32 -> bf16 (RTE) cast kernel as a NUL-terminated PTX module.
pub fn synthesize_f32_to_bf16_ptx() -> Vec<u8> {
    let mut ptx = String::with_capacity(2048);
    // bf16 cvt mnemonics require PTX ISA 7.8+; bump to 8.0 to match the
    // fused-LCE bf16 path's header policy (fused_linear_ce.rs:404-410).
    emit_header(&mut ptx, "8.0");
    emit_entry_preamble(&mut ptx, KERNEL_F32_TO_BF16);
    emit_narrow_body(&mut ptx, "cvt.rn.bf16.f32");
    ptx.push('\0');
    ptx.into_bytes()
}

/// Synthesize the bf16 -> f32 (exact) cast kernel as a NUL-terminated PTX module.
pub fn synthesize_bf16_to_f32_ptx() -> Vec<u8> {
    let mut ptx = String::with_capacity(2048);
    emit_header(&mut ptx, "8.0");
    emit_entry_preamble(&mut ptx, KERNEL_BF16_TO_F32);
    emit_widen_body(&mut ptx, "cvt.f32.bf16");
    ptx.push('\0');
    ptx.into_bytes()
}

/// Synthesize the f32 -> f16 (RTE) cast kernel as a NUL-terminated PTX module.
pub fn synthesize_f32_to_fp16_ptx() -> Vec<u8> {
    let mut ptx = String::with_capacity(2048);
    // f16 cvt mnemonics are valid at PTX 6.0+; stay at 7.0 to mirror the
    // fused-LCE F16 path's header policy.
    emit_header(&mut ptx, "7.0");
    emit_entry_preamble(&mut ptx, KERNEL_F32_TO_FP16);
    emit_narrow_body(&mut ptx, "cvt.rn.f16.f32");
    ptx.push('\0');
    ptx.into_bytes()
}

/// Synthesize the f16 -> f32 (exact) cast kernel as a NUL-terminated PTX module.
pub fn synthesize_fp16_to_f32_ptx() -> Vec<u8> {
    let mut ptx = String::with_capacity(2048);
    emit_header(&mut ptx, "7.0");
    emit_entry_preamble(&mut ptx, KERNEL_FP16_TO_F32);
    emit_widen_body(&mut ptx, "cvt.f32.f16");
    ptx.push('\0');
    ptx.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_text(bytes: &[u8]) -> &str {
        // Drop the trailing NUL for substring assertions.
        let end = bytes.len().saturating_sub(1);
        std::str::from_utf8(&bytes[..end]).expect("PTX must be ASCII")
    }

    fn assert_nul_terminated(bytes: &[u8], tag: &str) {
        assert!(
            bytes.last() == Some(&0),
            "{tag} PTX MUST be NUL-terminated (cuModuleLoadData reads bytes as C string)"
        );
        // Exactly one trailing NUL.
        assert!(
            !bytes[..bytes.len() - 1].contains(&0),
            "{tag} PTX must contain exactly ONE NUL byte (at the end)"
        );
    }

    fn assert_ascii_only(bytes: &[u8], tag: &str) {
        for &b in bytes {
            assert!(
                b < 128,
                "{tag} PTX must be ASCII-only (non-ASCII triggers \
                 CUDA_ERROR_INVALID_PTX under cudarc JIT); found 0x{b:02x}"
            );
        }
    }

    // ─── f32 -> bf16 ────────────────────────────────────────────────────────

    #[test]
    fn f32_to_bf16_structural() {
        let ptx = synthesize_f32_to_bf16_ptx();
        assert_nul_terminated(&ptx, "f32_to_bf16");
        assert_ascii_only(&ptx, "f32_to_bf16");
        let txt = as_text(&ptx);

        assert!(txt.contains(".version 8.0"), "bf16 narrowing MUST use PTX .version 8.0");
        assert!(txt.contains(".target sm_80"));
        assert!(txt.contains(".address_size 64"));
        assert!(
            txt.contains(".visible .entry nsl_cast_f32_to_bf16"),
            "kernel entry name pinned by FFI launcher"
        );
        assert!(
            txt.contains("cvt.rn.bf16.f32"),
            "narrowing cast MUST emit cvt.rn.bf16.f32 (RTE)"
        );
        assert!(txt.contains("ld.global.f32"), "source load MUST be f32");
        assert!(txt.contains("st.global.b16"), "destination store MUST be .b16");
        // Strict ban: mad.lo.u32 is invalid at PTX ISA 7.0 (and we keep the
        // same discipline at 8.0 — see MEMORY.md global GPU invariants).
        assert!(
            !txt.contains("mad.lo.u32"),
            "mad.lo.u32 is banned (PTX ISA 7.0 invariant); use mul.lo + add"
        );
        // Wrong rounding modifier (cvt.rz / cvt.rm / etc.) would be a silent
        // numerical bug — confirm rn is the only rounding modifier in use.
        assert!(!txt.contains("cvt.rz.bf16.f32"));
        assert!(!txt.contains("cvt.rm.bf16.f32"));
        // f16 cvt must NOT appear (cross-pair leak guard).
        assert!(!txt.contains("cvt.rn.f16.f32"));
    }

    // ─── bf16 -> f32 ────────────────────────────────────────────────────────

    #[test]
    fn bf16_to_f32_structural() {
        let ptx = synthesize_bf16_to_f32_ptx();
        assert_nul_terminated(&ptx, "bf16_to_f32");
        assert_ascii_only(&ptx, "bf16_to_f32");
        let txt = as_text(&ptx);

        assert!(txt.contains(".version 8.0"));
        assert!(txt.contains(".target sm_80"));
        assert!(txt.contains(".visible .entry nsl_cast_bf16_to_f32"));
        assert!(
            txt.contains("cvt.f32.bf16"),
            "widening cast MUST emit cvt.f32.bf16 (exact)"
        );
        assert!(txt.contains("ld.global.b16"), "source load MUST be .b16");
        assert!(txt.contains("st.global.f32"), "destination store MUST be f32");
        assert!(!txt.contains("mad.lo.u32"));
        assert!(!txt.contains("cvt.f32.f16"));
    }

    // ─── f32 -> fp16 ────────────────────────────────────────────────────────

    #[test]
    fn f32_to_fp16_structural() {
        let ptx = synthesize_f32_to_fp16_ptx();
        assert_nul_terminated(&ptx, "f32_to_fp16");
        assert_ascii_only(&ptx, "f32_to_fp16");
        let txt = as_text(&ptx);

        assert!(
            txt.contains(".version 7.0"),
            "f16 narrowing stays at PTX 7.0 (mirrors fused-LCE F16 header)"
        );
        assert!(txt.contains(".target sm_80"));
        assert!(txt.contains(".visible .entry nsl_cast_f32_to_fp16"));
        assert!(
            txt.contains("cvt.rn.f16.f32"),
            "narrowing cast MUST emit cvt.rn.f16.f32 (RTE)"
        );
        assert!(txt.contains("ld.global.f32"));
        assert!(txt.contains("st.global.b16"));
        assert!(!txt.contains("mad.lo.u32"));
        assert!(!txt.contains("cvt.rn.bf16.f32"));
    }

    // ─── fp16 -> f32 ────────────────────────────────────────────────────────

    #[test]
    fn fp16_to_f32_structural() {
        let ptx = synthesize_fp16_to_f32_ptx();
        assert_nul_terminated(&ptx, "fp16_to_f32");
        assert_ascii_only(&ptx, "fp16_to_f32");
        let txt = as_text(&ptx);

        assert!(txt.contains(".version 7.0"));
        assert!(txt.contains(".target sm_80"));
        assert!(txt.contains(".visible .entry nsl_cast_fp16_to_f32"));
        assert!(
            txt.contains("cvt.f32.f16"),
            "widening cast MUST emit cvt.f32.f16 (exact)"
        );
        assert!(txt.contains("ld.global.b16"));
        assert!(txt.contains("st.global.f32"));
        assert!(!txt.contains("mad.lo.u32"));
        assert!(!txt.contains("cvt.f32.bf16"));
    }

    // ─── Cross-cutting invariants ────────────────────────────────────────────

    #[test]
    fn all_four_kernels_use_grid_stride_loop() {
        for ptx_bytes in [
            synthesize_f32_to_bf16_ptx(),
            synthesize_bf16_to_f32_ptx(),
            synthesize_f32_to_fp16_ptx(),
            synthesize_fp16_to_f32_ptx(),
        ] {
            let txt = as_text(&ptx_bytes);
            assert!(txt.contains("CAST_LOOP:"), "grid-stride loop label missing");
            assert!(txt.contains("CAST_DONE:"), "loop-done label missing");
            assert!(
                txt.contains("add.u32 %r_idx, %r_idx, %r_stride;"),
                "grid-stride advance line missing"
            );
            // Stride is gridDim.x * blockDim.x (= nctaid.x * ntid.x).
            assert!(
                txt.contains("mul.lo.u32 %r_stride, %nctaid_x, %ntid_x;"),
                "stride must be nctaid.x * ntid.x"
            );
        }
    }

    #[test]
    fn all_four_kernels_have_three_param_ffi() {
        for ptx_bytes in [
            synthesize_f32_to_bf16_ptx(),
            synthesize_bf16_to_f32_ptx(),
            synthesize_f32_to_fp16_ptx(),
            synthesize_fp16_to_f32_ptx(),
        ] {
            let txt = as_text(&ptx_bytes);
            assert!(txt.contains(".param .u64 src_ptr"));
            assert!(txt.contains(".param .u64 dst_ptr"));
            assert!(txt.contains(".param .u64 numel"));
        }
    }

    #[test]
    fn kernel_name_constants_match_emitted_entries() {
        // Pin the exported constants to the actual emitted entry names so
        // the runtime launcher can't drift from the codegen side.
        assert!(
            as_text(&synthesize_f32_to_bf16_ptx()).contains(KERNEL_F32_TO_BF16)
        );
        assert!(
            as_text(&synthesize_bf16_to_f32_ptx()).contains(KERNEL_BF16_TO_F32)
        );
        assert!(
            as_text(&synthesize_f32_to_fp16_ptx()).contains(KERNEL_F32_TO_FP16)
        );
        assert!(
            as_text(&synthesize_fp16_to_f32_ptx()).contains(KERNEL_FP16_TO_F32)
        );
    }
}

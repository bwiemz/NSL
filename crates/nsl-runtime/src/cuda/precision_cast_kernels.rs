//! CFTP v7 — precision-cast PTX kernel launchers.
//!
//! Mirrors the four cast kernels emitted by
//! `nsl_codegen::precision_cast_ptx::synthesize_*` so the runtime can dispatch
//! GPU casts from `nsl_tensor_to_{bf16,fp16,f32}` without taking a build-time
//! dependency on `nsl-codegen` (the reverse direction — codegen depends on
//! runtime — is the established crate layering).
//!
//! ## Why embed the PTX here
//!
//! `nsl-codegen` already depends on `nsl-runtime`; reversing that creates a
//! cycle. The PTX is short, ASCII, and identical to the codegen-emitted
//! bytes, so we embed the four NUL-terminated strings as `static` arrays.
//! The codegen tests in `crates/nsl-codegen/src/precision_cast_ptx.rs` keep
//! pinning the emitted structure; the strings here are a CONSTANT
//! materialisation of that emitter at a single point in time — if the codegen
//! ever moves, the runtime strings here must move in lockstep (a
//! `cast_ptx_runtime_matches_codegen` test enforces this byte-for-byte).
//!
//! ## Module cache
//!
//! Reuses `crate::cuda::inner::kernel_launch`'s internal FNV-1a-keyed cache;
//! no per-launch `cuModuleLoadData`. (The first call seeds the cache; every
//! subsequent call hits it.)

#![allow(dead_code)]

#[cfg(feature = "cuda")]
use std::ffi::c_void;

// ─── Kernel names (must match codegen entries) ────────────────────────────

/// Kernel-name C strings (NUL-terminated). Pinned by codegen tests
/// (`kernel_name_constants_match_emitted_entries`).
pub(crate) const KNAME_F32_TO_BF16: &str = "nsl_cast_f32_to_bf16\0";
pub(crate) const KNAME_BF16_TO_F32: &str = "nsl_cast_bf16_to_f32\0";
pub(crate) const KNAME_F32_TO_FP16: &str = "nsl_cast_f32_to_fp16\0";
pub(crate) const KNAME_FP16_TO_F32: &str = "nsl_cast_fp16_to_f32\0";

/// Block dim every cast kernel launches with. Pinned by codegen
/// (`CAST_BLOCK_DIM_X = 256`).
pub(crate) const CAST_BLOCK_DIM_X: u32 = 256;

// ─── Embedded PTX (must be byte-identical to codegen output) ──────────────
//
// These four strings reproduce the codegen emitter's exact output. The
// `cast_ptx_runtime_matches_codegen` test (in `crates/nsl-codegen/tests/`)
// pins the equality. PTX is ASCII-only and NUL-terminated per the
// `cuModuleLoadData` C-string contract.

/// f32 -> bf16 (RTE). `.version 8.0` because bf16 cvt mnemonics need
/// PTX ISA 7.8+.
pub(crate) const PTX_F32_TO_BF16: &str = concat!(
    ".version 8.0\n",
    ".target sm_80\n",
    ".address_size 64\n\n",
    ".visible .entry nsl_cast_f32_to_bf16 (\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 numel\n",
    ")\n",
    "{\n",
    "    .reg .pred %p_done;\n",
    "    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;\n",
    "    .reg .u32 %r_numel, %r_idx, %r_stride, %r_tmp;\n",
    "    .reg .u64 %rd_src, %rd_dst, %rd_numel64;\n",
    "    .reg .u64 %rd_off, %rd_off_bytes, %rd_addr;\n",
    "    .reg .f32 %f_val;\n",
    "    .reg .b16 %h_val;\n\n",
    "    ld.param.u64 %rd_src, [src_ptr];\n",
    "    ld.param.u64 %rd_dst, [dst_ptr];\n",
    "    ld.param.u64 %rd_numel64, [numel];\n",
    "    cvt.u32.u64 %r_numel, %rd_numel64;\n\n",
    "    mov.u32 %tid_x, %tid.x;\n",
    "    mov.u32 %ctaid_x, %ctaid.x;\n",
    "    mov.u32 %ntid_x, %ntid.x;\n",
    "    mov.u32 %nctaid_x, %nctaid.x;\n\n",
    "    mul.lo.u32 %r_tmp, %ctaid_x, %ntid_x;\n",
    "    add.u32 %r_idx, %r_tmp, %tid_x;\n",
    "    mul.lo.u32 %r_stride, %nctaid_x, %ntid_x;\n\n",
    "CAST_LOOP:\n",
    "    setp.ge.u32 %p_done, %r_idx, %r_numel;\n",
    "    @%p_done bra CAST_DONE;\n",
    "    cvt.u64.u32 %rd_off, %r_idx;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 2;\n",
    "    add.u64 %rd_addr, %rd_src, %rd_off_bytes;\n",
    "    ld.global.f32 %f_val, [%rd_addr];\n",
    "    cvt.rn.bf16.f32 %h_val, %f_val;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 1;\n",
    "    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;\n",
    "    st.global.b16 [%rd_addr], %h_val;\n",
    "    add.u32 %r_idx, %r_idx, %r_stride;\n",
    "    bra CAST_LOOP;\n",
    "CAST_DONE:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

/// bf16 -> f32 (exact widening).
pub(crate) const PTX_BF16_TO_F32: &str = concat!(
    ".version 8.0\n",
    ".target sm_80\n",
    ".address_size 64\n\n",
    ".visible .entry nsl_cast_bf16_to_f32 (\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 numel\n",
    ")\n",
    "{\n",
    "    .reg .pred %p_done;\n",
    "    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;\n",
    "    .reg .u32 %r_numel, %r_idx, %r_stride, %r_tmp;\n",
    "    .reg .u64 %rd_src, %rd_dst, %rd_numel64;\n",
    "    .reg .u64 %rd_off, %rd_off_bytes, %rd_addr;\n",
    "    .reg .f32 %f_val;\n",
    "    .reg .b16 %h_val;\n\n",
    "    ld.param.u64 %rd_src, [src_ptr];\n",
    "    ld.param.u64 %rd_dst, [dst_ptr];\n",
    "    ld.param.u64 %rd_numel64, [numel];\n",
    "    cvt.u32.u64 %r_numel, %rd_numel64;\n\n",
    "    mov.u32 %tid_x, %tid.x;\n",
    "    mov.u32 %ctaid_x, %ctaid.x;\n",
    "    mov.u32 %ntid_x, %ntid.x;\n",
    "    mov.u32 %nctaid_x, %nctaid.x;\n\n",
    "    mul.lo.u32 %r_tmp, %ctaid_x, %ntid_x;\n",
    "    add.u32 %r_idx, %r_tmp, %tid_x;\n",
    "    mul.lo.u32 %r_stride, %nctaid_x, %ntid_x;\n\n",
    "CAST_LOOP:\n",
    "    setp.ge.u32 %p_done, %r_idx, %r_numel;\n",
    "    @%p_done bra CAST_DONE;\n",
    "    cvt.u64.u32 %rd_off, %r_idx;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 1;\n",
    "    add.u64 %rd_addr, %rd_src, %rd_off_bytes;\n",
    "    ld.global.b16 %h_val, [%rd_addr];\n",
    "    cvt.f32.bf16 %f_val, %h_val;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 2;\n",
    "    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;\n",
    "    st.global.f32 [%rd_addr], %f_val;\n",
    "    add.u32 %r_idx, %r_idx, %r_stride;\n",
    "    bra CAST_LOOP;\n",
    "CAST_DONE:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

/// f32 -> f16 (RTE). `.version 7.0` mirrors the fused-LCE F16 path header.
pub(crate) const PTX_F32_TO_FP16: &str = concat!(
    ".version 7.0\n",
    ".target sm_80\n",
    ".address_size 64\n\n",
    ".visible .entry nsl_cast_f32_to_fp16 (\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 numel\n",
    ")\n",
    "{\n",
    "    .reg .pred %p_done;\n",
    "    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;\n",
    "    .reg .u32 %r_numel, %r_idx, %r_stride, %r_tmp;\n",
    "    .reg .u64 %rd_src, %rd_dst, %rd_numel64;\n",
    "    .reg .u64 %rd_off, %rd_off_bytes, %rd_addr;\n",
    "    .reg .f32 %f_val;\n",
    "    .reg .b16 %h_val;\n\n",
    "    ld.param.u64 %rd_src, [src_ptr];\n",
    "    ld.param.u64 %rd_dst, [dst_ptr];\n",
    "    ld.param.u64 %rd_numel64, [numel];\n",
    "    cvt.u32.u64 %r_numel, %rd_numel64;\n\n",
    "    mov.u32 %tid_x, %tid.x;\n",
    "    mov.u32 %ctaid_x, %ctaid.x;\n",
    "    mov.u32 %ntid_x, %ntid.x;\n",
    "    mov.u32 %nctaid_x, %nctaid.x;\n\n",
    "    mul.lo.u32 %r_tmp, %ctaid_x, %ntid_x;\n",
    "    add.u32 %r_idx, %r_tmp, %tid_x;\n",
    "    mul.lo.u32 %r_stride, %nctaid_x, %ntid_x;\n\n",
    "CAST_LOOP:\n",
    "    setp.ge.u32 %p_done, %r_idx, %r_numel;\n",
    "    @%p_done bra CAST_DONE;\n",
    "    cvt.u64.u32 %rd_off, %r_idx;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 2;\n",
    "    add.u64 %rd_addr, %rd_src, %rd_off_bytes;\n",
    "    ld.global.f32 %f_val, [%rd_addr];\n",
    "    cvt.rn.f16.f32 %h_val, %f_val;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 1;\n",
    "    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;\n",
    "    st.global.b16 [%rd_addr], %h_val;\n",
    "    add.u32 %r_idx, %r_idx, %r_stride;\n",
    "    bra CAST_LOOP;\n",
    "CAST_DONE:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

/// f16 -> f32 (exact widening).
pub(crate) const PTX_FP16_TO_F32: &str = concat!(
    ".version 7.0\n",
    ".target sm_80\n",
    ".address_size 64\n\n",
    ".visible .entry nsl_cast_fp16_to_f32 (\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 numel\n",
    ")\n",
    "{\n",
    "    .reg .pred %p_done;\n",
    "    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;\n",
    "    .reg .u32 %r_numel, %r_idx, %r_stride, %r_tmp;\n",
    "    .reg .u64 %rd_src, %rd_dst, %rd_numel64;\n",
    "    .reg .u64 %rd_off, %rd_off_bytes, %rd_addr;\n",
    "    .reg .f32 %f_val;\n",
    "    .reg .b16 %h_val;\n\n",
    "    ld.param.u64 %rd_src, [src_ptr];\n",
    "    ld.param.u64 %rd_dst, [dst_ptr];\n",
    "    ld.param.u64 %rd_numel64, [numel];\n",
    "    cvt.u32.u64 %r_numel, %rd_numel64;\n\n",
    "    mov.u32 %tid_x, %tid.x;\n",
    "    mov.u32 %ctaid_x, %ctaid.x;\n",
    "    mov.u32 %ntid_x, %ntid.x;\n",
    "    mov.u32 %nctaid_x, %nctaid.x;\n\n",
    "    mul.lo.u32 %r_tmp, %ctaid_x, %ntid_x;\n",
    "    add.u32 %r_idx, %r_tmp, %tid_x;\n",
    "    mul.lo.u32 %r_stride, %nctaid_x, %ntid_x;\n\n",
    "CAST_LOOP:\n",
    "    setp.ge.u32 %p_done, %r_idx, %r_numel;\n",
    "    @%p_done bra CAST_DONE;\n",
    "    cvt.u64.u32 %rd_off, %r_idx;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 1;\n",
    "    add.u64 %rd_addr, %rd_src, %rd_off_bytes;\n",
    "    ld.global.b16 %h_val, [%rd_addr];\n",
    "    cvt.f32.f16 %f_val, %h_val;\n",
    "    shl.b64 %rd_off_bytes, %rd_off, 2;\n",
    "    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;\n",
    "    st.global.f32 [%rd_addr], %f_val;\n",
    "    add.u32 %r_idx, %r_idx, %r_stride;\n",
    "    bra CAST_LOOP;\n",
    "CAST_DONE:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

/// Pick the (PTX, kernel-name) pair for a (`src_dtype`, `target_dtype`) cast.
/// Returns `None` if the cast pair has no GPU kernel (caller must refuse / take
/// a different path).
///
/// `src_dtype` / `target_dtype` use `crate::tensor::DTYPE_*` constants
/// (F32=1, FP16=2, BF16=3).
pub(crate) fn pick_cast_kernel(src_dtype: u16, target_dtype: u16) -> Option<(&'static str, &'static str)> {
    use crate::tensor::{DTYPE_BF16, DTYPE_F32, DTYPE_FP16};
    match (src_dtype, target_dtype) {
        (DTYPE_F32, DTYPE_BF16) => Some((PTX_F32_TO_BF16, KNAME_F32_TO_BF16)),
        (DTYPE_BF16, DTYPE_F32) => Some((PTX_BF16_TO_F32, KNAME_BF16_TO_F32)),
        (DTYPE_F32, DTYPE_FP16) => Some((PTX_F32_TO_FP16, KNAME_F32_TO_FP16)),
        (DTYPE_FP16, DTYPE_F32) => Some((PTX_FP16_TO_F32, KNAME_FP16_TO_F32)),
        // Same-dtype "casts" (e.g. F32->F32 / BF16->BF16) are caller-handled
        // via cuMemcpyDtoD (no conversion needed).
        _ => None,
    }
}

/// Launch a precision-cast kernel with FFI signature
/// `(src_ptr: .u64, dst_ptr: .u64, numel: .u64)`.
///
/// * `src_dev` / `dst_dev` are device pointers (u64) — caller is responsible
///   for allocating `dst_dev` with `numel * sizeof(target_dtype)` bytes.
/// * Block size is fixed at `CAST_BLOCK_DIM_X = 256`.
/// * Grid is `ceil(numel / block)` (clamped to u32::MAX); the kernel uses a
///   grid-stride loop so any clamp still covers `numel` correctly.
/// * Shared mem = 0 (pure element-wise).
///
/// Returns 0 on success, non-zero CUresult on failure.
#[cfg(feature = "cuda")]
pub(crate) fn launch_cast(
    ptx: &str,
    kernel_name: &str,
    src_dev: u64,
    dst_dev: u64,
    numel: u64,
) -> u32 {
    let mut src_ptr = src_dev;
    let mut dst_ptr = dst_dev;
    let mut n_val = numel;

    let args: [*mut c_void; 3] = [
        &mut src_ptr as *mut _ as *mut c_void,
        &mut dst_ptr as *mut _ as *mut c_void,
        &mut n_val as *mut _ as *mut c_void,
    ];

    let block = CAST_BLOCK_DIM_X as i64;
    // Cap gridDim.x at u32::MAX (CUDA hardware limit); the kernel's grid-stride
    // loop covers anything above that. In practice tensors with > 2^31 elements
    // are not realistic but the grid-stride loop guarantees correctness.
    let raw_grid = (numel as i64 + block - 1) / block;
    let grid = raw_grid.min(u32::MAX as i64).max(1);

    let result = crate::cuda::inner::kernel_launch(
        ptx.as_ptr(),
        kernel_name.as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0, // no dynamic shared memory
    );
    result as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every embedded PTX must be NUL-terminated (cuModuleLoadData C-string contract).
    #[test]
    fn embedded_ptx_strings_are_nul_terminated() {
        for (tag, ptx) in [
            ("PTX_F32_TO_BF16", PTX_F32_TO_BF16),
            ("PTX_BF16_TO_F32", PTX_BF16_TO_F32),
            ("PTX_F32_TO_FP16", PTX_F32_TO_FP16),
            ("PTX_FP16_TO_F32", PTX_FP16_TO_F32),
        ] {
            assert!(
                ptx.ends_with('\0'),
                "{tag} must end in NUL (cuModuleLoadData reads as C string)"
            );
            // Exactly one trailing NUL.
            assert!(
                !ptx[..ptx.len() - 1].contains('\0'),
                "{tag} must contain exactly ONE NUL byte (at the end)"
            );
        }
    }

    /// PTX must be ASCII-only (cudarc JIT trips CUDA_ERROR_INVALID_PTX on
    /// non-ASCII; see global GPU invariants in MEMORY.md).
    #[test]
    fn embedded_ptx_strings_are_ascii() {
        for (tag, ptx) in [
            ("PTX_F32_TO_BF16", PTX_F32_TO_BF16),
            ("PTX_BF16_TO_F32", PTX_BF16_TO_F32),
            ("PTX_F32_TO_FP16", PTX_F32_TO_FP16),
            ("PTX_FP16_TO_F32", PTX_FP16_TO_F32),
        ] {
            assert!(ptx.is_ascii(), "{tag} must be ASCII-only");
        }
    }

    /// Kernel-name strings are pinned by the codegen FFI launcher.
    #[test]
    fn kernel_name_strings_are_nul_terminated() {
        for n in [KNAME_F32_TO_BF16, KNAME_BF16_TO_F32, KNAME_F32_TO_FP16, KNAME_FP16_TO_F32] {
            assert!(n.ends_with('\0'), "kernel name must be NUL-terminated");
        }
    }

    /// The PTX strings are `static`, so every call to `pick_cast_kernel` for
    /// the same pair returns pointers into the SAME memory. This is what
    /// makes the FNV-1a-keyed module cache in `kernel_launch` work
    /// efficiently — the bytes hashed are stable across calls.
    ///
    /// Pin the static-address invariant so a future refactor that returns
    /// freshly-allocated `Vec<u8>` from `pick_cast_kernel` would surface
    /// here (and not via a silent perf regression where the module cache
    /// re-loads every call).
    #[test]
    fn embedded_ptx_strings_have_stable_addresses_across_calls() {
        use crate::tensor::{DTYPE_BF16, DTYPE_F32};
        let (ptx_a, kname_a) = pick_cast_kernel(DTYPE_F32, DTYPE_BF16).unwrap();
        let (ptx_b, kname_b) = pick_cast_kernel(DTYPE_F32, DTYPE_BF16).unwrap();
        assert_eq!(
            ptx_a.as_ptr() as usize,
            ptx_b.as_ptr() as usize,
            "PTX must be `static` (cache-friendly): repeated lookup must \
             return identical address; otherwise the FNV-1a module cache \
             will re-load the module on every launch"
        );
        assert_eq!(
            kname_a.as_ptr() as usize,
            kname_b.as_ptr() as usize,
            "kernel-name lookup must also be `static`"
        );
    }

    #[test]
    fn pick_cast_kernel_covers_supported_pairs() {
        use crate::tensor::{DTYPE_BF16, DTYPE_F32, DTYPE_FP16};
        assert!(pick_cast_kernel(DTYPE_F32, DTYPE_BF16).is_some());
        assert!(pick_cast_kernel(DTYPE_BF16, DTYPE_F32).is_some());
        assert!(pick_cast_kernel(DTYPE_F32, DTYPE_FP16).is_some());
        assert!(pick_cast_kernel(DTYPE_FP16, DTYPE_F32).is_some());
        // Same-dtype: caller handles via memcpy.
        assert!(pick_cast_kernel(DTYPE_F32, DTYPE_F32).is_none());
        assert!(pick_cast_kernel(DTYPE_BF16, DTYPE_BF16).is_none());
        // bf16 <-> fp16 not supported — caller must stage via f32.
        assert!(pick_cast_kernel(DTYPE_BF16, DTYPE_FP16).is_none());
        assert!(pick_cast_kernel(DTYPE_FP16, DTYPE_BF16).is_none());
    }
}

//! Precision-cast kernel launchers (roadmap A2 step 7).
//!
//! The four element-wise casts — f32 <-> bf16 and f32 <-> f16 — are
//! described once, as KIR, in `nsl_kir::kernels::cast`. This module builds
//! them on first use and launches them.
//!
//! ## What this replaces
//!
//! The kernels used to be emitted as PTX text by
//! `nsl_codegen::precision_cast_ptx`. The runtime cannot call that:
//! `nsl-codegen` depends on `nsl-runtime`, so the reverse edge would be a
//! cycle. The bytes were therefore transcribed into this file as four
//! `static` strings, and a byte-for-byte parity test held the copy to the
//! emitter — two materialisations of one kernel, kept equal by CI.
//!
//! `nsl-kir` is a leaf crate, so both sides can depend on it. There is now
//! one description and no copy, which is what removed the parity test
//! along with the `__test_runtime_*` hooks that existed only to feed it.
//!
//! ## Built once, at first use
//!
//! `kernel_launch` keys its module cache on the PTX bytes, so the buffer
//! must live at a stable address for the cache to hit. A `OnceLock` gives
//! exactly that: the four modules are built on the first cast, and every
//! later lookup returns the same `&'static str`.

use std::sync::OnceLock;

use nsl_kir::kernels::cast::CastKind;

#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// Block dim every cast kernel launches with, from the kernel definition
/// rather than restated here.
// used by the CUDA-gated cast launcher
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const CAST_BLOCK_DIM_X: u32 = nsl_kir::kernels::cast::CAST_BLOCK_DIM_X;

/// One built cast kernel: its PTX module and the entry name to look up in
/// it. Both are NUL-terminated, because the CUDA driver reads them as C
/// strings.
struct CastModule {
    ptx: String,
    kernel_name: String,
}

static CAST_MODULES: OnceLock<[CastModule; 4]> = OnceLock::new();

/// The four modules, in `CastKind::ALL` order, built on first call.
fn modules() -> &'static [CastModule; 4] {
    CAST_MODULES.get_or_init(|| {
        CastKind::ALL.map(|kind| {
            let bytes = nsl_kir::kernels::cast::ptx(kind);
            debug_assert_eq!(bytes.last(), Some(&0), "PTX must be NUL-terminated");
            CastModule {
                // The printer emits ASCII only — non-ASCII in PTX trips
                // CUDA_ERROR_INVALID_PTX under the JIT — so this cannot
                // fail for a kernel this crate builds.
                ptx: String::from_utf8(bytes).expect("PTX must be ASCII"),
                kernel_name: format!("{}\0", kind.kernel_name()),
            }
        })
    })
}

/// Pick the (PTX, kernel-name) pair for a (`src_dtype`, `target_dtype`) cast.
/// Returns `None` if the cast pair has no GPU kernel (caller must refuse / take
/// a different path).
///
/// `src_dtype` / `target_dtype` use `crate::tensor::DTYPE_*` constants
/// (F32=1, FP16=2, BF16=3).
// Called from `tensor/precision_cast.rs`'s cuda-gated dispatch and from the
// unit tests here — dead only when neither applies.
#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
pub(crate) fn pick_cast_kernel(
    src_dtype: u16,
    target_dtype: u16,
) -> Option<(&'static str, &'static str)> {
    use crate::tensor::{DTYPE_BF16, DTYPE_F32, DTYPE_FP16};
    let kind = match (src_dtype, target_dtype) {
        (DTYPE_F32, DTYPE_BF16) => CastKind::F32ToBf16,
        (DTYPE_BF16, DTYPE_F32) => CastKind::Bf16ToF32,
        (DTYPE_F32, DTYPE_FP16) => CastKind::F32ToFp16,
        (DTYPE_FP16, DTYPE_F32) => CastKind::Fp16ToF32,
        // Same-dtype "casts" (e.g. F32->F32 / BF16->BF16) are caller-handled
        // via cuMemcpyDtoD (no conversion needed), and bf16 <-> f16 has no
        // direct kernel — the caller stages through f32.
        _ => return None,
    };
    let slot = CastKind::ALL.iter().position(|k| *k == kind).expect("kind is in ALL");
    let m = &modules()[slot];
    Some((m.ptx.as_str(), m.kernel_name.as_str()))
}

/// Launch a precision-cast kernel with FFI signature
/// `(src_ptr: .u64, dst_ptr: .u64, numel: .u64)`.
///
/// * `src_dev` / `dst_dev` are device pointers (u64) — caller is responsible
///   for allocating `dst_dev` with `numel * sizeof(target_dtype)` bytes.
/// * Block size is fixed at `CAST_BLOCK_DIM_X = 256`.
/// * Grid is `ceil(numel / block)` (clamped to u32::MAX); the kernel uses a
///   grid-stride loop so any clamp still covers `numel` correctly — and as
///   of CFTP v7 follow-on (finding-1/9 fix) the kernel uses u64 indices
///   and bounds, so >2^32-element casts complete correctly.
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

    // CFTP v7 follow-on (finding-5/9): div_ceil in u64 so pathological
    // numel values can't flip sign in signed i64 arithmetic and silently
    // collapse the grid to 1.  Cap gridDim.x at u32::MAX (CUDA hardware
    // limit); the u64 grid-stride loop in the kernel covers anything
    // above that correctly.
    let block_u64: u64 = CAST_BLOCK_DIM_X as u64;
    let raw_grid_u64: u64 = numel.div_ceil(block_u64);
    let grid: i64 = raw_grid_u64.min(u32::MAX as u64).max(1) as i64;
    let block: i64 = block_u64 as i64;

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

    /// Every built PTX must be NUL-terminated (cuModuleLoadData C-string contract).
    #[test]
    fn built_ptx_modules_are_nul_terminated() {
        for (kind, m) in CastKind::ALL.iter().zip(modules()) {
            assert!(
                m.ptx.ends_with('\0'),
                "{kind:?} must end in NUL (cuModuleLoadData reads as C string)"
            );
            // Exactly one trailing NUL.
            assert!(
                !m.ptx[..m.ptx.len() - 1].contains('\0'),
                "{kind:?} must contain exactly ONE NUL byte (at the end)"
            );
        }
    }

    /// PTX must be ASCII-only (cudarc JIT trips CUDA_ERROR_INVALID_PTX on
    /// non-ASCII; see global GPU invariants in MEMORY.md).
    #[test]
    fn built_ptx_modules_are_ascii() {
        for (kind, m) in CastKind::ALL.iter().zip(modules()) {
            assert!(m.ptx.is_ascii(), "{kind:?} must be ASCII-only");
        }
    }

    /// Each module must declare the entry its own name lookup asks for.
    /// A kernel name that does not appear in the module it is paired with
    /// is a `CUDA_ERROR_NOT_FOUND` at launch, and nothing before launch
    /// would have caught it.
    #[test]
    fn each_module_declares_the_entry_its_name_looks_up() {
        for (kind, m) in CastKind::ALL.iter().zip(modules()) {
            let name = m.kernel_name.trim_end_matches('\0');
            assert_eq!(name, kind.kernel_name(), "name must come from the kernel definition");
            assert!(
                m.ptx.contains(&format!(".visible .entry {name}")),
                "{kind:?}: module does not declare `{name}`"
            );
        }
    }

    /// Kernel names are NUL-terminated for the driver's C-string lookup.
    #[test]
    fn kernel_name_strings_are_nul_terminated() {
        for m in modules() {
            assert!(m.kernel_name.ends_with('\0'), "kernel name must be NUL-terminated");
        }
    }

    /// Building on first use must not mean building on *every* use. The
    /// module cache in `kernel_launch` is keyed on the PTX bytes, so a
    /// `pick_cast_kernel` that handed back a freshly-allocated buffer
    /// would miss the cache and re-load the module on every launch — a
    /// silent performance regression with no functional symptom. Pin the
    /// address so that refactor fails here instead.
    #[test]
    fn built_ptx_modules_have_stable_addresses_across_calls() {
        use crate::tensor::{DTYPE_BF16, DTYPE_F32};
        let (ptx_a, kname_a) = pick_cast_kernel(DTYPE_F32, DTYPE_BF16).unwrap();
        let (ptx_b, kname_b) = pick_cast_kernel(DTYPE_F32, DTYPE_BF16).unwrap();
        assert_eq!(
            ptx_a.as_ptr() as usize,
            ptx_b.as_ptr() as usize,
            "PTX must be built once (cache-friendly): repeated lookup must \
             return an identical address; otherwise the FNV-1a module cache \
             will re-load the module on every launch"
        );
        assert_eq!(
            kname_a.as_ptr() as usize,
            kname_b.as_ptr() as usize,
            "kernel-name lookup must also be stable"
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

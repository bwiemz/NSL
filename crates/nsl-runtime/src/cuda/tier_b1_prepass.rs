//! CSHA Tier B.1 production pre-pass kernels.
//!
//! Tier B.1's projection MMA expects inputs in NON-default layouts that the
//! standard CSHA pipeline doesn't produce:
//!
//!   * `csha_x_ptr` : the kernel reads it as f16 in chunks-major
//!                    `[d_model/chunk, seq, chunk]` layout, but the standard
//!                    pipeline writes f32 in row-major `[seq, d_model]`.
//!   * `csha_w*_ptr`: the kernel reads it as f16 in col-major-within-chunk
//!                    `[d_model/chunk, hd, chunk]` layout, but the standard
//!                    pipeline writes f32 in row-major `[d_model, hd]`.
//!
//! When the dispatched kernel is a Tier B.1 variant (detected via the
//! `_tier_b1` suffix on its name), `nsl_flash_attention_csha` orchestrates:
//!   1. RMSNorm + narrow + chunkify on `x` (per-step; dynamic input).
//!   2. Narrow + col-major-chunkify on `Wq/Wk/Wv` (per launch, like `x`).
//!   3. Launches the main Tier B.1 kernel with `csha.skip_rmsnorm_prologue=true`
//!      semantics, passing the chunkified scratch buffers.
//!   4. Frees the per-call scratch (x and the three weights) on completion.
//!
//! See `project_csha_tier_b1_numerical_correctness` memory for the full
//! design rationale (PR #180 closed the codegen bugs; this file closes the
//! caller-side pre-pass loop).


// ---------------------------------------------------------------------------
// The kernels
//
// Both are described as KIR in `nsl_kir::kernels::tier_b1_prepass` (roadmap
// A2 step 11), which documents their grids, signatures and layouts; this
// module builds each once and launches it. They replace two hand-written
// PTX modules; `nsl-codegen`'s `tier_b1_prepass_kir_equivalence` gate holds
// the two to the same bytes.
// ---------------------------------------------------------------------------

use nsl_kir::kernels::tier_b1_prepass::{Prepass, PREPASS_BLOCK};

/// `kind`'s module, built on first use and kept: `kernel_launch` keys its
/// module cache on the buffer's address, so every launch must pass the
/// same one. The text is NUL-terminated ASCII, as `cuModuleLoadData`
/// requires.
fn module(kind: Prepass) -> &'static str {
    static X: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    static W: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let slot = match kind {
        Prepass::X => &X,
        Prepass::W => &W,
    };
    // The printer emits ASCII only, so this cannot fail for a kernel
    // `nsl-kir` builds.
    slot.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::tier_b1_prepass::ptx(kind)).expect("PTX must be ASCII")
    })
}

/// The X pre-pass module (`csha_tier_b1_prepass_x`: RMSNorm + narrow +
/// chunkify), NUL-terminated.
pub fn csha_tier_b1_prepass_x_ptx() -> &'static str {
    module(Prepass::X)
}

/// The W pre-pass module (`csha_tier_b1_prepass_w`: narrow + col-major
/// chunkify), NUL-terminated.
pub fn csha_tier_b1_prepass_w_ptx() -> &'static str {
    module(Prepass::W)
}

// ---------------------------------------------------------------------------
// Launch wrappers
//
// There is no weight cache. One used to live here, keyed on the weight
// tensor's device address and never invalidated or freed; it served stale
// weights after an in-place update or when a freed address was reused by
// another tensor of the same shape. The caller converts the weights on each
// launch and frees the scratch with the launch (`PrepassScratch` in
// `flash_attention.rs`).
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUresult;

/// Launch the X pre-pass: read raw `x_in` (f32 `[seq, d_model]`), apply
/// RMSNorm with `gamma_in`, narrow to f16, and write to `x_out` in
/// `[d_model/chunk, seq, chunk]` chunks-major layout. `chunk` must be a
/// power of 2 (Tier B.1's `chunk_config::select` only emits 32/64/128).
///
/// All pointers are GPU-resident.
#[cfg(feature = "cuda")]
pub(crate) fn launch_x_prepass(
    x_in_ptr: u64,
    gamma_ptr: u64,
    x_out_ptr: u64,
    seq: u64,
    d_model: u64,
    chunk: u64,
    eps: f32,
) -> CUresult {
    debug_assert!(chunk.is_power_of_two(), "chunk must be a power of 2; got {}", chunk);
    let log2_chunk: u32 = chunk.trailing_zeros();
    let mut x_in = x_in_ptr;
    let mut gamma = gamma_ptr;
    let mut x_out = x_out_ptr;
    let mut seq_v = seq;
    let mut dm_v = d_model;
    let mut chunk_v = chunk;
    let mut log2c = log2_chunk;
    let mut eps_v = eps;
    let args: [*mut std::ffi::c_void; 8] = [
        &mut x_in as *mut _ as *mut std::ffi::c_void,
        &mut gamma as *mut _ as *mut std::ffi::c_void,
        &mut x_out as *mut _ as *mut std::ffi::c_void,
        &mut seq_v as *mut _ as *mut std::ffi::c_void,
        &mut dm_v as *mut _ as *mut std::ffi::c_void,
        &mut chunk_v as *mut _ as *mut std::ffi::c_void,
        &mut log2c as *mut _ as *mut std::ffi::c_void,
        &mut eps_v as *mut _ as *mut std::ffi::c_void,
    ];
    super::inner::kernel_launch(
        csha_tier_b1_prepass_x_ptx().as_ptr(),
        b"csha_tier_b1_prepass_x\0".as_ptr(),
        [seq as i64, 1, 1],
        [PREPASS_BLOCK as i64, 1, 1],
        &args,
        0,
    )
}

/// Launch the W pre-pass: read raw `w_in` (f32 `[d_model, hd]`), narrow
/// to f16, and write to `w_out` in `[d_model/chunk, hd, chunk]`
/// col-major-within-chunk layout. Both `chunk` and `hd` must be powers
/// of 2.
///
/// All pointers are GPU-resident. Use this on Wq/Wk/Wv, on every launch:
/// the output is a copy and goes stale when the weight is updated.
#[cfg(feature = "cuda")]
pub(crate) fn launch_w_prepass(
    w_in_ptr: u64,
    w_out_ptr: u64,
    d_model: u64,
    hd: u64,
    chunk: u64,
) -> CUresult {
    debug_assert!(chunk.is_power_of_two(), "chunk must be a power of 2; got {}", chunk);
    debug_assert!(hd.is_power_of_two(), "hd must be a power of 2; got {}", hd);
    let log2_chunk: u32 = chunk.trailing_zeros();
    let log2_hd: u32 = hd.trailing_zeros();
    let total: i64 = (d_model * hd) as i64;
    let block = PREPASS_BLOCK as i64;
    let grid_x = (total + block - 1) / block;
    let mut w_in = w_in_ptr;
    let mut w_out = w_out_ptr;
    let mut dm_v = d_model;
    let mut hd_v = hd;
    let mut chunk_v = chunk;
    let mut log2h = log2_hd;
    let mut log2c = log2_chunk;
    let args: [*mut std::ffi::c_void; 7] = [
        &mut w_in as *mut _ as *mut std::ffi::c_void,
        &mut w_out as *mut _ as *mut std::ffi::c_void,
        &mut dm_v as *mut _ as *mut std::ffi::c_void,
        &mut hd_v as *mut _ as *mut std::ffi::c_void,
        &mut chunk_v as *mut _ as *mut std::ffi::c_void,
        &mut log2h as *mut _ as *mut std::ffi::c_void,
        &mut log2c as *mut _ as *mut std::ffi::c_void,
    ];
    super::inner::kernel_launch(
        csha_tier_b1_prepass_w_ptx().as_ptr(),
        b"csha_tier_b1_prepass_w\0".as_ptr(),
        [grid_x, 1, 1],
        [block, 1, 1],
        &args,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modules_are_nul_terminated_ascii_with_their_entries() {
        for (ptx, name) in [
            (csha_tier_b1_prepass_x_ptx(), Prepass::X.kernel_name()),
            (csha_tier_b1_prepass_w_ptx(), Prepass::W.kernel_name()),
        ] {
            assert!(ptx.ends_with('\0'));
            assert!(ptx.is_ascii());
            assert!(ptx.contains(&format!(".visible .entry {name}(")), "{name}");
        }
        // The launchers pass these exact names.
        assert_eq!(Prepass::X.kernel_name(), "csha_tier_b1_prepass_x");
        assert_eq!(Prepass::W.kernel_name(), "csha_tier_b1_prepass_w");
    }

    #[test]
    fn each_module_is_built_once() {
        // One build, one address: the module cache keys on it.
        assert!(std::ptr::eq(csha_tier_b1_prepass_x_ptx(), csha_tier_b1_prepass_x_ptx()));
        assert!(std::ptr::eq(csha_tier_b1_prepass_w_ptx(), csha_tier_b1_prepass_w_ptx()));
    }

    #[test]
    fn the_chunk_index_is_a_shift_and_mask_and_the_output_is_f16() {
        // Not div / rem, which are expensive on the GPU and would mean the
        // caller has to pre-compute chunk_idx / c.
        for ptx in [csha_tier_b1_prepass_x_ptx(), csha_tier_b1_prepass_w_ptx()] {
            assert!(ptx.contains("shr.u64 ") && ptx.contains("and.b64 "));
            assert!(!ptx.contains("div.u64") && !ptx.contains("rem.u64"));
            assert!(ptx.contains("cvt.rn.f16.f32") && ptx.contains("st.global.b16"));
        }
    }
}

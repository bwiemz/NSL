//! FlashAttention-2 runtime launch wrappers.
//!
//! These functions compute grid/block dimensions from tensor shapes, marshal
//! arguments, and call `kernel_launch()` with pre-baked PTX from .rodata.
//! No PTX generation happens at runtime. On non-CUDA builds, falls back to
//! naive matmul+softmax attention path.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::tensor::NslTensor;
#[cfg(feature = "cuda")]
use crate::autodiff;

/// One-time log guard: prints the selected kernel variant (FA3 or FA2) only once.
#[cfg(feature = "cuda")]
static FA_VARIANT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Launch FlashAttention-3 (Hopper wgmma) kernel.
/// Returns 0 on success, -1 if the launch failed (caller should fall back to FA2).
#[cfg(feature = "cuda")]
fn flash_attention_hopper(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale: f32,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: bool,
) -> i64 {
    use crate::cuda::kernels_hopper::{generate_flash_attention_3_ptx, FA3Config};

    let block_q = 64;
    let block_kv = 64;

    let config = FA3Config {
        block_q,
        block_kv,
        head_dim: head_dim as usize,
        seq_len: seq_len as usize,
        batch_heads: (batch * heads) as usize,
        causal,
        fp8: false,
        scale,
    };

    // Generate PTX at runtime for this config
    let ptx = generate_flash_attention_3_ptx(block_q, block_kv, head_dim as usize, causal, false);
    let ptx_cstr = format!("{}\0", ptx);

    let mut q_data = q_ptr as u64;
    let mut k_data = k_ptr as u64;
    let mut v_data = v_ptr as u64;
    let mut o_data = out_ptr as u64;
    let mut lse_data = if logsumexp_ptr != 0 {
        logsumexp_ptr as u64
    } else {
        0u64
    };
    let mut scale_val = scale;
    let mut seq_val = seq_len as u64;
    let mut hd_val = head_dim as u64;
    let mut num_kv = config.num_kv_tiles() as u64;

    let args: [*mut c_void; 9] = [
        &mut q_data   as *mut _ as *mut c_void,
        &mut k_data   as *mut _ as *mut c_void,
        &mut v_data   as *mut _ as *mut c_void,
        &mut o_data   as *mut _ as *mut c_void,
        &mut lse_data as *mut _ as *mut c_void,
        &mut scale_val as *mut _ as *mut c_void,
        &mut seq_val  as *mut _ as *mut c_void,
        &mut hd_val   as *mut _ as *mut c_void,
        &mut num_kv   as *mut _ as *mut c_void,
    ];

    let grid  = config.grid();
    let block = config.block();
    let shared = config.shared_mem_bytes();

    let result = crate::cuda::inner::kernel_launch(
        ptx_cstr.as_ptr(),
        b"nsl_flash_attention_3\0".as_ptr(),
        grid,
        block,
        &args,
        shared,
    );

    if result as u32 != 0 {
        eprintln!(
            "[nsl] FA3 Hopper kernel launch failed ({:?}), falling back to FA2",
            result
        );
        return -1;
    }

    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    0
}

/// FlashAttention-2 kernel launch wrapper.
///
/// All params are i64 for Cranelift ABI compatibility (same pattern as nsl_kernel_launch).
/// f32 scale is passed as i64 and reconstructed via f32::from_bits(scale as u32).
///
/// Returns 0 on success, non-zero CUDA error code on failure.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// Non-CSHA entry: this path has no `segment_ids_ptr` parameter, so the runtime
/// gate is supplied `0` for that slot and always returns `false` — Tier B never
/// fires for non-CSHA configs. The extension is present to keep all 6 FFI entry
/// points uniformly shaped per planner spec §4.6.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,  // backward aux output (0 = skip, inference-only)
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    // Non-CSHA path has no segment_ids_ptr parameter; pass 0, gate always returns false.
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            0,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        let _scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        // M33: tree mask params (null for non-tree-mask variants)
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        // Backward pass: logsumexp auxiliary output
        let mut lse = logsumexp_ptr as u64;

        let args: [*mut c_void; 21] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
        ];

        // ── Ampere / Hopper dispatch ──────────────────────────────────────────
        // Detect SM version once; on Hopper (sm_90+) try FA3 (wgmma) first.
        // If FA3 launch fails fall through to FA2 with the caller-supplied PTX.
        let sm = crate::cuda::inner::detect_sm_version();
        let fa3_ok = if sm >= 90 {
            let fa3_result = flash_attention_hopper(
                q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr,
                f32::from_bits(scale_bits as u32),
                batch, heads, seq_len, head_dim,
                causal != 0,
            );
            if fa3_result == 0 {
                if !FA_VARIANT_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!("[nsl] Using FlashAttention-3 (Hopper wgmma, sm_90a)");
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        let result: i64 = if fa3_ok {
            // FA3 already launched and synced; treat as success.
            0
        } else {
            if !FA_VARIANT_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!("[nsl] Using FlashAttention-2 (Ampere mma.sync)");
            }
            crate::cuda::inner::kernel_launch(
                effective_ptx_ptr as *const u8,
                effective_name_ptr as *const u8,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                &args,
                shared_mem_bytes as u32,
            ) as i64
        };

        // Record tape op for backward pass if recording
        if autodiff::is_recording() {
            let scale = f32::from_bits(scale_bits as u32);
            // Bump refcounts on Q, K, V so they survive until backward
            NslTensor::from_ptr(q_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(k_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(v_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            // Bump refcount on out so it survives until backward (needed for D[i] = dO . O)
            NslTensor::from_ptr(out_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            // Bump refcount on logsumexp so it survives until backward
            NslTensor::from_ptr(logsumexp_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            autodiff::maybe_record(autodiff::TapeOp::FlashAttention {
                q: q_ptr, k: k_ptr, v: v_ptr,
                out: out_ptr,
                logsumexp: logsumexp_ptr,
                scale,
                batch, heads, seq_len, head_dim,
                causal: causal != 0,
                saved_q: q_ptr,
                saved_k: k_ptr,
                saved_v: v_ptr,
            });
        }

        result
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (effective_ptx_ptr, effective_name_ptr);
        eprintln!("[nsl] FlashAttention requires CUDA. Use naive path (no @flash_attention decorator).");
        -1
    }
}

/// Resolve an NSL tensor-struct pointer (`NslTensor*` wrapped as `i64`) into
/// the raw device data pointer expected by the CSHA PTX kernels.
///
/// Context: Cranelift-emitted call sites for the CSHA FFIs pass `q_val` /
/// `k_val` / ... as `NslTensor*` (the opaque tensor handle). The CSHA
/// kernels, however, declare their tensor parameters as `.param .u64` and
/// do raw HBM address arithmetic on the loaded value. Passing the host
/// `NslTensor*` straight through would make the kernel dereference the
/// struct's first fields (`magic`, `data`, `shape`, ...) as if they were
/// tensor elements, producing `CUDA_ERROR_ILLEGAL_ADDRESS` on the very
/// first `ld.global` — exactly the bug PR #79 surfaced.
///
/// Dual-path contract:
///   - **Production** (Cranelift-emitted code): always passes
///     `NslTensor*`. We auto-promote CPU→GPU via `nsl_tensor_to_device`
///     (idempotent) and read `.data` for the kernel.
///   - **Test-only** (`nsl_test_cuda_alloc` call sites in
///     `crates/nsl-codegen/tests/csha_cuda_launch_*.rs`): passes raw
///     device pointers from `cuMemAlloc_v2`. We detect this via
///     `cuPointerGetAttribute(MEMORY_TYPE)`: a CU_MEMORYTYPE_DEVICE
///     result means the pointer is already in HBM and we pass it
///     through unchanged.
///
/// Null passes through as `0` so the kernel's runtime null-guards still
/// work.
#[cfg(feature = "cuda")]
#[inline]
fn csha_tensor_data_ptr(tensor_ptr: i64) -> u64 {
    if tensor_ptr == 0 {
        return 0;
    }

    // Query the driver for the pointer's memory type. If it's already
    // device memory (test path via `nsl_test_cuda_alloc`), the caller
    // provided a raw device pointer and we pass it straight through.
    // The query also handles managed memory cleanly — managed pointers
    // have memory_type == HOST (or a cudaMemoryType mismatch) and we
    // treat them as NslTensor* for safety.
    //
    // NOTE: we check `MEMORY_TYPE` not `DEVICE_POINTER` because the
    // latter returns the same address for device allocations but also
    // succeeds (returning the managed alias) for managed memory, which
    // would false-positive on NslTensor* handles stored in managed
    // memory. MEMORY_TYPE returns CU_MEMORYTYPE_DEVICE only for
    // `cuMemAlloc`/`cuMemAllocPitch` results.
    unsafe {
        use cudarc::driver::sys::{
            cuPointerGetAttribute, CUpointer_attribute, CUmemorytype, CUresult,
        };
        let mut mem_type: u32 = 0;
        let rc = cuPointerGetAttribute(
            &mut mem_type as *mut u32 as *mut std::ffi::c_void,
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            tensor_ptr as cudarc::driver::sys::CUdeviceptr,
        );
        if rc == CUresult::CUDA_SUCCESS
            && mem_type == CUmemorytype::CU_MEMORYTYPE_DEVICE as u32
        {
            // Raw device pointer — test-only path. Pass through.
            return tensor_ptr as u64;
        }
    }

    // Host pointer (NslTensor* handle) — auto-promote CPU→GPU
    // (idempotent for already-GPU tensors) then extract `.data`.
    let t = NslTensor::from_ptr(tensor_ptr);
    let gpu_ptr = if t.device == 0 {
        crate::tensor::nsl_tensor_to_device(tensor_ptr, 1)
    } else {
        tensor_ptr
    };
    let gpu_t = NslTensor::from_ptr(gpu_ptr);
    gpu_t.data as u64
}

/// CSHA Tier A.1: FlashAttention FFI variant that carries per-layer CSHA
/// extras (paper §2). This entry point preserves the argument layout of
/// `nsl_flash_attention` and *appends* nine CSHA-specific arguments:
///
///   x_ptr            — pre-norm input tensor (what RMSNorm will read)
///   norm_weight_ptr  — RMSNorm gain vector
///   wq_ptr / wk_ptr / wv_ptr / wo_ptr — projection weights the fused
///                                       kernel will project in-SMEM
///   rmsnorm_eps_bits — f32 bits of the RMSNorm epsilon
///   active_heads     — number of heads the specialised kernel computes
///                      (0 = use `heads` from the base layout)
///   d_model          — feature dimension for the projection tiles
///
/// For Tier A.1 the CSHA PTX body is not yet emitted (A.2), so this FFI
/// launches the CSHA-tagged PTX variant directly with a 30-argument
/// launch list — the 21 base FlashAttention params plus the nine CSHA
/// extras declared by `emit_flash_attention_entry` (csha_x_ptr through
/// csha_d_model).
///
/// Pick the per-CTA threadcount (`block_x`) for the CSHA forward FFIs.
///
/// Tier B.1 codegen (`flash_attention_v2::tier_b1::synthesize`) emits a
/// `global_t = warp_id + local_t * 8` warp distribution that assumes 8
/// warps per CTA = 256 threads. Tier A v2 (and the classic FA path) uses
/// 4 warps = 128 threads. The codegen embeds a `_tier_b1` suffix in the
/// kernel name when Tier B.1 dispatch applies (`csha.level >= 2 &&
/// gpu_sm >= 80` per `flash_attention_v2::is_tier_b1_dispatch`). This
/// helper peeks at the null-terminated name string and selects the
/// right block_x.
///
/// Carrying the signal in the kernel name (rather than as a new FFI
/// arg) keeps the 37-arg ABI stable — every existing caller continues
/// to work; only the name string they pass through changes for the
/// Tier B.1 dispatch path.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string allocated by `CString` or similar.
#[cfg(feature = "cuda")]
fn csha_block_x_for_kernel(name_ptr: i64) -> i64 {
    if name_ptr == 0 {
        return 128;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    // Marker is short (8 bytes); scan is O(name_len * 8) per launch.
    if name_bytes.windows(b"_tier_b1".len()).any(|w| w == b"_tier_b1") {
        256
    } else {
        128
    }
}

/// Extract the chunk size encoded in a Tier B.1 kernel name (the
/// `_tier_b1_chunk<N>` suffix appended by
/// `flash_attention_kernel_name_v2`). Returns `None` for non-Tier-B.1
/// names. The runtime pre-pass uses this to size the chunkified
/// scratch buffers + launch the narrow-and-chunkify kernels at the
/// right output stride.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string.
#[cfg(feature = "cuda")]
fn csha_tier_b1_chunk_for_kernel(name_ptr: i64) -> Option<u32> {
    if name_ptr == 0 {
        return None;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    let marker = b"_tier_b1_chunk";
    let pos = name_bytes.windows(marker.len()).position(|w| w == marker)?;
    let rest = &name_bytes[pos + marker.len()..];
    let end = rest.iter().position(|&b| !b.is_ascii_digit()).unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    std::str::from_utf8(&rest[..end]).ok()?.parse::<u32>().ok()
}

/// Read the null-terminated kernel name into an owned `String` for
/// diagnostic logging. Returns `"<null>"` for a zero `name_ptr`. Truncates
/// at 256 bytes to bound an accidental run-away scan if the pointer
/// references uninitialised memory.
#[cfg(feature = "cuda")]
fn csha_kernel_name_for_diag(name_ptr: i64) -> String {
    if name_ptr == 0 {
        return "<null>".into();
    }
    unsafe {
        let c = name_ptr as *const u8;
        let mut end = 0usize;
        while end < 256 && *c.add(end) != 0 {
            end += 1;
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(c, end)).into_owned()
    }
}

/// Detect a per-document CTA kernel by name-suffix.
///
/// The per-doc CTA forward kernel emitted by
/// `nsl_codegen::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_forward`
/// appends the `_per_doc_cta` suffix to its entry-point name (mirrors the
/// `_tier_b1_chunk<N>` pattern used by Tier B.1).
///
/// When this returns `true`, the FFI dispatch uses:
///   * `grid_x = num_docs` (from the `num_docs_or_zero` trailing arg)
///     instead of `ceil(seq/block_q)`
///   * `block_x = 128` (per-doc kernel is 4 warps, same as FA-2 default)
///
/// The signal lives in the kernel name (not a new FFI arg) for the *kernel
/// identity* — `num_docs_or_zero` carries the *runtime grid_x*.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string.
#[cfg(feature = "cuda")]
fn csha_is_per_doc_cta_kernel(name_ptr: i64) -> bool {
    if name_ptr == 0 {
        return false;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    let marker = b"_per_doc_cta";
    name_bytes.windows(marker.len()).any(|w| w == marker)
}

/// Tier B.1 per-call x-scratch RAII holder. Drops free the GPU scratch
/// buffer after `cuCtxSynchronize`, ensuring the kernel reading from
/// it has completed.
///
/// **W is NOT held here.** Chunkified Wq/Wk/Wv buffers are owned by
/// the process-global cache in `cuda::tier_b1_prepass::w_cache` —
/// weights are static across inference calls, so caching them avoids
/// re-running `launch_w_prepass` on every call.
#[cfg(feature = "cuda")]
struct PrepassScratch {
    x_scratch: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
impl Drop for PrepassScratch {
    fn drop(&mut self) {
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize();
            if !self.x_scratch.is_null() {
                let _ = cudarc::driver::sys::cuMemFree_v2(self.x_scratch as cudarc::driver::sys::CUdeviceptr);
            }
        }
    }
}

/// Tier B.1 pre-pass orchestration. When the dispatched kernel is a
/// `_tier_b1_chunk<N>` variant, run the RMSNorm + narrow + chunkify on
/// `x` and the narrow + col-major chunkify on `Wq/Wk/Wv` (all on the
/// GPU via `cuda::tier_b1_prepass`), writing to freshly-allocated
/// scratch. Returns the substituted (x, Wq, Wk, Wv) pointers + RAII
/// scratch handle whose `Drop` runs `cuCtxSynchronize` + frees the
/// buffers. The caller MUST keep the handle alive until after the main
/// kernel launch.
///
/// Returns `None` when the kernel is not Tier B.1 (caller uses original
/// pointers unchanged).
#[cfg(feature = "cuda")]
fn csha_tier_b1_prepass_substitute(
    name_ptr: i64,
    x_data: u64, nw_data: u64,
    wq_data: u64, wk_data: u64, wv_data: u64,
    seq_len: i64, head_dim: i64, d_model: i64,
    rmsnorm_eps_bits: i64,
) -> Option<(u64, u64, u64, u64, PrepassScratch)> {
    let chunk = csha_tier_b1_chunk_for_kernel(name_ptr)? as u64;
    if x_data == 0 || nw_data == 0 || wq_data == 0 || wk_data == 0 || wv_data == 0 {
        return None;
    }
    let seq = seq_len as u64;
    let dm = d_model as u64;
    let hd = head_dim as u64;
    debug_assert_eq!(dm % chunk, 0, "d_model ({}) must be divisible by chunk ({})", dm, chunk);
    let n_chunks = dm / chunk;
    // X scratch is per-call (x changes every step). W is cached
    // process-globally (weights are static across inference calls).
    let x_bytes = (n_chunks * seq * chunk * 2) as usize;
    let x_scratch = crate::cuda::inner::alloc_device(x_bytes);
    if x_scratch.is_null() {
        return None;
    }
    let scratch = PrepassScratch { x_scratch };
    let eps = f32::from_bits(rmsnorm_eps_bits as u32);
    let rc_x = crate::cuda::tier_b1_prepass::launch_x_prepass(
        x_data, nw_data, scratch.x_scratch as u64, seq, dm, chunk, eps,
    );
    if rc_x as u32 != 0 {
        eprintln!("[csha-tier-b1] x prepass launch failed rc={:?}", rc_x);
        return None;
    }
    let wq_cached = crate::cuda::tier_b1_prepass::w_chunkified_cached(wq_data, dm, hd, chunk)?;
    let wk_cached = crate::cuda::tier_b1_prepass::w_chunkified_cached(wk_data, dm, hd, chunk)?;
    let wv_cached = crate::cuda::tier_b1_prepass::w_chunkified_cached(wv_data, dm, hd, chunk)?;
    Some((scratch.x_scratch as u64, wq_cached, wk_cached, wv_cached, scratch))
}

/// A.2.5: this replaces the pre-A.2.5 forwarder (which invoked
/// `nsl_flash_attention` and dropped the extras, causing a cuLaunch arg
/// count mismatch on kernels whose body now references the CSHA params).
/// The CSHA PTX scaffolds from A.2.2/A.2.3/A.2.4 still runtime-null-check
/// the pointers, so callers that pass NULL for any of Wq/Wk/Wv/Wo/
/// x_ptr/norm_weight_ptr (current A.1/A.2.1x state) execute the classic
/// Q-from-HBM path inside the kernel. This FFI's contract is therefore
/// a strict extension of `nsl_flash_attention`'s behaviour.
///
/// FA3 Hopper fallback is intentionally skipped here — the FA3 kernel
/// does not understand CSHA extras. Ampere/Hopper both run the Ampere
/// mma.sync FA2 path when CSHA is active.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_csha(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    // CSHA extras consumed by the PTX prologue/projection/epilogue
    // scaffolds (A.2.2/A.2.3/A.2.4). Each pointer is null-checked
    // inside the kernel so current NULL-passing call sites fall through
    // to the classic path without dereferencing garbage.
    x_ptr: i64,
    norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64,
    d_model: i64,
    // PCA Tier A: segment_ids device pointer for packed-sequence training.
    // Pass 0 on unpacked launches. The companion PTX kernel declares this
    // param in its prelude only when its FlashAttentionConfig has
    // segment_masked=true, in which case the pointer must be non-null and
    // reference a [B, S] u16 tensor in global memory.
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA §4.3: doc_starts device pointer for packed-sequence training
    // with document-aware RoPE position reset. Per-row layout —
    // [batch_size, MAX_NUM_DOCS+1] i32 tensor, total batch_size * 1028
    // bytes. Pass 0 to disable (identity positions). When non-zero,
    // kernel computes row offset as batch_idx * (MAX_NUM_DOCS+1) and
    // loads only its row's 1028-byte subtable. See spec §3.
    doc_starts_ptr: i64,
    // PCA per-doc CTA (Strategy 3 v1): number of documents in the batch.
    // Pass 0 for legacy (non-per-doc) topology; the standard
    // `grid_x = ceil(seq_len / block_q)` is used. Pass non-zero ONLY when
    // the dispatched kernel's name carries the `_per_doc_cta` suffix —
    // grid_x is then overridden to `num_docs_or_zero` per Strategy 3's
    // one-CTA-per-document launch contract. Mismatched signal/topology
    // (suffix present + zero count, or zero suffix + nonzero count)
    // returns -1 with an `eprintln!` diagnostic on `[nsl::flash_attention]`.
    //
    // Discussion: an alternative design reads the sentinel from
    // `doc_starts_ptr` via a synchronous `cuMemcpyDtoH_v2` per launch,
    // avoiding the ABI change. We rejected that because (1) the D2H
    // copy adds a host-device round-trip per kernel launch on the
    // hot per-step path, defeating Strategy 3's whole-point; (2) the
    // explicit arg makes mismatched callers a compile-time Cranelift
    // sig error rather than a silent grid_x miscalculation; (3) the
    // sentinel-0 default is byte-identical for every existing caller.
    num_docs_or_zero: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        // Grid / block identical to `nsl_flash_attention` — CSHA doesn't
        // change the outer tiling, except for A.4 weight-informed
        // specialisation: when `active_heads` is non-zero and smaller
        // than the full head count, the kernel variant was compiled
        // with a matching early-exit guard; shrink grid_y so each CTA
        // lands inside the active range. active_heads == 0 means "no
        // pruning — run the full head count".
        //
        // CSHA paper §5.2 v1 (cycle-2 audit): this is the PRIMARY tier
        // of dead-head elimination — blocks for pruned heads never
        // launch. The defense-in-depth in-kernel guard lives in
        // `crates/nsl-codegen/src/flash_attention_v2/phases/forward/
        // csha_hooks.rs::emit_active_heads_guard`; the same formula
        // appears at the scalar+fused backward launch sites further
        // down this file. Formula pinned by `a4_grid_y_*` unit tests
        // in the `#[cfg(test)] mod` below.
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };

        // PCA per-doc CTA (Strategy 3 v1) topology detection.
        // The kernel name carries `_per_doc_cta` when codegen emitted the
        // per-doc CTA forward kernel. Launch geometry differs from the
        // standard FA-2 path:
        //   * grid_x = num_docs (one CTA per document, NOT ceil(seq/block_q))
        //   * grid_y = batch * heads (unchanged)
        //   * grid_z = 1               (unchanged)
        //   * block_x = 128            (4 warps, same as FA-2 default)
        // The caller MUST pass `num_docs_or_zero > 0` AND
        // `doc_starts_ptr != 0` when launching a per-doc CTA variant.
        let per_doc_cta = csha_is_per_doc_cta_kernel(effective_name_ptr);
        if per_doc_cta {
            if num_docs_or_zero <= 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha: per-doc CTA kernel \
                     ({:?}) requires num_docs_or_zero > 0, got {}",
                    csha_kernel_name_for_diag(effective_name_ptr),
                    num_docs_or_zero,
                );
                return -1;
            }
            if doc_starts_ptr == 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha: per-doc CTA kernel \
                     ({:?}) requires doc_starts_ptr != 0",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
        } else if num_docs_or_zero > 0 {
            // Caller passed a per-doc grid_x but the dispatched kernel is
            // not a per-doc variant — likely a planner/dispatch bug.
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha: num_docs_or_zero={} provided but \
                 kernel name {:?} lacks the `_per_doc_cta` suffix",
                num_docs_or_zero,
                csha_kernel_name_for_diag(effective_name_ptr),
            );
            return -1;
        }

        let grid_x = if per_doc_cta {
            num_docs_or_zero
        } else {
            (seq_len + block_q - 1) / block_q
        };
        let grid_y = batch * effective_heads;
        let grid_z = 1i64;
        // Tier B.1 launch geometry: the pipelined-MMA codegen emits a
        // `global_t = warp_id + local_t * 8` warp distribution
        // assuming 8 warps per CTA. Tier A v2 (and the classic FA path)
        // use 4 warps. The codegen embeds a `_tier_b1` suffix in the
        // kernel name when Tier B.1 dispatch applies
        // (`csha.level >= 2 && gpu_sm >= 80` per
        // `flash_attention_v2::is_tier_b1_dispatch`). Peek at the
        // null-terminated name string here to pick the right block_x.
        //
        // Carrying the signal in the kernel name keeps the FFI ABI
        // stable — all existing 37-arg callers continue to work; only
        // the name string they pass through changes for Tier B.1.
        // (Use `effective_name_ptr` so the block_x reflects the actual
        // kernel that will be launched — if a Tier B PCA variant was
        // selected by `should_dispatch_tier_b_at_runtime`, its name is
        // what cuLaunchKernel resolves the function from.)
        let block_x = csha_block_x_for_kernel(effective_name_ptr);
        let block_y = 1i64;
        let block_z = 1i64;

        // 21 base args — must exactly mirror the non-CSHA path so the
        // shared PTX body works identically on NULL CSHA extras.
        //
        // PR #79 bug fix: each tensor arg below is a host `NslTensor*`
        // (Cranelift emits these as opaque tensor handles from
        // `compile_flash_attention_call`). The CSHA PTX expects raw
        // device (HBM) base pointers, so we resolve each handle via
        // `csha_tensor_data_ptr` which (a) auto-promotes CPU tensors to
        // GPU and (b) extracts `NslTensor.data` for the kernel.
        // Non-tensor args (batch, heads, block_size, ...) pass through
        // unchanged. Paged/ragged pointers (block_table, k_pool, ...)
        // are always null at today's call sites so they pass through as
        // `0`.
        let mut q = csha_tensor_data_ptr(q_ptr);
        let mut k = csha_tensor_data_ptr(k_ptr);
        let mut v = csha_tensor_data_ptr(v_ptr);
        let mut out = csha_tensor_data_ptr(out_ptr);
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = csha_tensor_data_ptr(cos_ptr);
        let mut sin = csha_tensor_data_ptr(sin_ptr);
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr);

        // 9 CSHA extras, matching the PTX param declarations in
        // `emit_flash_attention_entry`. eps is declared .f32; heads /
        // d_model are .u32. Widths matter — the launch wrapper reads
        // sizeof(param_type) bytes starting at each `*mut c_void`.
        let mut x = csha_tensor_data_ptr(x_ptr);
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr);
        let mut wq = csha_tensor_data_ptr(wq_ptr);
        let mut wk = csha_tensor_data_ptr(wk_ptr);
        let mut wv = csha_tensor_data_ptr(wv_ptr);
        let mut wo = csha_tensor_data_ptr(wo_ptr);
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;

        // CSHA Tier B.1 production pre-pass orchestration.
        // When the dispatched kernel's name contains `_tier_b1_chunk<N>`,
        // we run the host-side RMSNorm + narrow + chunkify on `x` and
        // the narrow + col-major chunkify on Wq/Wk/Wv (all GPU-side via
        // `cuda::tier_b1_prepass`), then substitute the chunkified
        // pointers in the args list. The kernel was compiled with
        // `skip_rmsnorm_prologue=true` (force-overridden by codegen
        // in the Tier B.1 dispatch path) so it reads the pre-pass'd x
        // directly without re-RMSNormalizing.
        //
        // `_prepass_handle` is an RAII guard: its Drop runs
        // `cuCtxSynchronize` and frees the scratch buffers. It MUST
        // live until after `kernel_launch` returns.
        let _prepass_handle = if let Some((nx, nwq, nwk, nwv, handle)) =
            csha_tier_b1_prepass_substitute(
                effective_name_ptr,
                x, nw, wq, wk, wv,
                seq_len, head_dim, d_model,
                rmsnorm_eps_bits,
            )
        {
            x = nx;
            wq = nwq;
            wk = nwk;
            wv = nwv;
            Some(handle)
        } else {
            None
        };
        // Tier C activation-save pointers — always null for this FFI; the
        // `nsl_flash_attention_csha_with_saves` FFI passes real pointers.
        let mut q_proj: u64 = 0;
        let mut k_proj: u64 = 0;
        let mut v_proj: u64 = 0;
        let mut rmax: u64 = 0;
        let mut rsum: u64 = 0;
        let mut xraw: u64 = 0;
        // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order).
        let mut seg_ids = segment_ids_ptr as u64;
        // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order).
        // Sentinel 0 = identity positions (RoPE reset disabled).
        let mut doc_starts = doc_starts_ptr as u64;

        let args: [*mut c_void; 38] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            // ── CSHA extras ───────────────────────────────────────
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            // ── Tier C activation-save pointers (null by default) ─────
            &mut q_proj as *mut _ as *mut c_void,
            &mut k_proj as *mut _ as *mut c_void,
            &mut v_proj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order)
            &mut seg_ids as *mut _ as *mut c_void,
            // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order)
            &mut doc_starts as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            effective_ptx_ptr as *const u8,
            effective_name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        ) as i64;

        // Tape recording mirrors `nsl_flash_attention` exactly — CSHA
        // is a forward-pass optimisation; the backward pass re-reads Q/K/V
        // from the same pointers the caller supplied.
        if autodiff::is_recording() {
            let scale = f32::from_bits(scale_bits as u32);
            NslTensor::from_ptr(q_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(k_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(v_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(out_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(logsumexp_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            autodiff::maybe_record(autodiff::TapeOp::FlashAttention {
                q: q_ptr, k: k_ptr, v: v_ptr,
                out: out_ptr,
                logsumexp: logsumexp_ptr,
                scale,
                batch, heads, seq_len, head_dim,
                causal: causal != 0,
                saved_q: q_ptr,
                saved_k: k_ptr,
                saved_v: v_ptr,
            });
        }

        result
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (x_ptr, norm_weight_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr);
        let _ = (rmsnorm_eps_bits, active_heads, d_model, segment_ids_ptr, doc_starts_ptr);
        let _ = (tier_b_ptx_ptr, tier_b_name_ptr, effective_ptx_ptr, effective_name_ptr);
        let _ = num_docs_or_zero;
        eprintln!("[nsl] CSHA FlashAttention requires CUDA; non-CUDA build cannot launch.");
        -1
    }
}

/// CSHA Tier C: FlashAttention FFI variant that forwards to the same PTX
/// body as `nsl_flash_attention_csha` but supplies non-null
/// activation-save pointers so the fused source-AD backward kernel has
/// post-RoPE Q/K/V + row_max/row_sum available in HBM.
///
/// Pointer contract:
///   q_proj_ptr / k_proj_ptr / v_proj_ptr — HBM buffers of shape
///     `[batch, heads, seq, head_dim]` f16, row-major. Typically
///     allocated via `nsl_csha_alloc_backward_activations`.
///   row_max_ptr / row_sum_ptr — HBM buffers of shape
///     `[batch, heads, seq]` f32.
///
/// Null-safety: any of the 5 pointers MAY be null (the kernel's per-tensor
/// null guards skip that store). A fully-null call is equivalent to
/// `nsl_flash_attention_csha`.
///
/// Callers MUST have compiled the kernel with
/// `CshaExtras::save_activations_for_backward=true`; otherwise the PTX
/// emits no save path and the pointers are ignored.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_csha_with_saves(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    // Tier C activation-save pointers.
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    // Tier C: raw-x save pointer (forward stages a copy of x BEFORE
    // RMSNorm overwrites csha_x_ptr in place, so the backward dRMSNorm
    // can read pre-norm x). Null = skip the save.
    x_raw_ptr: i64,
    // PCA Tier A: segment_ids device pointer for packed-sequence training.
    // Pass 0 on unpacked launches. The companion PTX kernel declares this
    // param in its prelude only when its FlashAttentionConfig has
    // segment_masked=true, in which case the pointer must be non-null and
    // reference a [B, S] u16 tensor in global memory.
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA §4.3: doc_starts device pointer for packed-sequence training
    // with document-aware RoPE position reset. Per-row layout —
    // [batch_size, MAX_NUM_DOCS+1] i32 tensor, total batch_size * 1028
    // bytes. Pass 0 to disable (identity positions). When non-zero,
    // kernel computes row offset as batch_idx * (MAX_NUM_DOCS+1) and
    // loads only its row's 1028-byte subtable. See spec §3.
    doc_starts_ptr: i64,
    // PCA per-doc CTA (Strategy 3 v1): number of documents in the batch.
    // See `nsl_flash_attention_csha` for the full contract; this variant
    // mirrors that behaviour exactly.
    num_docs_or_zero: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha_with_saves",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };

        // PCA per-doc CTA (Strategy 3 v1) topology — see the matching
        // block in `nsl_flash_attention_csha` for the full contract +
        // alternative-design discussion.
        let per_doc_cta = csha_is_per_doc_cta_kernel(effective_name_ptr);
        if per_doc_cta {
            if num_docs_or_zero <= 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: per-doc CTA kernel \
                     ({:?}) requires num_docs_or_zero > 0, got {}",
                    csha_kernel_name_for_diag(effective_name_ptr),
                    num_docs_or_zero,
                );
                return -1;
            }
            if doc_starts_ptr == 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: per-doc CTA kernel \
                     ({:?}) requires doc_starts_ptr != 0",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
        } else if num_docs_or_zero > 0 {
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: num_docs_or_zero={} provided but \
                 kernel name {:?} lacks the `_per_doc_cta` suffix",
                num_docs_or_zero,
                csha_kernel_name_for_diag(effective_name_ptr),
            );
            return -1;
        }

        let grid_x = if per_doc_cta {
            num_docs_or_zero
        } else {
            (seq_len + block_q - 1) / block_q
        };
        let grid_y = batch * effective_heads;
        let grid_z = 1i64;
        // Tier B.1 launch geometry signal via kernel name suffix; see
        // `csha_block_x_for_kernel` and the matching comment in
        // `nsl_flash_attention_csha`. Must use `effective_name_ptr` (the
        // kernel actually launched) — when Tier B PCA dispatch swaps in
        // a different variant, its name decides the block_x.
        let block_x = csha_block_x_for_kernel(effective_name_ptr);
        let block_y = 1i64;
        let block_z = 1i64;

        // PR #79 bug fix: user-provided tensor args arrive as
        // `NslTensor*` — resolve each via `csha_tensor_data_ptr` (auto-
        // promotes CPU→GPU, extracts `.data`). The 6 activation-save
        // pointers (q_proj_ptr, ..., x_raw_ptr) are ALREADY raw device
        // pointers allocated by `nsl_csha_alloc_backward_activations_into`
        // and pass through untouched.
        let mut q = csha_tensor_data_ptr(q_ptr);
        let mut k = csha_tensor_data_ptr(k_ptr);
        let mut v = csha_tensor_data_ptr(v_ptr);
        let mut out = csha_tensor_data_ptr(out_ptr);
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = csha_tensor_data_ptr(cos_ptr);
        let mut sin = csha_tensor_data_ptr(sin_ptr);
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr);
        let mut x = csha_tensor_data_ptr(x_ptr);
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr);
        let mut wq = csha_tensor_data_ptr(wq_ptr);
        let mut wk = csha_tensor_data_ptr(wk_ptr);
        let mut wv = csha_tensor_data_ptr(wv_ptr);
        let mut wo = csha_tensor_data_ptr(wo_ptr);
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;

        // CSHA Tier B.1 production pre-pass — same orchestration as
        // `nsl_flash_attention_csha`. When the dispatched kernel is a
        // Tier B.1 variant (`_tier_b1_chunk<N>` suffix in its name) the
        // host-side RMSNorm + narrow + chunkify runs on `x` and the
        // narrow + col-major chunkify runs on Wq/Wk/Wv (cached
        // process-globally), and the chunkified pointers are
        // substituted before the kernel launch. The kernel itself was
        // compiled with `skip_rmsnorm_prologue=true` so it reads the
        // pre-pass'd x directly.
        //
        // `_prepass_handle` is RAII: its Drop runs `cuCtxSynchronize`
        // and frees the per-call x-scratch. It MUST live until after
        // `kernel_launch` returns.
        let _prepass_handle = if let Some((nx, nwq, nwk, nwv, handle)) =
            csha_tier_b1_prepass_substitute(
                effective_name_ptr,
                x, nw, wq, wk, wv,
                seq_len, head_dim, d_model,
                rmsnorm_eps_bits,
            )
        {
            x = nx;
            wq = nwq;
            wk = nwk;
            wv = nwv;
            Some(handle)
        } else {
            None
        };

        // Activation-save pointers: raw device buffers — pass through.
        let mut q_proj = q_proj_ptr as u64;
        let mut k_proj = k_proj_ptr as u64;
        let mut v_proj = v_proj_ptr as u64;
        let mut rmax = row_max_ptr as u64;
        let mut rsum = row_sum_ptr as u64;
        let mut xraw = x_raw_ptr as u64;
        // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order).
        let mut seg_ids = segment_ids_ptr as u64;
        // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order).
        // Sentinel 0 = identity positions (RoPE reset disabled).
        let mut doc_starts = doc_starts_ptr as u64;

        let args: [*mut c_void; 38] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut q_proj as *mut _ as *mut c_void,
            &mut k_proj as *mut _ as *mut c_void,
            &mut v_proj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order)
            &mut seg_ids as *mut _ as *mut c_void,
            // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order)
            &mut doc_starts as *mut _ as *mut c_void,
        ];

        // NSL_CSHA_DUMP_GRADS=1 forward-side diag: read the PTX kernel name
        // out of name_ptr (null-terminated C string) and print it so a
        // mismatched PTX selection (inference PTX instead of the _with_saves
        // variant) is visible. Also prints the save pointers so they can be
        // confirmed non-null and distinct.
        let dump_on = std::env::var_os("NSL_CSHA_DUMP_GRADS").map(|v| v != "0" && v != "").unwrap_or(false);
        if dump_on {
            let kname = unsafe {
                let c = effective_name_ptr as *const u8;
                if c.is_null() { String::from("<null>") } else {
                    let mut end = 0usize;
                    while *c.add(end) != 0 && end < 256 { end += 1; }
                    String::from_utf8_lossy(std::slice::from_raw_parts(c, end)).into_owned()
                }
            };
            eprintln!(
                "[csha-dump-fwd] with_saves kernel=\"{}\" q_proj=0x{:x} k_proj=0x{:x} v_proj=0x{:x} \
                 row_max=0x{:x} row_sum=0x{:x} x_raw=0x{:x}",
                kname, q_proj, k_proj, v_proj, rmax, rsum, xraw
            );
            // Dump the full PTX to a file for offline inspection — only once
            // to avoid accumulating files.
            unsafe {
                let c = effective_ptx_ptr as *const u8;
                if !c.is_null() {
                    let mut end = 0usize;
                    while *c.add(end) != 0 && end < (1 << 20) { end += 1; }
                    if end > 0 {
                        let bytes = std::slice::from_raw_parts(c, end);
                        let _ = std::fs::write(
                            std::env::temp_dir().join("csha_with_saves.ptx"),
                            bytes,
                        );
                        eprintln!(
                            "[csha-dump-fwd] ptx written to {}/csha_with_saves.ptx ({} bytes)",
                            std::env::temp_dir().display(), end
                        );
                    }
                }
            }
        }

        let fwd_rc = crate::cuda::inner::kernel_launch(
            effective_ptx_ptr as *const u8,
            effective_name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

        // Immediately after the forward kernel, sync + read back q_proj and
        // row_sum so we can tell whether the save-write path fired.  If the
        // forward kernel didn't write these then something between forward
        // PTX synthesis and launch lost the `save_activations_for_backward`
        // flag OR the saves emit but to wrong addresses.
        if dump_on {
            unsafe { crate::cuda::inner::cu_ctx_synchronize(); }
            eprintln!("[csha-dump-fwd-post] launch_rc={:?}", fwd_rc);
            let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
            let stats_elems = (batch * heads * seq_len) as usize;
            let x_elems = (batch * heads * seq_len * head_dim) as usize;
            // q_proj (f16)
            if q_proj != 0 {
                let mut host: Vec<u16> = vec![0u16; qkv_elems];
                unsafe {
                    crate::cuda::inner::memcpy_dtoh(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        q_proj as *const std::ffi::c_void,
                        qkv_elems * 2,
                    );
                }
                let nz = host.iter().filter(|&&b| b != 0).count();
                let first8: Vec<f32> = host.iter().take(8)
                    .map(|&b| crate::tensor::f16_bits_to_f32(b)).collect();
                eprintln!(
                    "[csha-dump-fwd-post]  q_proj first8={:?} nonzero_bits={}/{}",
                    first8, nz, qkv_elems
                );
            }
            // row_sum (f32)
            if rsum != 0 {
                let mut host: Vec<f32> = vec![0.0f32; stats_elems];
                unsafe {
                    crate::cuda::inner::memcpy_dtoh(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        rsum as *const std::ffi::c_void,
                        stats_elems * 4,
                    );
                }
                let nz = host.iter().filter(|&&v| v != 0.0).count();
                eprintln!(
                    "[csha-dump-fwd-post]  row_sum first8={:?} nonzero={}/{}",
                    host.iter().take(8).cloned().collect::<Vec<_>>(),
                    nz, stats_elems
                );
            }
            // x_raw (f32)
            if xraw != 0 {
                let mut host: Vec<f32> = vec![0.0f32; x_elems];
                unsafe {
                    crate::cuda::inner::memcpy_dtoh(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        xraw as *const std::ffi::c_void,
                        x_elems * 4,
                    );
                }
                let nz = host.iter().filter(|&&v| v != 0.0).count();
                eprintln!(
                    "[csha-dump-fwd-post]  x_raw  first8={:?} nonzero={}/{}",
                    host.iter().take(8).cloned().collect::<Vec<_>>(),
                    nz, x_elems
                );
            }
        }

        fwd_rc as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (x_ptr, norm_weight_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr);
        let _ = (rmsnorm_eps_bits, active_heads, d_model);
        let _ = (q_proj_ptr, k_proj_ptr, v_proj_ptr, row_max_ptr, row_sum_ptr, x_raw_ptr, segment_ids_ptr, doc_starts_ptr);
        let _ = (tier_b_ptx_ptr, tier_b_name_ptr, effective_ptx_ptr, effective_name_ptr);
        let _ = num_docs_or_zero;
        eprintln!("[nsl] CSHA FlashAttention w/ saves requires CUDA.");
        -1
    }
}

/// CSHA Tier C: fused source-AD backward kernel launch.
///
/// Forwards to the PTX synthesised by
/// `nsl_codegen::flash_attention_v2::synthesize_backward`. Launch-list
/// layout: forward's 35 CSHA args (identical order so downstream
/// tooling sees one ABI), then 8 backward-specific appends:
/// `dO_ptr` (input) plus the 7 gradient outputs
/// (`dq_ptr`, `dk_ptr`, `dv_ptr`, `dwq_ptr`, `dwk_ptr`, `dwv_ptr`,
/// `dx_ptr`).
///
/// The five activation-save pointers
/// (`q_proj_ptr`, `k_proj_ptr`, `v_proj_ptr`, `row_max_ptr`,
/// `row_sum_ptr`) are inherited from the forward slot: the forward
/// kernel *wrote* into them when launched via
/// `nsl_flash_attention_csha_with_saves`; the backward kernel *reads*
/// from them here. Null pointers on the gradient outputs let callers
/// skip stores they don't need (e.g. freezing a weight).
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_csha_backward(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    // Saved activations (inputs to backward).
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    // Tier C raw-x save: forward staged a copy here before RMSNorm
    // overwrote csha_x_ptr in place; backward dRMSNorm reads it.
    x_raw_ptr: i64,
    // Tier C backward-specific.
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64,
    // Gap I.5 fix (Option A): 8th gradient output — dx_norm (gradient
    // w.r.t. the RMSNorm output, i.e. dy_norm). Consumed by the AD-side
    // `RmsNormGammaBackward` emission for correct dgamma under the fused
    // CSHA dispatch path. f32, shape [batch, seq, d_model]. Null when
    // caller doesn't need it (e.g. inference warm-ups).
    dx_norm_ptr: i64,
    // PCA Tier A Task 4B: segment_ids device pointer for packed-sequence training.
    // Pass 0 on unpacked launches. The companion backward PTX kernel (Task 4A)
    // declares this param in its prelude only when segment_masked=true; the
    // segment_masked=false variant doesn't read this slot, so passing 0 is safe
    // for all existing callers.
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA §4.3: doc_starts device pointer for packed-sequence training
    // with document-aware RoPE position reset. Per-row layout —
    // [batch_size, MAX_NUM_DOCS+1] i32 tensor, total batch_size * 1028
    // bytes. Pass 0 to disable (identity positions). When non-zero,
    // kernel computes row offset as batch_idx * (MAX_NUM_DOCS+1) and
    // loads only its row's 1028-byte subtable. See spec §3.
    doc_starts_ptr: i64,
    // CSHA Tier B.2 (Phase 3 T6): explicit flag selecting the 4-kernel hybrid
    // backward launch path. When non-zero, `ptx_ptr` is treated as a module
    // holding the four concatenated Tier B.2 backward entries
    // (`tier_b2_d_prepass`, `tier_b2_dq_kernel`, `tier_b2_dkdv_kernel`,
    // `tier_b2_proj_backward`) and they are launched in sequence, with a
    // D = rowsum(dO*O) scratch buffer allocated for the intermediate. When 0
    // (the default for all existing callers, including the wengert lowering
    // until T7 computes the flag), behavior is byte-identical to the scalar
    // single-kernel path below. Selection is an explicit flag rather than
    // kernel-name-suffix sniffing — the dispatch decision is made in codegen
    // (see `tier_b2_hybrid_backward_eligible`) and threaded here. See spec §7.
    tier_b2_active: i64,
    // PCA per-doc CTA backward (Sprint 5): number of documents in the
    // batch. Pass 0 for legacy (non-per-doc) topology; the standard
    // `q_blocks` outer loop drives grid_x=1 launches per q-block. Pass
    // non-zero ONLY when the dispatched backward kernel's name carries
    // the `_per_doc_cta` suffix — grid_x is then overridden to
    // `num_docs_or_zero` per Sprint 5's one-CTA-per-document backward
    // launch contract, and the `q_blocks` outer loop is skipped.
    // Mismatched signal/topology (suffix present + zero count, or zero
    // suffix + nonzero count) returns -1 with an `eprintln!` diagnostic.
    num_docs_or_zero: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha_backward",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    // CSHA Tier B.2 (Phase 3 T6): 4-kernel hybrid backward launch branch.
    // When the explicit flag is set, `ptx_ptr` holds the concatenated 4-entry
    // Tier B.2 module; launch d_prepass -> dq -> dkdv -> proj_backward in
    // sequence (same-stream, implicitly ordered), with a D=rowsum(dO*O) f32
    // scratch buffer for the intermediate consumed by dq/dkdv. Returns early
    // so the scalar single-kernel path below is reached only when the flag is
    // 0 (byte-identical to today for every existing caller).
    //
    // Kernels 1-3 (d_prepass/dq/dkdv) use the standalone-validated per-kernel
    // ABIs (ported from the Layer-1 GPU reference launchers). Kernel 4
    // (proj_backward) reuses the SCALAR backward param ABI — the exact 49-arg
    // list the scalar path below builds — so it is launched by re-running the
    // scalar else-branch with `effective_name_ptr` overridden to
    // "tier_b2_proj_backward". To keep the scalar path byte-identical, the
    // Tier B.2 branch is fully self-contained here.
    #[cfg(feature = "cuda")]
    {
        if tier_b2_active != 0 {
            return csha_tier_b2_backward_launch(
                ptx_ptr,
                batch, heads, seq_len, head_dim,
                block_q,
                active_heads, d_model,
                scale_bits,
                shared_mem_bytes,
                causal,
                // RMSNorm + projection (scalar-ABI) forward inputs.
                x_ptr, norm_weight_ptr,
                wq_ptr, wk_ptr, wv_ptr, wo_ptr,
                rmsnorm_eps_bits,
                // Forward-saved activations.
                q_proj_ptr, k_proj_ptr, v_proj_ptr,
                row_max_ptr, row_sum_ptr, x_raw_ptr,
                logsumexp_ptr, out_ptr,
                // dO seed + gradient buffers.
                do_ptr,
                dq_ptr, dk_ptr, dv_ptr,
                dwq_ptr, dwk_ptr, dwv_ptr,
                dx_ptr, dx_norm_ptr,
                // Packed-sequence pass-throughs.
                segment_ids_ptr, doc_starts_ptr,
                // Sprint 10: rope_q=true threads cos/sin to proj_backward's
                // emit_drope phase. Null on rope_q=false launches (the entire
                // dRoPE emission is skipped at codegen time then, so a null
                // here is never read).
                cos_ptr, sin_ptr,
            );
        }
    }

    #[cfg(feature = "cuda")]
    {
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };
        // PCA per-doc CTA (Sprint 5) topology detection. Mirrors the
        // forward `nsl_flash_attention_csha` dispatch: when the kernel
        // name carries `_per_doc_cta`, the backward launches with
        // grid_x=num_docs (one CTA per document) and skips the per-q-block
        // outer launch loop entirely (the kernel's per-doc prelude derives
        // %q_start from doc_starts[bid_x] instead of bid_x*block_q +
        // q_launch_base, so the slens-threaded q-block trick is unused).
        let per_doc_cta = csha_is_per_doc_cta_kernel(effective_name_ptr);
        if per_doc_cta {
            if num_docs_or_zero <= 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_backward: per-doc CTA \
                     kernel ({:?}) requires num_docs_or_zero > 0, got {}",
                    csha_kernel_name_for_diag(effective_name_ptr),
                    num_docs_or_zero,
                );
                return -1;
            }
            if doc_starts_ptr == 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_backward: per-doc CTA \
                     kernel ({:?}) requires doc_starts_ptr != 0",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
        } else if num_docs_or_zero > 0 {
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha_backward: num_docs_or_zero={} \
                 provided but kernel name {:?} lacks the `_per_doc_cta` suffix",
                num_docs_or_zero,
                csha_kernel_name_for_diag(effective_name_ptr),
            );
            return -1;
        }
        let q_blocks = (seq_len + block_q - 1) / block_q;
        let grid_x = if per_doc_cta { num_docs_or_zero } else { 1i64 };
        let grid_y = batch * effective_heads;
        let grid_z = 1i64;
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // PR #79 bug fix: user-provided tensor args arrive as
        // `NslTensor*` — resolve each via `csha_tensor_data_ptr` (auto-
        // promotes CPU→GPU, extracts `.data`).
        //
        // Passes through unchanged:
        //   * Forward-saved activation pointers (q_proj_ptr, ...,
        //     x_raw_ptr): raw device buffers from
        //     `nsl_csha_alloc_backward_activations_into`.
        //   * Paged/ragged pointers (block_table, k_pool, v_pool,
        //     seq_ids, seq_lens): all null at today's callers.
        //
        // Resolved via `csha_tensor_data_ptr`:
        //   * Primary forward inputs (q, k, v, out, logsumexp, cos,
        //     sin, x, norm_weight, wq, wk, wv, wo).
        //   * Backward seed `dO` plus the 7 gradient outputs
        //     (dq, dk, dv, dwq, dwk, dwv, dx): these are allocated by
        //     Cranelift via `nsl_tensor_zeros_f16_on` / `nsl_tensor_zeros_on`
        //     (see `wengert_lower.rs::FusedCshaBackward`), which return
        //     `NslTensor*` with device=1. The kernel needs the raw
        //     device data pointer to write gradients into.
        let mut q = csha_tensor_data_ptr(q_ptr);
        let mut k = csha_tensor_data_ptr(k_ptr);
        let mut v = csha_tensor_data_ptr(v_ptr);
        let mut out = csha_tensor_data_ptr(out_ptr);
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = csha_tensor_data_ptr(cos_ptr);
        let mut sin = csha_tensor_data_ptr(sin_ptr);
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr);
        let mut x = csha_tensor_data_ptr(x_ptr);
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr);
        let mut wq = csha_tensor_data_ptr(wq_ptr);
        let mut wk = csha_tensor_data_ptr(wk_ptr);
        let mut wv = csha_tensor_data_ptr(wv_ptr);
        let mut wo = csha_tensor_data_ptr(wo_ptr);
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;
        // Forward-saved activations: raw device buffers, pass through.
        let mut qp = q_proj_ptr as u64;
        let mut kpj = k_proj_ptr as u64;
        let mut vpj = v_proj_ptr as u64;
        let mut rmax = row_max_ptr as u64;
        let mut rsum = row_sum_ptr as u64;
        let mut xraw = x_raw_ptr as u64;
        // Backward seed + 8 gradient outputs (dq/dk/dv/dwq/dwk/dwv/dx/dx_norm):
        // NslTensor handles — resolve to device pointers via csha_tensor_data_ptr.
        // PR #74 added dx_norm as the 8th extract so dgamma can be computed
        // from the post-RMSNorm gradient directly.
        let mut d_o = csha_tensor_data_ptr(do_ptr);
        let mut d_q = csha_tensor_data_ptr(dq_ptr);
        let mut d_k = csha_tensor_data_ptr(dk_ptr);
        let mut d_v = csha_tensor_data_ptr(dv_ptr);
        let mut d_wq = csha_tensor_data_ptr(dwq_ptr);
        let mut d_wk = csha_tensor_data_ptr(dwk_ptr);
        let mut d_wv = csha_tensor_data_ptr(dwv_ptr);
        let mut d_x = csha_tensor_data_ptr(dx_ptr);
        let mut d_xn = csha_tensor_data_ptr(dx_norm_ptr);
        // PCA Tier A Task 4B: segment_ids device pointer — raw pass-through
        // (not an NslTensor handle). Null on unpacked launches; the
        // segment_masked kernel variant reads it only when set.
        let mut seg_ids = segment_ids_ptr as u64;
        // PCA §4.3: doc_starts device pointer — raw pass-through.
        // Sentinel 0 = identity positions (RoPE reset disabled).
        let mut doc_starts = doc_starts_ptr as u64;

        // Option A f32 scratch for dK/dV: accumulate in f32 to avoid the
        // f16-HBM saturation-to-inf that plagued the naive ld-add-store
        // finalize. Scratch is zeroed, each q-block kernel does serialized
        // ld.f32/add.f32/st.f32 RMW (deterministic — no atomics), and a
        // final conversion kernel writes f32 scratch → f16 dK/dV outputs.
        let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
        let scratch_bytes = qkv_elems * 4; // f32
        let dk_scratch_raw = if d_k != 0 {
            crate::cuda::inner::alloc_device(scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        let dv_scratch_raw = if d_v != 0 {
            crate::cuda::inner::alloc_device(scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        if !dk_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dk_scratch_raw, scratch_bytes);
        }
        if !dv_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dv_scratch_raw, scratch_bytes);
        }
        let mut dk_scratch = dk_scratch_raw as u64;
        let mut dv_scratch = dv_scratch_raw as u64;

        let args: [*mut c_void; 49] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut kpj as *mut _ as *mut c_void,
            &mut vpj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            // ── Tier C backward outputs ──
            &mut d_o as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut d_k as *mut _ as *mut c_void,
            &mut d_v as *mut _ as *mut c_void,
            &mut d_wq as *mut _ as *mut c_void,
            &mut d_wk as *mut _ as *mut c_void,
            &mut d_wv as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_xn as *mut _ as *mut c_void,
            // Option A: f32 scratch for dK/dV accumulation.
            &mut dk_scratch as *mut _ as *mut c_void,
            &mut dv_scratch as *mut _ as *mut c_void,
            // PCA Tier A Task 4B: segment_ids trailing slot.
            &mut seg_ids as *mut _ as *mut c_void,
            // PCA §4.3: doc_starts trailing slot.
            &mut doc_starts as *mut _ as *mut c_void,
        ];

        // The backward kernel is launched one q-block at a time with the
        // q-block base carried in the otherwise-unused seq_lens_ptr slot.
        // Each launch does serialized f32 RMW on dk_scratch / dv_scratch,
        // so the final scratch values accumulate correctly across all
        // q-blocks without atomics. dQ is written per-launch to disjoint
        // slices so `d_q` still needs a zero init before the first launch.
        if d_q != 0 {
            crate::cuda::inner::memset_d8(d_q as *mut c_void, qkv_elems * 2);
        }

        let mut rc = cudarc::driver::sys::CUresult::CUDA_SUCCESS;
        if per_doc_cta {
            // Per-doc CTA: ONE launch with grid_x=num_docs. The kernel's
            // per-doc prelude derives %q_start from doc_starts[bid_x],
            // ignoring %q_launch_base, so the slens-threaded q-block trick
            // is not needed (and would be incorrect — there is no global
            // q-block ordinal in the per-doc launch).
            slens = 0u64;
            let _ = std::hint::black_box(&slens);
            rc = crate::cuda::inner::kernel_launch(
                effective_ptx_ptr as *const u8,
                effective_name_ptr as *const u8,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                &args,
                shared_mem_bytes as u32,
            );
        } else {
            for q_block in 0..q_blocks {
                // Thread the q-block base into seq_lens_ptr slot via `slens`.
                // Raw-ptr use: `args[16]` holds `&mut slens`, and CUDA's
                // `cuLaunchKernel` reads the pointee at call time, so each
                // iteration's assignment is picked up by the next launch.
                // `black_box` prevents the optimizer from eliding the write
                // because dataflow analysis can't see the read-through-ptr.
                slens = (q_block * block_q) as u64;
                let _ = std::hint::black_box(&slens);
                rc = crate::cuda::inner::kernel_launch(
                    effective_ptx_ptr as *const u8,
                    effective_name_ptr as *const u8,
                    [grid_x, grid_y, grid_z],
                    [block_x, block_y, block_z],
                    &args,
                    shared_mem_bytes as u32,
                );
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    break;
                }
            }
        }

        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS
            && std::env::var_os("NSL_CSHA_DUMP_GRADS").map(|v| v != "0" && v != "").unwrap_or(false)
        {
            let dump_first_nonfinite_scratch = |name: &str, ptr: *mut c_void| {
                if ptr.is_null() || qkv_elems == 0 {
                    return;
                }
                let mut host = vec![0f32; qkv_elems];
                crate::cuda::inner::memcpy_dtoh(host.as_mut_ptr() as *mut c_void, ptr, scratch_bytes);
                if let Some((idx, value)) = host.iter().copied().enumerate().find(|(_, x)| !x.is_finite()) {
                    eprintln!(
                        "[csha-dump-bwd-scratch] first non-finite {name}[{idx}]={value:?}"
                    );
                } else {
                    let first = host.iter().take(4).copied().collect::<Vec<_>>();
                    eprintln!(
                        "[csha-dump-bwd-scratch] {name} first4={first:?} max_abs={:.6e}",
                        host.iter().copied().map(f32::abs).fold(0.0f32, f32::max)
                    );
                }
            };
            dump_first_nonfinite_scratch("dk_scratch", dk_scratch_raw);
            dump_first_nonfinite_scratch("dv_scratch", dv_scratch_raw);
        }

        // Convert f32 scratch -> f16 output for dV only.
        // Cycle-16 G16-1 fix: dK is now written as f16 directly by the kernel
        // (emit_store_dk_only Phase 4). The dk_scratch buffer is allocated and
        // passed to the kernel (param slot stays; register loaded but unused),
        // but we must NOT run the f32->f16 conversion for dK -- it would
        // overwrite the correct f16 dK the kernel already wrote to dk_ptr.
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS
            && !dv_scratch_raw.is_null() && d_v != 0
        {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dv_scratch_raw, d_v as *mut c_void, qkv_elems,
            );
            if c_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                rc = c_rc;
            }
        }

        if !dk_scratch_raw.is_null() {
            crate::cuda::inner::free_device(dk_scratch_raw);
        }
        if !dv_scratch_raw.is_null() {
            crate::cuda::inner::free_device(dv_scratch_raw);
        }

        // NSL_CSHA_DUMP_GRADS=1 diagnostic — zero-cost when unset. Reads back
        // the 6 forward-saved activations (q_proj, k_proj, v_proj, row_max,
        // row_sum, x_raw) and all 8 gradient outputs (dq/dk/dv/dwq/dwk/dwv/
        // dx/dx_norm) from device → host, prints the first 8 values, max|.|,
        // element-sum, and NaN count for each.
        //
        // Consumes `qp, kpj, vpj, rmax, rsum, xraw` and `d_o, d_q, d_k, d_v,
        // d_wq, d_wk, d_wv, d_x, d_xn` (already resolved to raw device
        // pointers above). Shape info is derived from {batch, heads, seq_len,
        // head_dim, d_model}. Synchronises the device first so any async
        // kernel-launch failure or stale writeback is visible.
        if std::env::var_os("NSL_CSHA_DUMP_GRADS").map(|v| v != "0" && v != "").unwrap_or(false) {
            unsafe { crate::cuda::inner::cu_ctx_synchronize(); }
            let kname = unsafe {
                let c = effective_name_ptr as *const u8;
                if c.is_null() { String::from("<null>") } else {
                    let mut end = 0usize;
                    while *c.add(end) != 0 && end < 256 { end += 1; }
                    String::from_utf8_lossy(std::slice::from_raw_parts(c, end)).into_owned()
                }
            };
            eprintln!("[csha-dump-bwd] kernel=\"{}\"", kname);
            eprintln!(
                "[csha-dump-bwd] saves: q_proj=0x{:x} k_proj=0x{:x} v_proj=0x{:x} \
                 row_max=0x{:x} row_sum=0x{:x} x_raw=0x{:x}",
                qp, kpj, vpj, rmax, rsum, xraw
            );
            eprintln!(
                "[csha-dump-bwd] grads: dO=0x{:x} dq=0x{:x} dk=0x{:x} dv=0x{:x} \
                 dwq=0x{:x} dwk=0x{:x} dwv=0x{:x} dx=0x{:x} dx_norm=0x{:x}",
                d_o, d_q, d_k, d_v, d_wq, d_wk, d_wv, d_x, d_xn
            );
            csha_dump_backward_buffers(
                rc as i64,
                batch, heads, seq_len, head_dim, d_model,
                qp, kpj, vpj, rmax, rsum, xraw,
                d_o, d_q, d_k, d_v, d_wq, d_wk, d_wv, d_x, d_xn,
            );
        }

        rc as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (x_ptr, norm_weight_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr);
        let _ = (rmsnorm_eps_bits, active_heads, d_model);
        let _ = (q_proj_ptr, k_proj_ptr, v_proj_ptr, row_max_ptr, row_sum_ptr, x_raw_ptr);
        let _ = (do_ptr, dq_ptr, dk_ptr, dv_ptr, dwq_ptr, dwk_ptr, dwv_ptr, dx_ptr, dx_norm_ptr);
        let _ = (segment_ids_ptr, doc_starts_ptr, tier_b2_active);
        let _ = (tier_b_ptx_ptr, tier_b_name_ptr, effective_ptx_ptr, effective_name_ptr);
        let _ = num_docs_or_zero;
        eprintln!("[nsl] CSHA backward requires CUDA.");
        -1
    }
}

/// CSHA cycle 19 T1 — dS probe FFI (variant-B new symbol).
///
/// **Scope:** cycle-18 DEGENERATE-PROBE meta-lesson gate. The existing
/// `nsl_flash_attention_csha_backward` 54-param signature is FROZEN — this
/// wrapper adds two trailing device-pointer slots (`probe_ds_out_ptr`,
/// `probe_dv_out_ptr`, each 8 f32 slots wide) reserved for a NON-DEGENERATE
/// probe site at `(batch=0, head=0, q_tile_iter=0, warp_row=1, lane=0,
/// causal=true)` where `P[1,0]≈0.5`, `P[1,1]≈0.5`, and `dS = 0.25 *
/// (dP[1,0] - dP[1,1])` is nonzero under random dO. The exact 0.25
/// coefficient assumes symmetric Q/K (i.e. `S[1,0] ≈ S[1,1]`). Under
/// fully randomized Q/K the coefficient varies but `dS` remains nonzero
/// — the coordinate is non-degenerate in both regimes. See cycle-18
/// defer log.
///
/// **T1 scope is FFI+decl scaffolding ONLY.** The PTX-side probe emission
/// (Step 6 of the cycle-19 T1 spec) — writing 8 f32 probe slots via
/// `st.global.f32` predicated on the warp/lane/tile/batch/head coordinates
/// — has NOT been wired here. The runtime body currently:
///   1. Delegates the existing 54-param backward launch verbatim, and
///   2. Zeros the probe slots (when non-null) as an honest sentinel.
///
/// A cycle-19 T2 follow-up must extend the backward PTX emitter to accept
/// the probe pointers via a prelude widening and populate the 8 slots
/// (row_max, row_sum, S_pre_mask, P, dP, rowsum_dP_P, dS, scale*dS).
/// Until then the probe-integration test at
/// `crates/nsl-codegen/tests/csha_cycle19_ds_probe.rs` is `#[ignore]`d
/// with an XFAIL marker (probe-gate meta-lesson: DO NOT ship a fix based
/// on probe readings within T1 — that is T4 scope).
///
/// **ABI:** 54 original params + `probe_ds_out_ptr: i64` + `probe_dv_out_ptr:
/// i64` = **56 total i64 params**. Byte-identical to the original signature
/// on the first 54 slots; sentinel `0` on either trailing slot disables that
/// half of the probe write.
#[cfg(feature = "csha_cycle19_probe")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_csha_backward_probe(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    x_raw_ptr: i64,
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64,
    dx_norm_ptr: i64,
    segment_ids_ptr: i64,
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    doc_starts_ptr: i64,
    tier_b2_active: i64,
    num_docs_or_zero: i64,
    // T1 additive trailing slots (variant-B).
    probe_ds_out_ptr: i64,
    probe_dv_out_ptr: i64,
) -> i64 {
    // T1 scaffolding: delegate to the existing 54-param body verbatim. The
    // PTX-side probe emission is deferred to T2 per the c18 meta-lesson.
    let rc = nsl_flash_attention_csha_backward(
        q_ptr, k_ptr, v_ptr,
        out_ptr,
        logsumexp_ptr,
        scale_bits,
        batch, heads, seq_len, head_dim,
        block_table_ptr,
        k_pool_ptr, v_pool_ptr,
        block_size,
        cos_ptr, sin_ptr,
        seq_ids_ptr, seq_lens_ptr,
        shared_mem_bytes,
        ptx_ptr, name_ptr,
        block_q, block_kv,
        causal,
        x_ptr, norm_weight_ptr,
        wq_ptr, wk_ptr, wv_ptr, wo_ptr,
        rmsnorm_eps_bits,
        active_heads, d_model,
        q_proj_ptr, k_proj_ptr, v_proj_ptr,
        row_max_ptr, row_sum_ptr,
        x_raw_ptr,
        do_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        dwq_ptr, dwk_ptr, dwv_ptr,
        dx_ptr,
        dx_norm_ptr,
        segment_ids_ptr,
        tier_b_ptx_ptr,
        tier_b_name_ptr,
        doc_starts_ptr,
        tier_b2_active,
        num_docs_or_zero,
    );

    // T1 scaffolding: probe pointers preserved; PTX-side st.global.f32
    // population deferred to c20. These branches exist so the parameters are
    // not dead-code-eliminated by rustc — they carry no runtime effect in T1.
    // The 8-slot layout the future PTX emitter will target is:
    // {row_max, row_sum, S_pre_mask, P, dP, rowsum_dP_P, dS, scale*dS}.
    // Sentinel `0` on either pointer disables that half of the probe write
    // once c20 wires up PTX-side population.
    if probe_ds_out_ptr != 0 {
        let _ = probe_ds_out_ptr;
    }
    if probe_dv_out_ptr != 0 {
        let _ = probe_dv_out_ptr;
    }

    rc
}

/// CSHA Tier B.2 (Phase 3) — `effective_bq` per the dQ/dK/dV-kernel SMEM
/// fallback schedule (`smem_layout::tier_b2_effective_bq`). The runtime cannot
/// depend on `nsl-codegen` (the dependency runs the other way), so the formula
/// is duplicated here. block_q is clamped to 32 at head_dim 128/256 (the
/// SMEM-pressure / register-pressure fallback); otherwise it is used as-is.
/// MUST stay in sync with `smem_layout::tier_b2_effective_bq`.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_effective_bq(block_q: i64, head_dim: i64) -> i64 {
    match head_dim {
        128 | 256 => block_q.min(32),
        _ => block_q,
    }
}

/// Total dynamic-SMEM bytes for the `tier_b2_dq_kernel`, mirroring
/// `smem_layout::tier_b2_dq_total_smem_bytes` (the offset chain summed band by
/// band). `eb == ek` in the Approach-A bq==bkv invariant, but both are computed
/// for fidelity. CHUNK = `TIER_B2_RMSNORM_CHUNK` = 4. All tiles f16 (2 B/elem).
/// MUST stay in sync with the codegen helper.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_dq_total_smem_bytes(block_q: i64, head_dim: i64) -> i64 {
    const CHUNK: i64 = 4;
    let eb = tier_b2_effective_bq(block_q, head_dim);
    let ek = eb; // bq == bkv invariant
    let hd = head_dim;
    // Band chain (low -> high address), matching tier_b2_dq_*_offset:
    //   Q, K, V, dO, dS, Wk-chunk, Wv-chunk, x_q-chunk, x_kv-chunk, K-colmajor.
    let mut off = 0i64;
    off += eb * hd * 2; // Q          -> K offset
    off += ek * hd * 2; // K          -> V offset
    off += ek * hd * 2; // V          -> dO offset
    off += eb * hd * 2; // dO         -> dS offset
    off += eb * ek * 2; // dS         -> Wk-chunk offset
    off += CHUNK * hd * 2; // Wk-chunk -> Wv-chunk offset
    off += CHUNK * hd * 2; // Wv-chunk -> x_q-chunk offset
    off += eb * CHUNK * 2; // x_q-chunk -> x_kv-chunk offset
    off += ek * CHUNK * 2; // x_kv-chunk -> K-colmajor offset
    off += ek * hd * 2; // K-colmajor band (capacity == row-major K tile)
    off
}

/// Total dynamic-SMEM bytes for the `tier_b2_dkdv_kernel`, mirroring
/// `smem_layout::tier_b2_dkdv_total_smem_bytes`. Same prefix as dQ through the
/// x_kv chunk, then the col-major B/A re-stage bands (Q-colmajor, dO-colmajor,
/// P-colmajor, dS-colmajor). MUST stay in sync with the codegen helper.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_dkdv_total_smem_bytes(block_q: i64, head_dim: i64) -> i64 {
    const CHUNK: i64 = 4;
    let eb = tier_b2_effective_bq(block_q, head_dim);
    let ek = eb; // bq == bkv invariant
    let hd = head_dim;
    let mut off = 0i64;
    off += eb * hd * 2; // Q          -> K offset
    off += ek * hd * 2; // K          -> V offset
    off += ek * hd * 2; // V          -> dO offset
    off += eb * hd * 2; // dO         -> Wk-chunk offset
    off += CHUNK * hd * 2; // Wk-chunk -> Wv-chunk offset
    off += CHUNK * hd * 2; // Wv-chunk -> x_q-chunk offset
    off += eb * CHUNK * 2; // x_q-chunk -> x_kv-chunk offset
    off += ek * CHUNK * 2; // x_kv-chunk -> Q-colmajor offset
    off += eb * hd * 2; // Q-colmajor -> dO-colmajor offset
    off += eb * hd * 2; // dO-colmajor -> P-colmajor offset
    off += eb * ek * 2; // P-colmajor -> dS-colmajor offset
    off += eb * ek * 2; // dS-colmajor band
    off
}

/// Compacted dynamic-SMEM bytes for the standalone `tier_b2_proj_backward`
/// kernel, mirroring `smem_layout::tier_b2_proj_backward_smem_bytes`.
///
/// The proj kernel reuses the scalar fused-backward prelude, which lays out the
/// dQ/dK/dV/v_in/x_norm/dx_norm/rms tiles AFTER the full forward layout
/// (`total_bytes` + P + dS). `synthesize_proj_backward` rebases `%shmem_base`
/// down by `backward_dq_offset` so those tiles start at allocation byte 0; in
/// the rebase the `total_bytes` and P/dS terms cancel, leaving exactly the sum
/// of the tail tiles below. All f32 except v_in (f16). seq == block_q in the
/// smoke scope so block_q/block_kv are the raw config values (NOT effective_bq).
/// MUST stay in sync with the codegen helper.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_proj_backward_smem_bytes(block_q: i64, block_kv: i64, head_dim: i64, d_model: i64) -> i64 {
    let dq = block_q * head_dim * 4;
    let dv = block_kv * head_dim * 4;
    let dk = block_kv * head_dim * 4;
    let v_in = block_kv * head_dim * 2;
    let x_norm = block_q * d_model * 4;
    let dx_norm = block_q * d_model * 4;
    let rms_strip = block_q * 4;
    dq + dv + dk + v_in + x_norm + dx_norm + rms_strip
}

/// CSHA Tier B.2 (Phase 3 T6): launch the 4-kernel hybrid backward sequence.
///
/// `ptx_ptr` is a single (null-terminated) PTX module containing all four
/// concatenated entries:
///   1. `tier_b2_d_prepass`    reads dO, O          -> writes D (scratch)
///   2. `tier_b2_dq_kernel`    reads q/k/v_saved, dO, row_max, row_sum, D -> dQ
///   3. `tier_b2_dkdv_kernel`  reads q/k/v_saved, dO, row_max, row_sum, D -> dK, dV
///   4. `tier_b2_proj_backward` reads dQ/dK/dV (HBM, read-only) + x_raw/Wq/Wk/Wv
///      + norm_weight -> dWq, dWk, dWv, dx, dx_norm  (scalar backward param ABI)
///
/// Buffer aliasing (spec §4): the dQ/dK/dV buffers written by kernels 2/3 ARE
/// the final gradient outputs that kernel 4 reads; only the D-scratch is new.
///
/// Per-kernel grid/block are ported from the validated Layer-1 GPU reference
/// launchers (`tier_b2_dq_kernel_cpu_reference.rs`,
/// `tier_b2_dkdv_kernel_cpu_reference.rs`); kernel 4 mirrors the scalar
/// backward launch (grid `(1, batch*effective_heads, 1)`, block `(128,1,1)`,
/// threading the per-q-block base through the `seq_lens` slot).
///
/// Launches are same-stream and therefore implicitly ordered; `NSL_CUDA_SYNC`
/// is honored inside `kernel_launch` (it synchronizes + surfaces async errors
/// after every launch). Returns the first non-success `CUresult` as `i64`, or
/// `CUDA_SUCCESS` (0) when all four launch cleanly.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn csha_tier_b2_backward_launch(
    ptx_ptr: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_q: i64,
    active_heads: i64, d_model: i64,
    scale_bits: i64,
    shared_mem_bytes: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64, x_raw_ptr: i64,
    logsumexp_ptr: i64, out_ptr: i64,
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64, dx_norm_ptr: i64,
    segment_ids_ptr: i64, doc_starts_ptr: i64,
    // Sprint 10: rope_q=true support. When non-zero, the cos/sin device
    // buffers are threaded through to the proj_backward kernel so its
    // emit_drope phase can de-rotate dQ/dK from post-RoPE to pre-RoPE
    // before emit_dproj computes the weight gradients. Null on rope_q=false
    // launches; emit_drope's internal null-guard then makes the rotation a
    // no-op (which it is even on the codegen side: the codegen-level gate
    // in proj_backward.rs skips emit_drope entirely when !config.rope_q,
    // so a null cos/sin here is never read).
    cos_ptr: i64, sin_ptr: i64,
) -> i64 {
    use cudarc::driver::sys::CUresult;

    let success = CUresult::CUDA_SUCCESS;

    // Resolve the user-facing NslTensor handles to raw device pointers (auto-
    // promotes CPU->GPU + extracts `.data`), exactly as the scalar path does.
    // The forward-saved activations (q/k/v_proj, row_max, row_sum, x_raw) are
    // raw device buffers from `nsl_csha_alloc_backward_activations_into` and
    // pass through unchanged.
    let d_o = csha_tensor_data_ptr(do_ptr);
    let d_q = csha_tensor_data_ptr(dq_ptr);
    let d_k = csha_tensor_data_ptr(dk_ptr);
    let d_v = csha_tensor_data_ptr(dv_ptr);
    let qp = q_proj_ptr as u64;
    let kpj = k_proj_ptr as u64;
    let vpj = v_proj_ptr as u64;
    let rmax = row_max_ptr as u64;
    let rsum = row_sum_ptr as u64;
    let xraw = x_raw_ptr as u64;

    // The D pre-pass (kernel 1) needs the forward attention output O to compute
    // D = rowsum(dO * O). The scalar Tier C path resolves O from the `out_ptr`
    // slot (`csha_tensor_data_ptr(out_ptr)`); the standalone dQ/dK/dV gates feed
    // `fwd.o`. We source O from the same `out_ptr` slot here.
    //
    // PARITY-GATE NOTE (T8): the wengert `FusedCshaBackward` lowering currently
    // passes `null` for `out_ptr` (it does not retain O as an NslTensor handle).
    // The production forward-save plumbing must thread O into this slot for the
    // hybrid; this is the single launch-arg the parity gate (T8) must confirm.
    // If O is null the D pre-pass yields D == 0 and the §8 zero-output guard
    // FAILS — so a missing O cannot pass vacuously. (Today the branch is dormant
    // — every caller passes tier_b2_active = 0 — so this is not yet exercised.)
    let o_for_prepass = csha_tensor_data_ptr(out_ptr);

    // D-scratch: D = rowsum(dO * O), f32, shape [batch, heads, seq_len].
    let d_elems = (batch * heads * seq_len) as usize;
    let d_scratch_raw = if d_elems > 0 {
        crate::cuda::inner::alloc_device(d_elems * 4)
    } else {
        std::ptr::null_mut()
    };
    if d_scratch_raw.is_null() {
        eprintln!("[nsl] CSHA Tier B.2 backward: D-scratch alloc failed");
        return CUresult::CUDA_ERROR_OUT_OF_MEMORY as i64;
    }
    crate::cuda::inner::memset_d8(d_scratch_raw, d_elems * 4);
    let d_scratch = d_scratch_raw as u64;

    // Sprint 1 T1.2 — DTYPE RECONCILIATION (production f16 dq/dk/dv vs kernel f32).
    //
    // Wengert allocates dq/dk/dv as f16 NslTensors (`nsl_tensor_zeros_f16_on` —
    // see `crates/nsl-codegen/src/wengert_lower.rs:1834-1846`). The downstream
    // optimizer / extract ops expect f16 — this is the production contract.
    //
    // The Tier B.2 dq and dkdv kernels write `st.global.f32` (4 bytes/elem) into
    // dq/dk/dv; the proj_backward kernel reads them back as `ld.global.f32`. Wiring
    // the kernels' f32 writes directly against the f16 production buffers would be
    // a 2x buffer overflow (the f16 allocation is half the size of the f32 write
    // footprint). The scalar Tier C dK/dV path solves the same mismatch with f32
    // scratch + `csha_bwd_convert_f32_to_f16` (see `flash_attention.rs:1521-1681`);
    // we apply the identical pattern to dq + dk + dv here.
    //
    // Layout per scratch: f32 [batch, heads, seq_len, head_dim].
    let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
    let scratch_bytes = qkv_elems * 4;
    let dq_scratch_raw = if qkv_elems > 0 {
        crate::cuda::inner::alloc_device(scratch_bytes)
    } else {
        std::ptr::null_mut()
    };
    let dk_scratch_raw = if qkv_elems > 0 {
        crate::cuda::inner::alloc_device(scratch_bytes)
    } else {
        std::ptr::null_mut()
    };
    let dv_scratch_raw = if qkv_elems > 0 {
        crate::cuda::inner::alloc_device(scratch_bytes)
    } else {
        std::ptr::null_mut()
    };
    let free_scratches = |dq_raw: *mut c_void, dk_raw: *mut c_void, dv_raw: *mut c_void,
                          d_raw: *mut c_void| {
        if !dq_raw.is_null() {
            crate::cuda::inner::free_device(dq_raw);
        }
        if !dk_raw.is_null() {
            crate::cuda::inner::free_device(dk_raw);
        }
        if !dv_raw.is_null() {
            crate::cuda::inner::free_device(dv_raw);
        }
        if !d_raw.is_null() {
            crate::cuda::inner::free_device(d_raw);
        }
    };
    if qkv_elems > 0 && (dq_scratch_raw.is_null() || dk_scratch_raw.is_null() || dv_scratch_raw.is_null()) {
        eprintln!("[nsl] CSHA Tier B.2 backward: dQ/dK/dV scratch alloc failed");
        free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
        return CUresult::CUDA_ERROR_OUT_OF_MEMORY as i64;
    }
    if !dq_scratch_raw.is_null() {
        crate::cuda::inner::memset_d8(dq_scratch_raw, scratch_bytes);
    }
    if !dk_scratch_raw.is_null() {
        crate::cuda::inner::memset_d8(dk_scratch_raw, scratch_bytes);
    }
    if !dv_scratch_raw.is_null() {
        crate::cuda::inner::memset_d8(dv_scratch_raw, scratch_bytes);
    }
    let dq_scratch = dq_scratch_raw as u64;
    let dk_scratch = dk_scratch_raw as u64;
    let dv_scratch = dv_scratch_raw as u64;

    // Null-terminated PTX + kernel-name C strings.
    let ptx = ptx_ptr as *const u8;
    let name_d_prepass = b"tier_b2_d_prepass\0";
    let name_dq = b"tier_b2_dq_kernel\0";
    let name_dkdv = b"tier_b2_dkdv_kernel\0";
    let name_proj = b"tier_b2_proj_backward\0";

    let eb = tier_b2_effective_bq(block_q, head_dim);

    // ── Kernel 1: D pre-pass ──
    // ABI: (d_o_ptr, o_ptr, d_out_ptr, seq_len:u32, heads:u32).
    // Grid (ceil(seq/32), heads, batch), block (32,1,1), no dynamic SMEM.
    {
        let mut a_do = d_o as u64;
        let mut a_o = o_for_prepass as u64;
        let mut a_dout = d_scratch;
        let mut a_seq = seq_len as u32;
        let mut a_heads = heads as u32;
        let args: [*mut c_void; 5] = [
            &mut a_do as *mut _ as *mut c_void,
            &mut a_o as *mut _ as *mut c_void,
            &mut a_dout as *mut _ as *mut c_void,
            &mut a_seq as *mut _ as *mut c_void,
            &mut a_heads as *mut _ as *mut c_void,
        ];
        let grid_x = (seq_len + 31) / 32;
        let rc = crate::cuda::inner::kernel_launch(
            ptx, name_d_prepass.as_ptr(),
            [grid_x, heads, batch], [32, 1, 1], &args, 0,
        );
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // ── Kernel 2: dQ ──
    // ABI: (q,k,v_saved, d_o, row_max, row_sum, d, segment_ids, d_q_out,
    //       seq:u32, heads:u32, batch:u32).
    // Grid (ceil(seq/eb), heads, batch), block (128,1,1), dynamic SMEM = dq total.
    {
        let mut a_q = qp;
        let mut a_k = kpj;
        let mut a_v = vpj;
        let mut a_do = d_o as u64;
        let mut a_rmax = rmax;
        let mut a_rsum = rsum;
        let mut a_d = d_scratch;
        let mut a_seg = segment_ids_ptr as u64;
        // Sprint 1 T1.2: dq kernel writes f32 → dq_scratch (not the production f16 buffer).
        let mut a_dq = dq_scratch;
        let mut a_seq = seq_len as u32;
        let mut a_heads = heads as u32;
        let mut a_batch = batch as u32;
        let args: [*mut c_void; 12] = [
            &mut a_q as *mut _ as *mut c_void,
            &mut a_k as *mut _ as *mut c_void,
            &mut a_v as *mut _ as *mut c_void,
            &mut a_do as *mut _ as *mut c_void,
            &mut a_rmax as *mut _ as *mut c_void,
            &mut a_rsum as *mut _ as *mut c_void,
            &mut a_d as *mut _ as *mut c_void,
            &mut a_seg as *mut _ as *mut c_void,
            &mut a_dq as *mut _ as *mut c_void,
            &mut a_seq as *mut _ as *mut c_void,
            &mut a_heads as *mut _ as *mut c_void,
            &mut a_batch as *mut _ as *mut c_void,
        ];
        let grid_x = (seq_len + eb - 1) / eb;
        let smem = tier_b2_dq_total_smem_bytes(block_q, head_dim) as u32;
        let rc = crate::cuda::inner::kernel_launch(
            ptx, name_dq.as_ptr(),
            [grid_x, heads, batch], [128, 1, 1], &args, smem,
        );
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // ── Kernel 3: dK/dV ──
    // ABI: (q,k,v_saved, d_o, row_max, row_sum, d, segment_ids, d_k_out,
    //       d_v_out, seq:u32, heads:u32, batch:u32). NOTE: d_k BEFORE d_v.
    // Grid (ceil(seq/eb), heads, batch), block (128,1,1), dynamic SMEM = dkdv total.
    {
        let mut a_q = qp;
        let mut a_k = kpj;
        let mut a_v = vpj;
        let mut a_do = d_o as u64;
        let mut a_rmax = rmax;
        let mut a_rsum = rsum;
        let mut a_d = d_scratch;
        let mut a_seg = segment_ids_ptr as u64;
        // Sprint 1 T1.2: dkdv kernel writes f32 → dk_scratch / dv_scratch.
        let mut a_dk = dk_scratch;
        let mut a_dv = dv_scratch;
        let mut a_seq = seq_len as u32;
        let mut a_heads = heads as u32;
        let mut a_batch = batch as u32;
        let args: [*mut c_void; 13] = [
            &mut a_q as *mut _ as *mut c_void,
            &mut a_k as *mut _ as *mut c_void,
            &mut a_v as *mut _ as *mut c_void,
            &mut a_do as *mut _ as *mut c_void,
            &mut a_rmax as *mut _ as *mut c_void,
            &mut a_rsum as *mut _ as *mut c_void,
            &mut a_d as *mut _ as *mut c_void,
            &mut a_seg as *mut _ as *mut c_void,
            &mut a_dk as *mut _ as *mut c_void,
            &mut a_dv as *mut _ as *mut c_void,
            &mut a_seq as *mut _ as *mut c_void,
            &mut a_heads as *mut _ as *mut c_void,
            &mut a_batch as *mut _ as *mut c_void,
        ];
        let grid_x = (seq_len + eb - 1) / eb;
        let smem = tier_b2_dkdv_total_smem_bytes(block_q, head_dim) as u32;
        let rc = crate::cuda::inner::kernel_launch(
            ptx, name_dkdv.as_ptr(),
            [grid_x, heads, batch], [128, 1, 1], &args, smem,
        );
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // ── Kernel 4: projection backward ──
    // Reuses the SCALAR backward param ABI (the same 49-arg list the scalar
    // else-branch builds). dQ/dK/dV are INPUTS (read from HBM into SMEM); only
    // dWq/dWk/dWv/dx/dx_norm are written. The dk_scratch/dv_scratch slots are
    // unused by proj_backward (no dK/dV finalize store) — pass null. Launch
    // dims mirror the scalar backward: grid (1, batch*effective_heads, 1),
    // block (128,1,1), incoming `shared_mem_bytes`, threading the per-q-block
    // base through the `seq_lens` slot.
    {
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };

        // q_ptr/k_ptr/v_ptr are unused by proj_backward (it reads the forward
        // saves, not the raw inputs). Pass 0 for all three.
        let mut k = 0u64;
        let mut v = 0u64;
        let mut out_p = o_for_prepass as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = 0u64;
        let mut kp = 0u64;
        let mut vp = 0u64;
        let mut bsz = 0u64;
        // Sprint 10: thread cos/sin to proj_backward so its emit_drope phase
        // can de-rotate dQ/dK to the pre-RoPE basis before dproj reads them.
        // Resolve through csha_tensor_data_ptr (same auto-promote + .data
        // extraction the scalar backward path uses for cos/sin at line
        // 1469-1470). On rope_q=false launches the caller passes 0, which
        // csha_tensor_data_ptr returns as 0 — emit_drope's internal null-
        // guard then short-circuits; under the Sprint-10 codegen-side gate
        // the entire dRoPE block is also skipped, so this is doubly safe.
        let mut cos = csha_tensor_data_ptr(cos_ptr) as u64;
        let mut sin = csha_tensor_data_ptr(sin_ptr) as u64;
        let mut sids = 0u64;
        let mut slens = 0u64; // per-q-block base, threaded in the loop below
        let mut dfs_enter = 0u64;
        let mut dfs_exit = 0u64;
        let mut num_tree_nodes = 0u64;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr) as u64;
        let mut x = csha_tensor_data_ptr(x_ptr) as u64;
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr) as u64;
        let mut wq = csha_tensor_data_ptr(wq_ptr) as u64;
        let mut wk = csha_tensor_data_ptr(wk_ptr) as u64;
        let mut wv = csha_tensor_data_ptr(wv_ptr) as u64;
        let mut wo = csha_tensor_data_ptr(wo_ptr) as u64;
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;
        let mut a_qp = qp;
        let mut a_kpj = kpj;
        let mut a_vpj = vpj;
        let mut a_rmax = rmax;
        let mut a_rsum = rsum;
        let mut a_xraw = xraw;
        let mut a_do = d_o as u64;
        // dQ/dK/dV: read-only inputs to proj_backward (from HBM).
        // Sprint 1 T1.2: proj_backward reads f32 dQ/dK/dV from the same scratches
        // the dq + dkdv kernels wrote to. The production f16 destinations are
        // populated AFTER proj_backward via `csha_bwd_convert_f32_to_f16`.
        let mut a_dq = dq_scratch;
        let mut a_dk = dk_scratch;
        let mut a_dv = dv_scratch;
        // Gradient outputs written by proj_backward.
        let mut a_dwq = csha_tensor_data_ptr(dwq_ptr) as u64;
        let mut a_dwk = csha_tensor_data_ptr(dwk_ptr) as u64;
        let mut a_dwv = csha_tensor_data_ptr(dwv_ptr) as u64;
        let mut a_dx = csha_tensor_data_ptr(dx_ptr) as u64;
        let mut a_dxn = csha_tensor_data_ptr(dx_norm_ptr) as u64;
        // proj_backward does not write dK/dV — no f32 scratch needed.
        let mut a_dk_scratch = 0u64;
        let mut a_dv_scratch = 0u64;
        let mut a_seg = segment_ids_ptr as u64;
        let mut a_docs = doc_starts_ptr as u64;
        let _ = causal; // proj/dRMSNorm are causal-agnostic; param resolved for ABI parity

        let args: [*mut c_void; 49] = [
            &mut k as *mut _ as *mut c_void, // q_ptr slot (unused; pass 0)
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out_p as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bsz as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut a_qp as *mut _ as *mut c_void,
            &mut a_kpj as *mut _ as *mut c_void,
            &mut a_vpj as *mut _ as *mut c_void,
            &mut a_rmax as *mut _ as *mut c_void,
            &mut a_rsum as *mut _ as *mut c_void,
            &mut a_xraw as *mut _ as *mut c_void,
            &mut a_do as *mut _ as *mut c_void,
            &mut a_dq as *mut _ as *mut c_void,
            &mut a_dk as *mut _ as *mut c_void,
            &mut a_dv as *mut _ as *mut c_void,
            &mut a_dwq as *mut _ as *mut c_void,
            &mut a_dwk as *mut _ as *mut c_void,
            &mut a_dwv as *mut _ as *mut c_void,
            &mut a_dx as *mut _ as *mut c_void,
            &mut a_dxn as *mut _ as *mut c_void,
            &mut a_dk_scratch as *mut _ as *mut c_void,
            &mut a_dv_scratch as *mut _ as *mut c_void,
            &mut a_seg as *mut _ as *mut c_void,
            &mut a_docs as *mut _ as *mut c_void,
        ];

        let grid_y = batch * effective_heads;
        let q_blocks = if block_q > 0 { (seq_len + block_q - 1) / block_q } else { 1 };
        // T8 fix: the proj kernel rebases %shmem_base to a COMPACTED layout
        // (see `synthesize_proj_backward` + `tier_b2_proj_backward_smem_bytes`),
        // so the incoming `shared_mem_bytes` (which sizes the FULL scalar fused
        // backward, ~137 KB at hd=64 — past the 99 KB device cap) must be
        // overridden with the compacted ~88 KB span the standalone kernel
        // actually uses. d_model from the active_heads-derived projection
        // contract (== head_dim under the smoke scope).
        let proj_smem = tier_b2_proj_backward_smem_bytes(
            block_q, block_q, head_dim, d_model,
        ) as u32;
        let _ = shared_mem_bytes; // superseded by the compacted proj footprint
        let mut rc = success;
        for q_block in 0..q_blocks {
            slens = (q_block * block_q) as u64;
            let _ = std::hint::black_box(&slens);
            rc = crate::cuda::inner::kernel_launch(
                ptx, name_proj.as_ptr(),
                [1, grid_y, 1], [128, 1, 1], &args, proj_smem,
            );
            if rc != success {
                break;
            }
        }
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // Sprint 1 T1.2: Convert f32 scratches → f16 production destinations. Each
    // call reads `qkv_elems` f32 elements from a scratch and writes the same
    // count of f16 elements to the destination (the production wengert-allocated
    // f16 buffer). `csha_bwd_convert_f32_to_f16` is the same helper the scalar
    // dK/dV path uses (see `flash_attention.rs:1654-1674`).
    if qkv_elems > 0 {
        if d_q != 0 && !dq_scratch_raw.is_null() {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dq_scratch_raw, d_q as *mut c_void, qkv_elems,
            );
            if c_rc != success {
                free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
                return c_rc as i64;
            }
        }
        if d_k != 0 && !dk_scratch_raw.is_null() {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dk_scratch_raw, d_k as *mut c_void, qkv_elems,
            );
            if c_rc != success {
                free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
                return c_rc as i64;
            }
        }
        if d_v != 0 && !dv_scratch_raw.is_null() {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dv_scratch_raw, d_v as *mut c_void, qkv_elems,
            );
            if c_rc != success {
                free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
                return c_rc as i64;
            }
        }
    }

    free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
    success as i64
}

/// f32 scratch → f16 output conversion kernel used by the CSHA backward
/// (Option A of the dK/dV saturation fix). Cached once per process via
/// `kernel_launch`'s content-hashed module cache.
///
/// Kernel body: one thread per element, bounds-checked against `n_elems`.
/// `src` is f32, `dst` is f16; conversion is `cvt.rn.f16.f32` (round to
/// nearest even). Overflow still saturates to ±inf — acceptable here
/// because the f32 scratch was accumulated deterministically and any
/// |x| > 65504 is a legitimate gradient value the downstream f16 store
/// can't represent regardless of accumulation strategy.
#[cfg(feature = "cuda")]
const CSHA_BWD_F32_TO_F16_PTX: &str = concat!(
    ".version 8.7\n",
    ".target sm_75\n",
    ".address_size 64\n",
    ".visible .entry nsl_csha_bwd_f32_to_f16(\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 n_elems\n",
    ")\n",
    "{\n",
    // User registers intentionally avoid %tid / %ntid / %ctaid names
    // — those are PTX built-in special registers and redefining them
    // as user regs triggers a ptxas parse failure (tid.x dot-syntax
    // clash).
    "    .reg .u32 %th, %bd, %bi, %gid, %tmp;\n",
    "    .reg .u64 %idx, %n, %src, %dst, %addr_src, %addr_dst;\n",
    "    .reg .f32 %f;\n",
    "    .reg .b16 %h;\n",
    "    .reg .pred %p;\n",
    "    mov.u32 %th, %tid.x;\n",
    "    mov.u32 %bd, %ntid.x;\n",
    "    mov.u32 %bi, %ctaid.x;\n",
    "    mul.lo.u32 %tmp, %bi, %bd;\n",
    "    add.u32 %gid, %tmp, %th;\n",
    "    cvt.u64.u32 %idx, %gid;\n",
    "    ld.param.u64 %src, [src_ptr];\n",
    "    ld.param.u64 %dst, [dst_ptr];\n",
    "    ld.param.u64 %n, [n_elems];\n",
    "    setp.ge.u64 %p, %idx, %n;\n",
    "    @%p bra F32F16_END;\n",
    "    shl.b64 %addr_src, %idx, 2;\n",
    "    add.u64 %addr_src, %src, %addr_src;\n",
    "    shl.b64 %addr_dst, %idx, 1;\n",
    "    add.u64 %addr_dst, %dst, %addr_dst;\n",
    "    ld.global.f32 %f, [%addr_src];\n",
    "    cvt.rn.f16.f32 %h, %f;\n",
    "    st.global.b16 [%addr_dst], %h;\n",
    "F32F16_END:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

#[cfg(feature = "cuda")]
const CSHA_BWD_F32_TO_F16_NAME: &str = "nsl_csha_bwd_f32_to_f16\0";

/// Launch the f32→f16 conversion kernel. `n_elems` is the element count
/// (src is f32*4 bytes, dst is f16*2 bytes).
#[cfg(feature = "cuda")]
fn csha_bwd_convert_f32_to_f16(
    src: *mut c_void,
    dst: *mut c_void,
    n_elems: usize,
) -> cudarc::driver::sys::CUresult {
    if n_elems == 0 {
        return cudarc::driver::sys::CUresult::CUDA_SUCCESS;
    }
    let mut src_ptr = src as u64;
    let mut dst_ptr = dst as u64;
    let mut n = n_elems as u64;
    let args: [*mut c_void; 3] = [
        &mut src_ptr as *mut _ as *mut c_void,
        &mut dst_ptr as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block: i64 = 128;
    let grid: i64 = ((n_elems as i64) + block - 1) / block;
    crate::cuda::inner::kernel_launch(
        CSHA_BWD_F32_TO_F16_PTX.as_ptr(),
        CSHA_BWD_F32_TO_F16_NAME.as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    )
}

/// NSL_CSHA_DUMP_GRADS diagnostic: d2h-read the 6 save buffers and the 8
/// gradient outputs, print first 8 values + max|.| + sum + NaN count.
/// Zero-cost when disabled (callers gate on env var).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn csha_dump_backward_buffers(
    launch_rc: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64, d_model: i64,
    q_proj_dev: u64, k_proj_dev: u64, v_proj_dev: u64,
    row_max_dev: u64, row_sum_dev: u64, x_raw_dev: u64,
    d_o_dev: u64, d_q_dev: u64, d_k_dev: u64, d_v_dev: u64,
    d_wq_dev: u64, d_wk_dev: u64, d_wv_dev: u64,
    d_x_dev: u64, d_xn_dev: u64,
) {
    eprintln!(
        "[csha-dump] launch_rc={} batch={} heads={} seq={} head_dim={} d_model={}",
        launch_rc, batch, heads, seq_len, head_dim, d_model
    );

    // qkv-shaped: f16 [batch, heads, seq, head_dim]
    let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
    // stats: f32 [batch, heads, seq]
    let stats_elems = (batch * heads * seq_len) as usize;
    // x_raw / dx / dx_norm: f32 [batch, seq, d_model] or [batch, heads, seq, head_dim]
    // (x_raw is B*H*S*D = 1024; dx / dx_norm are B*S*M = 1024 here since M=H*D=32).
    let x_elems = (batch * heads * seq_len * head_dim) as usize;
    let dx_elems = (batch * seq_len * d_model) as usize;
    // dwq/dwk/dwv: f16 [d_model, heads * head_dim]
    let dw_elems = (d_model * heads * head_dim) as usize;

    fn dump_f16(name: &str, dev_ptr: u64, n: usize) {
        if dev_ptr == 0 || n == 0 {
            eprintln!("[csha-dump] {:>9} = <null or empty> dev=0x{:x} n={}", name, dev_ptr, n);
            return;
        }
        let mut host: Vec<u16> = vec![0u16; n];
        unsafe {
            crate::cuda::inner::memcpy_dtoh(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                dev_ptr as *const std::ffi::c_void,
                n * 2,
            );
        }
        let mut nan_count: usize = 0;
        let mut max_abs: f32 = 0.0;
        let mut sum: f64 = 0.0;
        for &bits in &host {
            let v = crate::tensor::f16_bits_to_f32(bits);
            if v.is_nan() { nan_count += 1; }
            let a = v.abs();
            if a.is_finite() && a > max_abs { max_abs = a; }
            if v.is_finite() { sum += v as f64; }
        }
        let first8: Vec<f32> = host.iter().take(8).map(|&b| crate::tensor::f16_bits_to_f32(b)).collect();
        eprintln!(
            "[csha-dump] {:>9} [f16 n={}]: first8={:?} max|.|={:.6e} sum={:.6e} nan_count={}",
            name, n, first8, max_abs, sum, nan_count
        );
    }

    fn dump_f32(name: &str, dev_ptr: u64, n: usize) {
        if dev_ptr == 0 || n == 0 {
            eprintln!("[csha-dump] {:>9} = <null or empty> dev=0x{:x} n={}", name, dev_ptr, n);
            return;
        }
        let mut host: Vec<f32> = vec![0.0f32; n];
        unsafe {
            crate::cuda::inner::memcpy_dtoh(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                dev_ptr as *const std::ffi::c_void,
                n * 4,
            );
        }
        let mut nan_count: usize = 0;
        let mut max_abs: f32 = 0.0;
        let mut sum: f64 = 0.0;
        for &v in &host {
            if v.is_nan() { nan_count += 1; }
            let a = v.abs();
            if a.is_finite() && a > max_abs { max_abs = a; }
            if v.is_finite() { sum += v as f64; }
        }
        let first8: Vec<f32> = host.iter().take(8).cloned().collect();
        eprintln!(
            "[csha-dump] {:>9} [f32 n={}]: first8={:?} max|.|={:.6e} sum={:.6e} nan_count={}",
            name, n, first8, max_abs, sum, nan_count
        );
    }

    // Save buffers (inputs to backward kernel — forward populated them).
    dump_f16("q_proj",   q_proj_dev, qkv_elems);
    dump_f16("k_proj",   k_proj_dev, qkv_elems);
    dump_f16("v_proj",   v_proj_dev, qkv_elems);
    dump_f32("row_max",  row_max_dev, stats_elems);
    dump_f32("row_sum",  row_sum_dev, stats_elems);
    dump_f32("x_raw",    x_raw_dev,   x_elems);

    // Backward seed.
    // dO is f16 (attention output gradient), shape [batch, heads, seq, head_dim].
    dump_f16("dO",       d_o_dev,  qkv_elems);

    // Gradient outputs.
    dump_f16("dq",       d_q_dev,  qkv_elems);
    dump_f16("dk",       d_k_dev,  qkv_elems);
    dump_f16("dv",       d_v_dev,  qkv_elems);
    dump_f16("dwq",      d_wq_dev, dw_elems);
    dump_f16("dwk",      d_wk_dev, dw_elems);
    dump_f16("dwv",      d_wv_dev, dw_elems);
    dump_f32("dx",       d_x_dev,  dx_elems);
    dump_f32("dx_norm",  d_xn_dev, dx_elems);
}

/// M42b: Quantized FlashAttention — KV-cache in INT8/FP8, Q in f16/f32.
///
/// Same tiled FlashAttention-2 algorithm but with inline dequantization:
/// each tile load of K/V dequantizes INT8→f32 (using per-head scales from
/// meta_k/meta_v) before the Q@K^T dot product.
///
/// kv_quant_scheme:
///   0 = None (f32 KV, same as nsl_flash_attention)
///   1 = INT8 per-head (scale per attention head)
///   2 = INT8 per-token (scale per token position)
///   3 = INT4 per-group (scale + zero_point per group)
///   4 = FP8 E4M3 (no scale needed, direct cast)
///
/// meta_k/meta_v: pointers to KvBlockQuantMeta arrays (null for FP8/None).
///
/// Returns 0 on success, negative on error.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// Non-CSHA entry: this path has no `segment_ids_ptr` parameter, so the runtime
/// gate is supplied `0` for that slot and always returns `false` — Tier B never
/// fires for non-CSHA configs. The extension is present to keep all 6 FFI entry
/// points uniformly shaped per planner spec §4.6.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_quantized(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    meta_k: i64, meta_v: i64,
    kv_quant_scheme: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_quantized",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    // Non-CSHA path has no segment_ids_ptr parameter; pass 0, gate always returns false.
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            0,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        let _scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut mk = meta_k as u64;
        let mut mv = meta_v as u64;
        let mut qs = kv_quant_scheme as u64;

        let args: [*mut c_void; 16] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut mk as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
        ];

        // For scheme 0 (no quantization), delegate to standard FlashAttention PTX.
        // For scheme 1-4, the quantized PTX kernel handles dequantization inline:
        //   INT8: each tile load does `v_f32 = (int8_val as f32) * scale[head]`
        //   FP8:  each tile load does direct E4M3→f32 cast via LUT
        //   INT4: each tile load unpacks nibbles + applies group scale/zero_point
        let result = crate::cuda::inner::kernel_launch(
            effective_ptx_ptr as *const u8,
            effective_name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (meta_k, meta_v, kv_quant_scheme);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv);
        let _ = (effective_ptx_ptr, effective_name_ptr);
        eprintln!("[nsl] quantized FlashAttention requires CUDA.");
        -1
    }
}

/// RoPE + paged cache write kernel launch wrapper.
///
/// All params i64 for Cranelift ABI compatibility.
/// Grid: (num_tokens, num_heads, ceil(head_dim/2))
#[no_mangle]
pub extern "C" fn nsl_rope_cache_write(
    k_projected_ptr: i64, v_projected_ptr: i64,
    cos_ptr: i64, sin_ptr: i64,
    positions_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_table_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    num_tokens: i64, num_heads: i64, head_dim: i64, block_size: i64,
    ptx_ptr: i64, name_ptr: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let grid_x = num_tokens;
        let grid_y = num_heads;
        let grid_z = (head_dim + 1) / 2;

        let block_x = 1i64;
        let block_y = 1i64;
        let block_z = 1i64;

        let mut kp = k_projected_ptr as u64;
        let mut vp = v_projected_ptr as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut pos = positions_ptr as u64;
        let mut k_pool = k_pool_ptr as u64;
        let mut v_pool = v_pool_ptr as u64;
        let mut bt = block_table_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut nt = num_tokens as u64;
        let mut nh = num_heads as u64;
        let mut hd = head_dim as u64;
        let mut bs = block_size as u64;

        let args: [*mut c_void; 14] = [
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut k_pool as *mut _ as *mut c_void,
            &mut v_pool as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            0,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (k_projected_ptr, v_projected_ptr, cos_ptr, sin_ptr, positions_ptr);
        let _ = (k_pool_ptr, v_pool_ptr, block_table_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (num_tokens, num_heads, head_dim, block_size, ptx_ptr, name_ptr);
        eprintln!("[nsl] rope_cache_write requires CUDA.");
        -1
    }
}

// ── FlashAttention backward pass (CPU reference implementation) ──────────

/// CPU reference implementation of FlashAttention backward.
///
/// Computes dQ, dK, dV from the upstream gradient dO, using the saved Q, K, V
/// tensors and the logsumexp (L) from the forward pass. The attention matrix is
/// recomputed from L (never stored), so memory is O(N) not O(N^2).
///
/// All tensors are [batch, heads, seq_len, head_dim] except logsumexp which is
/// [batch, heads, seq_len]. The output tensor O is [batch, heads, seq_len, head_dim].
///
/// Algorithm per (batch, head):
///   D[i] = sum_d(dO[i,d] * O[i,d])                        -- correction term
///   For each query i, key j:
///     s = sum_d(Q[i,d] * K[j,d]) * scale
///     p = exp(s - L[i])                                     -- recomputed softmax
///     if causal and j > i: p = 0
///     dV[j] += p * dO[i]                                    -- P^T @ dO
///     dp = sum_d(dO[i,d] * V[j,d])                          -- dO @ V^T
///     ds = p * (dp - D[i])                                   -- softmax backward
///     dQ[i] += ds * K[j] * scale
///     dK[j] += ds * Q[i] * scale
/// §4.3 attention sinks (Sprint 1b cycle-7) FFI null-pointer guard.
///
/// Returns `Ok(())` when sinks are disabled (`num_sink_tokens == 0`) OR
/// when both `sink_k_ptr` and `sink_v_ptr` are non-null. Returns
/// `Err(...)` with a precise diagnostic naming which pointer is null
/// when sinks are enabled but one or both pointers were passed as zero.
///
/// **Why this exists**: when codegen emits a sinks-enabled kernel
/// (`num_sink_tokens > 0`), the kernel issues `ld.global.b16
/// [sink_k_ptr + ...]` / `ld.global.b16 [sink_v_ptr + ...]`
/// unconditionally for every thread in the cooperative pre-load. A
/// null pointer there causes `CUDA_ERROR_ILLEGAL_ADDRESS` at the first
/// access — silent-corruption-free, but with a useless device-side
/// diagnostic. Catching the null case host-side gives the user a
/// clear, actionable error pointing at the FFI dispatch.
///
/// **FFI wiring deferred**: the production `nsl_flash_attention` FFI
/// does not yet thread `sink_k_ptr`/`sink_v_ptr` through its signature
/// (the cascade of Cranelift call-site updates exceeds the Sprint 1b
/// 100-LOC budget per the spec's failure-policy guidance). The
/// validator is exposed here so the codegen-side eligibility check can
/// be paired with a host-side null-ptr check at the eventual FFI
/// landing site without refactoring this helper.
pub fn validate_sinks_pointers_nonnull(
    num_sink_tokens: u32,
    sink_k_ptr: i64,
    sink_v_ptr: i64,
) -> Result<(), String> {
    if num_sink_tokens == 0 {
        return Ok(());
    }
    if sink_k_ptr == 0 && sink_v_ptr == 0 {
        return Err(format!(
            "nsl_flash_attention: num_sink_tokens={} > 0 requires non-null sink_k_ptr and sink_v_ptr (got sink_k_ptr=0x0, sink_v_ptr=0x0)",
            num_sink_tokens
        ));
    }
    if sink_k_ptr == 0 {
        return Err(format!(
            "nsl_flash_attention: num_sink_tokens={} > 0 requires non-null sink_k_ptr (got sink_k_ptr=0x0)",
            num_sink_tokens
        ));
    }
    if sink_v_ptr == 0 {
        return Err(format!(
            "nsl_flash_attention: num_sink_tokens={} > 0 requires non-null sink_v_ptr (got sink_v_ptr=0x0)",
            num_sink_tokens
        ));
    }
    Ok(())
}

#[cfg(test)]
mod sinks_validator_tests {
    use super::validate_sinks_pointers_nonnull;

    #[test]
    fn zero_sinks_accepts_both_null() {
        // Sentinel: when sinks are disabled, the validator must accept
        // null pointers (callers will pass 0 for the disabled path).
        assert!(validate_sinks_pointers_nonnull(0, 0, 0).is_ok());
    }

    #[test]
    fn zero_sinks_accepts_arbitrary_ptrs() {
        // Sentinel disabled: pointer values are ignored.
        assert!(validate_sinks_pointers_nonnull(0, 0xdead, 0xbeef).is_ok());
    }

    #[test]
    fn nonzero_sinks_accepts_both_nonnull() {
        assert!(validate_sinks_pointers_nonnull(4, 0xdead, 0xbeef).is_ok());
    }

    #[test]
    fn nonzero_sinks_rejects_both_null_naming_both() {
        let err = validate_sinks_pointers_nonnull(4, 0, 0)
            .expect_err("both null with sinks enabled must reject");
        assert!(
            err.contains("sink_k_ptr") && err.contains("sink_v_ptr"),
            "error must name BOTH missing pointers: {}",
            err
        );
        assert!(err.contains("num_sink_tokens=4"));
    }

    #[test]
    fn nonzero_sinks_rejects_null_k_naming_k() {
        let err = validate_sinks_pointers_nonnull(4, 0, 0xbeef)
            .expect_err("null sink_k_ptr with sinks enabled must reject");
        assert!(
            err.contains("sink_k_ptr"),
            "error must name the missing sink_k_ptr: {}",
            err
        );
        assert!(
            !err.contains("sink_v_ptr"),
            "error must NOT name sink_v_ptr when only k is null: {}",
            err
        );
    }

    #[test]
    fn nonzero_sinks_rejects_null_v_naming_v() {
        let err = validate_sinks_pointers_nonnull(4, 0xdead, 0)
            .expect_err("null sink_v_ptr with sinks enabled must reject");
        assert!(
            err.contains("sink_v_ptr"),
            "error must name the missing sink_v_ptr: {}",
            err
        );
        assert!(
            !err.contains("sink_k_ptr"),
            "error must NOT name sink_k_ptr when only v is null: {}",
            err
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_cpu(
    q: &[f32], k: &[f32], v: &[f32],
    out: &[f32], logsumexp: &[f32], dout: &[f32],
    dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool,
) {
    // Strides for [batch, heads, seq_len, head_dim] layout
    let bh_stride = heads * seq_len * head_dim;
    let h_stride = seq_len * head_dim;
    let s_stride = head_dim;

    // Strides for logsumexp [batch, heads, seq_len]
    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;

    for b in 0..batch {
        for h in 0..heads {
            let qkv_base = b * bh_stride + h * h_stride;
            let lse_base = b * lse_bh_stride + h * lse_h_stride;

            // Phase 1: Compute correction D[i] = rowsum(dO[i] * O[i])
            let mut d_corr = vec![0.0f32; seq_len];
            for (i, d_corr_i) in d_corr.iter_mut().enumerate() {
                let row_base = qkv_base + i * s_stride;
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += dout[row_base + d] * out[row_base + d];
                }
                *d_corr_i = sum;
            }

            // Phase 2: Compute dQ, dK, dV
            for i in 0..seq_len {
                let q_base = qkv_base + i * s_stride;
                let lse_i = logsumexp[lse_base + i];

                let j_max = if causal { i + 1 } else { seq_len };
                for j in 0..j_max {
                    let k_base = qkv_base + j * s_stride;
                    let v_base = qkv_base + j * s_stride;

                    // Recompute S[i,j] = Q[i] . K[j] * scale
                    let mut s_val = 0.0f32;
                    for d in 0..head_dim {
                        s_val += q[q_base + d] * k[k_base + d];
                    }
                    s_val *= scale;

                    // Recompute P[i,j] = exp(S[i,j] - L[i])
                    let p_val = (s_val - lse_i).exp();

                    // dV[j] += p * dO[i]
                    for d in 0..head_dim {
                        dv[k_base + d] += p_val * dout[q_base + d];
                    }

                    // dp = dO[i] . V[j]
                    let mut dp_val = 0.0f32;
                    for d in 0..head_dim {
                        dp_val += dout[q_base + d] * v[v_base + d];
                    }

                    // dS = P * (dP - D[i])
                    let ds_val = p_val * (dp_val - d_corr[i]);

                    // dQ[i] += dS * K[j] * scale
                    // dK[j] += dS * Q[i] * scale
                    for d in 0..head_dim {
                        dq[q_base + d] += ds_val * k[k_base + d] * scale;
                        dk[k_base + d] += ds_val * q[q_base + d] * scale;
                    }
                }
            }
        }
    }
}

/// GQA-aware backward: Q has `heads` heads, K/V have `kv_heads` heads.
/// Each group of `gqa_groups` Q heads shares one KV head.
/// dQ has shape [batch, heads, seq, head_dim].
/// dK, dV have shape [batch, kv_heads, seq, head_dim].
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_cpu_gqa(
    q: &[f32], k: &[f32], v: &[f32],
    out: &[f32], logsumexp: &[f32], dout: &[f32],
    dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    batch: usize, heads: usize, kv_heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool, gqa_groups: usize,
) {
    // Q/dQ/out/dout strides: [batch, heads, seq_len, head_dim]
    let q_bh_stride = heads * seq_len * head_dim;
    let q_h_stride = seq_len * head_dim;
    // K/V/dK/dV strides: [batch, kv_heads, seq_len, head_dim]
    let kv_bh_stride = kv_heads * seq_len * head_dim;
    let kv_h_stride = seq_len * head_dim;
    let s_stride = head_dim;

    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;

    for b_idx in 0..batch {
        for h_idx in 0..heads {
            let kv_h_idx = h_idx / gqa_groups;
            let q_base = b_idx * q_bh_stride + h_idx * q_h_stride;
            let kv_base = b_idx * kv_bh_stride + kv_h_idx * kv_h_stride;
            let lse_base = b_idx * lse_bh_stride + h_idx * lse_h_stride;

            // Phase 1: D[i] = rowsum(dO[i] * O[i])
            let mut d_corr = vec![0.0f32; seq_len];
            for (i, d_corr_i) in d_corr.iter_mut().enumerate() {
                let row = q_base + i * s_stride;
                let mut sum = 0.0f32;
                for d_idx in 0..head_dim {
                    sum += dout[row + d_idx] * out[row + d_idx];
                }
                *d_corr_i = sum;
            }

            // Phase 2: dQ, dK, dV
            for i in 0..seq_len {
                let qi = q_base + i * s_stride;
                let lse_i = logsumexp[lse_base + i];

                let j_max = if causal { i + 1 } else { seq_len };
                for j in 0..j_max {
                    let kj = kv_base + j * s_stride;
                    let vj = kv_base + j * s_stride;

                    // S[i,j] = Q[i] . K[j] * scale
                    let mut s_val = 0.0f32;
                    for d_idx in 0..head_dim {
                        s_val += q[qi + d_idx] * k[kj + d_idx];
                    }
                    s_val *= scale;

                    // P[i,j] = exp(S[i,j] - L[i])
                    let p_val = (s_val - lse_i).exp();

                    // dV[j] += p * dO[i]
                    for d_idx in 0..head_dim {
                        dv[kj + d_idx] += p_val * dout[qi + d_idx];
                    }

                    // dp = dO[i] . V[j]
                    let mut dp_val = 0.0f32;
                    for d_idx in 0..head_dim {
                        dp_val += dout[qi + d_idx] * v[vj + d_idx];
                    }

                    // dS = P * (dP - D[i])
                    let ds_val = p_val * (dp_val - d_corr[i]);

                    // dQ[i] += dS * K[j] * scale
                    // dK[j] += dS * Q[i] * scale
                    for d_idx in 0..head_dim {
                        dq[qi + d_idx] += ds_val * k[kj + d_idx] * scale;
                        dk[kj + d_idx] += ds_val * q[qi + d_idx] * scale;
                    }
                }
            }
        }
    }
}

/// FFI entry point for FlashAttention backward pass (CPU reference).
/// Auto-compute logsumexp from Q, K when the forward didn't save it.
///
/// lse[b,h,i] = log(sum_j(exp(Q[b,h,i,:] . K[b,h,j,:] * scale)))
/// with optional causal masking (j <= i).
/// Auto-compute logsumexp with GQA support.
///
/// Q has `heads` heads, K has `kv_heads` heads. Each Q head group maps to one KV head.
/// lse[b,h,i] = log(sum_j(exp(Q[b,h,i,:] . K[b,h//gqa_groups,j,:] * scale)))
#[allow(clippy::too_many_arguments)]
fn compute_logsumexp_gqa(
    q: &[f32], k: &[f32],
    batch: usize, heads: usize, kv_heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool,
) -> Vec<f32> {
    let q_bh_stride = heads * seq_len * head_dim;
    let q_h_stride = seq_len * head_dim;
    let k_bh_stride = kv_heads * seq_len * head_dim;
    let k_h_stride = seq_len * head_dim;
    let s_stride = head_dim;
    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;
    let gqa_groups = if kv_heads > 0 { heads / kv_heads } else { 1 };

    let total_lse = batch * heads * seq_len;
    let mut lse = vec![0.0f32; total_lse];

    for b_idx in 0..batch {
        for h_idx in 0..heads {
            let kv_h_idx = h_idx / gqa_groups;
            let q_base_bh = b_idx * q_bh_stride + h_idx * q_h_stride;
            let k_base_bh = b_idx * k_bh_stride + kv_h_idx * k_h_stride;
            let lse_base = b_idx * lse_bh_stride + h_idx * lse_h_stride;

            for i in 0..seq_len {
                let q_row = q_base_bh + i * s_stride;
                let j_max = if causal { i + 1 } else { seq_len };

                // Numerically stable logsumexp: max + log(sum(exp(x - max)))
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..j_max {
                    let k_row = k_base_bh + j * s_stride;
                    let mut dot = 0.0f32;
                    for d_idx in 0..head_dim {
                        dot += q[q_row + d_idx] * k[k_row + d_idx];
                    }
                    let score = dot * scale;
                    if score > max_val {
                        max_val = score;
                    }
                }

                let mut sum_exp = 0.0f32;
                for j in 0..j_max {
                    let k_row = k_base_bh + j * s_stride;
                    let mut dot = 0.0f32;
                    for d_idx in 0..head_dim {
                        dot += q[q_row + d_idx] * k[k_row + d_idx];
                    }
                    let score = dot * scale;
                    sum_exp += (score - max_val).exp();
                }

                lse[lse_base + i] = max_val + sum_exp.ln();
            }
        }
    }

    lse
}

/// GPU PTX backward dispatch: launches Phase 1 (D-correction) and Phase 2 (dQ/dK/dV)
/// kernels entirely on GPU. No host-device transfer needed.
///
/// Returns NslList [dQ, dK, dV] as GPU tensors.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn flash_attention_backward_gpu(
    dout_ptr: i64, q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, _logsumexp_ptr: i64,
    scale: f32,
    b: usize, h: usize, s: usize, d: usize,
    is_causal: bool,
    phase1_ptx_ptr: i64, phase1_name_ptr: i64,
    phase2_ptx_ptr: i64, phase2_name_ptr: i64,
) -> i64 {
    use crate::cuda::inner;
    use std::ffi::c_void;

    // Sync before reading tensor data pointers
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);

    let total_qkv = b * h * s * d;

    // Block sizes must match compile-time PTX constants
    let block_q: i64 = 64;
    let block_kv: i64 = 64;

    // ── Allocate D correction vector [b*h*s] on GPU ──
    let d_buf = inner::alloc_managed(b * h * s * 4);

    // ── Allocate dQ on GPU (zero-initialized) ──
    let dq_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dq_data, total_qkv * 4);

    // ── Allocate dK, dV on GPU (zero-initialized) ──
    let dk_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dk_data, total_qkv * 4);
    let dv_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dv_data, total_qkv * 4);

    // ── Phase 1: D-correction vector ──
    // D[bh, i] = sum_d( dO[bh, i, d] * O[bh, i, d] )
    // Grid: (b*h, ceil(s/block_q), 1), Block: (block_q, 1, 1)
    {
        let mut dout_data = dout_t.data as u64;
        let mut out_data = out_t.data as u64;
        let mut d_data = d_buf as u64;
        let mut sl = s as u64;
        let mut hd = d as u64;

        let args: [*mut c_void; 5] = [
            &mut dout_data as *mut _ as *mut c_void,
            &mut out_data as *mut _ as *mut c_void,
            &mut d_data as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let grid = [(b * h) as i64, (s as i64 + block_q - 1) / block_q, 1];
        let block = [block_q, 1, 1];

        let res = inner::kernel_launch(
            phase1_ptx_ptr as *const u8,
            phase1_name_ptr as *const u8,
            grid, block, &args, 0,
        );
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("[flash-bwd] Phase 1 kernel launch failed: {:?}", res);
        }
    }

    // ── Phase 2: Main backward (dQ/dK/dV) ──
    // Grid: (b*h, ceil(s/block_kv), 1), Block: (block_q, 1, 1)
    // Shared memory: compute inline (matches backward_shared_mem_bytes)
    {
        let pad: i64 = 4;
        let hd_padded = d as i64 + pad;
        let tile_bytes = |rows: i64, cols: i64| -> i64 { rows * cols * 4 };
        let shmem = (tile_bytes(block_kv, hd_padded) * 2  // K, V tiles
            + tile_bytes(block_q, hd_padded) * 2           // Q, dO tiles
            + tile_bytes(block_kv, hd_padded) * 2          // dK, dV accumulators
            + tile_bytes(block_q, block_kv)                // S tile
            + block_q * 4                                  // D vector
            + block_q * 4                                  // L (logsumexp) vector
        ) as u32;

        // Compute logsumexp on GPU: reuse D buffer area as scratch for lse
        // For now, pass the logsumexp pointer if available, otherwise auto-compute
        // The Phase 2 kernel expects lse_data; we need to compute it.
        // Since _logsumexp_ptr might be 0, we compute lse on CPU and upload.
        let lse_data = {
            let total_lse = b * h * s;
            let lse_gpu = inner::alloc_managed(total_lse * 4);

            // Read Q and K to CPU to compute logsumexp
            let total_qkv_bytes = total_qkv * 4;
            let mut q_cpu = vec![0.0f32; total_qkv];
            let mut k_cpu = vec![0.0f32; total_qkv];
            inner::memcpy_dtoh(
                q_cpu.as_mut_ptr() as *mut c_void,
                q_t.data as *const c_void,
                total_qkv_bytes,
            );
            inner::memcpy_dtoh(
                k_cpu.as_mut_ptr() as *mut c_void,
                k_t.data as *const c_void,
                total_qkv_bytes,
            );

            let lse_cpu = compute_logsumexp_gqa(&q_cpu, &k_cpu, b, h, h, s, d, scale, is_causal);
            inner::memcpy_htod(
                lse_gpu,
                lse_cpu.as_ptr() as *const c_void,
                total_lse * 4,
            );
            lse_gpu
        };

        let mut dout_data = dout_t.data as u64;
        let mut q_data = q_t.data as u64;
        let mut k_data = k_t.data as u64;
        let mut v_data = v_t.data as u64;
        let mut dq_d = dq_data as u64;
        let mut dk_d = dk_data as u64;
        let mut dv_d = dv_data as u64;
        let mut d_d = d_buf as u64;
        let mut lse_d = lse_data as u64;
        let mut sc = scale;
        let mut sl = s as u64;
        let mut hd = d as u64;

        let args: [*mut c_void; 12] = [
            &mut dout_data as *mut _ as *mut c_void,
            &mut q_data as *mut _ as *mut c_void,
            &mut k_data as *mut _ as *mut c_void,
            &mut v_data as *mut _ as *mut c_void,
            &mut dq_d as *mut _ as *mut c_void,
            &mut dk_d as *mut _ as *mut c_void,
            &mut dv_d as *mut _ as *mut c_void,
            &mut d_d as *mut _ as *mut c_void,
            &mut lse_d as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let grid = [(b * h) as i64, (s as i64 + block_kv - 1) / block_kv, 1];
        let block = [block_q, 1, 1];

        let res = inner::kernel_launch(
            phase2_ptx_ptr as *const u8,
            phase2_name_ptr as *const u8,
            grid, block, &args, shmem,
        );
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("[flash-bwd] Phase 2 kernel launch failed: {:?}", res);
        }

        // Sync after all kernels
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

        // Free logsumexp scratch buffer
        inner::free_managed(lse_data);
    }

    // Free D correction buffer
    inner::free_managed(d_buf);

    // ── Build output NslTensor wrappers for dQ, dK, dV ──
    fn make_gpu_tensor(data: *mut c_void, shape: &[i64], total: usize) -> i64 {
        let ndim = shape.len() as i64;
        let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *shape_ptr.add(i) = s };
        }
        let strides = NslTensor::compute_strides(shape_ptr, ndim);
        let t = Box::new(NslTensor::new(
            data,
            shape_ptr,
            strides,
            ndim,
            total as i64,
            1, // device = GPU
            1, // dtype = f32
            1, // refcount
            0, // flags
        ));
        Box::into_raw(t) as i64
    }

    let shape = [b as i64, h as i64, s as i64, d as i64];
    let dq_ptr = make_gpu_tensor(dq_data, &shape, total_qkv);
    let dk_ptr = make_gpu_tensor(dk_data, &shape, total_qkv);
    let dv_ptr = make_gpu_tensor(dv_data, &shape, total_qkv);

    // Pack into NslList [dQ, dK, dV]
    let list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(list, dq_ptr);
    crate::list::nsl_list_push(list, dk_ptr);
    crate::list::nsl_list_push(list, dv_ptr);
    list
}

///
/// Allocates dQ, dK, dV tensors, runs the CPU backward, and returns them
/// as a tuple of three tensor pointers packed into an NslList.
///
/// This is called from the backward dispatch in autodiff/backward.rs.
/// Falls back to CPU when GPU PTX backward is not available (no PTX pointers
/// or GQA with different Q/KV head counts).
///
/// When `logsumexp_ptr == 0`, the logsumexp is auto-computed from Q, K, scale.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// **Two-phase backward structural note:** this entry uses `phase1_*`/`phase2_*` PTX
/// pairs rather than a single `ptx_ptr`/`name_ptr` pair, so the canonical
/// `(effective_ptx_ptr, effective_name_ptr)` dispatch substitution doesn't apply.
/// Non-CSHA + non-`segment_masked`: the runtime gate would never select Tier-B-on
/// even if a variant were provided. The sentinel assertion is retained for
/// uniformity per planner spec §4.6; the params are accepted but unused.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_backward(
    dout_ptr: i64,
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: i64,
    phase1_ptx_ptr: i64,
    phase1_name_ptr: i64,
    phase2_ptx_ptr: i64,
    phase2_name_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::assert_tier_b_sentinels;

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    // Two-phase backward has no canonical dispatch shape; the runtime gate is
    // structurally inapplicable here (non-CSHA + non-segment_masked). The
    // sentinel assertion still runs to enforce call-site discipline.
    assert_tier_b_sentinels(
        "nsl_flash_attention_backward",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );
    // Suppress "unused" — the params are accepted for FFI uniformity.
    let _ = (tier_b_ptx_ptr, tier_b_name_ptr);
    let scale = f32::from_bits(scale_bits as u32);
    let b = batch as usize;
    let h = heads as usize;
    let s = seq_len as usize;
    let d = head_dim as usize;
    let is_causal = causal != 0;

    // GPU PTX dispatch: if PTX pointers are provided and tensors are on GPU,
    // launch the backward kernels directly on the device (no CPU transfer).
    #[cfg(feature = "cuda")]
    {
        let dout_t = NslTensor::from_ptr(dout_ptr);
        let k_t = NslTensor::from_ptr(k_ptr);
        let kv_h = if k_t.ndim >= 2 {
            unsafe { *k_t.shape.add(1) as usize }
        } else {
            h
        };

        if dout_t.device > 0 && phase1_ptx_ptr != 0 && kv_h == h {
            return flash_attention_backward_gpu(
                dout_ptr, q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr,
                scale, b, h, s, d, is_causal,
                phase1_ptx_ptr, phase1_name_ptr,
                phase2_ptx_ptr, phase2_name_ptr,
            );
        }

        if dout_t.device > 0 {
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (phase1_ptx_ptr, phase1_name_ptr, phase2_ptx_ptr, phase2_name_ptr);
    }

    // Read input tensors
    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);

    // Detect GQA: K/V may have fewer heads than Q.
    // Read actual KV head count from K's shape (dim 1).
    let kv_h = if k_t.ndim >= 2 {
        unsafe { *k_t.shape.add(1) as usize }
    } else {
        h
    };

    let total_qkv = b * h * s * d;
    let total_kv = b * kv_h * s * d;
    let total_lse = b * h * s;

    // Helper to read tensor data as f32 slice (handles both f32 and f64 dtypes).
    // Handles GPU tensors by transferring data to CPU via cudaMemcpy.
    fn read_f32_data(t: &NslTensor, len: usize) -> Vec<f32> {
        let is_gpu = t.device > 0;

        if t.dtype == 1 {
            // f32 tensor
            let mut buf = vec![0.0f32; len];
            if is_gpu {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::memcpy_dtoh(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        t.data as *const std::ffi::c_void,
                        len * 4,
                    );
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("[flash-bwd] WARNING: GPU tensor but CUDA not enabled");
                }
            } else {
                for i in 0..len {
                    buf[i] = unsafe { *t.data_f32().add(i) };
                }
            }
            buf
        } else {
            // f64 -> f32
            if is_gpu {
                // GPU f64 tensors: transfer as f64, then convert
                #[allow(unused_mut)] // `mut` needed only under `cuda` feature
                let mut f64_buf = vec![0.0f64; len];
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::memcpy_dtoh(
                        f64_buf.as_mut_ptr() as *mut std::ffi::c_void,
                        t.data as *const std::ffi::c_void,
                        len * 8,
                    );
                }
                f64_buf.iter().map(|&v| v as f32).collect()
            } else {
                (0..len).map(|i| unsafe { *t.data_f64().add(i) as f32 }).collect()
            }
        }
    }

    let dout_data = read_f32_data(dout_t, total_qkv);
    let q_data = read_f32_data(q_t, total_qkv);
    let k_data = read_f32_data(k_t, total_kv);
    let v_data = read_f32_data(v_t, total_kv);
    let out_data = read_f32_data(out_t, total_qkv);

    // Read or auto-compute logsumexp.
    // When logsumexp_ptr == 0, the forward was decomposed (not fused FlashAttention)
    // and no logsumexp buffer was saved. Compute it from Q, K, scale, causal.
    let gqa_groups = if kv_h > 0 { h / kv_h } else { 1 };
    let lse_data = if logsumexp_ptr != 0 {
        let lse_t = NslTensor::from_ptr(logsumexp_ptr);
        read_f32_data(lse_t, total_lse)
    } else {
        compute_logsumexp_gqa(&q_data, &k_data, b, h, kv_h, s, d, scale, is_causal)
    };

    // Allocate gradient buffers (zero-initialized)
    // dQ has Q's shape [batch, heads, seq, head_dim]
    // dK, dV have KV's shape [batch, kv_heads, seq, head_dim]
    let mut dq_data = vec![0.0f32; total_qkv];
    let mut dk_data = vec![0.0f32; total_kv];
    let mut dv_data = vec![0.0f32; total_kv];

    // Run the CPU backward with GQA support
    flash_attention_backward_cpu_gqa(
        &q_data, &k_data, &v_data,
        &out_data, &lse_data, &dout_data,
        &mut dq_data, &mut dk_data, &mut dv_data,
        b, h, kv_h, s, d,
        scale, is_causal, gqa_groups,
    );

    // Create output tensors (f32 dtype, matching Q shape for dQ, KV shape for dK/dV)
    fn make_tensor(data: &[f32], shape: &[i64]) -> i64 {
        let ndim = shape.len() as i64;
        let total = data.len();
        let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *shape_ptr.add(i) = s };
        }
        let strides = NslTensor::compute_strides(shape_ptr, ndim);
        let data_size = std::mem::size_of_val(data);
        let data_ptr = crate::memory::checked_alloc(data_size) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, total);
        }
        let t = Box::new(NslTensor::new(
            data_ptr as *mut std::ffi::c_void,
            shape_ptr,
            strides,
            ndim,
            total as i64,
            0,
            1,
            1,
            0,
        ));
        Box::into_raw(t) as i64
    }

    let q_shape = [batch, heads, seq_len, head_dim];
    let kv_shape = [batch, kv_h as i64, seq_len, head_dim];
    let mut dq_ptr = make_tensor(&dq_data, &q_shape);
    let mut dk_ptr = make_tensor(&dk_data, &kv_shape);
    let mut dv_ptr = make_tensor(&dv_data, &kv_shape);

    // If inputs were on GPU, transfer gradient tensors to GPU to match device
    let input_device = q_t.device;
    if input_device > 0 {
        dq_ptr = crate::tensor::nsl_tensor_to_device(dq_ptr, input_device as i64);
        dk_ptr = crate::tensor::nsl_tensor_to_device(dk_ptr, input_device as i64);
        dv_ptr = crate::tensor::nsl_tensor_to_device(dv_ptr, input_device as i64);
    }

    // Pack into an NslList [dq, dk, dv]
    let list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(list, dq_ptr);
    crate::list::nsl_list_push(list, dk_ptr);
    crate::list::nsl_list_push(list, dv_ptr);
    list
}

// ── CSHA backward-activation buffer allocator ─────────────────────────────

/// The five HBM buffers that the CSHA forward kernel fills when
/// `save_activations_for_backward = true`.  All fields are device
/// pointers represented as `i64` (matching the rest of the NSL runtime
/// ABI).  A value of `0` indicates allocation failure.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CshaBackwardActivations {
    pub q_proj: i64,
    pub k_proj: i64,
    pub v_proj: i64,
    pub row_max: i64,
    pub row_sum: i64,
    /// Raw (pre-RMSNorm) x copy. Forward RMSNorm writes x_normed back into
    /// `csha_x_ptr` in place; the backward dRMSNorm closed form needs the
    /// pre-norm x, so the forward save path stages a copy here. Layout:
    /// `[batch, heads, seq, head_dim]` f32, row-major (matches the x_ptr
    /// the kernel sees).
    pub x_raw: i64,
}

/// Allocate the 5 HBM buffers forward fills when
/// `csha.save_activations_for_backward = true`. Called by the compiler
/// before the forward launch in training mode. All returned pointers
/// are non-zero on success; a zero pointer indicates allocation failure.
/// Caller must call `nsl_csha_free_backward_activations` to release.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_alloc_backward_activations(
    batch: i64, heads: i64, seq: i64, head_dim: i64,
) -> CshaBackwardActivations {
    let qkv_bytes = batch * heads * seq * head_dim * 2;  // f16 = 2 bytes
    let stats_bytes = batch * heads * seq * 4;           // f32 = 4 bytes
    // x_raw mirrors the kernel-visible x buffer: [batch, heads, seq, head_dim] f32.
    let xraw_bytes = batch * heads * seq * head_dim * 4;
    #[cfg(feature = "cuda")]
    {
        use crate::cuda::inner;
        CshaBackwardActivations {
            q_proj: inner::alloc_device(qkv_bytes as usize) as i64,
            k_proj: inner::alloc_device(qkv_bytes as usize) as i64,
            v_proj: inner::alloc_device(qkv_bytes as usize) as i64,
            row_max: inner::alloc_device(stats_bytes as usize) as i64,
            row_sum: inner::alloc_device(stats_bytes as usize) as i64,
            x_raw: inner::alloc_device(xraw_bytes as usize) as i64,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (qkv_bytes, stats_bytes, xraw_bytes);
        CshaBackwardActivations {
            q_proj: 0, k_proj: 0, v_proj: 0, row_max: 0, row_sum: 0, x_raw: 0,
        }
    }
}

/// Free the 5 HBM buffers allocated by `nsl_csha_alloc_backward_activations`.
/// Safe to call with zero pointers — they are silently skipped.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_free_backward_activations(
    a: CshaBackwardActivations,
) {
    #[cfg(feature = "cuda")]
    {
        use crate::cuda::inner;
        if a.q_proj != 0 { inner::free_managed(a.q_proj as *mut std::ffi::c_void); }
        if a.k_proj != 0 { inner::free_managed(a.k_proj as *mut std::ffi::c_void); }
        if a.v_proj != 0 { inner::free_managed(a.v_proj as *mut std::ffi::c_void); }
        if a.row_max != 0 { inner::free_managed(a.row_max as *mut std::ffi::c_void); }
        if a.row_sum != 0 { inner::free_managed(a.row_sum as *mut std::ffi::c_void); }
        if a.x_raw != 0 { inner::free_managed(a.x_raw as *mut std::ffi::c_void); }
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = a; }
}

/// Gap A: codegen-friendly wrapper around
/// `nsl_csha_alloc_backward_activations` that writes the 6 device pointers
/// into `out_ptr` as a contiguous `[i64; 6]` array.
///
/// Struct-by-value returns are fiddly to emit from Cranelift (sret),
/// so the compiler-side emission uses this i64-array variant instead.
/// Layout matches `CshaBackwardActivations` field order:
///   \[0] q_proj, \[1] k_proj, \[2] v_proj,
///   \[3] row_max, \[4] row_sum, \[5] x_raw
///
/// Returns `0` on success. A zero pointer in any slot indicates allocation
/// failure (matches `nsl_csha_alloc_backward_activations` semantics).
///
/// SAFETY: `out_ptr` must point to at least 48 bytes (6 × i64) of writable
/// memory and be 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_alloc_backward_activations_into(
    batch: i64, heads: i64, seq: i64, head_dim: i64,
    out_ptr: i64,
) -> i64 {
    let a = nsl_csha_alloc_backward_activations(batch, heads, seq, head_dim);
    if out_ptr == 0 {
        // No slot to write into — free immediately to avoid leak and fail.
        nsl_csha_free_backward_activations(a);
        return -1;
    }
    let slots = out_ptr as *mut i64;
    // SAFETY: caller guarantees at least 6 i64 slots.
    slots.add(0).write(a.q_proj);
    slots.add(1).write(a.k_proj);
    slots.add(2).write(a.v_proj);
    slots.add(3).write(a.row_max);
    slots.add(4).write(a.row_sum);
    slots.add(5).write(a.x_raw);
    0
}

/// Gap A: codegen-friendly free variant taking 6 individual i64 pointers
/// (matches the i64-array produced by
/// `nsl_csha_alloc_backward_activations_into`). Avoids passing
/// `CshaBackwardActivations` by-value across the Cranelift ABI boundary.
///
/// Safe to call with zero pointers — they are silently skipped, mirroring
/// `nsl_csha_free_backward_activations`.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_free_backward_activations_from(
    q_proj: i64, k_proj: i64, v_proj: i64,
    row_max: i64, row_sum: i64, x_raw: i64,
) {
    let a = CshaBackwardActivations {
        q_proj, k_proj, v_proj, row_max, row_sum, x_raw,
    };
    nsl_csha_free_backward_activations(a);
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn nsl_csha_alloc_backward_activations_allocates_five_buffers() {
        // Shape: batch=2, heads=4, seq=32, head_dim=32.
        // Expected sizes:
        //   q_proj, k_proj, v_proj: 2*4*32*32 f16 = 8192 bytes each
        //   row_max, row_sum: 2*4*32 f32 = 1024 bytes each
        let r = unsafe { nsl_csha_alloc_backward_activations(2, 4, 32, 32) };
        assert_ne!(r.q_proj, 0, "q_proj alloc failed");
        assert_ne!(r.k_proj, 0, "k_proj alloc failed");
        assert_ne!(r.v_proj, 0, "v_proj alloc failed");
        assert_ne!(r.row_max, 0, "row_max alloc failed");
        assert_ne!(r.row_sum, 0, "row_sum alloc failed");
        assert_ne!(r.x_raw, 0, "x_raw alloc failed");
        unsafe { nsl_csha_free_backward_activations(r); }
    }

    /// Naive attention forward for reference: O = softmax(Q @ K^T * scale) @ V
    fn naive_attention_forward(
        q: &[f32], k: &[f32], v: &[f32],
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        // Returns (output, logsumexp)
        let bh_stride = heads * seq_len * head_dim;
        let h_stride = seq_len * head_dim;
        let s_stride = head_dim;
        let lse_bh_stride = heads * seq_len;
        let lse_h_stride = seq_len;

        let total_qkv = batch * heads * seq_len * head_dim;
        let total_lse = batch * heads * seq_len;
        let mut output = vec![0.0f32; total_qkv];
        let mut logsumexp = vec![0.0f32; total_lse];

        for b in 0..batch {
            for h in 0..heads {
                let qkv_base = b * bh_stride + h * h_stride;
                let lse_base = b * lse_bh_stride + h * lse_h_stride;

                for i in 0..seq_len {
                    let q_base = qkv_base + i * s_stride;

                    // Compute scores S[i,:] = Q[i] . K[:] * scale
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }

                    // Apply causal mask
                    if causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }

                    // Online softmax and logsumexp
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for &s in &scores {
                        sum_exp += (s - max_score).exp();
                    }
                    logsumexp[lse_base + i] = max_score + sum_exp.ln();

                    // Softmax probs
                    let lse = logsumexp[lse_base + i];
                    let probs: Vec<f32> = scores.iter().map(|&s| (s - lse).exp()).collect();

                    // O[i] = sum_j P[i,j] * V[j]
                    for j in 0..seq_len {
                        let v_base = qkv_base + j * s_stride;
                        for d in 0..head_dim {
                            output[q_base + d] += probs[j] * v[v_base + d];
                        }
                    }
                }
            }
        }

        (output, logsumexp)
    }

    /// Naive attention backward for reference (full S^2 computation)
    fn naive_attention_backward(
        q: &[f32], k: &[f32], v: &[f32],
        dout: &[f32],
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let bh_stride = heads * seq_len * head_dim;
        let h_stride = seq_len * head_dim;
        let s_stride = head_dim;
        let total_qkv = batch * heads * seq_len * head_dim;

        let mut dq = vec![0.0f32; total_qkv];
        let mut dk = vec![0.0f32; total_qkv];
        let mut dv = vec![0.0f32; total_qkv];

        for b in 0..batch {
            for h in 0..heads {
                let qkv_base = b * bh_stride + h * h_stride;

                // Compute full attention scores and softmax probs
                let mut probs = vec![vec![0.0f32; seq_len]; seq_len];
                for i in 0..seq_len {
                    let q_base = qkv_base + i * s_stride;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }
                    if causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_exp += *s;
                    }
                    for j in 0..seq_len {
                        probs[i][j] = scores[j] / sum_exp;
                    }
                }

                // dV[j] = sum_i P[i,j] * dO[i]
                for j in 0..seq_len {
                    let v_base = qkv_base + j * s_stride;
                    for i in 0..seq_len {
                        let do_base = qkv_base + i * s_stride;
                        for d in 0..head_dim {
                            dv[v_base + d] += probs[i][j] * dout[do_base + d];
                        }
                    }
                }

                // dP[i,j] = dO[i] . V[j]
                // dS[i,j] = P[i,j] * (dP[i,j] - sum_k(dP[i,k] * P[i,k]))
                //         = P[i,j] * (dP[i,j] - D[i])
                // where D[i] = sum_j P[i,j] * dP[i,j] = sum_j P[i,j] * (dO[i] . V[j])
                //            = dO[i] . (sum_j P[i,j] * V[j]) = dO[i] . O[i]
                for i in 0..seq_len {
                    let do_base = qkv_base + i * s_stride;

                    // D[i] = sum_j P[i,j] * (dO[i] . V[j])
                    let mut d_i = 0.0f32;
                    for j in 0..seq_len {
                        let v_base = qkv_base + j * s_stride;
                        let mut dp_ij = 0.0f32;
                        for d in 0..head_dim {
                            dp_ij += dout[do_base + d] * v[v_base + d];
                        }
                        d_i += probs[i][j] * dp_ij;
                    }

                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let v_base = qkv_base + j * s_stride;

                        // dP[i,j]
                        let mut dp_ij = 0.0f32;
                        for d in 0..head_dim {
                            dp_ij += dout[do_base + d] * v[v_base + d];
                        }

                        // dS[i,j]
                        let ds_ij = probs[i][j] * (dp_ij - d_i);

                        // dQ[i] += dS[i,j] * K[j] * scale
                        let q_base = qkv_base + i * s_stride;
                        for d in 0..head_dim {
                            dq[q_base + d] += ds_ij * k[k_base + d] * scale;
                            dk[k_base + d] += ds_ij * q[q_base + d] * scale;
                        }
                    }
                }
            }
        }

        (dq, dk, dv)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[allow(dead_code)]
    fn rel_error(a: &[f32], b: &[f32]) -> f32 {
        let mut max_rel = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let denom = x.abs().max(y.abs()).max(1e-8);
            let rel = (x - y).abs() / denom;
            if rel > max_rel { max_rel = rel; }
        }
        max_rel
    }

    #[test]
    fn test_flash_backward_matches_naive_non_causal() {
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        // Deterministic test data
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        // Forward pass (naive reference)
        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);

        // Random-ish upstream gradient
        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.7 + 3.0).cos() * 0.3).collect();

        // Naive backward
        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, false,
        );

        // Flash backward
        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, false,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);

        assert!(dq_err < tol, "dQ max abs diff = {dq_err} exceeds tolerance {tol}");
        assert!(dk_err < tol, "dK max abs diff = {dk_err} exceeds tolerance {tol}");
        assert!(dv_err < tol, "dV max abs diff = {dv_err} exceeds tolerance {tol}");
    }

    #[test]
    fn test_flash_backward_matches_naive_causal() {
        let batch = 1;
        let heads = 2;
        let seq_len = 6;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.4).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.4).collect();

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, true);

        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, true,
        );

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, true,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);

        assert!(dq_err < tol, "dQ max abs diff = {dq_err} exceeds tolerance {tol} (causal)");
        assert!(dk_err < tol, "dK max abs diff = {dk_err} exceeds tolerance {tol} (causal)");
        assert!(dv_err < tol, "dV max abs diff = {dv_err} exceeds tolerance {tol} (causal)");
    }

    #[test]
    fn test_flash_backward_batched_multihead() {
        let batch = 2;
        let heads = 3;
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.11 + 0.5).cos() * 0.3).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.19 + 1.5).sin() * 0.3).collect();

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);

        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.29 + 2.5).cos() * 0.2).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, false,
        );

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, false,
        );

        let tol = 1e-4;
        assert!(max_abs_diff(&dq_naive, &dq_flash) < tol, "dQ mismatch in batched test");
        assert!(max_abs_diff(&dk_naive, &dk_flash) < tol, "dK mismatch in batched test");
        assert!(max_abs_diff(&dv_naive, &dv_flash) < tol, "dV mismatch in batched test");
    }

    #[test]
    fn test_flash_backward_causal_gradient_mask() {
        // Verify that for a causal mask, gradients for keys past the query position are zero
        // in the sense that they don't receive "future" contributions.
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        // Use identity-like Q/K to make attention sharp
        let mut q = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        let v: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Make Q[0] only attend to K[0] (with causal, it can't see K[1..3] anyway)
        for d in 0..head_dim {
            q[0 * head_dim + d] = if d == 0 { 5.0 } else { 0.0 };
            k[0 * head_dim + d] = if d == 0 { 5.0 } else { 0.0 };
            // Other positions get different patterns
            for i in 1..seq_len {
                q[i * head_dim + d] = if d == i % head_dim { 5.0 } else { 0.0 };
                k[i * head_dim + d] = if d == i % head_dim { 5.0 } else { 0.0 };
            }
        }

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, true);

        // Gradient only on the first query row
        let mut dout = vec![0.0f32; total];
        for d in 0..head_dim {
            dout[d] = 1.0;
        }

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, true,
        );

        // With causal mask, Q[0] only attends to K[0], so:
        // - dK for positions j>0 should be zero from Q[0]'s gradient
        //   (since dout is only nonzero for row 0 and row 0 can only see j=0)
        // Check dK[1..] are zero
        for j in 1..seq_len {
            for d in 0..head_dim {
                assert!(dk_flash[j * head_dim + d].abs() < 1e-6,
                    "dK[{j},{d}] = {} should be zero for causal mask with dout only at row 0",
                    dk_flash[j * head_dim + d]);
            }
        }
        // dV for positions j>0 should also be zero (P[0,j]=0 for j>0)
        for j in 1..seq_len {
            for d in 0..head_dim {
                assert!(dv_flash[j * head_dim + d].abs() < 1e-6,
                    "dV[{j},{d}] = {} should be zero for causal mask with dout only at row 0",
                    dv_flash[j * head_dim + d]);
            }
        }
    }

    #[test]
    fn test_flash_backward_finite_difference() {
        // Numerical gradient check using finite differences
        let batch = 1;
        let heads = 1;
        let seq_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        let eps = 1e-3f32;

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);
        let dout: Vec<f32> = vec![1.0; total]; // all-ones upstream gradient = computing d(sum(O))/d(param)

        // Analytic gradients
        let mut dq_analytic = vec![0.0f32; total];
        let mut dk_analytic = vec![0.0f32; total];
        let mut dv_analytic = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_analytic, &mut dk_analytic, &mut dv_analytic,
            batch, heads, seq_len, head_dim, scale, false,
        );

        // Finite-difference check for dQ
        for idx in 0..total.min(12) {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[idx] += eps;
            q_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q_plus, &k, &v, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q_minus, &k, &v, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dq_analytic[idx] - fd_grad).abs();
            let denom = dq_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dQ[{idx}] analytic={} fd={} rel_err={}",
                dq_analytic[idx], fd_grad, diff / denom);
        }

        // Finite-difference check for dK
        for idx in 0..total.min(12) {
            let mut k_plus = k.clone();
            let mut k_minus = k.clone();
            k_plus[idx] += eps;
            k_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q, &k_plus, &v, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q, &k_minus, &v, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dk_analytic[idx] - fd_grad).abs();
            let denom = dk_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dK[{idx}] analytic={} fd={} rel_err={}",
                dk_analytic[idx], fd_grad, diff / denom);
        }

        // Finite-difference check for dV
        for idx in 0..total.min(12) {
            let mut v_plus = v.clone();
            let mut v_minus = v.clone();
            v_plus[idx] += eps;
            v_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q, &k, &v_plus, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q, &k, &v_minus, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dv_analytic[idx] - fd_grad).abs();
            let denom = dv_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dV[{idx}] analytic={} fd={} rel_err={}",
                dv_analytic[idx], fd_grad, diff / denom);
        }
    }

    #[test]
    fn test_logsumexp_only_auxiliary_storage() {
        // Verify that the backward pass uses O(N) auxiliary storage (logsumexp)
        // not O(N^2) (full attention matrix). The logsumexp tensor is
        // [batch, heads, seq_len] while the attention matrix would be
        // [batch, heads, seq_len, seq_len].
        let batch = 2;
        let heads = 4;
        let seq_len = 16;
        let _head_dim = 8;
        let total_lse = batch * heads * seq_len;
        let total_attn = batch * heads * seq_len * seq_len;

        // logsumexp is O(N), attention matrix would be O(N^2)
        assert!(total_lse < total_attn,
            "logsumexp ({total_lse}) should be much smaller than full attention ({total_attn})");
        assert_eq!(total_lse, batch * heads * seq_len);
    }

    // ── GQA helpers ─────────────────────────────────────────────────────

    /// Naive GQA-aware forward: Q [b, h_q, s, d], K/V [b, h_kv, s, d]
    /// Each group of h_q/h_kv Q heads shares one KV head.
    fn naive_attention_forward_gqa(
        q: &[f32], k: &[f32], v: &[f32],
        batch: usize, h_q: usize, h_kv: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let groups = h_q / h_kv;
        let q_bh = h_q * seq_len * head_dim;
        let kv_bh = h_kv * seq_len * head_dim;
        let total_q = batch * h_q * seq_len * head_dim;
        let total_lse = batch * h_q * seq_len;
        let mut output = vec![0.0f32; total_q];
        let mut logsumexp = vec![0.0f32; total_lse];

        for b in 0..batch {
            for hq in 0..h_q {
                let hkv = hq / groups;
                let q_base = b * q_bh + hq * seq_len * head_dim;
                let kv_base = b * kv_bh + hkv * seq_len * head_dim;
                let lse_base = b * h_q * seq_len + hq * seq_len;

                for i in 0..seq_len {
                    let qi = q_base + i * head_dim;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim { dot += q[qi + dd] * k[kj + dd]; }
                        scores[j] = dot * scale;
                    }
                    if causal { for j in (i+1)..seq_len { scores[j] = f32::NEG_INFINITY; } }

                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for &s in &scores { sum_exp += (s - max_s).exp(); }
                    logsumexp[lse_base + i] = max_s + sum_exp.ln();
                    let lse = logsumexp[lse_base + i];

                    let probs: Vec<f32> = scores.iter().map(|&s| (s - lse).exp()).collect();
                    for j in 0..seq_len {
                        let vj = kv_base + j * head_dim;
                        for dd in 0..head_dim {
                            output[qi + dd] += probs[j] * v[vj + dd];
                        }
                    }
                }
            }
        }
        (output, logsumexp)
    }

    /// Naive GQA backward reference: returns (dQ [b,h_q,s,d], dK [b,h_kv,s,d], dV [b,h_kv,s,d])
    fn naive_attention_backward_gqa(
        q: &[f32], k: &[f32], v: &[f32], dout: &[f32],
        batch: usize, h_q: usize, h_kv: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let groups = h_q / h_kv;
        let q_bh = h_q * seq_len * head_dim;
        let kv_bh = h_kv * seq_len * head_dim;
        let total_q = batch * h_q * seq_len * head_dim;
        let total_kv = batch * h_kv * seq_len * head_dim;

        let mut dq = vec![0.0f32; total_q];
        let mut dk = vec![0.0f32; total_kv];
        let mut dv = vec![0.0f32; total_kv];

        for b in 0..batch {
            for hq in 0..h_q {
                let hkv = hq / groups;
                let q_base = b * q_bh + hq * seq_len * head_dim;
                let kv_base = b * kv_bh + hkv * seq_len * head_dim;

                // Compute softmax probs
                let mut probs = vec![vec![0.0f32; seq_len]; seq_len];
                for i in 0..seq_len {
                    let qi = q_base + i * head_dim;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim { dot += q[qi + dd] * k[kj + dd]; }
                        scores[j] = dot * scale;
                    }
                    if causal { for j in (i+1)..seq_len { scores[j] = f32::NEG_INFINITY; } }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in &mut scores { *s = (*s - max_s).exp(); sum_exp += *s; }
                    for j in 0..seq_len { probs[i][j] = scores[j] / sum_exp; }
                }

                // dV[j] += sum_i P[i,j] * dO[i]  (accumulated across GQA group)
                for j in 0..seq_len {
                    let vj_idx = kv_base + j * head_dim;
                    for i in 0..seq_len {
                        let doi = q_base + i * head_dim;
                        for dd in 0..head_dim {
                            dv[vj_idx + dd] += probs[i][j] * dout[doi + dd];
                        }
                    }
                }

                // dQ, dK
                for i in 0..seq_len {
                    let doi = q_base + i * head_dim;
                    let mut d_i = 0.0f32;
                    for j in 0..seq_len {
                        let vj = kv_base + j * head_dim;
                        let mut dp_ij = 0.0f32;
                        for dd in 0..head_dim { dp_ij += dout[doi + dd] * v[vj + dd]; }
                        d_i += probs[i][j] * dp_ij;
                    }

                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let vj = kv_base + j * head_dim;
                        let mut dp_ij = 0.0f32;
                        for dd in 0..head_dim { dp_ij += dout[doi + dd] * v[vj + dd]; }
                        let ds_ij = probs[i][j] * (dp_ij - d_i);

                        let qi = q_base + i * head_dim;
                        for dd in 0..head_dim {
                            dq[qi + dd] += ds_ij * k[kj + dd] * scale;
                            dk[kj + dd] += ds_ij * q[qi + dd] * scale;
                        }
                    }
                }
            }
        }
        (dq, dk, dv)
    }

    #[test]
    fn test_flash_attention_backward_gqa_cpu() {
        // GQA: 4 Q heads, 2 KV heads (groups=2)
        let b = 1;
        let h_q = 4;
        let h_kv = 2;
        let s = 16;
        let d = 16;
        let scale = 1.0 / (d as f32).sqrt();
        let groups = h_q / h_kv;

        let total_q = b * h_q * s * d;
        let total_kv = b * h_kv * s * d;

        // Deterministic test data
        let q: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        // GQA forward
        let (out, lse) = naive_attention_forward_gqa(&q, &k, &v, b, h_q, h_kv, s, d, scale, false);

        let dout: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.7 + 3.0).cos() * 0.3).collect();

        // Naive GQA backward (reference)
        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward_gqa(
            &q, &k, &v, &dout, b, h_q, h_kv, s, d, scale, false,
        );

        // Flash GQA backward (under test)
        let mut dq_flash = vec![0.0f32; total_q];
        let mut dk_flash = vec![0.0f32; total_kv];
        let mut dv_flash = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            b, h_q, h_kv, s, d,
            scale, false, groups,
        );

        // Verify shapes implicitly via lengths
        assert_eq!(dq_flash.len(), total_q, "dQ should have Q shape [b, h_q, s, d]");
        assert_eq!(dk_flash.len(), total_kv, "dK should have KV shape [b, h_kv, s, d]");
        assert_eq!(dv_flash.len(), total_kv, "dV should have KV shape [b, h_kv, s, d]");

        // Verify non-zero gradients
        let dq_norm: f32 = dq_flash.iter().map(|x| x * x).sum();
        let dk_norm: f32 = dk_flash.iter().map(|x| x * x).sum();
        let dv_norm: f32 = dv_flash.iter().map(|x| x * x).sum();
        assert!(dq_norm > 1e-8, "dQ should be non-zero, got norm={dq_norm}");
        assert!(dk_norm > 1e-8, "dK should be non-zero, got norm={dk_norm}");
        assert!(dv_norm > 1e-8, "dV should be non-zero, got norm={dv_norm}");

        // Match naive reference
        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);
        assert!(dq_err < tol, "GQA dQ max abs diff = {dq_err} exceeds tol {tol}");
        assert!(dk_err < tol, "GQA dK max abs diff = {dk_err} exceeds tol {tol}");
        assert!(dv_err < tol, "GQA dV max abs diff = {dv_err} exceeds tol {tol}");

        // Cross-check: summing dK from non-GQA (h_q heads, same K for each group)
        // should match the GQA dK. Run a non-GQA backward with K replicated.
        let mut k_expanded = vec![0.0f32; total_q]; // [b, h_q, s, d]
        let mut v_expanded = vec![0.0f32; total_q];
        for bb in 0..b {
            for hq in 0..h_q {
                let hkv = hq / groups;
                for si in 0..s {
                    for dd in 0..d {
                        let q_idx = bb * h_q * s * d + hq * s * d + si * d + dd;
                        let kv_idx = bb * h_kv * s * d + hkv * s * d + si * d + dd;
                        k_expanded[q_idx] = k[kv_idx];
                        v_expanded[q_idx] = v[kv_idx];
                    }
                }
            }
        }

        // Non-GQA backward with expanded K/V
        let (out_exp, lse_exp) = naive_attention_forward(
            &q, &k_expanded, &v_expanded, b, h_q, s, d, scale, false,
        );
        let mut dk_expanded = vec![0.0f32; total_q];
        let mut dq_expanded = vec![0.0f32; total_q];
        let mut dv_expanded = vec![0.0f32; total_q];
        flash_attention_backward_cpu(
            &q, &k_expanded, &v_expanded, &out_exp, &lse_exp, &dout,
            &mut dq_expanded, &mut dk_expanded, &mut dv_expanded,
            b, h_q, s, d, scale, false,
        );

        // Sum dk_expanded across groups to get per-KV-head gradients
        let mut dk_summed = vec![0.0f32; total_kv];
        for bb in 0..b {
            for hkv in 0..h_kv {
                for g in 0..groups {
                    let hq = hkv * groups + g;
                    for si in 0..s {
                        for dd in 0..d {
                            let kv_idx = bb * h_kv * s * d + hkv * s * d + si * d + dd;
                            let q_idx = bb * h_q * s * d + hq * s * d + si * d + dd;
                            dk_summed[kv_idx] += dk_expanded[q_idx];
                        }
                    }
                }
            }
        }

        let dk_cross_err = max_abs_diff(&dk_summed, &dk_flash);
        assert!(dk_cross_err < tol,
            "GQA dK should match sum of expanded non-GQA dK, err={dk_cross_err}");
    }

    #[test]
    fn test_flash_attention_backward_gqa_causal() {
        // GQA with causal mask: 4 Q heads, 2 KV heads
        let b = 1;
        let h_q = 4;
        let h_kv = 2;
        let s = 8;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let groups = h_q / h_kv;

        let total_q = b * h_q * s * d;
        let total_kv = b * h_kv * s * d;

        let q: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.4).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.4).collect();

        let (out, lse) = naive_attention_forward_gqa(&q, &k, &v, b, h_q, h_kv, s, d, scale, true);

        let dout: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward_gqa(
            &q, &k, &v, &dout, b, h_q, h_kv, s, d, scale, true,
        );

        let mut dq_flash = vec![0.0f32; total_q];
        let mut dk_flash = vec![0.0f32; total_kv];
        let mut dv_flash = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            b, h_q, h_kv, s, d,
            scale, true, groups,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);
        assert!(dq_err < tol, "GQA causal dQ err = {dq_err} exceeds tol {tol}");
        assert!(dk_err < tol, "GQA causal dK err = {dk_err} exceeds tol {tol}");
        assert!(dv_err < tol, "GQA causal dV err = {dv_err} exceeds tol {tol}");

        // Causal check: with dout only at position 0, positions j > 0
        // should not receive gradient from Q[0] for any head
        let mut dout_first = vec![0.0f32; total_q];
        for hq in 0..h_q {
            for dd in 0..d {
                dout_first[hq * s * d + dd] = 1.0;
            }
        }

        let mut dk_causal = vec![0.0f32; total_kv];
        let mut dq_causal = vec![0.0f32; total_q];
        let mut dv_causal = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout_first,
            &mut dq_causal, &mut dk_causal, &mut dv_causal,
            b, h_q, h_kv, s, d,
            scale, true, groups,
        );

        // With causal mask and dout only at row 0, K/V positions j>0
        // should get zero gradient contribution from row 0
        for hkv in 0..h_kv {
            for j in 1..s {
                for dd in 0..d {
                    let idx = hkv * s * d + j * d + dd;
                    assert!(dk_causal[idx].abs() < 1e-5,
                        "dK[hkv={hkv},j={j},d={dd}] = {} should be ~0 for causal with dout at row 0",
                        dk_causal[idx]);
                    assert!(dv_causal[idx].abs() < 1e-5,
                        "dV[hkv={hkv},j={j},d={dd}] = {} should be ~0 for causal with dout at row 0",
                        dv_causal[idx]);
                }
            }
        }
    }

    // ── CSHA A.2.5: FFI launch path tests ────────────────────────────

    /// Pre-A.2.5 the `_csha` FFI forwarded to `nsl_flash_attention` and
    /// dropped the nine CSHA extras. Any unit test that could observe
    /// the difference without a GPU was really testing the forwarder's
    /// fallback. Post-A.2.5, the CSHA FFI launches the PTX variant
    /// directly — we can still exercise it on a non-CUDA build: the
    /// function must return a clean error code (-1) rather than
    /// panicking, and must accept the full 30-argument FFI signature
    /// without unused-parameter lints.
    ///
    /// On a CUDA build, the same call with null pointers would skip the
    /// PTX prologue/projection/epilogue scaffolds (runtime null-checks)
    /// and execute the classic Q-from-HBM path, so behaviour is a
    /// strict extension of `nsl_flash_attention`.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a25_csha_ffi_noncuda_returns_error_without_panic() {
        let r = nsl_flash_attention_csha(
            0, 0, 0, 0, 0, 1.0f32.to_bits() as i64,
            1, 1, 16, 64,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0,
            64, 64, 0,
            // CSHA extras (all null / zero — matches current call-site state).
            0, 0, 0, 0, 0, 0,
            1e-5f32.to_bits() as i64,
            0, 0,
            // PCA Tier A: segment_ids_ptr (0 = unpacked path).
            0,
            // PCA Tier B Planner: tier_b sentinel pair (both zero = disabled).
            0, 0,
            // PCA §4.3: doc_starts_ptr (0 = identity positions).
            0,
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero (0 = legacy topology).
            0,
        );
        assert_eq!(r, -1, "non-CUDA build must return -1 for the CSHA FFI");
    }

    /// A.4 grid-sizing logic, extracted as a pure function so it can be
    /// exercised without a GPU. Mirrors the `effective_heads` computation
    /// inside `nsl_flash_attention_csha`'s cuda branch — keep the two
    /// in sync.
    fn a4_effective_heads(heads: i64, active_heads: i64) -> i64 {
        if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        }
    }

    #[test]
    fn a4_grid_y_uses_full_heads_when_active_heads_zero() {
        // active_heads == 0 is the "no pruning" signal — grid_y must
        // use the full head count.
        assert_eq!(a4_effective_heads(8, 0), 8);
    }

    #[test]
    fn a4_grid_y_shrinks_when_active_heads_prunes() {
        // Common weight-informed specialisation: 3 of 8 heads pruned,
        // kernel launches with grid_y = batch * 5.
        assert_eq!(a4_effective_heads(8, 5), 5);
    }

    #[test]
    fn a4_grid_y_falls_back_to_heads_when_active_exceeds() {
        // Defensive: if the plan somehow reports active_heads > heads
        // (shouldn't happen but could during bring-up), use the full
        // count — the kernel's A.4 PTX guard would otherwise eject
        // every block.
        assert_eq!(a4_effective_heads(8, 16), 8);
    }

    #[test]
    fn a4_grid_y_full_heads_when_active_equals_heads() {
        // active_heads == heads is semantically "no pruning expressed
        // differently"; fall back to the full count so we don't emit
        // the guard branch unnecessarily.
        assert_eq!(a4_effective_heads(8, 8), 8);
    }

    /// CSHA paper §5.2 v1 (Sprint 2 cycle-2 audit pin): the canonical
    /// "half the heads pruned" case used by `csha_cuda_launch_fused`'s
    /// h=8/active=4 fixture. Co-pinned alongside the in-source doc on
    /// `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs
    /// ::emit_active_heads_guard` so the launcher and kernel guard
    /// cannot silently disagree.
    #[test]
    fn a4_grid_y_half_pruned_canonical_case() {
        assert_eq!(a4_effective_heads(8, 4), 4);
        // Also pin the table of paper §5.2 v1 inputs we ship today.
        for (heads, active, expected) in [
            (4i64, 0i64, 4i64),  // sentinel zero -> full
            (4,    1,    1),      // single live head (paper §5.2 extreme)
            (4,    2,    2),
            (4,    4,    4),      // active == heads -> full
            (8,    0,    8),
            (8,    4,    4),      // canonical half-pruned (this test fixture)
            (8,    8,    8),
        ] {
            assert_eq!(
                a4_effective_heads(heads, active),
                expected,
                "effective_heads({heads}, {active}) must be {expected}"
            );
        }
    }

    /// Smoke test: the CSHA FFI symbol resolves and the parameter signature
    /// is wired correctly through the `extern "C"` ABI. This catches A.2.5
    /// signature regressions (e.g. an accidentally dropped extras arg) at
    /// link time — the forwarder version would have silently ignored
    /// wrong-count calls.
    /// Updated in PCA Tier A Task 3C: trailing segment_ids_ptr added.
    /// Updated in PCA Tier B Planner P-3.4: two trailing tier_b_* params added.
    /// Updated in PCA §4.3 Task 3: trailing doc_starts_ptr added.
    /// Updated in PCA per-doc CTA Strategy 3 v1: trailing num_docs_or_zero added.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a25_csha_ffi_signature_has_thirty_params() {
        let _: extern "C" fn(
            i64, i64, i64, i64, i64, i64,
            i64, i64, i64, i64,
            i64, i64, i64, i64,
            i64, i64, i64, i64,
            i64, i64, i64,
            i64, i64, i64,
            // extras
            i64, i64, i64, i64, i64, i64,
            i64, i64, i64,
            // PCA Tier A: segment_ids_ptr
            i64,
            // PCA Tier B Planner: tier_b_ptx_ptr, tier_b_name_ptr
            i64, i64,
            // PCA §4.3: doc_starts_ptr
            i64,
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero
            i64,
        ) -> i64 = nsl_flash_attention_csha;
    }
}
